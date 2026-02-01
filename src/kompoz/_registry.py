"""Expression parser and registry for config-driven rules."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Generic

from kompoz._caching import CachedPredicate
from kompoz._core import Combinator, _And, _Or, _Not, _Then, _IfThenElse
from kompoz._predicate import Predicate, PredicateFactory
from kompoz._retry import Retry
from kompoz._transform import Transform, TransformFactory
from kompoz._types import T, _cache_store
from kompoz._utility import Always, Never


# =============================================================================
# Registry & Expression Parser
# =============================================================================


class ExpressionParser:
    """
    Parser for human-readable rule expressions.

    Supports two equivalent syntaxes:
        Symbol style:  is_admin & ~is_banned & account_older_than(30)
        Word style:    is_admin AND NOT is_banned AND account_older_than(30)

    Operators (by precedence, lowest to highest):
        IF/THEN/ELSE, ? : - Conditional branching (lowest precedence)
        |, OR       - Any must pass
        >>, THEN    - Sequence (run both)
        &, AND      - All must pass
        ~, NOT, !   - Invert result
        :modifier   - Apply modifier (highest precedence)

    Grouping:
        ( )         - Override precedence

    Rules:
        rule_name                   - Simple rule
        rule_name(arg)              - Rule with one argument
        rule_name(arg1, arg2)       - Rule with multiple arguments

    Conditional (if/else):
        IF condition THEN action ELSE alternative
        condition ? action : alternative

    Modifiers (postfix syntax):
        rule:retry(n)               - Retry up to n times
        rule:retry(n, backoff)      - With backoff delay in seconds
        rule:retry(n, backoff, true)  - Exponential backoff
        rule:retry(n, backoff, true, jitter)  - With jitter
        rule:cached                 - Cache results within use_cache() scope
        (expr):modifier             - Apply to grouped expression
        rule:mod1:mod2              - Chain multiple modifiers

    Multi-line expressions are supported (newlines are ignored).

    Examples:
        is_admin & is_active
        is_admin AND is_active
        is_admin | is_premium
        is_admin OR is_premium
        ~is_banned
        NOT is_banned
        is_admin & (is_active | is_premium)
        account_older_than(30) & credit_above(700)

        # Conditional
        IF is_premium THEN apply_discount ELSE charge_full
        is_premium ? apply_discount : charge_full

        # Modifiers
        fetch_data:retry(3)
        fetch_data:retry(5, 1.0, true)
        expensive_check:cached
        (primary | fallback):retry(3)

        # Multi-line
        is_admin
        & ~is_banned
        & account_older_than(30)
    """

    # Token types
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    THEN = "THEN"  # >> operator and THEN keyword
    IF = "IF"
    ELSE = "ELSE"
    QUESTION = "QUESTION"  # ?
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    COMMA = "COMMA"
    COLON = "COLON"  # For modifier syntax (:retry, :cached) and ternary
    IDENT = "IDENT"
    NUMBER = "NUMBER"
    STRING = "STRING"
    BOOL = "BOOL"  # For true/false literals
    EOF = "EOF"

    # Reserved modifier keywords
    MODIFIERS = {"retry", "cached"}

    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.tokens: list[tuple[str, Any]] = []
        self.token_pos = 0
        self._tokenize()

    def _tokenize(self) -> None:
        """Convert text into tokens."""
        while self.pos < len(self.text):
            ch = self.text[self.pos]

            # Skip whitespace and newlines
            if ch in " \t\n\r":
                self.pos += 1
                continue

            # Skip comments
            if ch == "#":
                while self.pos < len(self.text) and self.text[self.pos] != "\n":
                    self.pos += 1
                continue

            # Operators
            if ch == "&":
                self.tokens.append((self.AND, "&"))
                self.pos += 1
            elif ch == "|":
                self.tokens.append((self.OR, "|"))
                self.pos += 1
            elif (
                ch == ">"
                and self.pos + 1 < len(self.text)
                and self.text[self.pos + 1] == ">"
            ):
                self.tokens.append((self.THEN, ">>"))
                self.pos += 2
            elif ch in "~!":
                self.tokens.append((self.NOT, ch))
                self.pos += 1
            elif ch == "?":
                self.tokens.append((self.QUESTION, "?"))
                self.pos += 1
            elif ch == "(":
                self.tokens.append((self.LPAREN, "("))
                self.pos += 1
            elif ch == ")":
                self.tokens.append((self.RPAREN, ")"))
                self.pos += 1
            elif ch == ",":
                self.tokens.append((self.COMMA, ","))
                self.pos += 1
            elif ch == ":":
                self.tokens.append((self.COLON, ":"))
                self.pos += 1

            # Strings
            elif ch in "\"'":
                self.tokens.append((self.STRING, self._read_string(ch)))

            # Numbers
            elif ch.isdigit() or (
                ch == "-"
                and self.pos + 1 < len(self.text)
                and self.text[self.pos + 1].isdigit()
            ):
                self.tokens.append((self.NUMBER, self._read_number()))

            # Identifiers and keywords
            elif ch.isalpha() or ch == "_":
                ident = self._read_ident()
                upper = ident.upper()
                lower = ident.lower()
                if upper == "AND":
                    self.tokens.append((self.AND, ident))
                elif upper == "OR":
                    self.tokens.append((self.OR, ident))
                elif upper == "NOT":
                    self.tokens.append((self.NOT, ident))
                elif upper == "IF":
                    self.tokens.append((self.IF, ident))
                elif upper == "THEN":
                    # Same token as >> operator - context determines meaning
                    self.tokens.append((self.THEN, ident))
                elif upper == "ELSE":
                    self.tokens.append((self.ELSE, ident))
                elif lower in ("true", "false"):
                    self.tokens.append((self.BOOL, lower == "true"))
                else:
                    self.tokens.append((self.IDENT, ident))

            else:
                raise ValueError(f"Unexpected character: {ch!r} at position {self.pos}")

        self.tokens.append((self.EOF, None))

    def _read_string(self, quote: str) -> str:
        """Read a quoted string with escape sequence processing."""
        self.pos += 1  # skip opening quote
        result = []
        while self.pos < len(self.text) and self.text[self.pos] != quote:
            if self.text[self.pos] == "\\" and self.pos + 1 < len(self.text):
                next_ch = self.text[self.pos + 1]
                # Handle common escape sequences
                escape_map = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\"}
                if next_ch in escape_map:
                    result.append(escape_map[next_ch])
                else:
                    # For \' or \" or any other, just use the escaped char
                    result.append(next_ch)
                self.pos += 2
            else:
                result.append(self.text[self.pos])
                self.pos += 1
        if self.pos >= len(self.text):
            raise ValueError("Unterminated string literal")
        self.pos += 1  # skip closing quote
        return "".join(result)

    def _read_number(self) -> int | float:
        """Read a number."""
        start = self.pos
        if self.text[self.pos] == "-":
            self.pos += 1
        while self.pos < len(self.text) and (
            self.text[self.pos].isdigit() or self.text[self.pos] == "."
        ):
            self.pos += 1
        text = self.text[start : self.pos]
        return float(text) if "." in text else int(text)

    def _read_ident(self) -> str:
        """Read an identifier."""
        start = self.pos
        while self.pos < len(self.text) and (
            self.text[self.pos].isalnum() or self.text[self.pos] == "_"
        ):
            self.pos += 1
        return self.text[start : self.pos]

    def _peek(self) -> tuple[str, Any]:
        """Look at current token without consuming."""
        return self.tokens[self.token_pos]

    def _consume(self) -> tuple[str, Any]:
        """Consume and return current token."""
        token = self.tokens[self.token_pos]
        self.token_pos += 1
        return token

    def _expect(self, token_type: str) -> tuple[str, Any]:
        """Consume token and verify its type."""
        token = self._consume()
        if token[0] != token_type:
            raise ValueError(f"Expected {token_type}, got {token[0]}")
        return token

    def parse(self) -> dict | str:
        """
        Parse expression and return config dict.

        Grammar:
            expr      = if_expr
            if_expr   = 'IF' or_expr 'THEN' or_expr 'ELSE' or_expr
                      | or_expr ('?' or_expr ':' or_expr)?
            or_expr   = then_expr (('|' | 'OR') then_expr)*
            then_expr = and_expr (('>>' | 'THEN') and_expr)*
            and_expr  = not_expr (('&' | 'AND') not_expr)*
            not_expr  = ('~' | 'NOT' | '!')? postfix
            postfix   = primary (':' MODIFIER args?)*
            primary   = IDENT args? | '(' expr ')'
            args      = '(' arg_list? ')'
            arg_list  = arg (',' arg)*
            arg       = NUMBER | STRING | IDENT | BOOL

        Conditionals:
            IF condition THEN then_branch ELSE else_branch
            condition ? then_branch : else_branch

        Modifiers:
            :retry(max_attempts, [backoff], [exponential], [jitter])
            :cached
        """
        result = self._parse_if()
        if self._peek()[0] != self.EOF:
            raise ValueError(f"Unexpected token: {self._peek()}")
        return result

    def _parse_if(self) -> dict | str:
        """Parse IF/THEN/ELSE or ternary expression (lowest precedence)."""
        # Check for keyword IF
        if self._peek()[0] == self.IF:
            self._consume()  # consume IF
            # Parse condition allowing OR but not THEN sequences
            # This means IF a >> b THEN c ELSE d needs parentheses: IF (a >> b) THEN c ELSE d
            condition = self._parse_or_no_then()
            self._expect(self.THEN)  # expect THEN keyword
            then_branch = self._parse_if()  # Allow nested if/ternary in branches
            self._expect(self.ELSE)  # expect ELSE
            else_branch = self._parse_if()  # Allow nested if/ternary in branches
            return {"if": {"cond": condition, "then": then_branch, "else": else_branch}}

        # Otherwise parse or_expr and check for ternary ?:
        condition = self._parse_or()

        if self._peek()[0] == self.QUESTION:
            self._consume()  # consume ?
            then_branch = self._parse_if()  # Allow nested if/ternary
            self._expect(self.COLON)  # expect :
            else_branch = self._parse_if()  # Allow nested if/ternary
            return {"if": {"cond": condition, "then": then_branch, "else": else_branch}}

        return condition

    def _parse_or_no_then(self) -> dict | str:
        """Parse OR expression without THEN sequences (for IF conditions)."""
        left = self._parse_and()

        items = [left]
        while self._peek()[0] == self.OR:
            self._consume()
            items.append(self._parse_and())

        if len(items) == 1:
            return items[0]
        return {"or": items}

    def _parse_or(self) -> dict | str:
        """Parse OR expression."""
        left = self._parse_then()

        items = [left]
        while self._peek()[0] == self.OR:
            self._consume()
            items.append(self._parse_then())

        if len(items) == 1:
            return items[0]
        return {"or": items}

    def _parse_then(self) -> dict | str:
        """Parse THEN expression (sequence, runs both)."""
        left = self._parse_and()

        items = [left]
        while self._peek()[0] == self.THEN:
            self._consume()
            items.append(self._parse_and())

        if len(items) == 1:
            return items[0]
        return {"then": items}

    def _parse_and(self) -> dict | str:
        """Parse AND expression."""
        left = self._parse_not()

        items = [left]
        while self._peek()[0] == self.AND:
            self._consume()
            items.append(self._parse_not())

        if len(items) == 1:
            return items[0]
        return {"and": items}

    def _parse_not(self) -> dict | str:
        """Parse NOT expression (highest precedence)."""
        if self._peek()[0] == self.NOT:
            self._consume()
            inner = self._parse_not()  # Allow chained NOT
            return {"not": inner}
        return self._parse_postfix()

    def _parse_postfix(self) -> dict | str:
        """Parse primary with optional :modifier suffixes."""
        result = self._parse_primary()

        # Check for chained modifiers like :retry(3):cached
        while self._peek()[0] == self.COLON:
            # Peek ahead - if this is part of ternary ?: we should stop
            # Save position in case we need to backtrack
            saved_pos = self.token_pos
            self._consume()  # consume ':'

            # Check if this looks like a modifier (IDENT)
            if self._peek()[0] != self.IDENT:
                # Not a modifier, backtrack (this is probably ternary :)
                self.token_pos = saved_pos
                break

            modifier_name = self._peek()[1]
            modifier = modifier_name.lower()

            # Peek at what comes after the identifier
            next_pos = self.token_pos + 1
            next_token = (
                self.tokens[next_pos][0] if next_pos < len(self.tokens) else self.EOF
            )

            if modifier not in self.MODIFIERS:
                # Only raise error if it looks like a modifier call (has parens)
                # This allows ternary `a ? b : c` to work (c is not followed by parens)
                if next_token == self.LPAREN:
                    raise ValueError(
                        f"Unknown modifier '{modifier_name}'. Valid modifiers: {', '.join(sorted(self.MODIFIERS))}"
                    )
                else:
                    # Could be ternary colon, backtrack
                    self.token_pos = saved_pos
                    break

            # It's a modifier, consume and process
            self._consume()  # consume modifier name

            # Check for optional arguments
            args: list[Any] = []
            if self._peek()[0] == self.LPAREN:
                self._consume()  # (
                args = self._parse_args()
                self._expect(self.RPAREN)  # )

            # Wrap result with modifier
            if modifier == "retry":
                result = {"retry": {"inner": result, "args": args}}
            elif modifier == "cached":
                result = {"cached": result}

        return result

    def _parse_primary(self) -> dict | str:
        """Parse primary expression (identifier or grouped expr)."""
        token = self._peek()

        if token[0] == self.LPAREN:
            self._consume()
            expr = self._parse_if()  # Allow full expressions including if/then/else
            self._expect(self.RPAREN)
            return expr

        if token[0] == self.IDENT:
            name = self._consume()[1]

            # Check for arguments
            if self._peek()[0] == self.LPAREN:
                self._consume()  # (
                args = self._parse_args()
                self._expect(self.RPAREN)
                return {name: args}

            return name

        raise ValueError(f"Unexpected token: {token}")

    def _parse_args(self) -> list:
        """Parse argument list."""
        args = []

        if self._peek()[0] == self.RPAREN:
            return args

        args.append(self._parse_arg())

        while self._peek()[0] == self.COMMA:
            self._consume()
            args.append(self._parse_arg())

        return args

    def _parse_arg(self) -> Any:
        """Parse single argument."""
        token = self._peek()

        if token[0] == self.NUMBER:
            return self._consume()[1]
        if token[0] == self.STRING:
            return self._consume()[1]
        if token[0] == self.BOOL:
            return self._consume()[1]
        if token[0] == self.IDENT:
            # Treat bare identifiers as strings
            return self._consume()[1]

        raise ValueError(f"Invalid argument: {token}")


def parse_expression(text: str) -> dict | str:
    """
    Parse a rule expression into a config structure.

    Args:
        text: Expression string like "is_admin & ~is_banned"

    Returns:
        Config dict compatible with Registry.load()

    Example:
        >>> parse_expression("is_admin & ~is_banned")
        {'and': ['is_admin', {'not': 'is_banned'}]}
    """
    return ExpressionParser(text).parse()


class _CachedCombinatorWrapper(Combinator[T]):
    """
    Internal wrapper to cache any combinator's result.

    Used by Registry when :cached modifier is applied to non-Predicate combinators.
    The cache is keyed by object id and is shared across all instances.
    """

    _cache: dict[int, tuple[bool, Any]] = {}

    def __init__(self, inner: Combinator[T]):
        self.inner = inner

    def _execute(self, ctx: T) -> tuple[bool, T]:
        # Check if caching is enabled via use_cache()
        cache = _cache_store.get()
        if cache is not None:
            key = f"_wrapped:{id(self.inner)}:{id(ctx)}"
            if key in cache:
                return cache[key]
            result = self.inner._execute(ctx)
            cache[key] = result
            return result

        # No cache scope, just execute
        return self.inner._execute(ctx)

    def __repr__(self) -> str:
        return f"Cached({self.inner!r})"


class Registry(Generic[T]):
    """
    Registry for named predicates and transforms.

    Allows loading combinator chains from human-readable expressions.

    Example:
        reg: Registry[User] = Registry()

        @reg.predicate
        def is_admin(u: User) -> bool:
            return u.is_admin

        @reg.predicate
        def older_than(u: User, days: int) -> bool:
            return u.age > days

        # Load from expression
        loaded = reg.load("is_admin & older_than(30)")

        # Also supports word syntax
        loaded = reg.load("is_admin AND older_than(30)")
    """

    def __init__(self) -> None:
        self._predicates: dict[str, Predicate[T] | PredicateFactory[T]] = {}
        self._transforms: dict[str, Transform[T] | TransformFactory[T]] = {}

    def predicate(self, fn: Callable[..., bool]) -> Predicate[T] | PredicateFactory[T]:
        """
        Decorator to register a predicate.

        For simple predicates (single arg), returns Predicate[T].
        For parameterized predicates (multiple args), returns PredicateFactory[T].
        """
        params = list(inspect.signature(fn).parameters)

        if len(params) == 1:
            p: Predicate[T] = Predicate(fn, fn.__name__)
            self._predicates[fn.__name__] = p
            return p
        else:
            factory: PredicateFactory[T] = PredicateFactory(fn, fn.__name__)
            self._predicates[fn.__name__] = factory
            return factory

    def transform(self, fn: Callable[..., T]) -> Transform[T] | TransformFactory[T]:
        """
        Decorator to register a transform.

        For simple transforms (single arg), returns Transform[T].
        For parameterized transforms (multiple args), returns TransformFactory[T].
        """
        params = list(inspect.signature(fn).parameters)

        if len(params) == 1:
            t: Transform[T] = Transform(fn, fn.__name__)
            self._transforms[fn.__name__] = t
            return t
        else:
            factory: TransformFactory[T] = TransformFactory(fn, fn.__name__)
            self._transforms[fn.__name__] = factory
            return factory

    def load(self, expr: str) -> Combinator[T]:
        """
        Load a combinator chain from a human-readable expression.

        Expression format:
            # Simple rules
            is_admin
            is_active

            # Operators (symbols or words)
            is_admin & is_active          # AND
            is_admin AND is_active        # AND (same as &)
            is_admin | is_premium         # OR
            is_admin OR is_premium        # OR (same as |)
            ~is_banned                    # NOT
            NOT is_banned                 # NOT (same as ~)
            !is_banned                    # NOT (same as ~)

            # Grouping with parentheses
            is_admin | (is_active & ~is_banned)

            # Parameterized rules
            account_older_than(30)
            credit_above(700)
            in_role("admin", "moderator")

            # Conditional (if/then/else)
            IF is_premium THEN apply_discount ELSE charge_full
            is_premium ? apply_discount : charge_full

            # Modifiers (postfix syntax)
            fetch_data:retry(3)                   # Retry up to 3 times
            fetch_data:retry(3, 1.0)              # With 1s backoff
            fetch_data:retry(3, 1.0, true)        # Exponential backoff
            fetch_data:retry(3, 1.0, true, 0.1)   # With jitter
            expensive_check:cached                # Cache results
            (fetch_a | fetch_b):retry(5)          # Retry grouped expr
            slow_query:cached:retry(3)            # Chain modifiers

            # Multi-line (newlines are ignored)
            is_admin
            & is_active
            & ~is_banned
            & account_older_than(30)

            # Comments
            is_admin  # must be admin
            & ~is_banned  # and not banned

        Operator precedence (lowest to highest):
            OR  (|)  - evaluated last
            AND (&)  - evaluated second
            NOT (~)  - evaluated first
            Modifiers (:) - highest, binds to immediate left
        """
        config = parse_expression(expr)
        return self._build(config)

    def load_file(self, path: str) -> Combinator[T]:
        """Load combinator chain from an expression file."""
        from pathlib import Path

        content = Path(path).read_text()
        return self.load(content)

    def _build(self, node: dict | str) -> Combinator[T]:
        """Build a combinator from a parsed expression config."""
        # String: simple predicate or transform reference
        if isinstance(node, str):
            return self._resolve(node)

        # Dict: operator or parameterized call
        if isinstance(node, dict):
            if len(node) != 1:
                raise ValueError(f"Config node must have exactly one key: {node}")

            key, value = next(iter(node.items()))

            # Operators
            if key == "and":
                items = [self._build(item) for item in value]
                return self._combine_and(items)

            elif key == "or":
                items = [self._build(item) for item in value]
                return self._combine_or(items)

            elif key == "not":
                return ~self._build(value)

            elif key == "then":
                items = [self._build(item) for item in value]
                return self._combine_seq(items)

            elif key == "if":
                # If/then/else conditional
                condition = self._build(value["cond"])
                then_branch = self._build(value["then"])
                else_branch = self._build(value["else"])
                return _IfThenElse(condition, then_branch, else_branch)

            # Modifier: retry
            elif key == "retry":
                inner = self._build(value["inner"])
                args = value.get("args", [])

                # Parse retry args: max_attempts, [backoff], [exponential], [jitter]
                max_attempts = int(args[0]) if len(args) > 0 else 3
                backoff = float(args[1]) if len(args) > 1 else 0.0
                exponential = bool(args[2]) if len(args) > 2 else False
                jitter = float(args[3]) if len(args) > 3 else 0.0

                return Retry(
                    inner,
                    max_attempts=max_attempts,
                    backoff=backoff,
                    exponential=exponential,
                    jitter=jitter,
                )

            # Modifier: cached
            elif key == "cached":
                inner = self._build(value)
                # Wrap in CachedPredicate if it's a predicate
                if isinstance(inner, Predicate):
                    return CachedPredicate(inner.fn, inner.name)
                else:
                    # For non-predicates, wrap with a generic caching combinator
                    return _CachedCombinatorWrapper(inner)

            # Parameterized predicate/transform
            else:
                return self._resolve(key, value)

        raise ValueError(f"Invalid config node: {node}")

    def _resolve(self, name: str, args: Any = None) -> Combinator[T]:
        """Resolve a name to a predicate or transform, optionally with args."""
        # Check predicates first, then transforms
        if name in self._predicates:
            pred_or_factory = self._predicates[name]
            if args is None:
                if isinstance(pred_or_factory, PredicateFactory):
                    raise ValueError(f"Predicate '{name}' requires arguments")
                return pred_or_factory
            else:
                if isinstance(pred_or_factory, Predicate):
                    raise ValueError(f"Predicate '{name}' does not take arguments")
                if isinstance(args, list):
                    return pred_or_factory(*args)
                elif isinstance(args, dict):
                    return pred_or_factory(**args)
                else:
                    return pred_or_factory(args)

        elif name in self._transforms:
            trans_or_factory = self._transforms[name]
            if args is None:
                if isinstance(trans_or_factory, TransformFactory):
                    raise ValueError(f"Transform '{name}' requires arguments")
                return trans_or_factory
            else:
                if isinstance(trans_or_factory, Transform):
                    raise ValueError(f"Transform '{name}' does not take arguments")
                if isinstance(args, list):
                    return trans_or_factory(*args)
                elif isinstance(args, dict):
                    return trans_or_factory(**args)
                else:
                    return trans_or_factory(args)

        raise ValueError(f"Unknown predicate or transform: '{name}'")

    def _combine_and(self, items: list[Combinator[T]]) -> Combinator[T]:
        if not items:
            return Always()
        result = items[0]
        for item in items[1:]:
            result = result & item
        return result

    def _combine_or(self, items: list[Combinator[T]]) -> Combinator[T]:
        if not items:
            return Never()
        result = items[0]
        for item in items[1:]:
            result = result | item
        return result

    def _combine_seq(self, items: list[Combinator[T]]) -> Combinator[T]:
        if not items:
            return Always()
        result = items[0]
        for item in items[1:]:
            result = result >> item
        return result
