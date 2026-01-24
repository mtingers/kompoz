"""
Kompoz - Composable Predicate & Transform Combinators

A Python library for building composable, declarative rule chains using
operator overloading. Supports boolean logic (AND, OR, NOT), data pipelines,
and config-driven rules via custom DSL.

Operators:
    &  = "and then" (sequence, short-circuits on failure)
    |  = "or else" (fallback, short-circuits on success)
    ~  = "not" / "inverse"
    >> = "then" (always runs both, keeps second result)

Example:
    from kompoz import rule, rule_args

    @rule
    def is_admin(user):
        return user.is_admin

    @rule
    def is_active(user):
        return user.is_active

    @rule_args
    def account_older_than(user, days):
        return user.account_age_days > days

    # Combine with operators
    can_access = is_admin | (is_active & account_older_than(30))

    # Use it
    ok, _ = can_access.run(user)
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Your Name"
__all__ = [
    "Combinator",
    "Predicate",
    "PredicateFactory",
    "Transform",
    "TransformFactory",
    "Try",
    "Always",
    "Never",
    "Debug",
    "Registry",
    "parse_expression",
    "ExpressionParser",
    "rule",
    "rule_args",
    "pipe",
    "pipe_args",
    # Aliases for backwards compatibility
    "predicate",
    "predicate_factory",
    "transform",
    "transform_factory",
]

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")


# =============================================================================
# Core Combinator Base
# =============================================================================


class Combinator(ABC, Generic[T]):
    """
    Base class for all combinators.

    A combinator takes a context and returns (success: bool, new_context).
    Combinators can be composed using operators:
        &  = and then (short-circuits on failure)
        |  = or else (short-circuits on success)
        ~  = not (inverts success/failure)
        >> = then (always runs both)
    """

    @abstractmethod
    def run(self, ctx: T) -> tuple[bool, T]:
        """Execute the combinator and return (success, new_context)."""
        ...

    def __and__(self, other: Combinator[T]) -> Combinator[T]:
        """a & b = run b only if a succeeds."""
        return _And(self, other)

    def __or__(self, other: Combinator[T]) -> Combinator[T]:
        """a | b = run b only if a fails."""
        return _Or(self, other)

    def __invert__(self) -> Combinator[T]:
        """~a = invert success/failure."""
        return _Not(self)

    def __rshift__(self, other: Combinator[T]) -> Combinator[T]:
        """a >> b = run b regardless of a's result (keep b's result)."""
        return _Then(self, other)

    def __call__(self, ctx: T) -> tuple[bool, T]:
        """Shorthand for run()."""
        return self.run(ctx)


@dataclass
class _And(Combinator[T]):
    left: Combinator[T]
    right: Combinator[T]

    def run(self, ctx: T) -> tuple[bool, T]:
        ok, ctx = self.left.run(ctx)
        if not ok:
            return False, ctx
        return self.right.run(ctx)


@dataclass
class _Or(Combinator[T]):
    left: Combinator[T]
    right: Combinator[T]

    def run(self, ctx: T) -> tuple[bool, T]:
        ok, ctx = self.left.run(ctx)
        if ok:
            return True, ctx
        return self.right.run(ctx)


@dataclass
class _Not(Combinator[T]):
    inner: Combinator[T]

    def run(self, ctx: T) -> tuple[bool, T]:
        ok, ctx = self.inner.run(ctx)
        return not ok, ctx


@dataclass
class _Then(Combinator[T]):
    left: Combinator[T]
    right: Combinator[T]

    def run(self, ctx: T) -> tuple[bool, T]:
        _, ctx = self.left.run(ctx)
        return self.right.run(ctx)


# =============================================================================
# Predicate Combinator
# =============================================================================


class Predicate(Combinator[T]):
    """
    A combinator that checks a condition without modifying context.

    Example:
        is_valid: Predicate[int] = Predicate(lambda x: x > 0, "is_positive")
        ok, _ = is_valid.run(5)  # (True, 5)
    """

    def __init__(self, fn: Callable[[T], bool], name: str | None = None):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "predicate")

    def run(self, ctx: T) -> tuple[bool, T]:
        return self.fn(ctx), ctx

    def __repr__(self) -> str:
        return f"Predicate({self.name})"


class PredicateFactory(Generic[T]):
    """
    A factory that creates Predicates when called with arguments.

    Used for parameterized predicates like `older_than(30)`.
    """

    def __init__(self, fn: Callable[..., bool], name: str):
        self._fn = fn
        self._name = name
        self.__name__ = name

    def __call__(self, *args: Any, **kwargs: Any) -> Predicate[T]:
        name = f"{self._name}({', '.join(map(repr, args))})"
        return Predicate(lambda ctx: self._fn(ctx, *args, **kwargs), name)

    def __repr__(self) -> str:
        return f"PredicateFactory({self._name})"


def rule(fn: Callable[[T], bool]) -> Predicate[T]:
    """
    Decorator to create a simple rule/predicate (single context argument).

    Example:
        @rule
        def is_admin(user: User) -> bool:
            return user.is_admin

        ok, _ = is_admin.run(user)

    For parameterized rules, use @rule_args instead.
    """
    return Predicate(fn, fn.__name__)


def rule_args(fn: Callable[..., bool]) -> PredicateFactory[Any]:
    """
    Decorator to create a parameterized rule factory.

    Example:
        @rule_args
        def older_than(user: User, days: int) -> bool:
            return user.account_age_days > days

        r = older_than(30)  # Returns Predicate
        ok, _ = r.run(user)

    For simple rules (single argument), use @rule instead.
    """
    return PredicateFactory(fn, fn.__name__)


# Aliases for backwards compatibility
predicate = rule
predicate_factory = rule_args


# =============================================================================
# Transform Combinator
# =============================================================================


class Transform(Combinator[T]):
    """
    A combinator that transforms the context.

    Succeeds unless an exception is raised.

    Example:
        double: Transform[int] = Transform(lambda x: x * 2, "double")
        ok, result = double.run(5)  # (True, 10)
    """

    def __init__(self, fn: Callable[[T], T], name: str | None = None):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "transform")

    def run(self, ctx: T) -> tuple[bool, T]:
        try:
            return True, self.fn(ctx)
        except Exception:
            return False, ctx

    def __repr__(self) -> str:
        return f"Transform({self.name})"


class TransformFactory(Generic[T]):
    """
    A factory that creates Transforms when called with arguments.

    Used for parameterized transforms like `add(10)`.
    """

    def __init__(self, fn: Callable[..., T], name: str):
        self._fn = fn
        self._name = name
        self.__name__ = name

    def __call__(self, *args: Any, **kwargs: Any) -> Transform[T]:
        name = f"{self._name}({', '.join(map(repr, args))})"
        return Transform(lambda ctx: self._fn(ctx, *args, **kwargs), name)

    def __repr__(self) -> str:
        return f"TransformFactory({self._name})"


def pipe(fn: Callable[[T], T]) -> Transform[T]:
    """
    Decorator to create a simple transform/pipe (single context argument).

    Example:
        @pipe
        def double(data: int) -> int:
            return data * 2

        ok, result = double.run(5)

    For parameterized transforms, use @pipe_args instead.
    """
    return Transform(fn, fn.__name__)


def pipe_args(fn: Callable[..., Any]) -> TransformFactory[Any]:
    """
    Decorator to create a parameterized transform factory.

    Example:
        @pipe_args
        def multiply(data: int, factor: int) -> int:
            return data * factor

        t = multiply(10)  # Returns Transform
        ok, result = t.run(5)

    For simple transforms (single argument), use @pipe instead.
    """
    return TransformFactory(fn, fn.__name__)


# Aliases for backwards compatibility
transform = pipe
transform_factory = pipe_args


# =============================================================================
# Try Combinator
# =============================================================================


class Try(Combinator[T]):
    """
    A combinator that catches exceptions and converts them to failure.

    Useful for wrapping operations that might fail.

    Example:
        fetch = Try(fetch_from_api, "fetch_api")
        ok, result = fetch.run(request)
    """

    def __init__(self, fn: Callable[[T], T], name: str | None = None):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "try")

    def run(self, ctx: T) -> tuple[bool, T]:
        try:
            return True, self.fn(ctx)
        except Exception:
            return False, ctx

    def __repr__(self) -> str:
        return f"Try({self.name})"


# =============================================================================
# Utility Combinators
# =============================================================================


class Always(Combinator[T]):
    """A combinator that always succeeds."""

    def run(self, ctx: T) -> tuple[bool, T]:
        return True, ctx

    def __repr__(self) -> str:
        return "Always()"


class Never(Combinator[T]):
    """A combinator that always fails."""

    def run(self, ctx: T) -> tuple[bool, T]:
        return False, ctx

    def __repr__(self) -> str:
        return "Never()"


class Debug(Combinator[T]):
    """A combinator that prints context and always succeeds."""

    def __init__(self, label: str = "debug"):
        self.label = label

    def run(self, ctx: T) -> tuple[bool, T]:
        print(f"[{self.label}] {ctx}")
        return True, ctx

    def __repr__(self) -> str:
        return f"Debug({self.label!r})"


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
        |, OR       - Any must pass (lowest precedence)
        &, AND      - All must pass
        ~, NOT, !   - Invert result (highest precedence)

    Grouping:
        ( )         - Override precedence

    Rules:
        rule_name                   - Simple rule
        rule_name(arg)              - Rule with one argument
        rule_name(arg1, arg2)       - Rule with multiple arguments

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

        # Multi-line
        is_admin
        & ~is_banned
        & account_older_than(30)
    """

    # Token types
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    COMMA = "COMMA"
    IDENT = "IDENT"
    NUMBER = "NUMBER"
    STRING = "STRING"
    EOF = "EOF"

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
            elif ch in "~!":
                self.tokens.append((self.NOT, ch))
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
                if upper == "AND":
                    self.tokens.append((self.AND, ident))
                elif upper == "OR":
                    self.tokens.append((self.OR, ident))
                elif upper == "NOT":
                    self.tokens.append((self.NOT, ident))
                else:
                    self.tokens.append((self.IDENT, ident))

            else:
                raise ValueError(f"Unexpected character: {ch!r} at position {self.pos}")

        self.tokens.append((self.EOF, None))

    def _read_string(self, quote: str) -> str:
        """Read a quoted string."""
        self.pos += 1  # skip opening quote
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] != quote:
            if self.text[self.pos] == "\\":
                self.pos += 2  # skip escape
            else:
                self.pos += 1
        result = self.text[start : self.pos]
        self.pos += 1  # skip closing quote
        return result

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
            expr     = or_expr
            or_expr  = and_expr (('|' | 'OR') and_expr)*
            and_expr = not_expr (('&' | 'AND') not_expr)*
            not_expr = ('~' | 'NOT' | '!')? primary
            primary  = IDENT args? | '(' expr ')'
            args     = '(' arg_list? ')'
            arg_list = arg (',' arg)*
            arg      = NUMBER | STRING | IDENT
        """
        result = self._parse_or()
        if self._peek()[0] != self.EOF:
            raise ValueError(f"Unexpected token: {self._peek()}")
        return result

    def _parse_or(self) -> dict | str:
        """Parse OR expression (lowest precedence)."""
        left = self._parse_and()

        items = [left]
        while self._peek()[0] == self.OR:
            self._consume()
            items.append(self._parse_and())

        if len(items) == 1:
            return items[0]
        return {"or": items}

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
        return self._parse_primary()

    def _parse_primary(self) -> dict | str:
        """Parse primary expression (identifier or grouped expr)."""
        token = self._peek()

        if token[0] == self.LPAREN:
            self._consume()
            expr = self._parse_or()
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
