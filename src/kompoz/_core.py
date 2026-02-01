"""Core Combinator base class and composite combinators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic

from kompoz._types import T, _trace_config, _trace_hook


class Combinator(ABC, Generic[T]):
    """
    Base class for all combinators.

    A combinator takes a context and returns (success: bool, new_context).
    Combinators can be composed using operators:
        &  = and then (short-circuits on failure)
        |  = or else (short-circuits on success)
        ~  = not (inverts success/failure)
        >> = then (always runs both)

    Conditional branching:
        condition.if_else(then_branch, else_branch)
        if_then_else(condition, then_branch, else_branch)

    Tracing:
        Use `with use_tracing(hook):` to trace all run() calls within scope.
        Or use `run_traced(combinator, ctx, hook)` for explicit tracing.
    """

    @abstractmethod
    def _execute(self, ctx: T) -> tuple[bool, T]:
        """Internal execution - subclasses implement this."""
        ...

    def run(self, ctx: T) -> tuple[bool, T]:
        """
        Execute the combinator and return (success, new_context).

        If tracing is enabled via use_tracing(), this will automatically
        trace the execution.
        """
        hook = _trace_hook.get()
        if hook is not None:
            config = _trace_config.get()
            from kompoz._tracing import _traced_run

            return _traced_run(self, ctx, hook, config, depth=0)

        return self._execute(ctx)

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

    def if_else(
        self, then_branch: Combinator[T], else_branch: Combinator[T]
    ) -> Combinator[T]:
        """
        Create a conditional: if self succeeds, run then_branch; else run else_branch.

        Example:
            is_premium.if_else(apply_discount, charge_full_price)

        This is equivalent to:
            IF is_premium THEN apply_discount ELSE charge_full_price

        Unlike OR (|), this always executes exactly one branch based on the condition.
        """
        return _IfThenElse(self, then_branch, else_branch)

    def __call__(self, ctx: T) -> tuple[bool, T]:
        """Shorthand for run()."""
        return self.run(ctx)


def _is_composite(combinator: Combinator) -> bool:
    """Check if combinator is a composite type that needs stack-based execution."""
    return isinstance(combinator, (_And, _Or, _Not, _Then, _IfThenElse))


def _execute_iterative(root: Combinator[T], ctx: T) -> tuple[bool, T]:
    """
    Execute a combinator tree iteratively using an explicit stack.

    This avoids deep recursion for complex nested combinators.
    Uses continuation-passing style with a work stack.
    """
    result_stack: list[tuple[bool, T]] = []
    work_stack: list[tuple[Combinator[T], T, int, Any]] = [(root, ctx, 0, None)]

    while work_stack:
        combinator, current_ctx, phase, saved_data = work_stack.pop()

        if isinstance(combinator, _And):
            if phase == 0:
                work_stack.append((combinator, current_ctx, 1, None))
                if _is_composite(combinator.left):
                    work_stack.append((combinator.left, current_ctx, 0, None))
                else:
                    ok, new_ctx = combinator.left._execute(current_ctx)
                    result_stack.append((ok, new_ctx))
            elif phase == 1:
                ok, new_ctx = result_stack.pop()
                if not ok:
                    result_stack.append((False, new_ctx))
                else:
                    work_stack.append((combinator, new_ctx, 2, None))
                    if _is_composite(combinator.right):
                        work_stack.append((combinator.right, new_ctx, 0, None))
                    else:
                        ok2, new_ctx2 = combinator.right._execute(new_ctx)
                        result_stack.append((ok2, new_ctx2))
            else:
                pass

        elif isinstance(combinator, _Or):
            if phase == 0:
                work_stack.append((combinator, current_ctx, 1, None))
                if _is_composite(combinator.left):
                    work_stack.append((combinator.left, current_ctx, 0, None))
                else:
                    ok, new_ctx = combinator.left._execute(current_ctx)
                    result_stack.append((ok, new_ctx))
            elif phase == 1:
                ok, new_ctx = result_stack.pop()
                if ok:
                    result_stack.append((True, new_ctx))
                else:
                    work_stack.append((combinator, new_ctx, 2, None))
                    if _is_composite(combinator.right):
                        work_stack.append((combinator.right, new_ctx, 0, None))
                    else:
                        ok2, new_ctx2 = combinator.right._execute(new_ctx)
                        result_stack.append((ok2, new_ctx2))
            else:
                pass

        elif isinstance(combinator, _Not):
            if phase == 0:
                work_stack.append((combinator, current_ctx, 1, None))
                if _is_composite(combinator.inner):
                    work_stack.append((combinator.inner, current_ctx, 0, None))
                else:
                    ok, new_ctx = combinator.inner._execute(current_ctx)
                    result_stack.append((ok, new_ctx))
            else:
                ok, new_ctx = result_stack.pop()
                result_stack.append((not ok, new_ctx))

        elif isinstance(combinator, _Then):
            if phase == 0:
                work_stack.append((combinator, current_ctx, 1, None))
                if _is_composite(combinator.left):
                    work_stack.append((combinator.left, current_ctx, 0, None))
                else:
                    ok, new_ctx = combinator.left._execute(current_ctx)
                    result_stack.append((ok, new_ctx))
            elif phase == 1:
                _, new_ctx = result_stack.pop()
                work_stack.append((combinator, new_ctx, 2, None))
                if _is_composite(combinator.right):
                    work_stack.append((combinator.right, new_ctx, 0, None))
                else:
                    ok2, new_ctx2 = combinator.right._execute(new_ctx)
                    result_stack.append((ok2, new_ctx2))
            else:
                pass

        elif isinstance(combinator, _IfThenElse):
            if phase == 0:
                work_stack.append((combinator, current_ctx, 1, None))
                if _is_composite(combinator.condition):
                    work_stack.append((combinator.condition, current_ctx, 0, None))
                else:
                    ok, new_ctx = combinator.condition._execute(current_ctx)
                    result_stack.append((ok, new_ctx))
            elif phase == 1:
                cond_ok, new_ctx = result_stack.pop()
                work_stack.append((combinator, new_ctx, 2, None))
                if cond_ok:
                    if _is_composite(combinator.then_branch):
                        work_stack.append((combinator.then_branch, new_ctx, 0, None))
                    else:
                        ok2, new_ctx2 = combinator.then_branch._execute(new_ctx)
                        result_stack.append((ok2, new_ctx2))
                else:
                    if _is_composite(combinator.else_branch):
                        work_stack.append((combinator.else_branch, new_ctx, 0, None))
                    else:
                        ok2, new_ctx2 = combinator.else_branch._execute(new_ctx)
                        result_stack.append((ok2, new_ctx2))
            else:
                pass

        else:
            ok, new_ctx = combinator._execute(current_ctx)
            result_stack.append((ok, new_ctx))

    return result_stack[-1] if result_stack else (False, ctx)


@dataclass
class _And(Combinator[T]):
    left: Combinator[T]
    right: Combinator[T]

    def _execute(self, ctx: T) -> tuple[bool, T]:
        return _execute_iterative(self, ctx)


@dataclass
class _Or(Combinator[T]):
    left: Combinator[T]
    right: Combinator[T]

    def _execute(self, ctx: T) -> tuple[bool, T]:
        return _execute_iterative(self, ctx)


@dataclass
class _Not(Combinator[T]):
    inner: Combinator[T]

    def _execute(self, ctx: T) -> tuple[bool, T]:
        return _execute_iterative(self, ctx)


@dataclass
class _Then(Combinator[T]):
    left: Combinator[T]
    right: Combinator[T]

    def _execute(self, ctx: T) -> tuple[bool, T]:
        return _execute_iterative(self, ctx)


@dataclass
class _IfThenElse(Combinator[T]):
    """
    Conditional combinator: if condition succeeds, run then_branch; else run else_branch.

    Unlike OR which short-circuits on success, this explicitly branches:
    - condition ? then_branch : else_branch
    - IF condition THEN then_branch ELSE else_branch
    """

    condition: Combinator[T]
    then_branch: Combinator[T]
    else_branch: Combinator[T]

    def _execute(self, ctx: T) -> tuple[bool, T]:
        return _execute_iterative(self, ctx)


def if_then_else(
    condition: Combinator[T], then_branch: Combinator[T], else_branch: Combinator[T]
) -> Combinator[T]:
    """
    Create a conditional combinator: if condition succeeds, run then_branch; else run else_branch.

    Example:
        from kompoz import if_then_else, rule

        @rule
        def is_premium(user):
            return user.is_premium

        @rule
        def apply_discount(user):
            user.discount = 0.2
            return True

        @rule
        def charge_full_price(user):
            user.discount = 0
            return True

        pricing = if_then_else(is_premium, apply_discount, charge_full_price)
        ok, user = pricing.run(user)

    DSL equivalent:
        IF is_premium THEN apply_discount ELSE charge_full_price
        is_premium ? apply_discount : charge_full_price

    Unlike OR (|) which is a fallback (try a, if fail try b), if_then_else
    explicitly branches based on the condition result.
    """
    return _IfThenElse(condition, then_branch, else_branch)
