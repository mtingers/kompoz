"""Validation combinators with error messages."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, overload
import inspect

from kompoz._core import Combinator, _And, _Or, _Not, _Then, _IfThenElse, _is_composite, _execute_iterative
from kompoz._types import T


def _get_combinator_name(combinator: Combinator) -> str:
    """Get a human-readable name for a combinator (simple version for validation)."""
    if hasattr(combinator, "name"):
        return getattr(combinator, "name")
    return repr(combinator)


@dataclass
class ValidationResult:
    """
    Result of validation with error messages.

    Attributes:
        ok: Whether all checks passed
        errors: List of error messages from failed checks
        ctx: The (possibly transformed) context
    """

    ok: bool
    errors: list[str]
    ctx: Any

    def __bool__(self) -> bool:
        return self.ok

    def raise_if_invalid(self, exception_class: type = ValueError) -> None:
        """Raise an exception if validation failed."""
        if not self.ok:
            raise exception_class("; ".join(self.errors))


class ValidatingCombinator(Combinator[T]):
    """
    Base class for combinators that support validation with error messages.

    Subclasses must implement the validate() method.
    """

    @abstractmethod
    def validate(self, ctx: T) -> ValidationResult:
        """Run validation and return result with errors."""
        ...

    def __and__(self, other: Combinator[T]) -> ValidatingCombinator[T]:
        """Override & to create validating AND."""
        return _ValidatingAnd(self, other)

    def __or__(self, other: Combinator[T]) -> ValidatingCombinator[T]:
        """Override | to create validating OR."""
        return _ValidatingOr(self, other)

    def __invert__(self) -> ValidatingCombinator[T]:
        """Override ~ to create validating NOT."""
        return _ValidatingNot(self)


class ValidatingPredicate(ValidatingCombinator[T]):
    """
    A predicate that provides an error message on failure.

    Example:
        @vrule(error="User must be an admin")
        def is_admin(user):
            return user.is_admin

        result = is_admin.validate(user)
        if not result.ok:
            print(result.errors)  # ["User must be an admin"]
    """

    def __init__(
        self,
        fn: Callable[[T], bool],
        name: str | None = None,
        error: str | Callable[[T], str] | None = None,
    ):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "predicate")
        self._error = error

    def _execute(self, ctx: T) -> tuple[bool, T]:
        return self.fn(ctx), ctx

    def get_error(self, ctx: T) -> str:
        """Get the error message for this predicate."""
        if self._error is None:
            return f"Check failed: {self.name}"
        if callable(self._error):
            return self._error(ctx)
        # String interpolation with ctx
        try:
            return self._error.format(ctx=ctx)
        except (KeyError, AttributeError, IndexError):
            return self._error

    def validate(self, ctx: T) -> ValidationResult:
        """Run validation and return result with errors."""
        ok, result = self._execute(ctx)
        errors = [] if ok else [self.get_error(ctx)]
        return ValidationResult(ok=ok, errors=errors, ctx=result)

    def __repr__(self) -> str:
        return f"ValidatingPredicate({self.name})"


def _is_validating_composite(combinator: Combinator) -> bool:
    """Check if combinator is a validating composite type."""
    return isinstance(combinator, (_ValidatingAnd, _ValidatingOr, _ValidatingNot))


def _execute_validating_iterative(root: Combinator[T], ctx: T) -> tuple[bool, T]:
    """
    Execute a validating combinator tree iteratively using an explicit stack.
    """
    result_stack: list[tuple[bool, T]] = []
    work_stack: list[tuple[Combinator[T], T, int, Any]] = [(root, ctx, 0, None)]

    while work_stack:
        combinator, current_ctx, phase, saved_data = work_stack.pop()

        if isinstance(combinator, _ValidatingAnd):
            if phase == 0:
                work_stack.append((combinator, current_ctx, 1, None))
                if _is_validating_composite(combinator.left) or _is_composite(
                    combinator.left
                ):
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
                    if _is_validating_composite(combinator.right) or _is_composite(
                        combinator.right
                    ):
                        work_stack.append((combinator.right, new_ctx, 0, None))
                    else:
                        ok2, new_ctx2 = combinator.right._execute(new_ctx)
                        result_stack.append((ok2, new_ctx2))
            else:
                pass

        elif isinstance(combinator, _ValidatingOr):
            if phase == 0:
                work_stack.append((combinator, current_ctx, 1, None))
                if _is_validating_composite(combinator.left) or _is_composite(
                    combinator.left
                ):
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
                    if _is_validating_composite(combinator.right) or _is_composite(
                        combinator.right
                    ):
                        work_stack.append((combinator.right, new_ctx, 0, None))
                    else:
                        ok2, new_ctx2 = combinator.right._execute(new_ctx)
                        result_stack.append((ok2, new_ctx2))
            else:
                pass

        elif isinstance(combinator, _ValidatingNot):
            if phase == 0:
                work_stack.append((combinator, current_ctx, 1, None))
                if _is_validating_composite(combinator.inner) or _is_composite(
                    combinator.inner
                ):
                    work_stack.append((combinator.inner, current_ctx, 0, None))
                else:
                    ok, new_ctx = combinator.inner._execute(current_ctx)
                    result_stack.append((ok, new_ctx))
            else:
                ok, new_ctx = result_stack.pop()
                result_stack.append((not ok, new_ctx))

        # Handle standard composite combinators
        elif _is_composite(combinator):
            ok, new_ctx = _execute_iterative(combinator, current_ctx)
            result_stack.append((ok, new_ctx))

        else:
            ok, new_ctx = combinator._execute(current_ctx)
            result_stack.append((ok, new_ctx))

    return result_stack[-1] if result_stack else (False, ctx)


class _ValidatingAnd(ValidatingCombinator[T]):
    """AND combinator that collects all validation errors."""

    def __init__(self, left: Combinator[T], right: Combinator[T]):
        self.left = left
        self.right = right

    def _execute(self, ctx: T) -> tuple[bool, T]:
        return _execute_validating_iterative(self, ctx)

    def validate(self, ctx: T) -> ValidationResult:
        """Validate both sides and collect all errors (iteratively)."""
        errors: list[str] = []

        # Flatten the AND chain iteratively
        to_validate: list[Combinator[T]] = []
        stack: list[Combinator[T]] = [self]
        while stack:
            current = stack.pop()
            if isinstance(current, _ValidatingAnd):
                stack.append(current.right)
                stack.append(current.left)
            else:
                to_validate.append(current)

        # Validate each item
        for item in to_validate:
            if isinstance(item, ValidatingCombinator):
                result = item.validate(ctx)
                errors.extend(result.errors)
                ctx = result.ctx
            else:
                ok, ctx = item._execute(ctx)
                if not ok:
                    errors.append(f"Check failed: {_get_combinator_name(item)}")

        return ValidationResult(ok=len(errors) == 0, errors=errors, ctx=ctx)


class _ValidatingOr(ValidatingCombinator[T]):
    """OR combinator for validation - passes if any succeeds."""

    def __init__(self, left: Combinator[T], right: Combinator[T]):
        self.left = left
        self.right = right

    def _execute(self, ctx: T) -> tuple[bool, T]:
        return _execute_validating_iterative(self, ctx)

    def validate(self, ctx: T) -> ValidationResult:
        """Validate - passes if any in the chain passes (iteratively)."""
        # Flatten the OR chain iteratively
        to_validate: list[Combinator[T]] = []
        stack: list[Combinator[T]] = [self]
        while stack:
            current = stack.pop()
            if isinstance(current, _ValidatingOr):
                stack.append(current.right)
                stack.append(current.left)
            else:
                to_validate.append(current)

        # Try each item until one passes
        last_result: ValidationResult | None = None
        for item in to_validate:
            if isinstance(item, ValidatingCombinator):
                result = item.validate(ctx)
                if result.ok:
                    return result
                last_result = result
            else:
                ok, result_ctx = item._execute(ctx)
                if ok:
                    return ValidationResult(ok=True, errors=[], ctx=result_ctx)
                last_result = ValidationResult(
                    ok=False,
                    errors=[f"Check failed: {_get_combinator_name(item)}"],
                    ctx=result_ctx,
                )

        return last_result or ValidationResult(
            ok=False, errors=["No conditions to check"], ctx=ctx
        )


class _ValidatingNot(ValidatingCombinator[T]):
    """NOT combinator for validation - inverts the result."""

    def __init__(self, inner: Combinator[T], error: str | None = None):
        self.inner = inner
        self._error = error

    def _execute(self, ctx: T) -> tuple[bool, T]:
        return _execute_validating_iterative(self, ctx)

    def validate(self, ctx: T) -> ValidationResult:
        """Validate - inverts the inner result."""
        if isinstance(self.inner, ValidatingCombinator):
            inner_result = self.inner.validate(ctx)
            if inner_result.ok:
                error_msg = self._error or "NOT condition failed (inner passed)"
                return ValidationResult(
                    ok=False, errors=[error_msg], ctx=inner_result.ctx
                )
            else:
                return ValidationResult(ok=True, errors=[], ctx=inner_result.ctx)
        else:
            ok, result = self.inner._execute(ctx)
            if ok:
                error_msg = (
                    self._error or f"NOT {_get_combinator_name(self.inner)} failed"
                )
                return ValidationResult(ok=False, errors=[error_msg], ctx=result)
            else:
                return ValidationResult(ok=True, errors=[], ctx=result)


@overload
def vrule(
    fn: Callable[[T], bool], *, error: str | Callable[[T], str] | None = None
) -> ValidatingPredicate[T]: ...


@overload
def vrule(
    fn: None = None, *, error: str | Callable[[T], str] | None = None
) -> Callable[[Callable[[T], bool]], ValidatingPredicate[T]]: ...


def vrule(
    fn: Callable[[T], bool] | None = None,
    *,
    error: str | Callable[[T], str] | None = None,
) -> ValidatingPredicate[T] | Callable[[Callable[[T], bool]], ValidatingPredicate[T]]:
    """
    Decorator to create a validating rule with an error message.

    Example:
        @vrule(error="User {ctx.name} must be an admin")
        def is_admin(user):
            return user.is_admin

        @vrule(error=lambda u: f"{u.name} is banned!")
        def is_not_banned(user):
            return not user.is_banned

        result = is_admin.validate(user)
        result = (is_admin & is_not_banned).validate(user)  # Collects all errors
    """

    def decorator(f: Callable[[T], bool]) -> ValidatingPredicate[T]:
        return ValidatingPredicate(f, f.__name__, error)

    if fn is not None:
        return decorator(fn)
    return decorator


@overload
def vrule_args(
    fn: Callable[..., bool], *, error: str | Callable[..., str] | None = None
) -> Callable[..., ValidatingPredicate]: ...


@overload
def vrule_args(
    fn: None = None, *, error: str | Callable[..., str] | None = None
) -> Callable[[Callable[..., bool]], Callable[..., ValidatingPredicate]]: ...


def vrule_args(
    fn: Callable[..., bool] | None = None,
    *,
    error: str | Callable[..., str] | None = None,
) -> (
    Callable[..., ValidatingPredicate]
    | Callable[[Callable[..., bool]], Callable[..., ValidatingPredicate]]
):
    def decorator(f: Callable[..., bool]) -> Callable[..., ValidatingPredicate]:
        # 1. Inspect the function signature to enable param name
        sig = inspect.signature(f)

        def factory(*args: Any, **kwargs: Any) -> ValidatingPredicate:
            name = f"{f.__name__}({', '.join(map(repr, args))})"

            # 2. Helper to resolve parameter names from args
            def get_bound_params():
                # We assume the first argument of 'f' is 'ctx', which isn't in *args here.
                # We bind a dummy value for the first argument to align *args correctly.
                try:
                    bound = sig.bind_partial(None, *args, **kwargs)
                    bound.apply_defaults()
                    # Remove the first argument (the context placeholder)
                    params = dict(bound.arguments)
                    first_param_name = list(sig.parameters.keys())[0]
                    if first_param_name in params:
                        del params[first_param_name]
                    return params
                except TypeError:
                    # Fallback if binding fails
                    return kwargs

            err_msg: str | Callable[[Any], str] | None

            if error is None:
                err_msg = None

            # CASE A: Error is a callable (custom function)
            elif callable(error):
                error_fn: Callable[..., str] = error

                def make_error_fn(ctx: Any) -> str:
                    return error_fn(ctx, *args, **kwargs)

                err_msg = make_error_fn

            # CASE B: Error is a string (template)
            else:
                # This tells the type checker: "Inside this block, we are 100% sure this is a string."
                template_str: str = error

                def make_formatted_error(ctx: Any) -> str:
                    # 1. Standard {arg0}, {arg1} support
                    format_context = {f"arg{i}": v for i, v in enumerate(args)}

                    # 2. Parameter name support ({score})
                    format_context.update(get_bound_params())

                    # 3. Context support ({ctx})
                    format_context["ctx"] = ctx

                    try:
                        # Now we use 'template_str' instead of 'error'
                        return template_str.format(**format_context)
                    except (KeyError, IndexError, AttributeError):
                        return template_str

                err_msg = make_formatted_error

            def predicate_fn(ctx: Any) -> bool:
                return f(ctx, *args, **kwargs)

            return ValidatingPredicate(predicate_fn, name, err_msg)

        factory.__name__ = f.__name__
        return factory

    if fn is not None:
        return decorator(fn)
    return decorator


