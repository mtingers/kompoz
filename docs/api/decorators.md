# Decorators

## @rule

```python
@rule
def predicate_name(ctx: T) -> bool: ...
```

Create a simple predicate from a function that takes one argument and returns `bool`.

Returns: `Predicate[T]`

```python
@rule
def is_admin(user: User) -> bool:
    return user.is_admin
```

---

## @rule_args

```python
@rule_args
def predicate_name(ctx: T, *args) -> bool: ...
```

Create a parameterized predicate factory. Call the result with arguments to get a `Predicate[T]`.

Returns: `PredicateFactory[T]`

```python
@rule_args
def credit_above(user: User, threshold: int) -> bool:
    return user.credit_score > threshold

check = credit_above(700)  # Predicate[User]
```

---

## @pipe

```python
@pipe
def transform_name(ctx: T) -> T: ...
```

Create a simple transform from a function that takes one argument and returns the transformed value.

Returns: `Transform[T]`

```python
@pipe
def double(x: int) -> int:
    return x * 2
```

---

## @pipe_args

```python
@pipe_args
def transform_name(ctx: T, *args) -> T: ...
```

Create a parameterized transform factory.

Returns: `TransformFactory[T]`

```python
@pipe_args
def add(x: int, n: int) -> int:
    return x + n

add_ten = add(10)  # Transform[int]
```

---

## @vrule

```python
@vrule(error="Error message with {ctx.field}")
def predicate_name(ctx: T) -> bool: ...
```

Create a validating rule with an error message. The error string can reference `ctx` for template formatting, or be a callable `(ctx) -> str`.

Returns: `ValidatingPredicate[T]`

```python
@vrule(error="User {ctx.name} must be an admin")
def is_admin(user):
    return user.is_admin

@vrule(error=lambda u: f"{u.name} is banned!")
def not_banned(user):
    return not user.is_banned
```

---

## @vrule_args

```python
@vrule_args(error="Error with {param}")
def predicate_name(ctx: T, param) -> bool: ...
```

Create a parameterized validating rule. Extra arguments are available in the error template.

Returns: Factory producing `ValidatingPredicate[T]`

```python
@vrule_args(error="Account must be older than {days} days")
def account_older_than(user, days):
    return user.account_age_days > days
```

---

## @async_rule

```python
@async_rule
async def predicate_name(ctx: T) -> bool: ...
```

Create an async predicate.

Returns: `AsyncPredicate[T]`

---

## @async_rule_args

```python
@async_rule_args
async def predicate_name(ctx: T, *args) -> bool: ...
```

Create a parameterized async predicate factory.

Returns: `AsyncPredicateFactory[T]`

---

## @async_pipe

```python
@async_pipe
async def transform_name(ctx: T) -> T: ...
```

Create an async transform.

Returns: `AsyncTransform[T]`

---

## @async_pipe_args

```python
@async_pipe_args
async def transform_name(ctx: T, *args) -> T: ...
```

Create a parameterized async transform factory.

Returns: `AsyncTransformFactory[T]`

---

## @async_vrule

```python
@async_vrule(error="Error message")
async def predicate_name(ctx: T) -> bool: ...
```

Create an async validating rule with error message.

Returns: `AsyncValidatingPredicate[T]`

---

## @async_vrule_args

```python
@async_vrule_args(error="Error with {param}")
async def predicate_name(ctx: T, param) -> bool: ...
```

Create a parameterized async validating rule.

Returns: Factory producing `AsyncValidatingPredicate[T]`

---

## @cached_rule

```python
@cached_rule
def predicate_name(ctx: T) -> bool: ...

@cached_rule(key=lambda ctx: ctx.id)
def predicate_name(ctx: T) -> bool: ...
```

Create a rule with result caching. Results are cached within a `use_cache()` scope.

Optionally provide a `key` function to control the cache key.

Returns: `CachedPredicate[T]`

---

## @async_cached_rule

```python
@async_cached_rule
async def predicate_name(ctx: T) -> bool: ...

@async_cached_rule(key=lambda ctx: ctx.id)
async def predicate_name(ctx: T) -> bool: ...
```

Create an async rule with result caching.

Returns: `AsyncCachedPredicate[T]`
