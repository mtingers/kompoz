# Utility Combinators

## Always

```python
class Always(Combinator[T])
```

A combinator that always succeeds. Returns `(True, ctx)` for any input.

```python
from kompoz import Always

always = Always()
ok, result = always.run(anything)  # (True, anything)
```

---

## Never

```python
class Never(Combinator[T])
```

A combinator that always fails. Returns `(False, ctx)` for any input.

```python
from kompoz import Never

never = Never()
ok, result = never.run(anything)  # (False, anything)
```

---

## Debug

```python
class Debug(Combinator[T])
```

A combinator that prints the context and always succeeds. Useful for debugging pipelines.

**Constructor:**

```python
Debug(label: str = "")
```

**Example:**

```python
from kompoz import Debug

# Insert into a pipeline to inspect intermediate values
pipeline = step_a & Debug("after step_a") & step_b
```
