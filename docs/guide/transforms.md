# Transforms (Data Pipelines)

Transforms modify the context as it flows through a pipeline. Unlike predicates (which check conditions), transforms change data.

## Basic Transforms

```python
from kompoz import pipe, pipe_args, rule

@pipe
def parse_int(data):
    return int(data)

@pipe
def double(data):
    return data * 2

@pipe_args
def add(data, n):
    return data + n

@rule
def is_positive(data):
    return data > 0

# Build a pipeline
pipeline = parse_int & is_positive & double & add(10)

ok, result = pipeline.run("21")
# ok=True, result=52  (21 * 2 + 10)

ok, result = pipeline.run("-5")
# ok=False, result=-5  (stopped at is_positive)
```

## Error Tracking

Transforms track exceptions via the `last_error` attribute:

```python
@pipe
def risky_transform(data):
    return int(data)  # May raise ValueError

ok, result = risky_transform.run("not a number")
if not ok:
    print(f"Failed: {risky_transform.last_error}")
    # Failed: invalid literal for int() with base 10: 'not a number'
```

This also works for async transforms:

```python
@async_pipe
async def fetch_data(url):
    async with aiohttp.get(url) as resp:
        return await resp.json()

ok, result = await fetch_data.run("https://api.example.com")
if not ok:
    print(f"Request failed: {fetch_data.last_error}")
```

### Thread-safe alternative: `run_with_error()`

The `last_error` attribute is mutated on each call, which is not safe when the same
transform instance is used from multiple threads or async tasks. Use `run_with_error()`
instead -- it returns the error in the result tuple without mutating the instance:

```python
ok, result, error = risky_transform.run_with_error("not a number")
if not ok:
    print(f"Failed: {error}")
```

!!! note
    `run_with_error()` is available on both `Transform` and `AsyncTransform`.
