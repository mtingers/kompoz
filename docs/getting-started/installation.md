# Installation

## Requirements

- **Python 3.13+**
- **Zero dependencies** -- the core library has no external dependencies

## Install

=== "pip"

    ```bash
    pip install kompoz
    ```

=== "uv"

    ```bash
    uv add kompoz
    ```

=== "Poetry"

    ```bash
    poetry add kompoz
    ```

## Optional Extras

### OpenTelemetry

For distributed tracing integration:

```bash
pip install kompoz[opentelemetry]
```

This installs `opentelemetry-api`, `opentelemetry-sdk`, and `opentelemetry-exporter-otlp`.

## Verify Installation

```python
import kompoz
print(kompoz.__version__)
```
