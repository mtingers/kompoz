# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in Kompoz, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, please email: **matth@mtingers.com**

Include the following in your report:

- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact
- Suggested fix (if any)

You should receive an acknowledgment within 48 hours, and a detailed response within 7 days indicating next steps.

## Scope

The following areas are in scope for security reports:

- **Expression parser input handling** -- malicious or malformed expressions passed to `Registry.load()` or `parse_expression()`
- **File loading** -- path traversal or injection via `Registry.load_file()`
- **OpenTelemetry integration** -- data leakage or injection through tracing hooks
- **Cache poisoning** -- manipulation of cached results in `use_cache()` or `use_cache_shared()`

## Out of Scope

- Vulnerabilities in dependencies (report these to the respective projects)
- Denial of service through intentionally complex expressions (resource limits are the caller's responsibility)
