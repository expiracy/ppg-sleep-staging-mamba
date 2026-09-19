def format_model_name(pretty_name, params=None):
    """Format model name as PrettyName[key=value, ...].

    Returns just pretty_name if params is None or empty.
    Nested dicts become nested brackets: attention=[type=windowed, window_size=12]
    Lists become bracketed: tcn_dilations=[1, 2, 4, 8]
    """
    if not params:
        return pretty_name
    formatted = ", ".join(f"{k}={_format_value(v)}" for k, v in params.items())
    return f"{pretty_name}[{formatted}]"


def _format_value(value) -> str:
    """Recursively format a value for bracket display."""
    if isinstance(value, dict):
        inner = ", ".join(f"{k}={_format_value(v)}" for k, v in value.items())
        return f"[{inner}]"
    if isinstance(value, (list, tuple)):
        inner = ", ".join(_format_value(v) for v in value)
        return f"[{inner}]"
    if isinstance(value, bool):
        return str(value)
    return str(value)
