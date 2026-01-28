"""
Runtime stub for CoolProp.

Any attempt to use CoolProp in solver/runtime code should fail fast with a clear error.
"""


def _raise_disabled(name: str) -> None:
    raise RuntimeError(
        "CoolProp is disabled in solver runtime. "
        "Use p2db-based properties. "
        "For offline fitting, install optional deps from requirements-fit.txt. "
        f"Attempted to access: {name}"
    )


def PropsSI(*_args, **_kwargs):
    _raise_disabled("PropsSI")


class AbstractState:
    def __init__(self, *_args, **_kwargs):
        _raise_disabled("AbstractState")
