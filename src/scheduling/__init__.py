"""Smart scheduling module for the AI trading system."""

# Lazy imports to avoid cascading import failures when
# main.py imports individual submodules directly.
__all__ = ['LightCheck', 'SmartScheduler']


def __getattr__(name):
    if name == 'LightCheck':
        from src.scheduling.light_check import LightCheck
        return LightCheck
    if name == 'SmartScheduler':
        from src.scheduling.scheduler import SmartScheduler
        return SmartScheduler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
