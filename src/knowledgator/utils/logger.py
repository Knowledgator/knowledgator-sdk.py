from __future__ import annotations
import time

class Logger:
    """A minimal stdout logger with level filtering."""

    _LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}

    def __init__(self, name: str = "SDK Logger", level: str | int = "INFO",
                 fmt: str = "{ts} {level:<7} {name}: {msg}") -> None:
        self.name = name
        self.level = self._coerce_level(level)
        self.fmt = fmt

    def debug(self, msg: str, *args, **kwargs) -> None:
        self._log("DEBUG", msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        self._log("INFO", msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self._log("WARNING", msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self._log("ERROR", msg, *args, **kwargs)

    def _coerce_level(self, level: str | int) -> int:
        if isinstance(level, int):
            return level
        return self._LEVELS.get(level.upper(), 20)

    def _log(self, level_name: str, msg: str, *args, **kwargs) -> None:
        if self._LEVELS[level_name] < self.level:
            return  # filtered out
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        text = msg.format(*args, **kwargs)
        print(self.fmt.format(ts=ts, level=level_name, name=self.name, msg=text))