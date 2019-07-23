from __future__ import print_function
from enum import Enum
class LogLevels(Enum):
    DEBUG = 10
    WARNING = 20
    INFO = 30
    ERROR = 40
    NONE = 100


class MessageLogger:
    def __init__(self, logLevel=LogLevels.INFO):
        self._logLevel = logLevel

    def logMessage(self, message, level=LogLevels.INFO, newline=True):
        if level.value < self._logLevel.value:
            return
        if newline:
            print(str(level.name).ljust(8) + ": | " + message)
        else:
            print(str(level.name).ljust(8) + ": | " + message, end="")