from __future__ import print_function
from enum import Enum
class LogLevels(Enum):
    DEBUG = 10
    WARNING = 20
    ERROR = 40
    NONE = 100

def logMessage(message, level="info", newline=True):
    if newline:
        print(message)
    else:
        print(message, end="")