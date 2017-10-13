def logMessage(message, level="info", newline=True):
    if newline:
        print(message)
    else:
        print message,