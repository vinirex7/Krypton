"""Exchange failures that the execution loop can handle without repeating an entry."""


class ExchangeError(RuntimeError):
    def __init__(self, message, status, code, *, rejected=False, crossed=False):
        super().__init__(message)
        self.status = status
        self.code = code
        self.rejected = rejected
        self.crossed = crossed
