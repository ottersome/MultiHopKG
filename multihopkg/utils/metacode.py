
class StaleCodeError(Exception):
    """Exception raised when potentially stale code is called."""
    pass

def stale_code(func):
    """Decorator to mark code as stale/unused and prevent execution."""
    def wrapper(*args, **kwargs):
        raise StaleCodeError(
            f"Class/Function '{func.__name__}' has not been used for a while. "
            "Please review before using."
        )
    return wrapper

