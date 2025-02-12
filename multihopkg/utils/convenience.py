from typing import Type, TypeVar, Any
import torch

# Define a generic type variable
T = TypeVar("T")


# TOREM: When we finish the first stage of debugging
def create_placeholder(expected_type: Type[T], name: str, location: str) -> Any:
    """Creates a placeholder function that raises NotImplementedError.
    Args:
        expected_type: The expected return type of the function.
    Returns:
        A function that raises NotImplementedError.
    """

    def placeholder(*args, **kwargs) -> T:
        raise NotImplementedError(
            f"{name}, at {location} is a placeholder and is expected to return {expected_type.__name__}.\n"
            "If you see this error it means you commited to changing this later"
        )

    return placeholder

def tensor_normalization(tensor: torch.Tensor) -> torch.Tensor:
    """Normalizes a tensor by its mean and standard deviation.
    Args:
        tensor: The tensor to normalize.
    Returns:
        The normalized tensor.
    """
    mean = tensor.mean() + 1e-8
    std = tensor.std()
    return (tensor - mean) / std
