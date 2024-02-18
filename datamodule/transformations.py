import torch
from torchvision import transforms

class Standardize(object):
    """
    Move a tensor from range [0.0, 1.0] to [-1.0, 1.0].
    """

    def __call__(self, tensor):
        """
        Args:
            tensor (torch.Tensor): tensor whose range should be changed.
        Returns:
            Tensor: a tensor in the range [-1.0, 1.0]
        """
        return (tensor - 0.5) * 2.0

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
    
class Reverse(object):
    """
    Reverse the pixel values of a tensor w.r.t 1, i.e. 1 goes to 0, and vice versa.
    """

    def __call__(self, tensor):
        """
        Args:
            tensor (torch.Tensor): tensor to be reversed.
        Returns:
            Tensor: a tensor in the range [0.0, 1.0]
        """
        return 1-tensor

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
    
class TensorClip(object):
    """
    Clip the values of a tensor to the range [-1.0, 1.0].
    """

    def __call__(self, tensor):
        """
        Args:
            tensor (torch.Tensor): tensor to be clipped.
        Returns:
            Tensor: a tensor in the range [-1.0, 1.0]
        """
        return torch.clip(tensor, -1, 1)

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
    
# ALL_TRANSFORMATIONS = {
#     "standard": Standardize(),
#     "reverse": Reverse(),
#     "tensor": transforms.ToTensor(),
#     "normalize": transforms.Normalize(),
#     "clip": TensorClip(),
# }
