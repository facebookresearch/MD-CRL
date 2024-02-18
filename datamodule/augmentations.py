# import torch.utils.data
# import torch
# import math
# import os
# import numpy as np
# import h5py
# from typing import Callable, Optional, Sequence
# from PIL import Image
# import torchvision.transforms.functional as TF
# from torchvision import transforms

# OWN_PATH = os.path.split(os.path.abspath(__file__))[0]


PIL_to_th = lambda x: x  # transforms.PILToTensor()


def random_rotate(im, order: int = 3):
    """
    Parameters:
    im - an image to be rotated
    order - the number of dimensions we want for the latent space -
    i.e. are we estimating postion (order 1), position + velocity (order 2), etc.
    """
    d = float((torch.rand(1) - 0.5) * 90)  # uniform [-45, 45]
    theta = d / 180 * math.pi
    A_ = torch.tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    A = torch.zeros((2 * order, 2 * order))
    for i in range(order):
        A[i * 2 : (1 + i) * 2, i * 2 : (1 + i) * 2] = A_
    b = torch.zeros(
        order * 2,
    )
    if isinstance(im, list):
        return [PIL_to_th(TF.rotate(Image.fromarray(i), d, fill=255)) for i in im], A, b
    else:
        return PIL_to_th(TF.rotate(Image.fromarray(im), d, fill=255)), A, b


def hflip(im, order: int = 3):
    A = torch.diag(torch.tensor([-1.0, 1.0] * order))
    b = torch.zeros(
        order * 2,
    )
    if isinstance(im, list):
        return [PIL_to_th(TF.hflip(Image.fromarray(i))) for i in im], A, b
    else:
        return PIL_to_th(TF.hflip(Image.fromarray(im))), A, b


def vflip(im, order: int = 3):
    A = torch.diag(torch.tensor([1.0, -1.0] * order))
    b = torch.zeros(
        order * 2,
    )
    if isinstance(im, list):
        return [PIL_to_th(TF.vflip(Image.fromarray(i))) for i in im], A, b
    else:
        return PIL_to_th(TF.vflip(Image.fromarray(im))), A, b


def shift(im):
    b = np.random.uniform(-0.2, 0.2, size=(2,))
    # shift only 1 axis at a time
    if np.random.rand() < 0.5:
        b[0] = 0
    else:
        b[1] = 0
    A = torch.eye(2)
    if isinstance(im, list):
        transformed = [
            PIL_to_th(
                TF.affine(
                    Image.fromarray(i),
                    angle=0.0,
                    translate=list(b * 64),
                    scale=1.0,
                    shear=0.0,
                    fill=255,
                )
            )
            for i in im
        ]
    else:
        transformed = PIL_to_th(
            TF.affine(
                Image.fromarray(im),
                angle=0.0,
                translate=list(b * 64),
                scale=1.0,
                shear=0.0,
                fill=255,
            )
        )
    b = torch.tensor(b)
    return (
        transformed,
        A,
        b,
    )  # 64 because the images are 64 pixels wide so the mechanism is in z space


def scale(im, order: int = 3):
    c = float(1 + 0.2 * (np.random.rand() - 0.5))  # uniform [0.9, 1.1]
    b = torch.zeros(
        2 * order,
    )
    A = c * torch.eye(2 * order)
    if isinstance(im, list):
        return (
            [
                PIL_to_th(
                    TF.affine(
                        Image.fromarray(i),
                        angle=0.0,
                        translate=[0, 0],
                        scale=c,
                        shear=0.0,
                        fill=255,
                    )
                )
                for i in im
            ],
            A,
            b,
        )
    else:
        return (
            PIL_to_th(
                TF.affine(
                    Image.fromarray(im),
                    angle=0.0,
                    translate=[0, 0],
                    scale=c,
                    shear=0.0,
                    fill=255,
                )
            ),
            A,
            b,
        )


# augmentations = [random_rotate, hflip, vflip, shift]  # , scale]
ALL_AUGMENTATIONS = {
    "shift": shift,
    "hflip": hflip,
    "vflip": vflip,
    "scale": scale,
    "rotate": random_rotate,
}
