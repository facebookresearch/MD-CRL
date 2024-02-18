import torch.utils.data
import torch
import os
import numpy as np
from typing import Callable, Optional
import pygame
from pygame import gfxdraw

if "SDL_VIDEODRIVER" not in os.environ:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dsp"

COLOURS_ = [
    [2, 156, 154],
    [222, 100, 100],
    [149, 59, 123],
    [74, 114, 179],
    [27, 159, 119],
    [218, 95, 2],
    [117, 112, 180],
    [232, 41, 139],
    [102, 167, 30],
    [231, 172, 2],
    [167, 118, 29],
    [102, 102, 102],
]


SCREEN_DIM = 64
Y_SHIFT = -0.9


def circle(
    x_,
    y_,
    surf,
    color=(204, 204, 0),
    radius=0.1,
    screen_width=SCREEN_DIM,
    y_shift=Y_SHIFT,
    offset=None,
):
    if offset is None:
        offset = screen_width / 2
    scale = screen_width
    x = scale * x_ + offset
    y = scale * y_ + offset

    gfxdraw.aacircle(
        surf, int(x), int(y - offset * y_shift), int(radius * scale), color
    )
    gfxdraw.filled_circle(
        surf, int(x), int(y - offset * y_shift), int(radius * scale), color
    )

    temp_surf = pygame.Surface((screen_width, screen_width), pygame.SRCALPHA)

    gfxdraw.aacircle(
        temp_surf, int(x), int(y - offset * y_shift), int(radius * scale), color
        )
    gfxdraw.filled_circle(
        temp_surf, int(x), int(y - offset * y_shift), int(radius * scale), color
    )

    temp_surf_pos = (0,0)
    ball_mask = pygame.mask.from_surface(temp_surf)

    # mask -› surface
    new_temp_surf = ball_mask.to_surface()
    # do the same flip as the one occurring for the screen
    new_temp_surf = pygame.transform.flip(new_temp_surf, False, True)
    new_temp_surf.set_colorkey((0,0,0))

    # print(np.array(pygame.surfarray.pixels3d(temp_surf))[:, :, :1].shape)
    # print(np.array(pygame.surfarray.pixels3d(new_temp_surf))[:, :, :1].shape)
    # return np.array(pygame.surfarray.pixels3d(new_temp_surf))[:, :, :1] # [screen_width, screen_width, 1]
    return np.transpose(np.array(pygame.surfarray.pixels3d(new_temp_surf)), axes=(1, 0, 2))[:, :, :1] # [screen_width, screen_width, 1]



class BallsDataset(torch.utils.data.Dataset):
    ball_rad = 0.04 # 0.04, 0.12
    screen_dim = 64

    def __init__(
        self,
        transform: Optional[Callable] = None, # type: ignore
        augmentations: list = [],
        human_mode: bool = False,
        # n_colours: int = 1,
        num_samples: int = 20000,
        **kwargs,
    ):
        super(BallsDataset, self).__init__()
        if transform is None:

            def transform(x):
                return x

        # self.n_colours = n_colours
        self.num_samples = num_samples
        self.transform = transform
        self.human_mode = human_mode
        self.augmentations = augmentations

        if kwargs.get("color_selection", None) is not None:
            self.color_selection = kwargs["color_selection"]
        else:
            self.color_selection = "cyclic_fixed"

    def __len__(self) -> int:
        return self.num_samples

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {len(self)}"]

        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " + line for line in body]
        return "\n".join(lines)

    def _draw_scene(self, z, colours=None):
        self.surf.fill((255, 255, 255))
        # getting the background segmentation mask
        self.bg_surf = pygame.Surface((self.screen_dim, self.screen_dim), pygame.SRCALPHA)

        obj_masks = []
        if z.ndim == 1:
            z = z.reshape((1, 2))
        if colours is None:
            colours = [COLOURS_[3]] * z.shape[0]
        for i in range(z.shape[0]):
            obj_masks.append(
                circle(
                    z[i, 0],
                    z[i, 1],
                    self.surf,
                    color=colours[i],
                    radius=self.ball_rad,
                    screen_width=self.screen_dim,
                    y_shift=0.0,
                    offset=0.0,
                )
            )
            _ = circle(
                z[i, 0],
                z[i, 1],
                self.bg_surf,
                color=colours[i],
                radius=self.ball_rad,
                screen_width=self.screen_dim,
                y_shift=0.0,
                offset=0.0,
            )

        bg_surf_pos = (0,0)
        bg_mask = pygame.mask.from_surface(self.bg_surf)
        bg_mask.invert() # so that mask bits for balls are cleared and the bg gets set.

        # mask -› surface
        new_bg_surf = bg_mask.to_surface()
        new_bg_surf.set_colorkey((0,0,0))
        # do the same flip as the one occurring for the screen
        new_bg_surf = pygame.transform.flip(new_bg_surf, False, True)

        # print(np.array(pygame.surfarray.pixels3d(new_bg_surf)).shape)
        # bg_mask = np.array(pygame.surfarray.pixels3d(new_bg_surf))[:, :, :1] # [screen_width, screen_width, 1]
        bg_mask = np.transpose(np.array(pygame.surfarray.pixels3d(new_bg_surf)), axes=(1, 0, 2))[:, :, :1] # [screen_width, screen_width, 1]
        # ------------------------------------------ #
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.human_mode:
            pygame.display.flip()
        return (
            np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
                )
            , np.array([bg_mask] + obj_masks)
        )

    def _setup(self, screen_dim=None):
        screen_dim = screen_dim if screen_dim is not None else self.screen_dim
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_dim, self.screen_dim))
        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))

    def _teardown(self):
        del self.surf
        del self.screen
        pygame.quit()

