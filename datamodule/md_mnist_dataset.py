from .abstract_mnist_dataset import MNISTBase
import torch
import torchvision
from torchvision import transforms
import numpy as np
from typing import Callable, Optional
import utils.general as utils
log = utils.get_logger(__name__)
import random
from tqdm import tqdm
log_ = False

COLORS = [
    [1., 0., 0.], # red
    [0., 1., 0.], # green
    [0., 0., 1.], # blue
    [1., 1., 0.], # yellow
    [1., 0., 1.], # magenta
    [0., 1., 1.], # cyan
    [1., 1., 1.], # white
    [0., 0., 0.], # black
]
class MNISTMultiDomainDataset(MNISTBase):
    """
    This class instantiates a torch.utils.data.Dataset object.
    """

    def __init__(
        self,
        transform: Optional[Callable] = None, # type: ignore
        num_samples: int = 20000,
        path: str = "/network/datasets/torchvision",
        path_narval: str = "/home/aminm/scratch/",
        num_domains: int = 2,
        domain_lengths: Optional[list] = None,
        generation_strategy: str = "auto",
        spurious_method: str = "digits",
        # list of list color indices for each domain
        domain_color_list: Optional[list] = None,
        domain_color_probs: Optional[list] = None,
        normalization: str = "z_score", # z_score, none
        **kwargs,
    ):
        super(MNISTMultiDomainDataset, self).__init__(transform
                                            , num_samples
                                            , path
                                            ,**kwargs
                                            )
        if transform is None:
            def transform(x):
                return x

        self.transform = transform
        self.generation_strategy = generation_strategy
        self.spurious_method = spurious_method
        self.correlated_z = kwargs["correlated_z"]
        self.corr_prob = kwargs["corr_prob"]
        self.dynamic_corr_prob = kwargs["dynamic_corr_prob"]
        self.path = path
        self.path_narval = path_narval
        self.split = kwargs.get("split", "train")
        # self.num_samples = num_samples
        self.num_domains = num_domains
        self.domain_lengths = domain_lengths if generation_strategy == "manual" else [1 / num_domains] * num_domains
        self.domain_color_list = domain_color_list
        self.domain_color_probs = domain_color_probs
        self.normalization = normalization
        self.data = self._generate_data()

    def _generate_data(self):
        new_data = {}
        if self.split == "train":
            try:
                data = torchvision.datasets.MNIST(self.path, True, transform=self.transform)
            except:
                data = torchvision.datasets.MNIST(self.path_narval, True, transform=self.transform)
        else:
            try:
                data = torchvision.datasets.MNIST(self.path, False, transform=self.transform)
            except:
                data = torchvision.datasets.MNIST(self.path_narval, False, transform=self.transform)

        min_value_channel = data[0][0].min()
        # create the new data by extending each image with two more color channels
        # example: domain_indices[0] = [0, 2, 5, 6, 8, 9, 10], domain_indices[1] = [1, 3, 4, 7]
        # domains_indices, mapping = random_split(list(range(self.num_samples)), [self.num_samples * x for x in self.domain_lengths])
        domains_indices, mapping = random_split(list(range(len(data))), [len(data) * x for x in self.domain_lengths])

        # for each split, create the splits of colors
        for domain_idx in range(self.num_domains):
            # example: domain_indices[0] = domain_indices_ = [0, 2, 5, 6, 8, 9, 10]
            domain_indices_ = domains_indices[domain_idx] # some indices of the range(len(data))

            if self.generation_strategy == "manual":
                # splits of the range(len(domain_indices_)), i.e., domain_color_indices[0] is a list of indices of domain domain_idx that have the first color, and so on
                # example: domain_color_indices[0] = [0, 1, 4, 5], domain_color_indices[1] = [2, 3, 6]
                # , with the mapping to domain_indices_ being: domain_color_indices[0] = [0, 1, 4, 5] 
                # corresponding to domain_indices_[domain_color_indices[0]] = domain_indices_[[0, 1, 4, 5]] = [0, 2, 8, 9]
                domain_color_indices, _ = random_split(list(range(len(domain_indices_))), [len(domain_indices_) * x for x in self.domain_color_probs[domain_idx]])
                for color_idx in range(len(domain_color_indices)): # number of colors per each domain
                    color_indices = domain_color_indices[color_idx] # indices of domain domain_idx that have the color color_idx, e.g., [0, 1, 4, 5]
                    color = self.domain_color_list[domain_idx][color_idx] # selected from the list of colors for domain domain_idx
                    for idx in color_indices: # for idx in [0, 1, 4, 5]
                        img, label = data[domain_indices_[idx]] # domain_indices_[idx] = [0, 2, 8, 9]
                        img = img.permute(1, 2, 0).repeat(1, 1, 3)
                        # change the color of bright pixels to color
                        digit_pixels = img[:, :, 0] > 0.3
                        bg_pixels = img[:, :, 0] <= 0.3
                        img[digit_pixels, :] = img[digit_pixels, :] * torch.tensor(COLORS[color]).float()
                        img[bg_pixels, :] = torch.tensor([min_value_channel, min_value_channel, min_value_channel]).float()

                        new_data[domain_indices_[idx]] = (img.permute(2, 0, 1), label, mapping[domain_indices_[idx]], color)
            
            else:
                # in each domain, we will sample the colors of the digits by a linear 
                # combination of RGB, where the coefficients of the linear combination are
                # sampled from a uniform distribution for which the low and high are different
                # in each domain, i.e, those are sampled as well.

                # sample the low and high of Red coefficient from a uniform distribution
                low_r, high_r = random.random(), random.random()
                while low_r > high_r:
                    low_r, high_r = random.random(), random.random()
                # sample the low and high of Green coefficient from a uniform distribution
                low_g, high_g = random.random(), random.random()
                while low_g > high_g:
                    low_g, high_g = random.random(), random.random()
                # sample the low and high of Blue coefficient from a uniform distribution
                low_b, high_b = random.random(), random.random()
                while low_b > high_b:
                    low_b, high_b = random.random(), random.random()

                if not self.correlated_z:
                    # sample the coefficients of red, green, and blue from the corresponding uniform distributions
                    r = torch.rand(len(domain_indices_)) * (high_r - low_r) + low_r
                    g = torch.rand(len(domain_indices_)) * (high_g - low_g) + low_g
                    b = torch.rand(len(domain_indices_)) * (high_b - low_b) + low_b
                else:

                    domain_indices_0_4 = torch.tensor([data[idx][1] < 5 for idx in domain_indices_]).bool()
                    # domain_indices_0_4_indices = torch.nonzero(domain_indices_0_4).squeeze()
                    domain_indices_5_9 = torch.tensor([data[idx][1] >= 5 for idx in domain_indices_]).bool()
                    # domain_indices_5_9_indices = torch.nonzero(domain_indices_5_9).squeeze()
                    # for digits 0 to 4, sample the coefficients of red, green, and blue from the first half of the corresponding uniform distributions
                    # for digits 5 to 9, sample the coefficients of red, green, and blue from the second half of the corresponding uniform distributions
                    r_0_4 = torch.rand(domain_indices_0_4.sum().item()) * ((high_r - low_r) / 2) + low_r
                    r_5_9 = torch.rand(domain_indices_5_9.sum().item()) * ((high_r - low_r) / 2) + ((high_r - low_r) / 2) + low_r
                    
                    # # toss a coin and for 95% percent of the time when digit is 0 to 4, sample the coefficients of red, green, and blue from the other half corresponding to digits 5-9
                    # # and for 95% percent of the time when digit is 5 to 9, sample the coefficients of red, green, and blue from the other half corresponding to digits 0-4
                    # # this is to create a dynamic SCM between the digit and the color
                    # coin = torch.rand(len(domain_indices_)) > self.dynamic_corr_prob
                    # r_0_4[coin[domain_indices_0_4]] = torch.rand(coin[domain_indices_0_4].sum().item()) * ((high_r - low_r) / 2) + ((high_r - low_r) / 2) + low_r
                    # r_5_9[coin[domain_indices_5_9]] = torch.rand(coin[domain_indices_5_9].sum().item()) * ((high_r - low_r) / 2) + low_r

                    # concat r_0_4 and r_5_9 to get r such that their indices align with their position in domain_indices_
                    # i.e., r[i] should be r_0_4 if domain_indices_[i] is in domain_indices_0_4, and r[i] should be r_5_9 if domain_indices_[i] is in domain_indices_5_9

                    r = torch.zeros(len(domain_indices_))
                    r[domain_indices_0_4] = r_0_4
                    r[domain_indices_5_9] = r_5_9


                    g_0_4 = torch.rand(domain_indices_0_4.sum().item()) * ((high_g - low_g) / 2) + low_g
                    g_5_9 = torch.rand(domain_indices_5_9.sum().item()) * ((high_g - low_g) / 2) + ((high_g - low_g) / 2) + low_g
                    # # dynamic SCM
                    # g_0_4[coin[domain_indices_0_4]] = torch.rand(coin[domain_indices_0_4].sum().item()) * ((high_g - low_g) / 2) + ((high_g - low_g) / 2) + low_g
                    # g_5_9[coin[domain_indices_5_9]] = torch.rand(coin[domain_indices_5_9].sum().item()) * ((high_g - low_g) / 2) + low_g

                    g = torch.zeros(len(domain_indices_))
                    g[domain_indices_0_4] = g_0_4
                    g[domain_indices_5_9] = g_5_9


                    b_0_4 = torch.rand(domain_indices_0_4.sum().item()) * ((high_b - low_b) / 2) + low_b
                    b_5_9 = torch.rand(domain_indices_5_9.sum().item()) * ((high_b - low_b) / 2) + ((high_b - low_b) / 2) + low_b
                    # # dynamic SCM
                    # b_0_4[coin[domain_indices_0_4]] = torch.rand(coin[domain_indices_0_4].sum().item()) * ((high_b - low_b) / 2) + ((high_b - low_b) / 2) + low_b
                    # b_5_9[coin[domain_indices_5_9]] = torch.rand(coin[domain_indices_5_9].sum().item()) * ((high_b - low_b) / 2) + low_b

                    b = torch.zeros(len(domain_indices_))
                    b[domain_indices_0_4] = b_0_4
                    b[domain_indices_5_9] = b_5_9

                    # now resample the colors of the digits with probability 1-corr_prob from the whole range of the uniform distribution
                    r_ = torch.rand(len(domain_indices_)) * (high_r - low_r) + low_r
                    g_ = torch.rand(len(domain_indices_)) * (high_g - low_g) + low_g
                    b_ = torch.rand(len(domain_indices_)) * (high_b - low_b) + low_b
                    resample = torch.rand(len(domain_indices_)) > self.corr_prob
                    r[resample] = r_[resample]
                    g[resample] = g_[resample]
                    b[resample] = b_[resample]
                    

                if self.spurious_method == "digits":
                    # for each image in domain domain_idx, change the color of bright pixels to a linear combination of RGB
                    for idx in tqdm(range(len(domain_indices_))):
                        img, label = data[domain_indices_[idx]]
                        img = img.permute(1, 2, 0).repeat(1, 1, 3)
                        # change the color of bright pixels to color
                        digit_pixels = img[:, :, 0] > 0.3
                        bg_pixels = img[:, :, 0] <= 0.3
                        img[digit_pixels, :] = img[digit_pixels, :] * torch.tensor([r[idx], g[idx], b[idx]]).float()
                        img[bg_pixels, :] = torch.tensor([min_value_channel, min_value_channel, min_value_channel]).float()

                        new_data[domain_indices_[idx]] = (img.permute(2, 0, 1), label, mapping[domain_indices_[idx]], torch.tensor([r[idx], g[idx], b[idx]]).float())

                elif self.spurious_method == "background":
                    # for each image in domain domain_idx, change the color of background pixels to a linear combination of RGB
                    for idx in tqdm(range(len(domain_indices_))):
                        img, label = data[domain_indices_[idx]]
                        img = img.permute(1, 2, 0).repeat(1, 1, 3)
                        # change the color of bright pixels to color
                        digit_pixels = img[:, :, 0] > 0.3
                        bg_pixels = img[:, :, 0] <= 0.3
                        img[bg_pixels, :] = torch.tensor([r[idx], g[idx], b[idx]]).float()

                        new_data[domain_indices_[idx]] = (img.permute(2, 0, 1), label, mapping[domain_indices_[idx]], torch.tensor([r[idx], g[idx], b[idx]]).float())

                elif self.spurious_method == "fixed_corner":
                    # for each image in domain domain_idx, change the color of the top-left corner to a linear combination of RGB
                    corner_side = 7
                    for idx in tqdm(range(len(domain_indices_))):
                        img, label = data[domain_indices_[idx]]
                        img = img.permute(1, 2, 0).repeat(1, 1, 3)
                        img[0:corner_side, 0:corner_side, :] = torch.tensor([r[idx], g[idx], b[idx]]).float()

                        new_data[domain_indices_[idx]] = (img.permute(2, 0, 1), label, mapping[domain_indices_[idx]], torch.tensor([r[idx], g[idx], b[idx]]).float())
                
                elif self.spurious_method == "random_corner":
                    # for each image in domain domain_idx, change the color of a random corner to a linear combination of RGB
                    corner_side = 7
                    for idx in tqdm(range(len(domain_indices_))):
                        img, label = data[domain_indices_[idx]]
                        img = img.permute(1, 2, 0).repeat(1, 1, 3)
                        corner = random.randint(0, 3)
                        if corner == 0:
                            img[0:corner_side, 0:corner_side, :] = torch.tensor([r[idx], g[idx], b[idx]]).float()
                        elif corner == 1:
                            img[0:corner_side, 28-corner_side:, :] = torch.tensor([r[idx], g[idx], b[idx]]).float()
                        elif corner == 2:
                            img[28-corner_side:, 0:corner_side, :] = torch.tensor([r[idx], g[idx], b[idx]]).float()
                        else:
                            img[28-corner_side:, 28-corner_side:, :] = torch.tensor([r[idx], g[idx], b[idx]]).float()

                        new_data[domain_indices_[idx]] = (img.permute(2, 0, 1), label, mapping[domain_indices_[idx]], torch.tensor([r[idx], g[idx], b[idx]]).float())
        
        new_data_ = list(new_data.values())
        if self.normalization == "z_score":
            # normalize the img values using the z_score method
            # find the mean of all pixel values of all images and their std
            mean = torch.cat([x[0].flatten() for x in new_data_]).mean()
            std = torch.cat([x[0].flatten() for x in new_data_]).std()
            self.mean = mean.detach().numpy()
            self.std = std.detach().numpy()
            print(f"mean, std: {mean}, {std}")
            for idx in range(len(new_data)):
                img, label, domain, color = new_data[idx]
                img = (img - mean) / std
                new_data[idx] = (img, label, domain, color)
        else:
            min_ = torch.cat([x[0].flatten() for x in new_data_]).min()
            max_ = torch.cat([x[0].flatten() for x in new_data_]).max()
            self.min_ = min_.detach().numpy()
            self.max_ = max_.detach().numpy()
            print(f"min, max: {min_}, {max_}")
            for idx in range(len(new_data)):
                img, label, domain, color = new_data[idx]
                img = (img - min_) / (max_ - min_)
                new_data[idx] = (img, label, domain, color)

        return new_data

    def __getitem__(self, idx):
        return {"image": self.data[idx][0], "label": self.data[idx][1], "domain": self.data[idx][2], "color": self.data[idx][3]}
    
    def renormalize(self):
        if self.normalization == "z_score":
            # return a lambda function that re-adjusts the values of the normalized images
            # to be in the range [0, 1]
            return lambda x: (x * self.std) + self.mean
        else:
            # return a lambda function that reverses the minmax normalization
            # to be in the range [0, 1]
            return lambda x: (x * (self.max_ - self.min_)) + self.min_


def random_split(array, sizes):
    if sum(sizes) != len(array):
        raise ValueError("Sum of sizes must equal the length of the array")

    indices = list(range(len(array)))
    random.shuffle(indices)

    splits = []
    start = 0
    split_id = 0
    mapping = {}
    for size in sizes:
        end = int(start + size)
        split_indices = indices[start:end]
        split = [array[i] for i in indices[start:end]]
        splits.append(split)
        for idx in split_indices:
            mapping[array[idx]] = split_id
        start = end
        split_id += 1

    return splits, mapping

