import torch
import torchvision
from torchvision import transforms
import numpy as np
import random
from typing import Callable, Optional
import utils.general as utils
log = utils.get_logger(__name__)
from tqdm import tqdm
import copy
log_ = True
from .md_mixing_dataset_pickle import MDMixingPickleable

class SyntheticMixingDataset(torch.utils.data.Dataset):
    """
    This class instantiates a torch.utils.data.Dataset object.
    """

    def __init__(
        self,
        transform: Optional[Callable] = None, # type: ignore
        generation_strategy: str = "auto",
        num_samples: int = 20000,
        num_domains: int = 2,
        z_dim: int = 10,
        z_dim_invariant: int = 5,
        x_dim: int = 20,
        domain_lengths: Optional[list] = None,
        domain_dist_ranges: Optional[list] = None,
        domain_dist_ranges_pos: Optional[list] = None,
        domain_dist_ranges_neg: Optional[list] = None,
        invariant_dist_params: Optional[list] = None,
        linear: bool = True,
        **kwargs,
    ):
        super(SyntheticMixingDataset, self).__init__()

        if transform is None:
            def transform(x):
                return x

        self.transform = transform
        self.generation_strategy = generation_strategy
        self.z_dim = z_dim
        self.z_dim_invariant = z_dim_invariant
        self.z_dim_spurious = int(z_dim - z_dim_invariant)
        self.x_dim = x_dim
        self.num_samples = num_samples
        self.num_domains = num_domains
        self.domain_lengths = domain_lengths if generation_strategy == "manual" else [1 / num_domains] * num_domains
        self.domain_dist_ranges = domain_dist_ranges
        # self.domain_dist_ranges_pos = domain_dist_ranges_pos
        # self.domain_dist_ranges_neg = domain_dist_ranges_neg
        self.invariant_dist_params = invariant_dist_params
        self.mixing_architecture_config = kwargs["mixing_architecture"]
        self.linear = linear
        self.non_linearity = kwargs["non_linearity"]
        self.polynomial_degree = kwargs["polynomial_degree"]
        self.correlated_z = kwargs["correlated_z"]
        self.corr_prob = kwargs["corr_prob"]
        self._mixing_G = self._generate_mixing_G(linear, z_dim, x_dim)
        self.data = self._generate_data()
        self.pickleable_dataset = MDMixingPickleable(self, self.data)

    def __len__(self) -> int:
        return self.num_samples

    def _generate_data(self):

        # data is a tensor of size [num_samples, z_dim] where the first z_dim_invariant dimensions are sampled from uniform [0,1]
        z_data = torch.zeros(self.num_samples, self.z_dim)
        # the first z_dim_invariant dimensions are sampled from uniform [0,1]
        z_data_invar = torch.rand(self.num_samples, self.z_dim_invariant) * (self.invariant_dist_params[1] - self.invariant_dist_params[0]) + self.invariant_dist_params[0]
        z_data[:, :self.z_dim_invariant] = z_data_invar

        self.spurious_lows = np.zeros((self.z_dim_spurious, self.num_domains))
        self.spurious_highs = np.zeros((self.z_dim_spurious, self.num_domains))

        if self.generation_strategy == "manual":
            # for each domain, create its data, i.e., a tensor of size [num_samples, z_dim-z_dim_invariant] 
            # where each dimension is sampled from uniform [domain_dist_ranges[domain_idx][0], domain_dist_ranges[domain_idx][0]]
            domain_mask = torch.zeros(self.num_samples, 1)
            start = 0
            for domain_idx in range(self.num_domains):
                domain_size = int(self.domain_lengths[domain_idx] * self.num_samples)
                end = domain_size + start
                domain_data = torch.rand(domain_size, self.z_dim_spurious) * (self.domain_dist_ranges[domain_idx][1] - self.domain_dist_ranges[domain_idx][0]) + self.domain_dist_ranges[domain_idx][0]
                z_data[start:end, self.z_dim_invariant:] = domain_data
                domain_mask[start:end] = domain_idx
                start = end
        else:
            # for each dimension of the spurious part of z, for each domain, first decide whether it is positive or negative
            # by tossing a fair coin. Then sample the low and high of this domain from the corresponding domain_dist_ranges_pos
            # or domain_dist_ranges_neg. Then sample from uniform [low, high] for each domain. 
            # Repeat this procedure for all spurious dimensions of z.
            # note that domain mask in this case is not 
            domain_mask = torch.zeros(self.num_samples, 1)
            for dim_idx in range(self.z_dim_spurious):
                for domain_idx in range(self.num_domains):
                    # toss a fair coin to decide whether this dimension is positive or negative
                    # coin = random.randint(0, 1)
                    # if coin == 0:
                    #     # sample low and high of the domain distribution from the range
                    #     # specified in negative domain distribution. Make sure low < high
                    #     low = random.random() * (self.domain_dist_ranges_neg[1] - self.domain_dist_ranges_neg[0]) + self.domain_dist_ranges_neg[0]
                    #     high = random.random() * (self.domain_dist_ranges_neg[1] - self.domain_dist_ranges_neg[0]) + self.domain_dist_ranges_neg[0]
                    #     while low > high:
                    #         low = random.random() * (self.domain_dist_ranges_neg[1] - self.domain_dist_ranges_neg[0]) + self.domain_dist_ranges_neg[0]
                    #         high = random.random() * (self.domain_dist_ranges_neg[1] - self.domain_dist_ranges_neg[0]) + self.domain_dist_ranges_neg[0]
                    # else:
                    #     # sample low and high of the domain distribution from the range
                    #     # specified in positive domain distribution. Make sure low < high
                    #     low = random.random() * (self.domain_dist_ranges_pos[1] - self.domain_dist_ranges_pos[0]) + self.domain_dist_ranges_pos[0]
                    #     high = random.random() * (self.domain_dist_ranges_pos[1] - self.domain_dist_ranges_pos[0]) + self.domain_dist_ranges_pos[0]
                    #     while low > high:
                    #         low = random.random() * (self.domain_dist_ranges_pos[1] - self.domain_dist_ranges_pos[0]) + self.domain_dist_ranges_pos[0]
                    #         high = random.random() * (self.domain_dist_ranges_pos[1] - self.domain_dist_ranges_pos[0]) + self.domain_dist_ranges_pos[0]
                    low = random.random() * (self.domain_dist_ranges[1] - self.domain_dist_ranges[0]) + self.domain_dist_ranges[0]
                    high = random.random() * (self.domain_dist_ranges[1] - self.domain_dist_ranges[0]) + self.domain_dist_ranges[0]
                    while low > high:
                        low = random.random() * (self.domain_dist_ranges[1] - self.domain_dist_ranges[0]) + self.domain_dist_ranges[0]
                        high = random.random() * (self.domain_dist_ranges[1] - self.domain_dist_ranges[0]) + self.domain_dist_ranges[0]
                    
                    self.spurious_lows[dim_idx, domain_idx] = low
                    self.spurious_highs[dim_idx, domain_idx] = high

                    # sample from uniform [low, high] for each domain
                    domain_size = int(self.domain_lengths[domain_idx] * self.num_samples)
                    start = int(domain_size * domain_idx)
                    end = start + domain_size
                    domain_data = torch.rand(domain_size) * (high - low) + low
                    z_data[start:end, self.z_dim_invariant + dim_idx] = domain_data
                    domain_mask[start:end] = domain_idx # this is repetitive after the first loop over domain_idx, but it's ok

        # now the data is a tensor of size [num_samples, z_dim] where the first z_dim_invariant
        # dimensions are sampled from uniform [0,1], and the rest are sampled according to domain distributions
        # now shuffle the data and the domain mask similarly and create the final data
        indices = list(range(self.num_samples))
        random.shuffle(indices)
        z_data = z_data[indices]
        domain_mask = domain_mask[indices]
        if self.correlated_z:
            z_data = self._correlate_z(z_data, domain_mask)
        x_data = self._mixing_G(z_data)

        if not self.linear and self.non_linearity == "polynomial":
            # x_data: [n, poly_size]. Below we normalize and transform it to [n, x_dim]
            x1 = torch.matmul(x_data[:, :1+self.z_dim], torch.tensor(self.coff_matrix[:1+self.z_dim, :], dtype=x_data.dtype))
            # print('X1')
            # print('Min', torch.min(torch.abs(x1)), 'Max', torch.max(torch.abs(x1)), 'Mean', torch.mean(torch.abs(x1)))

            x2 = torch.matmul(x_data[:, 1+self.z_dim:], torch.tensor(self.coff_matrix[1+self.z_dim:, :], dtype=x_data.dtype))
            norm_factor = 0.5 * torch.max(torch.abs(x2)) / torch.max(torch.abs(x1)) 
            x2 = x2 / norm_factor
            # print('X2')
            # print('Min', torch.min(torch.abs(x2)), 'Max', torch.max(torch.abs(x2)), 'Mean', torch.mean(torch.abs(x2)))
            x_data = (x1+x2)

        return x_data, z_data, domain_mask

    def __getitem__(self, idx):
        return {"x": self.data[0][idx], "z": self.data[1][idx], "domain": self.data[2][idx]}

    def _generate_mixing_G(self, linear, z_dim, x_dim):
        if linear:
            # create an invertible matrix mapping from z_dim to x_dim for which
            # the entries are sampled from uniform [0,1]
            # this matrix should be operating on batches of size [num_samples, z_dim]

            # TODO: This G should be used by all splits of the dataset!
            G = torch.rand(z_dim, x_dim)
            # make sure the above G is full rank

            while np.linalg.matrix_rank(G) < min(z_dim, x_dim):
                G = torch.rand(z_dim, x_dim)
            self.G = G
            # return a lambda function that takes a batch of z and returns Gz
            return lambda z: torch.matmul(z, G)

        else:
            if self.non_linearity == "polynomial":
                # return a function that takes a batch of z and returns a multivariate
                # polynomial of z

                poly_size = compute_total_polynomial_terms(self.polynomial_degree, self.z_dim)
                # Generate random coefficients for the polynomial
                self.coff_matrix = np.random.multivariate_normal(np.zeros(poly_size), np.eye(poly_size), size=self.x_dim).T

                from functools import partial
                self.G = partial(compute_decoder_polynomial, self.polynomial_degree)

                # Define the polynomial function using lambda
                # return lambda z: torch.tensor(np.concatenate(list(map(lambda idx: np.matmul(self.G(z[idx, :]), self.coff_matrix), range(z.shape[0]))), axis=0), dtype=torch.float32)
                return lambda z: torch.tensor(np.concatenate(list(map(lambda idx: self.G(z[idx, :]), range(z.shape[0]))), axis=0), dtype=torch.float32)
                

            elif self.non_linearity == "mlp":
                # instantiate the non-linear G with hydra
                self.G = torch.nn.Sequential(
                *[layer_config for _, layer_config in self.mixing_architecture_config.items()]
                )

                # print all of the parameters of self.G
                if log_:
                    log.info("G params:")
                    for name, param in self.G.named_parameters():
                        log.info(f"{name}: {param}")
                # return a lambda function that takes a batch of z and returns Gz
                # make sure the output does not require any grads, and is simply a torch tensor
                return lambda z: self.G(z).detach()

    def _correlate_z(self, z, domain_mask):

        # # z: [n, z_dim]
        # # sample a rotation matrix from scipy to rotate z, make sure the dtype is float32
        # from scipy.stats import special_ortho_group
        # rotation_matrix = special_ortho_group.rvs(z.shape[1]).astype(np.float32) # [z_dim, z_dim]

        # # rotate z
        # z = np.matmul(z, rotation_matrix)

        # # clamp the invariant part of z to self.invariant_dist_params
        # z[:, :self.z_dim_invariant] = np.clip(z[:, :self.z_dim_invariant], self.invariant_dist_params[0], self.invariant_dist_params[1])

        # # clamp the spurious part of z to self.spurious_lows and self.spurious_highs corresponding 
        # # to each domain
        # for dim_idx in range(self.z_dim_spurious):
        #     for domain_idx in range(self.num_domains):
        #         low = self.spurious_lows[dim_idx, domain_idx]
        #         high = self.spurious_highs[dim_idx, domain_idx]
        #         domain_indices = (domain_mask == domain_idx).squeeze()
        #         z[domain_indices, self.z_dim_invariant + dim_idx] = np.clip(z[domain_indices, self.z_dim_invariant + dim_idx], low, high)

        # change the spurious dimensions as follows: for each sample and z_dim spurious, toss a coin that with
        # probability p comes head and with probability 1-p comes tail. If it comes head,
        # add the z_dim_invariant to the spurious dimension, otherwise leave it as is
        offset = torch.zeros((z.shape[0], self.z_dim_spurious))
        coin = np.random.binomial(1, self.corr_prob, size=(z.shape[0], self.z_dim_spurious)) # [n, z_dim_spurious]
        for dim_idx in range(self.z_dim_spurious):
            offset[:, dim_idx] = torch.tensor(coin[:, dim_idx]) * z[:, dim_idx]

        # coin = np.random.randint(0, 2, size=(z.shape[0], self.z_dim_spurious)) # [n, z_dim_spurious]
        # for dim_idx in range(self.z_dim_spurious):
        #     offset[:, dim_idx] = coin[:, dim_idx] * z[:, self.z_dim_invariant]

        z[:, self.z_dim_invariant:] = z[:, self.z_dim_invariant:] + offset

        return z

def compute_total_polynomial_terms(poly_degree, latent_dim):
    count=0
    for degree in range(poly_degree+1):
        count+= pow(latent_dim, degree)
    return count

def compute_kronecker_product(degree, latent):
    if degree ==0:
        out = np.array([1])
    else:
        out = copy.deepcopy(latent)
        for idx in range(1, degree):
            out= np.kron(out, latent)
    return out

def compute_decoder_polynomial(poly_degree, latent):
    out = []
    for degree in range(poly_degree+1):
        out.append(compute_kronecker_product(degree, latent))

    out = np.concatenate(out)
    out = np.reshape(out, (1,out.shape[0]))
    return out
