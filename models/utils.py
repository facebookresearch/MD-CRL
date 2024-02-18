import numpy as np
import torch
from torch import nn
import collections.abc

def set_seed(args):
    np.random.seed(args.seed)

    # This makes sure that the seed is used for random initialization of nn modules provided by nn init
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


#     random.seed(SEED)
#     np.random.seed(SEED)
#     torch.manual_seed(SEED)
#     torch.cuda.manual_seed(SEED)
# Shouldn't we use this last one?
#     torch.backends.cudnn.deterministic = True


# Don't worry about randomization and seed here. It's taken care of by set_seed above, and pl seed_everything
def init_weights(model):
    for name, param in model.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def update(d, u):
    """Performs a multilevel overriding of the values in dictionary d with the values of dictionary u"""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


from torch.nn import functional as F
def penalty_loss_minmax(z, domains, num_domains, z_dim_invariant, *args):

    top_k = args[0]
    loss_transform = args[1]
    # domain_z_mins is a torch tensor of shape [num_domains, d, top_k] containing the top_k smallest
    # values of the first d dimensions of z for each domain
    # domain_z_maxs is a torch tensor of shape [num_domains, d, top_k] containing the top_k largest
    # values of the first d dimensions of z for each domain
    domain_z_mins = torch.zeros((num_domains, z_dim_invariant, top_k))
    domain_z_maxs = torch.zeros((num_domains, z_dim_invariant, top_k))
    domain_skip = torch.zeros(num_domains)

    # z is [batch_size, latent_dim], so is domains. For the first d dimensions
    # of z, find the top_k smallest values of that dimension in each domain
    # find the mask of z's for each domain
    # for each domain, and for each of the first d dimensions, 
    # find the top_k smallest values of that z dimension in that domain
    for domain_idx in range(num_domains):
        domain_mask = (domains == domain_idx).squeeze()
        domain_z = z[domain_mask]

        if domain_z.shape[0] == 0:
            domain_skip[domain_idx] = 1
            continue
        elif domain_z.shape[0] < top_k:
            print(f"WARNING: domain_z.shape[0] < top_k for domain {domain_idx}")
            domain_z_mins[domain_idx, :, domain_z.shape[0]:] = 0.
            domain_z_maxs[domain_idx, :, domain_z.shape[0]:] = 0.
        else:
            for i in range(z_dim_invariant):
                # for each dimension i among the first d dimensions of z, find the top_k
                # smallest values of dimension i in domain_z
                domain_z_sorted, _ = torch.sort(domain_z[:, i], dim=0)
                domain_z_sorted = domain_z_sorted.squeeze()
                domain_z_sorted = domain_z_sorted[:top_k]
                if domain_z.shape[0] < top_k:
                    domain_z_mins[domain_idx, i, :domain_z.shape[0]] = domain_z_sorted
                else:
                    domain_z_mins[domain_idx, i, :] = domain_z_sorted

                # find the top_k largest values of dimension i in domain_z
                domain_z_sorted, _ = torch.sort(domain_z[:, i], dim=0, descending=True)
                domain_z_sorted = domain_z_sorted.squeeze()
                if domain_z.shape[0] < top_k:
                    domain_z_maxs[domain_idx, i, :domain_z.shape[0]] = domain_z_sorted
                else:
                    domain_z_sorted = domain_z_sorted[:top_k]
                    domain_z_maxs[domain_idx, i, :] = domain_z_sorted

    # compute the pairwise mse of domain_z_mins and add them all together. Same for domain_z_maxs
    mse_mins = 0
    mse_maxs = 0
    if loss_transform == "mse":
        for i in range(num_domains):
            if domain_skip[i] == 1:
                continue
            else:
                for j in range(i+1, num_domains):
                    if domain_skip[j] == 1:
                        continue
                    else:
                        mse_mins += F.mse_loss(domain_z_mins[i], domain_z_mins[j], reduction="mean")
                        mse_maxs += F.mse_loss(domain_z_maxs[i], domain_z_maxs[j], reduction="mean")
    elif loss_transform == "l1":
        for i in range(num_domains):
            if domain_skip[i] == 1:
                continue
            else:
                for j in range(i+1, num_domains):
                    if domain_skip[j] == 1:
                        continue
                    else:
                        mse_mins += F.l1_loss(domain_z_mins[i], domain_z_mins[j], reduction="mean")
                        mse_maxs += F.l1_loss(domain_z_maxs[i], domain_z_maxs[j], reduction="mean")
    elif loss_transform == "log_mse":
        for i in range(num_domains):
            for j in range(i+1, num_domains):
                # mse_mins += F.mse_loss(torch.log(domain_z_mins[i]), torch.log(domain_z_mins[j]), reduction="mean")
                # mse_maxs += F.mse_loss(torch.log(domain_z_maxs[i]), torch.log(domain_z_maxs[j]), reduction="mean")
                # mse_mins += torch.log(F.mse_loss(domain_z_mins[i], domain_z_mins[j], reduction="none")).mean()
                # mse_maxs += torch.log(F.mse_loss(domain_z_maxs[i], domain_z_maxs[j], reduction="none")).mean()
                mse_mins += torch.log((domain_z_mins[i] - domain_z_mins[j])**2).mean()
                mse_maxs += torch.log((domain_z_maxs[i] - domain_z_maxs[j])**2).mean()
    elif loss_transform == "log_l1":
        for i in range(num_domains):
            for j in range(i+1, num_domains):
                # mse_mins += F.l1_loss(torch.log(domain_z_mins[i]), torch.log(domain_z_mins[j]), reduction="mean")
                # mse_maxs += F.l1_loss(torch.log(domain_z_maxs[i]), torch.log(domain_z_maxs[j]), reduction="mean")
                mse_mins += torch.log(F.l1_loss(domain_z_mins[i], domain_z_mins[j], reduction="none")).mean()
                mse_maxs += torch.log(F.l1_loss(domain_z_maxs[i], domain_z_maxs[j], reduction="none")).mean()

    # hinge_loss_value = hinge_loss(z, domains, num_domains, z_dim_invariant, *args[2:])
    # return (mse_mins + mse_maxs) + hinge_loss_value, hinge_loss_value
    return (mse_mins + mse_maxs)

def penalty_loss_stddev(z, domains, num_domains, z_dim_invariant, *args):
    
    # domain_z_invariant_stddev is a torch tensor of shape [num_domains, d] containing the standard deviation
    # of the first d dimensions of z for each domain
    domain_z_invariant_stddev = torch.zeros((num_domains, z_dim_invariant)).to(z.device)

    # z is [batch_size, latent_dim], so is domains. For the first d dimensions
    # of z, find the standard deviation of that dimension in each domain
    for domain_idx in range(num_domains):
        domain_mask = (domains == domain_idx).squeeze()
        domain_z = z[domain_mask]
        # for each dimension i among the first d dimensions of z, find the standard deviation
        # of dimension i in domain_z
        for i in range(z_dim_invariant):
            domain_z_stddev = torch.std(domain_z[:, i], dim=0)
            domain_z_invariant_stddev[domain_idx, i] = domain_z_stddev
    
    # for each of the d dimensions, compute the pairwise mse of its stddev across domains
    # and add them all together in mse_stddev. mse_stddev is a tensor of size [d]
    mse_stddev = torch.zeros(z_dim_invariant).to(z.device)
    for i in range(z_dim_invariant):
        for j in range(num_domains):
            for k in range(j+1, num_domains):
                mse_stddev[i] += F.mse_loss(domain_z_invariant_stddev[j, i], domain_z_invariant_stddev[k, i], reduction="mean")

    # hinge_loss_value = hinge_loss(z, domains, num_domains, z_dim_invariant, *args)
    # return mse_stddev.sum() + hinge_loss_value, hinge_loss_value
    return mse_stddev.sum()


def hinge_loss(z, domains, num_domains, z_dim_invariant, *args):
    
    gamma = args[0]
    epsilon = args[1]
    hinge_loss_weight = args[2]

    # compute the variance regularization term using the hinge loss along each dimension of z.
    # The hinge loss is 0 if the variance is greater than gamma, and gamma - sqrt(variance + epsilon)
    # otherwise. The variance regularization term is the sum of the hinge losses along each dimension
    # of z
    variance_reg = torch.zeros(num_domains).to(z.device)
    for domain_idx in range(num_domains):
        domain_mask = (domains == domain_idx).squeeze()
        domain_z = z[domain_mask]
        # for each dimension i among the first d dimensions of z, find the variance
        # of dimension i in domain_z
        for i in range(z_dim_invariant):
            variance_reg[domain_idx] += F.relu(gamma - torch.sqrt(torch.var(domain_z[:, i], dim=0) + epsilon))
            # print(f"-----1-----:{torch.sqrt(torch.var(domain_z[:, i], dim=0) + epsilon)}\n-----2-----:{gamma - torch.sqrt(torch.var(domain_z[:, i], dim=0) + epsilon)}-----3-----:{F.relu(gamma - torch.sqrt(torch.var(domain_z[:, i], dim=0) + epsilon))}\n-----4-----:{variance_reg[domain_idx]}\n-----5-----:{hinge_loss_weight}")
        # take its mean over z_dim_invariant dimensions
        variance_reg[domain_idx] = variance_reg[domain_idx] / z_dim_invariant

    return variance_reg.sum() # * hinge_loss_weight

def penalty_domain_classification(z, domains, num_domains, z_dim_invariant, *args):

    multinomial_logistic_regression_model = args[0]
    classification_loss = args[1]
    pred_domains_log_proba = multinomial_logistic_regression_model(z[:, :z_dim_invariant])
    domain_classification_loss = classification_loss(pred_domains_log_proba, domains.squeeze().long()).to(z.device)

    # compute the classification accuracy
    pred_domains = torch.argmax(pred_domains_log_proba, dim=1)
    domain_classification_accuracy = (pred_domains == domains.squeeze().long()).sum().float() / pred_domains.shape[0]

    # we want to return a penalty, i.e., a positive number. So we return the negative cross entropy loss
    # so that the model tries to increase the cross entropy loss as much as possible, resulting in z_inv
    # not being predictive of domain as much as possible
    return -domain_classification_loss, domain_classification_accuracy
    # return torch.tensor(-domain_classification_loss, device=z.device).unsqueeze(0), 0.


# def compute_rbf_kernel(x, y, sigma=1.0):
#     """
#     Compute the Gaussian (RBF) kernel matrix between two sets of samples x and y.

#     Args:
#         x (Tensor): A tensor of shape (n_samples_x, n_features).
#         y (Tensor): A tensor of shape (n_samples_y, n_features).
#         sigma (float): The bandwidth of the RBF kernel.

#     Returns:
#         Tensor: The RBF kernel matrix of shape (n_samples_x, n_samples_y).
#     """

#     # Compute squared Euclidean distances between samples
#     xx = torch.sum(x * x, dim=1).view(-1, 1)
#     yy = torch.sum(y * y, dim=1).view(1, -1)
#     xy = torch.matmul(x, y.t())

#     distances = xx - 2 * xy + yy

#     # Compute RBF kernel matrix
#     kernel_matrix = torch.exp(-distances / (2 * sigma**2))

#     return kernel_matrix

# def mmd_loss(x, y, sigma=1.0):
#     """
#     Compute the Maximum Mean Discrepancy (MMD) between two sets of samples x and y.

#     Args:
#         x (Tensor): A tensor of shape (n_samples_x, n_features).
#         y (Tensor): A tensor of shape (n_samples_y, n_features).
#         sigma (float): The bandwidth of the RBF kernel.

#     Returns:
#         Tensor: The MMD loss.
#     """
#     kernel_xx = compute_rbf_kernel(x, x, sigma)
#     kernel_yy = compute_rbf_kernel(y, y, sigma)
#     kernel_xy = compute_rbf_kernel(x, y, sigma)

#     mmd = torch.mean(kernel_xx) - 2 * torch.mean(kernel_xy) + torch.mean(kernel_yy)

#     return mmd

def mmd_loss(MMD, z, domains, num_domains, z_dim_invariant, *args):
    # compute the MMD loss between the invariant dimensions of z
    # across all pairs of domains, and return it as the penalty

    mmd_loss = 0.
    domain_skip = torch.zeros(num_domains)
    for domain_idx in range(num_domains):
        domain_mask = (domains == domain_idx).squeeze()
        domain_z = z[domain_mask]

        if domain_z.shape[0] == 0:
            domain_skip[domain_idx] = 1
            continue

    # compute the pairwise mmd loss of domain_z across pairs of domains and add them all together.
    for i in range(num_domains):
        if domain_skip[i] == 1:
            continue
        else:
            for j in range(i+1, num_domains):
                if domain_skip[j] == 1:
                    continue
                else:
                    domain_i_z = z[(domains == i).squeeze()]
                    domain_j_z = z[(domains == j).squeeze()]
                    min_length = min(domain_i_z.shape[0], domain_j_z.shape[0])
                    mmd_loss += MMD(domain_i_z[:min_length, :z_dim_invariant],domain_j_z[:min_length, :z_dim_invariant])

    return mmd_loss


class MMD_loss(nn.Module):
    def __init__(self, kernel_multiplier = 2.0, kernel_number = 5, fix_sigma = None):
        super(MMD_loss, self).__init__()
        self.kernel_multiplier = kernel_multiplier
        self.kernel_number = kernel_number
        self.fix_sigma = fix_sigma
        return
    
    def guassian_kernel(self, source, target, kernel_multiplier=None, kernel_number=None, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        kernel_multiplier = kernel_multiplier if kernel_multiplier is not None else self.kernel_multiplier
        kernel_number = kernel_number if kernel_number is not None else self.kernel_number
        fix_sigma = fix_sigma if fix_sigma is not None else self.fix_sigma
        
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_multiplier ** (kernel_number // 2)
        bandwidth_list = [bandwidth * (kernel_multiplier**i) for i in range(kernel_number)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)
    
    def forward(self, source, target, kernel_multiplier=None, kernel_number=None, fix_sigma=None):
        kernel_multiplier = kernel_multiplier if kernel_multiplier is not None else self.kernel_multiplier
        kernel_number = kernel_number if kernel_number is not None else self.kernel_number
        fix_sigma = fix_sigma if fix_sigma is not None else self.fix_sigma
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_multiplier=self.kernel_multiplier, kernel_number=self.kernel_number, fix_sigma=self.fix_sigma)

        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss