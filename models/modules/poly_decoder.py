# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

class PolyDecoder(torch.nn.Module):
    
    def __init__(self, data_dim, latent_dim, poly_degree):
        super(PolyDecoder, self).__init__()        
        
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.poly_degree = poly_degree
        
        self.total_poly_terms = self.compute_total_polynomial_terms()
        
        self.coff_matrix = nn.Sequential(
                    nn.Linear(self.total_poly_terms, self.data_dim),  
                )        
        
    def forward(self, z):
        
        x = []
        for idx in range(z.shape[0]):
            x.append(self.compute_decoder_polynomial(z[idx, :]))
        x = torch.cat(x, dim=0)
        x = self.coff_matrix(x)
        
        return x
    
    
    def compute_total_polynomial_terms(self):
        count = 0
        for degree in range(self.poly_degree + 1):
            count += pow(self.latent_dim, degree)
        return count

    
    def compute_kronecker_product(self, degree, latent):
        if degree == 0:
            out = torch.tensor([1]).to(latent.device)        
        else:
            out = torch.clone(latent)
            for idx in range(1, degree):
                out = torch.kron(out, latent)
        return out

    def compute_decoder_polynomial(self, latent):
        out = []
        for degree in range(self.poly_degree + 1):
            out.append(self.compute_kronecker_product(degree, latent))

        out= torch.cat(out)
        out= out.view((1,out.shape[0]))    
        return out
    