import torch
from torch import nn
from torch import optim

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import (    
    AffineCouplingTransform
)

from shapeaxi.saxi_layers import *

class SaxiCouplingFlow(Flow):
    def __init__(self, features, num_layers, net_fn=None):
        base_dist = StandardNormal(shape=[features])
        
        mask = torch.ones(features)
        mask[features//2:] = 0

        if net_fn is None:
            def net_fn(in_features, out_features):
                return ProjectionHead(input_dim=in_features, hidden_dim=out_features, output_dim=out_features, dropout=0.1, bias=True)  

        transforms = []
        for _ in range(num_layers):    
            transforms.append(AffineCouplingTransform(mask=mask, transform_net_create_fn=net_fn))
        transform = CompositeTransform(transforms)
        
        super().__init__(transform=transform, distribution=base_dist)