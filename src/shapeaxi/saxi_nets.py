import math
import numpy as np 
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
import torchmetrics
import monai

import pandas as pd

from pytorch3d.structures import (
    Meshes,
    Pointclouds,)

from pytorch3d.renderer import (
        FoVPerspectiveCameras, PerspectiveCameras, look_at_rotation, 
        RasterizationSettings, MeshRenderer, MeshRasterizer, MeshRendererWithFragments,
        HardPhongShader, AmbientLights, TexturesVertex
)
from pytorch3d.ops import (sample_points_from_meshes,
                           knn_points, 
                           knn_gather)

from pytorch3d.loss import (
    chamfer_distance,
    point_mesh_edge_distance, 
    point_mesh_face_distance
)

from pytorch3d.utils import ico_sphere
from torch.nn.utils.rnn import pad_sequence

import json
import os

from positional_encodings.torch_encodings import PositionalEncoding2D

from shapeaxi import utils
from shapeaxi.saxi_layers import *
from shapeaxi.saxi_transforms import GaussianNoise, AvgPoolImages
from shapeaxi.colors import bcolors
from shapeaxi.saxi_losses import saxi_point_triangle_distance


class AttentionRing(nn.Module):
    def __init__(self, in_units, out_units, neigh_orders):
        super().__init__()
        self.num_heads = 8
        # neigh_order: (Nviews previous level, Neighbors next level)
        self.neigh_orders = neigh_orders
        #MHA
        self.Att = SelfAttention(in_units, out_units, dim=2)

    def forward(self, query, values):
        # Apply attention to the input sequence
        # query: (batch, views, features)
        # values: (batch, views, features)

        #Some neighbours are 7 and some are 6 so we repeat the first one to reach 7
        for data in self.neigh_orders:
            if len(data) == 6:
                repeat_neighbor = data[0]
                data.append(repeat_neighbor)
        
        query = query[:, self.neigh_orders] # (batch, Nv_{n-1}, Idx_{n}, features)
        values = values[:, self.neigh_orders] # (batch, Nv_{n-1}, Idx_{n}, features)

        # x, _ = self.MHA(query, values, values)
        context_vector, score = self.Att(query, values)

        return context_vector, score


class AttentionRings(nn.Module):
    def __init__(self, in_units, out_v_units, out_q_units, neigh_orders):
        super(AttentionRings, self).__init__()
        self.Q = nn.Linear(in_units, out_q_units)
        self.Att = AttentionRing(in_units, out_v_units, neigh_orders)
        
    def forward(self, x):
        query = self.Q(x)
        context_vector, score = self.Att(query, x)
        return context_vector, score





#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                                   SaxiRing Teeth Data                                                                             #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                       Architecture (SaxiRing) ued to train the model of Quality Control which is available on Github                                              #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################




class MHAEncoder(nn.Module):
    def __init__(self,  input_dim=3, output_dim=1, stages=[16], num_heads=[16], dropout=0.1, pooling_factor=None, pooling_hidden_dim=None, pooling_K=None, score_pooling=False, feed_forward_hidden_dim=None, use_layer_norm=False, return_v=False, time_dim=None, context_dim=None, use_mean_proj=False):
        super(MHAEncoder, self).__init__()

        assert len(stages) == len(num_heads)        
        assert feed_forward_hidden_dim is None or len(stages) == len(feed_forward_hidden_dim)        
        assert pooling_factor is None or len(stages) == len(pooling_factor) and len(pooling_factor) == len(pooling_hidden_dim) and len(pooling_factor) == len(pooling_K)

        
        self.num_heads = num_heads                
        self.stages = stages
        self.dropout = dropout        
        self.pooling_factor = pooling_factor
        self.pooling_hidden_dim = pooling_hidden_dim
        self.pooling_K = pooling_K
        self.score_pooling = score_pooling
        self.feed_forward_hidden_dim = feed_forward_hidden_dim        
        self.use_layer_norm = use_layer_norm
        self.return_v = return_v        

        self.embedding = ProjectionHead(input_dim, hidden_dim=self.stages[0], output_dim=self.stages[0])
        
        for i, st in enumerate(self.stages):

            if time_dim is not None:
                setattr(self, f"time_proj_{i}", ProjectionHead(time_dim, hidden_dim=st, output_dim=st))

            if context_dim is not None:
                setattr(self, f"context_proj_{i}", ProjectionHead(context_dim, hidden_dim=st, output_dim=st))

            if self.num_heads is not None and self.num_heads[i] is not None:
                setattr(self, f"mha_{i}", Residual(MHA(embed_dim=st, num_heads=num_heads[i], dropout=dropout)))
                if self.use_layer_norm:
                    setattr(self, f"norm_mha_{i}", nn.LayerNorm(st))

            if self.feed_forward_hidden_dim is not None and feed_forward_hidden_dim[i] is not None:
                setattr(self, f"ff_{i}", Residual(FeedForward(embed_dim=st, hidden_dim=feed_forward_hidden_dim[i], dropout=dropout)))
                if self.use_layer_norm:
                    setattr(self, f"norm_ff_{i}", nn.LayerNorm(st))

            if self.pooling_factor is not None and self.pooling_factor[i] is not None and self.pooling_hidden_dim is not None and self.pooling_hidden_dim[i] is not None and self.pooling_K is not None and self.pooling_K[i] is not None: 
                setattr(self, f"pool_{i}", AttentionPooling_V(embed_dim=st, hidden_dim=self.pooling_hidden_dim[i], K=self.pooling_K[i], pooling_factor=self.pooling_factor[i], score_pooling=self.score_pooling))
            
            if i+1 < len(self.stages):
                st_n = self.stages[i+1] 
                setattr(self, f"output_{i}", ProjectionHead(st, hidden_dim=st_n, output_dim=st_n))

        if use_mean_proj:
            setattr(self, f"mean_proj", nn.Sequential(
                    ProjectionHead(self.stages[-1], hidden_dim=self.stages[-1], output_dim=self.stages[-1]), 
                    ProjectionHead(self.stages[-1], hidden_dim=self.stages[-1], output_dim=self.stages[-1]), 
                    ProjectionHead(self.stages[-1], hidden_dim=self.stages[-1], output_dim=output_dim)
                )
            )
            setattr(self, f"std_proj", nn.Sequential(
                    ProjectionHead(self.stages[-1], hidden_dim=self.stages[-1], output_dim=self.stages[-1]), 
                    ProjectionHead(self.stages[-1], hidden_dim=self.stages[-1], output_dim=self.stages[-1]), 
                    ProjectionHead(self.stages[-1], hidden_dim=self.stages[-1], output_dim=output_dim)
                )
            )
        else:
            setattr(self, f"final_proj", nn.Sequential(
                    ProjectionHead(self.stages[-1], hidden_dim=self.stages[-1], output_dim=self.stages[-1]), 
                    ProjectionHead(self.stages[-1], hidden_dim=self.stages[-1], output_dim=self.stages[-1]), 
                    ProjectionHead(self.stages[-1], hidden_dim=self.stages[-1], output_dim=output_dim)
                )
            )

        
    def forward(self, x, x_v=None, context=None, time=None, beta=None):
        
        x = self.embedding(x)
        context_proj = 0
        time_proj = 0

        if beta is not None:
            batch_size = x.shape[0]
            beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
            context = context.view(batch_size, 1, -1)   # (B, 1, F)

            time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)

            context = torch.cat([time_emb, context], dim=-1) 

        
        for i, st in enumerate(self.stages):

            if hasattr(self, f"time_proj_{i}"):
                time_proj = getattr(self, f"time_proj_{i}")(time)
                x = x + time_proj

            if hasattr(self, f"context_proj_{i}"):
                context_proj = getattr(self, f"context_proj_{i}")(context)
                x = x + context_proj
            
            if hasattr(self, f"mha_{i}"):
                x = getattr(self, f"mha_{i}")(x)
            if hasattr(self, f"norm_mha_{i}"):
                x = getattr(self, f"norm_mha_{i}")(x)                

            if hasattr(self, f"time_proj_{i}"):
                x = x + time_proj
            if hasattr(self, f"context_proj_{i}"):
                x = x + context_proj

            if hasattr(self, f"ff_{i}"):
                x = getattr(self, f"ff_{i}")(x)
            if hasattr(self, f"norm_ff_{i}"):
                x = getattr(self, f"norm_ff_{i}")(x)

            if hasattr(self, f"pool_{i}"):
                x, x_v_next, x_s, _, _ = getattr(self, f"pool_{i}")(x, x_v)                
                x_v = x_v_next

            if hasattr(self, f"time_proj_{i}"):
                x = x + time_proj
            if hasattr(self, f"context_proj_{i}"):
                x = x + context_proj

            if hasattr(self, f"output_{i}"):
                x = getattr(self, f"output_{i}")(x)

        if hasattr(self, f"mean_proj"):
            x_m = self.mean_proj(x)
            x_s = self.std_proj(x)
            return x_m, x_s
        else:
            x = self.final_proj(x)
            return x
    

class MHADecoder(nn.Module):
    def __init__(self, input_dim=3, embed_dim=128, output_dim=3, num_heads=4, sample_levels=1, K=4, dropout=0.1, return_sorted=True):
        super(MHADecoder, self).__init__()

        self.input_dim = input_dim        
        self.K = K
        self.embed_dim = embed_dim
        self.num_heads = num_heads        
        self.sample_levels = sample_levels
        self.dropout = dropout
        self.return_sorted = return_sorted

        self.embedding = nn.Linear(input_dim, embed_dim, bias=False)

        for i in range(sample_levels):
            setattr(self, f"unpool_mha_knn_{i}", UnpoolMHA_KNN(MHA_KNN(embed_dim=embed_dim, num_heads=num_heads, K=K, dropout=dropout)))

        self.output = nn.Linear(embed_dim, output_dim, bias=False)
        
    
    def sample_points(self, x, Ns):
        """
        Samples Ns points from each batch in a tensor of shape (Bs, N, F).

        Args:
            x (torch.Tensor): Input tensor of shape (Bs, N, F).
            Ns (int): Number of points to sample from each batch.

        Returns:
            torch.Tensor: Output tensor of shape (Bs, Ns, F).
        """
        Bs, N, F = x.shape

        # Generate random indices for sampling
        indices = torch.randint(low=0, high=N, size=(Bs, Ns), device=x.device).unsqueeze(-1)

        # Gather the sampled points
        x = knn_gather(x, indices).squeeze(-2).contiguous()

        return x, indices
        
    def forward(self, x):

        x = self.embedding(x)
        
        for i in range(self.sample_levels):
            x = getattr(self, f"unpool_mha_knn_{i}")(x)            
        
        x = self.output(x)

        return x

class MHAEncoder_V(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, K=[27], num_heads=[16], stages=[16], dropout=0.1, pooling_factor=[0.125], score_pooling=False, pooling_hidden_dim=[8], feed_forward_hidden_dim=[8], return_sorted=True):
        super(MHAEncoder_V, self).__init__()
        
        self.num_heads = num_heads
        self.K = K
        self.stages = stages
        self.pooling_factor = pooling_factor
        self.dropout = dropout
        self.return_sorted = return_sorted
        self.feed_forward_hidden_dim = feed_forward_hidden_dim
        self.pooling_hidden_dim = pooling_hidden_dim
        
        self.embedding = nn.Linear(input_dim, self.stages[0])

        for i, st in enumerate(stages):
            setattr(self, f"mha_{i}", MHA_KNN_V(embed_dim=st, num_heads=num_heads[i], K=K[i], return_weights=True, dropout=dropout, return_sorted=return_sorted, use_direction=False))

            if self.feed_forward_hidden_dim is not None and feed_forward_hidden_dim[i] is not None:
                setattr(self, f"ff_{i}", Residual(FeedForward(embed_dim=st, hidden_dim=feed_forward_hidden_dim[i], dropout=dropout)))

            if self.pooling_factor is not None and pooling_factor[i] is not None and self.pooling_hidden_dim is not None and self.pooling_hidden_dim[i] is not None: 
                setattr(self, f"pool_{i}", AttentionPooling_V(embed_dim=st, pooling_factor=pooling_factor[i], hidden_dim=pooling_hidden_dim[i], K=K[i], score_pooling=score_pooling))
            
            st_n = stages[i+1] if i+1 < len(stages) else output_dim
            setattr(self, f"output_{i}", nn.Linear(st, st_n))
        
    def forward(self, x, x_v):
        
        x = self.embedding(x, x_v)
        x_s_idx = []
        
        for i, _ in enumerate(self.stages):
            # the mha will select optimal points from the input
            x, x_w = getattr(self, f"mha_{i}")(x, x_v)
            if self.feed_forward_hidden_dim is not None and self.feed_forward_hidden_dim[i] is not None:
                x = getattr(self, f"ff_{i}")(x)
            if self.pooling_factor is not None and self.pooling_factor[i] is not None and self.pooling_hidden_dim is not None and self.pooling_hidden_dim[i] is not None: 
                x, x_v, x_s, x_idx = getattr(self, f"pool_{i}")(x, x_v)
                x_s_idx.append((x_s, x_idx))
            x = getattr(self, f"output_{i}")(x)

        return x, x_v, x_s_idx


class MHAIcoEncoder(nn.Module):
    def __init__(self, input_dim=3, sample_levels=5, embed_dim=128, hidden_dim=64, num_heads=128, output_dim=128, feed_forward_hidden_dim=None, use_layer_norm=False, dropout=0.1):
        super(MHAIcoEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.sample_levels = sample_levels
        self.dropout = dropout
        self.feed_forward_hidden_dim = feed_forward_hidden_dim
        self.use_layer_norm = use_layer_norm

        self.embedding = nn.Linear(input_dim, embed_dim)

        self.init_neighs(sample_levels)

        for i in range(sample_levels):

            setattr(self, f"mha_{i}", Residual(MHA_Idx(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)))
            if self.use_layer_norm:
                setattr(self, f"norm_mha_{i}", nn.LayerNorm(embed_dim))
            if self.feed_forward_hidden_dim is not None:
                setattr(self, f"ff_{i}", Residual(FeedForward(embed_dim=embed_dim, hidden_dim=feed_forward_hidden_dim, dropout=dropout)))
                if self.use_layer_norm:
                    setattr(self, f"norm_ff_{i}", nn.LayerNorm(embed_dim))
            setattr(self, f"pool_{i}", AttentionPooling_Idx(embed_dim=embed_dim, hidden_dim=hidden_dim))
        
        self.output = nn.Linear(embed_dim, output_dim)
    
    def init_neighs(self, L):
        
        self.ico_neighs = []
        
        for l in range(L, 0, - 1):

            ico_s_current = ico_sphere(l).cuda()
            ico_v_current = ico_s_current.verts_packed().unsqueeze(0)
            ico_f_current = ico_s_current.faces_packed().unsqueeze(0)

            neigh_current = []
                
            for pid in range(ico_v_current.shape[1]):
                neigh = utils.GetNeighborsT(ico_f_current.squeeze(), pid)
                neigh_current.append(neigh)

            neigh_current = pad_sequence(neigh_current, batch_first=True, padding_value=0)
            self.ico_neighs.append(neigh_current)

        self.ico_pooling_neighs = []
        for l in range(L, 0, -1):

            # Find current level icosahedron
            ico_s_current = ico_sphere(l).cuda()
            ico_v_current = ico_s_current.verts_packed().unsqueeze(0)
            ico_f_current = ico_s_current.faces_packed().unsqueeze(0)

            # Find next level icosahedron
            ico_s_next = ico_sphere(l-1).cuda()
            ico_v_next = ico_s_next.verts_packed().unsqueeze(0)
            # ico_f_next = ico_s_next.faces_packed().unsqueeze(0)

            # Find the closest points in the current level icosahedron using the next level icosahedron
            dist = knn_points(ico_v_next, ico_v_current, K=1)

            # Find the neighbors of each point in the current level icosahedron
            neigh_current = []
            
            for pid in dist.idx.squeeze():
                neigh = utils.GetNeighborsT(ico_f_current.squeeze(), pid)
                neigh_current.append(neigh)

            neigh_current = pad_sequence(neigh_current, batch_first=True, padding_value=0)
            # neigh_current = torch.stack(neigh_current, dim=0)
            # print(neigh_current.shape)

            # The shape of neigh_current is (N - 1, 6) where N is the number of points in the next level icosahedron
            # However, the ids in neigh_current are the ids of the points in the current level icosahedron. We can use this to pool features
            # and go to the next level
            # print(neigh_current.shape, ico_v_current.shape, ico_v_next.shape)

            
            self.ico_pooling_neighs.append(neigh_current)
        
    def forward(self, x):
        
        batch_size = x.shape[0]
        x = self.embedding(x)
        
        for i in range(self.sample_levels - 1):
            
            neigh = self.ico_neighs[i]
            repeats = [batch_size, 1, 1]
            neigh = neigh.repeat(repeats)
            # the mha will select optimal points from the input
            x = getattr(self, f"mha_{i}")(x, neigh)
            if self.use_layer_norm:
                x = getattr(self, f"norm_mha_{i}")(x)
            if self.feed_forward_hidden_dim is not None:
                x = getattr(self, f"ff_{i}")(x)
                if self.use_layer_norm:
                    x = getattr(self, f"norm_ff_{i}")(x)

            neigh_next = self.ico_pooling_neighs[i]
            neigh_next = neigh_next.repeat(repeats)
            x, x_s = getattr(self, f"pool_{i}")(x, neigh_next)

        #output layer
        x = self.output(x)
        return x
    
class MHAIcoDecoder(nn.Module):
    def __init__(self, input_dim=3, sample_levels=5, embed_dim=128, hidden_dim=64, num_heads=128, output_dim=3, feed_forward_hidden_dim=None, use_layer_norm=False, dropout=0.1):
        super(MHAIcoDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.sample_levels = sample_levels
        self.dropout = dropout
        self.feed_forward_hidden_dim = feed_forward_hidden_dim
        self.use_layer_norm = use_layer_norm


        self.embedding = nn.Linear(input_dim, embed_dim)

        self.init_neighs(sample_levels)

        for i in range(sample_levels):

            setattr(self, f"pool_{i}", AttentionPooling_Idx(embed_dim=embed_dim, hidden_dim=hidden_dim))
            setattr(self, f"mha_{i}", MHA_Idx(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout))
            if self.use_layer_norm:
                setattr(self, f"norm_mha_{i}", nn.LayerNorm(embed_dim))
            
            if self.feed_forward_hidden_dim is not None:
                setattr(self, f"ff_{i}", Residual(FeedForward(embed_dim, hidden_dim=feed_forward_hidden_dim, dropout=dropout)))
                if use_layer_norm:
                    setattr(self, f"norm_ff_{i}", nn.LayerNorm(embed_dim))
            
        
        self.output = nn.Linear(embed_dim, output_dim)
    
    def init_neighs(self, L):

        self.ico_neighs = []
        
        for l in range(1, L):

            ico_s_current = ico_sphere(l).cuda()
            ico_v_current = ico_s_current.verts_packed().unsqueeze(0)
            ico_f_current = ico_s_current.faces_packed().unsqueeze(0)

            neigh_current = []
                
            for pid in range(ico_v_current.shape[1]):
                neigh = utils.GetNeighborsT(ico_f_current.squeeze(), pid)
                neigh_current.append(neigh)

            neigh_current = pad_sequence(neigh_current, batch_first=True, padding_value=0)
            self.ico_neighs.append(neigh_current)

        self.ico_pooling_neighs = []
        for l in range(L - 1):

            # Find current level icosahedron
            ico_s_current = ico_sphere(l).cuda()
            ico_v_current = ico_s_current.verts_packed().unsqueeze(0)
            ico_f_current = ico_s_current.faces_packed().unsqueeze(0)

            # Find next level icosahedron
            ico_s_next = ico_sphere(l+1).cuda()
            ico_v_next = ico_s_next.verts_packed().unsqueeze(0)
            # ico_f_next = ico_s_next.faces_packed().unsqueeze(0)

            # Find the closest points in the current level icosahedron using the next level icosahedron
            dist = knn_points(ico_v_next, ico_v_current, K=1)

            # Find the neighbors of each point in the current level icosahedron
            neigh_current = []
            
            for pid in dist.idx.squeeze():
                neigh = utils.GetNeighborsT(ico_f_current.squeeze(), pid)
                neigh_current.append(neigh)

            neigh_current = pad_sequence(neigh_current, batch_first=True, padding_value=0)
            # neigh_current = torch.stack(neigh_current, dim=0)
            # print(neigh_current.shape)

            # The shape of neigh_current is (N - 1, 6) where N is the number of points in the next level icosahedron
            # However, the ids in neigh_current are the ids of the points in the current level icosahedron. We can use this to pool features
            # and go to the next level
            # print(neigh_current.shape, ico_v_current.shape, ico_v_next.shape)

            
            self.ico_pooling_neighs.append(neigh_current)
        
    def forward(self, x):
        
        batch_size = x.shape[0]
        x = self.embedding(x)

        repeats = [batch_size, 1, 1]
        
        for i in range(self.sample_levels - 1):

            neigh_next = self.ico_pooling_neighs[i]
            neigh_next = neigh_next.repeat(repeats)
            x, x_s = getattr(self, f"pool_{i}")(x, neigh_next)
            
            neigh = self.ico_neighs[i]
            neigh = neigh.repeat(repeats)
            # the mha will select optimal points from the input
            x = getattr(self, f"mha_{i}")(x, neigh)

            if self.use_layer_norm:
                x = getattr(self, f"norm_mha_{i}")(x)
            
            if self.feed_forward_hidden_dim is not None:
                x = getattr(self, f"ff_{i}")(x)
                if self.use_layer_norm:
                    x = getattr(self, f"norm_ff_{i}")(x)
            # print(x.shape)
            # x = getattr(self, f"ff_{i}")(x)

        #output layer
        x = self.output(x)
        return x

class MHAIdxEncoder(nn.Module):
    def __init__(self,  input_dim=3, output_dim=1, K=[27], num_heads=[16], stages=[16], dropout=0.1, pooling_factor=[0.125], pooling_hidden_dim=[8], pooling_K=[27], score_pooling=False, conv_v_kernel_size=None, feed_forward_hidden_dim=None, return_sorted=True, use_skip_connection=False, use_layer_norm=False, return_v=False, use_direction=False, time_embed_dim=None):
        super(MHAIdxEncoder, self).__init__()

        
        self.num_heads = num_heads        
        self.K = K
        self.stages = stages
        self.dropout = dropout        
        self.pooling_factor = pooling_factor
        self.pooling_hidden_dim = pooling_hidden_dim
        self.pooling_K = pooling_K
        self.score_pooling = score_pooling
        self.feed_forward_hidden_dim = feed_forward_hidden_dim
        self.use_skip_connection = use_skip_connection
        self.conv_v_kernel_size = conv_v_kernel_size
        self.use_layer_norm = use_layer_norm
        self.return_v = return_v
        self.time_embed_dim = time_embed_dim
        

        self.embedding = nn.Linear(input_dim, self.stages[0], bias=False)

        if self.pooling_factor is not None:
            assert len(self.pooling_factor) == len(self.pooling_hidden_dim) and len(self.pooling_factor) == len(self.pooling_K)

        for i, st in enumerate(self.stages):

            if time_embed_dim is not None:
                setattr(self, f"time_proj_{i}", nn.Linear(time_embed_dim, st, bias=False))
                setattr(self, f"norm_time_0_{i}", nn.LayerNorm(st))
                setattr(self, f"norm_time_1_{i}", nn.LayerNorm(st))
                setattr(self, f"norm_time_2_{i}", nn.LayerNorm(st))


            if self.conv_v_kernel_size is not None and self.conv_v_kernel_size[i] is not None:
                setattr(self, f"conv_0_{i}", Residual(ConvBlock_V(in_channels=st, out_channels=st, K=self.conv_v_kernel_size[i], kernel_size=self.conv_v_kernel_size[i], bias=False)))
                setattr(self, f"conv_1_{i}", Residual(ConvBlock_V(in_channels=st, out_channels=st, K=self.conv_v_kernel_size[i], kernel_size=self.conv_v_kernel_size[i], bias=False)))
                setattr(self, f"conv_2_{i}", Residual(ConvBlock_V(in_channels=st, out_channels=st, K=self.conv_v_kernel_size[i], kernel_size=self.conv_v_kernel_size[i], bias=False)))

            if self.K is not None and self.K[i] is not None:
                setattr(self, f"mha_{i}", Residual(MHA_KNN_V(embed_dim=st, num_heads=num_heads[i], K=K[i], dropout=dropout, return_sorted=return_sorted, use_direction=use_direction)))
                if self.use_layer_norm:
                    setattr(self, f"norm_mha_{i}", nn.LayerNorm(st))

            if self.feed_forward_hidden_dim is not None and feed_forward_hidden_dim[i] is not None:
                setattr(self, f"ff_{i}", Residual(FeedForward(embed_dim=st, hidden_dim=feed_forward_hidden_dim[i], dropout=dropout)))
                if self.use_layer_norm:
                    setattr(self, f"norm_ff_{i}", nn.LayerNorm(st))

            if self.pooling_factor is not None and self.pooling_factor[i] is not None and self.pooling_hidden_dim is not None and self.pooling_hidden_dim[i] is not None and self.pooling_K is not None and self.pooling_K[i] is not None: 
                setattr(self, f"pool_{i}", AttentionPooling_V(embed_dim=st, hidden_dim=self.pooling_hidden_dim[i], K=self.pooling_K[i], pooling_factor=self.pooling_factor[i], score_pooling=self.score_pooling))
            
            st_n = self.stages[i+1] if i+1 < len(self.stages) else output_dim
            setattr(self, f"output_{i}", nn.Linear(st, st_n, bias=False))
        
    def forward(self, x, x_v, time=None):
        
        x = self.embedding(x)

        unpooling_idxs = []
        skip_connections = []
        
        for i, st in enumerate(self.stages):

            time_proj = None
            if time is not None:
                time_proj = getattr(self, f"time_proj_{i}")(time)
                time_proj = time_proj.unsqueeze(1)
                
                x = x + time_proj
                x = getattr(self, f"norm_time_0_{i}")(x)

            if hasattr(self, f"conv_0_{i}"):
                x = getattr(self, f"conv_0_{i}")(x, x_v)
                x = getattr(self, f"conv_1_{i}")(x, x_v)
                x = getattr(self, f"conv_2_{i}")(x, x_v)

            if time is not None:
                x = x + time_proj
                x = getattr(self, f"norm_time_1_{i}")(x)
            
            if hasattr(self, f"mha_{i}"):
                x = getattr(self, f"mha_{i}")(x, x_v)
            if hasattr(self, f"norm_mha_{i}"):
                x = getattr(self, f"norm_mha_{i}")(x)

            if time is not None:
                x = x + time_proj
                x = getattr(self, f"norm_time_2_{i}")(x)

            if hasattr(self, f"ff_{i}"):
                x = getattr(self, f"ff_{i}")(x)
            if hasattr(self, f"norm_ff_{i}"):
                x = getattr(self, f"norm_ff_{i}")(x)

            if self.use_skip_connection:
                skip_connections.insert(0, x)

            if hasattr(self, f"pool_{i}"):
                x, x_v_next, x_s, pooling_idx, unpooling_idx = getattr(self, f"pool_{i}")(x, x_v)
                unpooling_idxs.insert(0, (x_v, unpooling_idx, x_s))
                x_v = x_v_next

            x = getattr(self, f"output_{i}")(x)
        
        if self.use_skip_connection:
            if self.return_v:
                return x, x_v, unpooling_idxs, skip_connections
            return x, unpooling_idxs, skip_connections
        if self.return_v:
            return x, x_v, unpooling_idxs
        return x, unpooling_idxs
    
class MHAIdxDecoder(nn.Module):
    def __init__(self,  input_dim=3, output_dim=1, K=[27], num_heads=[16], stages=[16], dropout=0.1, pooling_hidden_dim=[8], conv_v_kernel_size=None, feed_forward_hidden_dim=None, return_sorted=True, use_skip_connection=False, use_layer_norm=False, use_direction=False, time_embed_dim=None):
        super(MHAIdxDecoder, self).__init__()

        assert len(stages) == len(num_heads) == len(K) == len(pooling_hidden_dim)

        self.num_heads = num_heads        
        self.K = K
        self.stages = stages
        self.dropout = dropout
        self.pooling_hidden_dim = pooling_hidden_dim
        self.feed_forward_hidden_dim = feed_forward_hidden_dim
        self.use_skip_connection = use_skip_connection
        self.use_layer_norm = use_layer_norm
        self.conv_v_kernel_size = conv_v_kernel_size
        
        self.embedding = nn.Linear(input_dim, self.stages[0], bias=False)

        for i, st in enumerate(self.stages):

            if time_embed_dim is not None:
                setattr(self, f"time_proj_{i}", nn.Linear(time_embed_dim, st, bias=False))
                setattr(self, f"norm_time_0_{i}", nn.LayerNorm(st))
                setattr(self, f"norm_time_1_{i}", nn.LayerNorm(st))
                setattr(self, f"norm_time_2_{i}", nn.LayerNorm(st))

            setattr(self, f"pool_{i}", AttentionPooling_Idx(embed_dim=st, hidden_dim=pooling_hidden_dim[i]))

            if self.use_skip_connection:
                setattr(self, f"proj_{i}", Residual(ProjectionHead(input_dim=st*2, hidden_dim=st, output_dim=st, dropout=dropout)))

            if self.conv_v_kernel_size is not None and self.conv_v_kernel_size[i] is not None:
                setattr(self, f"conv_0_{i}", Residual(ConvBlock_V(in_channels=st, out_channels=st, K=self.conv_v_kernel_size[i], kernel_size=self.conv_v_kernel_size[i], bias=False)))
                setattr(self, f"conv_1_{i}", Residual(ConvBlock_V(in_channels=st, out_channels=st, K=self.conv_v_kernel_size[i], kernel_size=self.conv_v_kernel_size[i], bias=False)))
                setattr(self, f"conv_2_{i}", Residual(ConvBlock_V(in_channels=st, out_channels=st, K=self.conv_v_kernel_size[i], kernel_size=self.conv_v_kernel_size[i], bias=False)))

            if self.K is not None and self.K[i] is not None:
                setattr(self, f"mha_{i}", Residual(MHA_KNN_V(embed_dim=st, num_heads=num_heads[i], K=K[i], dropout=dropout, return_sorted=return_sorted, use_direction=use_direction)))
                if self.use_layer_norm:
                    setattr(self, f"norm_mha_{i}", nn.LayerNorm(st))

            if self.feed_forward_hidden_dim is not None and feed_forward_hidden_dim[i] is not None:
                setattr(self, f"ff_{i}", Residual(FeedForward(embed_dim=st, hidden_dim=feed_forward_hidden_dim[i], dropout=dropout)))
                if self.use_layer_norm:
                    setattr(self, f"norm_ff_{i}", nn.LayerNorm(st))
            
            st_n = self.stages[i+1] if i+1 < len(self.stages) else output_dim
            setattr(self, f"output_{i}", nn.Linear(st, st_n, bias=False))
        
    def forward(self, x, unpooling_idxs, skip_connections=None, time=None):
        
        assert len(unpooling_idxs) == len(self.stages)
                
        x = self.embedding(x)
        
        for i, unp_idx in enumerate(unpooling_idxs):

            x_v, unpooling_idx, _ = unp_idx

            x, x_s = getattr(self, f"pool_{i}")(x, unpooling_idx)

            if self.use_skip_connection:
                x = getattr(self, f"proj_{i}")(x, skip_connections[i])

            time_proj = None
            if time is not None:
                time_proj = getattr(self, f"time_proj_{i}")(time)
                time_proj = time_proj.unsqueeze(1)
                x = x + time_proj
                x = getattr(self, f"norm_time_0_{i}")(x)

            if hasattr(self, f"conv_0_{i}"):
                x = getattr(self, f"conv_0_{i}")(x, x_v)
                x = getattr(self, f"conv_1_{i}")(x, x_v)
                x = getattr(self, f"conv_2_{i}")(x, x_v)

            if time is not None:
                x = x + time_proj
                x = getattr(self, f"norm_time_1_{i}")(x)

            if hasattr(self, f"mha_{i}"):
                x = getattr(self, f"mha_{i}")(x, x_v)
            if hasattr(self, f"norm_mha_{i}"):
                x = getattr(self, f"norm_mha_{i}")(x)

            if time is not None:
                x = x + time_proj
                x = getattr(self, f"norm_time_2_{i}")(x)

            if hasattr(self, f"ff_{i}"):
                x = getattr(self, f"ff_{i}")(x)

            if hasattr(self, f"norm_ff_{i}"):
                x = getattr(self, f"norm_ff_{i}")(x)

            x = getattr(self, f"output_{i}")(x)
        
        return x

class ContextModulatedNet(nn.Module):

    def __init__(self, input_dim=3, stages=[128, 256, 512, 256, 128, 3], context_dim=256):
        super().__init__()

        self.stages = stages
        
        in_dim = input_dim        

        for i, st in enumerate(stages):
            activation = None
            if i < len(stages) - 1:
                activation = nn.LeakyReLU
            setattr(self, f"context_modulated{i}", ContextModulated(input_dim=in_dim, output_dim=st, context_dim=context_dim, activation=activation))
            in_dim = st           
        

    def forward(self, x, context, beta=None, time=None):

        batch_size = x.shape[0]

        if beta is not None:
            
            beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
            context = context.view(batch_size, 1, -1)   # (B, 1, F)

            time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)

            context = torch.cat([time_emb, context], dim=-1) 

        if time is not None:
            time = time.view(batch_size, 1, -1)
            context = torch.cat([time, context], dim=-1)

        if x.shape[1] != context.shape[1]:
            context = context.repeat(1, x.shape[1], 1)
        
        out = x
        for i, _ in enumerate(self.stages):
            out = getattr(self, f"context_modulated{i}")(out, context)
        
        return x + out
    
class PointNetEncoder(nn.Module):
    def __init__(self, zdim, input_dim=3):
        super().__init__()
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        # Mapping to [c], cmean
        self.fc1_m = nn.Linear(512, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, zdim)
        self.fc_bn1_m = nn.BatchNorm1d(256)
        self.fc_bn2_m = nn.BatchNorm1d(128)

        # Mapping to [c], cmean
        self.fc1_v = nn.Linear(512, 256)
        self.fc2_v = nn.Linear(256, 128)
        self.fc3_v = nn.Linear(128, zdim)
        self.fc_bn1_v = nn.BatchNorm1d(256)
        self.fc_bn2_v = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.transpose(1, 2)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        
        x = torch.max(x, 2, keepdim=True)[0]
        
        x = x.view(-1, 512)        

        m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        m = self.fc3_m(m)
        v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
        v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
        v = self.fc3_v(v)

        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return m, v

class HilbertSort3D(nn.Module):    
    def __init__(self, origin=(0.0, 0.0, 0.0), radius=1.25, bins=32):
        super().__init__()
        """
        Initialize HilbertSort3D.
        :param origin: Tuple of floats, the origin point for the Hilbert sorting.
        :param radius: Float, radius of the space.
        :param bins: Int, number of bins (must be a power of 2).
        """
        origin = torch.tensor(origin, dtype=torch.float32)
        radius = torch.tensor(radius)
        bins = torch.tensor(bins)
        curve = self._generate_hilbert_curve(bins)
        self.register_buffer('origin', origin)
        self.register_buffer('radius', radius)
        self.register_buffer('bins', bins)
        self.register_buffer('curve', curve)

    def _generate_hilbert_curve(self, bins):
        """
        Generate the 3D Hilbert curve indices for given bins.
        Returns a tensor mapping bin (x, y, z) coordinates to their 1D Hilbert order.
        """
        size = bins
        indices = torch.arange(size**3).view(size, size, size)
        return indices

    def forward(self, point_cloud):
        """
        Sort a batch of point clouds using Hilbert sorting.
        :param point_cloud: Tensor of shape (B, N, 3), where B is batch size, N is number of points.
        :return: Sorted tensor of shape (B, N, 3).
        """
        B, N, _ = point_cloud.shape

        # Center and normalize data
        point_cloud = point_cloud - self.origin
        bin_interval = (self.radius * 2) / self.bins
        bins = ((point_cloud / bin_interval) + (self.bins // 2)).long()
        bins = torch.clamp(bins, 0, self.bins - 1)

        # Flatten the bin coordinates into a 1D Hilbert index
        hilbert_indices = self.curve[bins[:, :, 0], bins[:, :, 1], bins[:, :, 2]]

        # Sort each batch of point clouds by their Hilbert indices
        sorted_indices = torch.argsort(hilbert_indices, dim=1)
        sorted_points = torch.gather(point_cloud, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, 3))
        return sorted_points, sorted_indices

# class NeRF(nn.Module):
#     def __init__(self, input_dim=3, pos_dim=10, view_dim=4, hidden_dim=256) -> None:
#         super().__init__()
        
#         self.act = nn.ReLU()

#         self.pos_dim = pos_dim
#         p_dim = pos_dim*input_dim*2+input_dim
        
#         self.view_dim = view_dim
#         v_dim = view_dim*input_dim*2+input_dim

#         self.block1 = nn.Sequential(nn.Linear(p_dim, hidden_dim), 
#         nn.ReLU(),
#         nn.Linear(hidden_dim, hidden_dim),
#         nn.ReLU(),
#         nn.Linear(hidden_dim, hidden_dim),
#         nn.ReLU(),
#         nn.Linear(hidden_dim, hidden_dim),
#         nn.ReLU())
        
#         self.block2 = nn.Sequential(nn.Linear(p_dim + hidden_dim, hidden_dim),
#         nn.ReLU(),
#         nn.Linear(hidden_dim, hidden_dim),
#         nn.ReLU(),
#         nn.Linear(hidden_dim, hidden_dim),
#         nn.ReLU(),
#         nn.Linear(hidden_dim, hidden_dim)) # No activation

#         self.final_sigma = nn.Sequential(nn.Linear(v_dim + hidden_dim, 1), 
#         nn.ReLU())

#         self.final_rgb = nn.Sequential(nn.Linear(v_dim + hidden_dim, hidden_dim),
#         nn.ReLU(),
#         nn.Linear(hidden_dim, 3),
#         nn.Sigmoid())

#     def encoding(self, x, L=10):
#         res = [x]
#         for i in range(L):
#             for fn in [torch.sin, torch.cos]:
#                 res.append(fn(2 ** i * torch.pi * x))
#         return torch.cat(res,dim=-1)
        
#     def forward(self, x_p, x_v):

#         # parameters:
#         # x_p: torch.Size([4, N_P, N_Samples, 3]) N_P is the number of points, N_Samples is the number of samples/bins
#         # x_v: torch.Size([4, N_P, N_Samples, 3]) N_V is the number of view directions

#         x_p_pos_enc = self.encoding(x_p, L=self.pos_dim)
#         x_p = self.act(x_p_pos_enc)

#         x_p = self.block1(x_p)

#         x_p = torch.cat([x_p, x_p_pos_enc], dim=-1)

#         x_p = self.block2(x_p)
        
#         x_v = self.encoding(x_v, L=self.view_dim)

#         x = torch.cat([x_p, x_v], dim=-1)

#         sigma = self.final_sigma(x)
#         rgb = self.final_rgb(x)

#         return rgb, sigma

class NeRF(nn.Module):
    def __init__(self, pos_enc_dim=63, view_enc_dim=27, hidden=256) -> None:
        super().__init__()
        
        self.linear1 = nn.Sequential(nn.Linear(pos_enc_dim,hidden),nn.ReLU())

        self.pre_skip_linear = nn.Sequential()
        for _ in range(4):
            self.pre_skip_linear.append(nn.Linear(hidden,hidden))
            self.pre_skip_linear.append(nn.ReLU())

        self.linear_skip = nn.Sequential(nn.Linear(pos_enc_dim+hidden,hidden),nn.ReLU())

        self.post_skip_linear = nn.Sequential()
        for _ in range(2):
            self.post_skip_linear.append(nn.Linear(hidden,hidden))
            self.post_skip_linear.append(nn.ReLU())

        self.density_layer = nn.Sequential(nn.Linear(hidden,1),nn.ReLU())

        self.linear2 = nn.Linear(hidden,hidden)

        self.color_linear1 = nn.Sequential(nn.Linear(hidden+view_enc_dim,hidden//2),nn.ReLU())
        self.color_linear2 = nn.Sequential(nn.Linear(hidden//2,3),nn.Sigmoid())

    def encoding(self, x, L=10):
        res = [x]
        for i in range(L):
            for fn in [torch.sin, torch.cos]:
                res.append(fn(2 ** i * torch.pi * x))
        return torch.cat(res,dim=-1)
        
    def forward(self, positions, view_dirs):
        # Encode
        pos_enc = self.encoding(positions, L=10)
        view_enc = self.encoding(view_dirs, L=4)

        x = self.linear1(pos_enc)
        x = self.pre_skip_linear(x)

        # Skip connection
        x = torch.cat([x, pos_enc],dim=-1)
        x = self.linear_skip(x)

        x = self.post_skip_linear(x)

        # Density
        sigma = self.density_layer(x)

        x = self.linear2(x)

        # View Encoding
        x = torch.cat([x,view_enc],dim=-1)
        x = self.color_linear1(x)

        # Color Prediction
        rgb = self.color_linear2(x)

        return rgb, sigma