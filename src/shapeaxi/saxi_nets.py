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
    def __init__(self, input_dim=3, embed_dim=256, hidden_dim=64, num_heads=256, K=32, output_dim=256, sample_levels=[40962, 10242, 2562, 642, 162], dropout=0.1, return_sorted=True):
        super(MHAEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.K = K
        self.sample_levels = sample_levels
        self.dropout = dropout
        self.return_sorted = return_sorted

        self.embedding = nn.Linear(input_dim, embed_dim)

        for i, sl in enumerate(sample_levels):
            setattr(self, f"mha_{i}", MHA_KNN(embed_dim=embed_dim, num_heads=num_heads, K=K, return_weights=True, dropout=dropout))
            setattr(self, f"ff_{i}", Residual(FeedForward(embed_dim, hidden_dim=hidden_dim, dropout=dropout)))
        
        self.output = nn.Linear(embed_dim, output_dim)
    
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

        weights = torch.zeros(x.shape[0], x.shape[1], device=x.device)
        idx = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        
        indices = []
        
        for i, sl in enumerate(self.sample_levels):
            
            if i > 0:
                # select the first sl points a.k.a. downsample/pooling                
                x, x_i = self.sample_points(x, sl)

                # initialize idx with the index of the current level
                idx = x_i
                
                for idx_prev in reversed(indices): # go through the list of the previous ones in reverse
                    idx = knn_gather(idx_prev, idx).squeeze(-2).contiguous() # using the indices of the previous level update idx, at the end idx should have the indices of the first level
                
                idx = idx.squeeze(-1)
                indices.append(x_i)
            
            # the mha will select optimal points from the input
            x, x_w = getattr(self, f"mha_{i}")(x)
            x = getattr(self, f"ff_{i}")(x)
            
            weights.scatter_add_(1, idx, x_w)

        #output layer
        x = self.output(x)
        return x, weights
    

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
    def __init__(self,  input_dim=3, output_dim=1, K=[27], num_heads=[16], stages=[16], dropout=0.1, 
                pooling_factor=None, pooling_hidden_dim=None, score_pooling=False, 
                 feed_forward_hidden_dim=None, return_sorted=True, use_skip_connection=False, 
                 use_layer_norm=False, return_v=True):
        super(MHAIdxEncoder, self).__init__()

        
        self.num_heads = num_heads        
        self.K = K
        self.stages = stages
        self.dropout = dropout        
        self.pooling_factor = pooling_factor
        self.pooling_hidden_dim = pooling_hidden_dim
        self.score_pooling = score_pooling
        self.feed_forward_hidden_dim = feed_forward_hidden_dim
        self.use_skip_connection = use_skip_connection
        self.use_layer_norm = use_layer_norm
        self.return_v = return_v

        self.embedding = nn.Linear(input_dim, self.stages[0], bias=False)

        for i, st in enumerate(self.stages):
            setattr(self, f"mha_{i}", Residual(MHA_KNN_V(embed_dim=st, num_heads=num_heads[i], K=K[i], dropout=dropout, return_sorted=return_sorted, use_direction=False)))
            if self.use_layer_norm:
                setattr(self, f"norm_mha_{i}", nn.LayerNorm(st))

            if self.feed_forward_hidden_dim is not None and feed_forward_hidden_dim[i] is not None:
                setattr(self, f"ff_{i}", Residual(FeedForward(embed_dim=st, hidden_dim=feed_forward_hidden_dim[i], dropout=dropout)))
                if self.use_layer_norm:
                    setattr(self, f"norm_ff_{i}", nn.LayerNorm(st))

            if self.pooling_factor is not None and self.pooling_factor[i] is not None and self.pooling_hidden_dim is not None and self.pooling_hidden_dim[i] is not None: 
                setattr(self, f"pool_{i}", AttentionPooling_V(embed_dim=st, hidden_dim=self.pooling_hidden_dim[i], K=self.K[i], pooling_factor=self.pooling_factor[i], score_pooling=self.score_pooling))
            
            st_n = self.stages[i+1] if i+1 < len(self.stages) else output_dim
            setattr(self, f"output_{i}", nn.Linear(st, st_n, bias=False))
        
    def forward(self, x, x_v, x_v_fixed=None):
        
        x = self.embedding(x)

        unpooling_idxs = []
        skip_connections = []
        
        for i, st in enumerate(self.stages):
            
            x, x_w = getattr(self, f"mha_{i}")(x, x_v, x_v_fixed=x_v_fixed)
            if self.use_layer_norm:
                x = getattr(self, f"norm_mha_{i}")(x)
            
            if self.feed_forward_hidden_dim is not None and self.feed_forward_hidden_dim[i] is not None:
                x = getattr(self, f"ff_{i}")(x)
                if self.use_layer_norm:
                    x = getattr(self, f"norm_ff_{i}")(x)

            if self.use_skip_connection:
                skip_connections.insert(0, x)

            if self.pooling_factor is not None and self.pooling_factor[i] is not None and self.pooling_hidden_dim is not None and self.pooling_hidden_dim[i] is not None: 
                # pooling_idx, unpooling_idx, x_v_next = self.get_pooling_idx(x_v, self.pooling_factor[i], self.K[i])
                x, x_v_next, x_s, pooling_idx, unpooling_idx, x_v_fixed = getattr(self, f"pool_{i}")(x, x_v, x_v_fixed=x_v_fixed)
                unpooling_idxs.insert(0, (x_v, unpooling_idx, x_s))
                x_v = x_v_next

            x = getattr(self, f"output_{i}")(x)
        
        if self.use_skip_connection:
            if self.return_v:
                return x, x_v, unpooling_idxs, skip_connections
        if self.return_v:
            return x, x_v, unpooling_idxs
        return x, unpooling_idxs
    
class MHAIdxDecoder(nn.Module):
    def __init__(self,  input_dim=3, output_dim=1, K=[27], num_heads=[16], stages=[16], dropout=0.1, pooling_hidden_dim=[8], feed_forward_hidden_dim=None, return_sorted=True, use_skip_connection=False, use_layer_norm=False):
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

        # self.embedding = KNN_Embedding_V(input_dim=input_dim, embed_dim=self.stages[0], K=self.K[0])
        self.embedding = nn.Linear(input_dim, self.stages[0], bias=False)

        for i, st in enumerate(self.stages):

            setattr(self, f"pool_{i}", AttentionPooling_Idx(embed_dim=st, hidden_dim=pooling_hidden_dim[i]))

            if self.use_skip_connection:
                setattr(self, f"proj_{i}", ProjectionHead(input_dim=st*2, hidden_dim=st, output_dim=st, dropout=dropout))

            setattr(self, f"mha_{i}", Residual(MHA_KNN_V(embed_dim=st, num_heads=num_heads[i], K=K[i], dropout=dropout, return_sorted=return_sorted, use_direction=False)))
            if self.use_layer_norm:
                setattr(self, f"norm_mha_{i}", nn.LayerNorm(st))

            if self.feed_forward_hidden_dim is not None and feed_forward_hidden_dim[i] is not None:
                setattr(self, f"ff_{i}", Residual(FeedForward(embed_dim=st, hidden_dim=feed_forward_hidden_dim[i], dropout=dropout)))
                if self.use_layer_norm:
                    setattr(self, f"norm_ff_{i}", nn.LayerNorm(st))
            
            st_n = self.stages[i+1] if i+1 < len(self.stages) else output_dim
            setattr(self, f"output_{i}", nn.Linear(st, st_n, bias=False))
        
    def forward(self, x, unpooling_idxs, skip_connections=None):
        
        assert len(unpooling_idxs) == len(self.stages)
                
        x = self.embedding(x)
        
        for i, unp_idx in enumerate(unpooling_idxs):

            x_v, unpooling_idx, _ = unp_idx
            
            x, x_s = getattr(self, f"pool_{i}")(x, unpooling_idx)

            if self.use_skip_connection:
                x = torch.cat([x, skip_connections[i]], dim=-1)
                x = getattr(self, f"proj_{i}")(x)
            
            x = getattr(self, f"mha_{i}")(x, x_v)
            if self.use_layer_norm:
                x = getattr(self, f"norm_mha_{i}")(x)

            if self.feed_forward_hidden_dim is not None and self.feed_forward_hidden_dim[i] is not None:
                x = getattr(self, f"ff_{i}")(x)
                if self.use_layer_norm:
                    x = getattr(self, f"norm_ff_{i}")(x)            

            x = getattr(self, f"output_{i}")(x)
        
        return x