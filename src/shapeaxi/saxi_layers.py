import torch
from torch import nn
from pytorch3d.ops import knn_points, knn_gather, sample_farthest_points

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence as pack_sequence, pad_packed_sequence as unpack_sequence


from copy import deepcopy
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.pool import voxel_grid
from torch_scatter import segment_csr

import einops
from timm.models.layers import DropPath
# import pointops

# This file contains the definition of the IcosahedronConv2d, IcosahedronConv1d and IcosahedronLinear classes, which are used to perform convolution and linear operations on icosahedral meshes

def sample_points(x, Ns):
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
        indices = torch.sort(indices, dim=1).values

        # Gather the sampled points
        x = knn_gather(x, indices).squeeze(-2).contiguous()

        return x, indices

class IcosahedronConv2d(nn.Module):
    def __init__(self,module,verts,list_edges):
        super().__init__()
        self.module = module
        self.verts = verts
        self.list_edges = list_edges
        # self.nbr_vert = np.max(self.list_edges)+1
        self.nbr_vert = len(self.verts)

        self.list_neighbors = self.get_neighbors()
        self.list_neighbors = self.sort_neighbors()
        self.list_neighbors = self.sort_rotation()
        mat_neighbors = self.get_mat_neighbors()

        self.register_buffer("mat_neighbors", mat_neighbors)


    def get_neighbors(self):
        neighbors = [[] for i in range(self.nbr_vert)]
        for edge in self.list_edges:
            v1 = edge[0].item()
            v2 = edge[1].item()
            neighbors[v1].append(v2)
            neighbors[v2].append(v1)
        return neighbors

    def sort_neighbors(self):
        new_neighbors = [[] for i in range(self.nbr_vert)]
        for i in range(self.nbr_vert):
            neighbors = self.list_neighbors[i].copy()
            vert = neighbors[0]
            new_neighbors[i].append(vert)
            neighbors.remove(vert)
            while len(neighbors) != 0:
                common_neighbors = list(set(neighbors).intersection(self.list_neighbors[vert]))
                vert = common_neighbors[0]
                new_neighbors[i].append(vert)
                neighbors.remove(vert)
        return new_neighbors

    def sort_rotation(self):
        new_neighbors = [[] for i in range(self.nbr_vert)]
        for i in range(self.nbr_vert):
            p0 = self.verts[i]
            p1 = self.verts[self.list_neighbors[i][0]]
            p2 = self.verts[self.list_neighbors[i][1]]
            v1 = p1 - p0
            v2 = p2 - p1
            vn = torch.cross(v1,v2)
            n = vn/torch.norm(vn)


            milieu = p1 + v2/2
            v3 = milieu - p0
            cg = p0 + 2*v3/3

            if (torch.dot(n,cg) > 1 ):
                new_neighbors[i] = self.list_neighbors[i]
            else:
                self.list_neighbors[i].reverse()
                new_neighbors[i] = self.list_neighbors[i]

        return new_neighbors

    def get_mat_neighbors(self):
        mat = torch.zeros(self.nbr_vert,self.nbr_vert*9)
        for index_cam in range(self.nbr_vert):
            mat[index_cam][index_cam*9] = 1
            for index_neighbor in range(len(self.list_neighbors[index_cam])):
                mat[self.list_neighbors[index_cam][index_neighbor]][index_cam*9+index_neighbor+1] = 1
        return mat


    def forward(self,x):
        batch_size,nbr_cam,nbr_features = x.size()
        x = x.permute(0,2,1)
        size_reshape = [batch_size*nbr_features,nbr_cam]
        x = x.contiguous().view(size_reshape)

        x = torch.mm(x,self.mat_neighbors)
        size_reshape2 = [batch_size,nbr_features,nbr_cam,3,3]
        x = x.contiguous().view(size_reshape2)
        x = x.permute(0,2,1,3,4)

        size_reshape3 = [batch_size*nbr_cam,nbr_features,3,3]
        x = x.contiguous().view(size_reshape3)

        output = self.module(x)
        output_channels = self.module.out_channels
        size_initial = [batch_size,nbr_cam,output_channels]
        output = output.contiguous().view(size_initial)

        return output


class IcosahedronConv1d(nn.Module):
    def __init__(self,module,verts,list_edges):
        super().__init__()
        self.module = module
        self.verts = verts
        self.list_edges = list_edges
        self.nbr_vert = np.max(self.list_edges)+1

        self.list_neighbors = self.get_neighbors()
        self.list_neighbors = self.sort_neighbors()
        self.list_neighbors = self.sort_rotation()
        mat_neighbors = self.get_mat_neighbors()

        self.register_buffer("mat_neighbors", mat_neighbors)


    def get_neighbors(self):
        neighbors = [[] for i in range(self.nbr_vert)]
        for edge in self.list_edges:
            v1 = edge[0]
            v2 = edge[1]
            neighbors[v1].append(v2)
            neighbors[v2].append(v1)
        return neighbors

    def sort_neighbors(self):
        new_neighbors = [[] for i in range(self.nbr_vert)]
        for i in range(self.nbr_vert):
            neighbors = self.list_neighbors[i].copy()
            vert = neighbors[0]
            new_neighbors[i].append(vert)
            neighbors.remove(vert)
            while len(neighbors) != 0:
                common_neighbors = list(set(neighbors).intersection(self.list_neighbors[vert]))
                vert = common_neighbors[0]
                new_neighbors[i].append(vert)
                neighbors.remove(vert)
        return new_neighbors

    def sort_rotation(self):
        new_neighbors = [[] for i in range(self.nbr_vert)]
        for i in range(self.nbr_vert):
            p0 = self.verts[i]
            p1 = self.verts[self.list_neighbors[i][0]]
            p2 = self.verts[self.list_neighbors[i][1]]
            v1 = p1 - p0
            v2 = p2 - p1
            vn = torch.cross(v1,v2)
            n = vn/torch.norm(vn)


            milieu = p1 + v2/2
            v3 = milieu - p0
            cg = p0 + 2*v3/3

            if (torch.dot(n,cg) > 1 ):
                new_neighbors[i] = self.list_neighbors[i]
            else:
                self.list_neighbors[i].reverse()
                new_neighbors[i] = self.list_neighbors[i]

        return new_neighbors

    def get_mat_neighbors(self):
        mat = torch.zeros(self.nbr_vert,self.nbr_vert*7)
        for index_cam in range(self.nbr_vert):
            mat[index_cam][index_cam*7] = 1
            for index_neighbor in range(len(self.list_neighbors[index_cam])):
                mat[self.list_neighbors[index_cam][index_neighbor]][index_cam*7+index_neighbor+1] = 1
        return mat


    def forward(self,x):
        batch_size,nbr_cam,nbr_features = x.size()
        x = x.permute(0,2,1)
        size_reshape = [batch_size*nbr_features,nbr_cam]
        x = x.contiguous().view(size_reshape)

        x = torch.mm(x,self.mat_neighbors)
        size_reshape2 = [batch_size,nbr_features,nbr_cam,7]
        x = x.contiguous().view(size_reshape2)
        x = x.permute(0,2,1,3)

        size_reshape3 = [batch_size*nbr_cam,nbr_features,7]
        x = x.contiguous().view(size_reshape3)

        output = self.module(x)
        output_channels = self.module.out_channels
        size_initial = [batch_size,nbr_cam,output_channels]
        output = output.contiguous().view(size_initial)

        return output

class IcosahedronLinear(nn.Module):
    def __init__(self,module,verts,list_edges):
        super().__init__()
        self.module = module
        self.out_channels = module.out_features
        self.verts = verts
        self.list_edges = list_edges
        self.nbr_vert = np.max(self.list_edges)+1

        self.list_neighbors = self.get_neighbors()
        self.list_neighbors = self.sort_neighbors()
        self.list_neighbors = self.sort_rotation()
        mat_neighbors = self.get_mat_neighbors()

        self.register_buffer("mat_neighbors", mat_neighbors)


    def get_neighbors(self):
        neighbors = [[] for i in range(self.nbr_vert)]
        for edge in self.list_edges:
            v1 = edge[0]
            v2 = edge[1]
            neighbors[v1].append(v2)
            neighbors[v2].append(v1)
        return neighbors

    def sort_neighbors(self):
        new_neighbors = [[] for i in range(self.nbr_vert)]
        for i in range(self.nbr_vert):
            neighbors = self.list_neighbors[i].copy()
            vert = neighbors[0]
            new_neighbors[i].append(vert)
            neighbors.remove(vert)
            while len(neighbors) != 0:
                common_neighbors = list(set(neighbors).intersection(self.list_neighbors[vert]))
                vert = common_neighbors[0]
                new_neighbors[i].append(vert)
                neighbors.remove(vert)
        return new_neighbors

    def sort_rotation(self):
        new_neighbors = [[] for i in range(self.nbr_vert)]
        for i in range(self.nbr_vert):
            p0 = self.verts[i]
            p1 = self.verts[self.list_neighbors[i][0]]
            p2 = self.verts[self.list_neighbors[i][1]]
            v1 = p1 - p0
            v2 = p2 - p1
            vn = torch.cross(v1,v2)
            n = vn/torch.norm(vn)


            milieu = p1 + v2/2
            v3 = milieu - p0
            cg = p0 + 2*v3/3

            if (torch.dot(n,cg) > 1 ):
                new_neighbors[i] = self.list_neighbors[i]
            else:
                self.list_neighbors[i].reverse()
                new_neighbors[i] = self.list_neighbors[i]

        return new_neighbors

    def get_mat_neighbors(self):
        mat = torch.zeros(self.nbr_vert,self.nbr_vert*7)
        for index_cam in range(self.nbr_vert):
            mat[index_cam][index_cam*7] = 1
            for index_neighbor in range(len(self.list_neighbors[index_cam])):
                mat[self.list_neighbors[index_cam][index_neighbor]][index_cam*7+index_neighbor+1] = 1
        return mat


    def forward(self,x):
        batch_size,nbr_cam,nbr_features = x.size()
        x = x.permute(0,2,1)
        size_reshape = [batch_size*nbr_features,nbr_cam]
        x = x.contiguous().view(size_reshape)

        x = torch.mm(x,self.mat_neighbors)
        size_reshape2 = [batch_size,nbr_features,nbr_cam,7]
        x = x.contiguous().view(size_reshape2)
        x = x.permute(0,2,1,3)

        size_reshape3 = [batch_size*nbr_cam,nbr_features*7]
        x = x.contiguous().view(size_reshape3)

        output = self.module(x)
        size_initial = [batch_size,nbr_cam,self.out_channels]
        output = output.contiguous().view(size_initial)

        return output
    
class TimeDistributed(nn.Module):
    def __init__(self, module, time_dim=1):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.time_dim = time_dim

    def forward(self, input_seq, *args, **kwargs):
        assert len(input_seq.size()) > 2

        # reshape input data --> (samples * timesteps, input_size)
        # squash timesteps
        size = list(input_seq.size())
        batch_size = size[0]
        time_steps = size.pop(self.time_dim)
        size_reshape = [batch_size * time_steps] + list(size[1:])
        reshaped_input = input_seq.contiguous().view(size_reshape)

        # Pass the additional arguments to the module
        output = self.module(reshaped_input, *args, **kwargs)

        if isinstance(output, tuple):
            output = list(output)
            for i in range(len(output)):
                output_size = output[i].size()
                output_size = [batch_size, time_steps] + list(output_size[1:])
                output[i] = output[i].contiguous().view(output_size)
            
        else:
            output_size = output.size()
            output_size = [batch_size, time_steps] + list(output_size[1:])
            output = output.contiguous().view(output_size)

        return output

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Norm(nn.Module):
    def __call__(self, x):
        return torch.nn.functional.normalize(x, p=2, dim=-1)

class ProjectionHead(nn.Module):
    # Projection MLP
    def __init__(self, input_dim=1280, hidden_dim=1280, output_dim=128, dropout=0.1):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False),
        )

    def forward(self, x):
        x = self.model(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, dim=1):
        super().__init__()

        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        self.Tanh = nn.Tanh()
        self.Sigmoid = nn.Sigmoid()
        self.dim = dim

    def forward(self, query, values):
        
        score = self.Sigmoid(self.V(self.Tanh(self.W1(query))))

        attention_weights = score/torch.sum(score, dim=self.dim, keepdim=True)

        context_vector = attention_weights * values
        context_vector = torch.sum(context_vector, dim=self.dim)

        return context_vector, score

class MaxPoolImages(nn.Module):
    def __init__(self, nbr_images = 12):
        super().__init__()
        self.nbr_images = nbr_images
        self.max_pool = nn.MaxPool1d(self.nbr_images)

    def forward(self,x):
        x = x.permute(0,2,1)
        output = self.max_pool(x)
        output = output.squeeze(dim=2)

        return output

class MHA(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, return_weights=False):
        super(MHA, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.return_weights = return_weights
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=False, batch_first=True)
    
    def forward(self, x):
        attn_output, attn_output_weights = self.attention(x, x, x)
        if self.return_weights:
            return attn_output, attn_output_weights
        return attn_output
    
class MHA_KNN(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, return_weights=False, K=6, return_sorted=True, random=False, return_v=False, use_direction=True):
        super(MHA_KNN, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.return_weights = return_weights
        self.K = K
        self.return_sorted = return_sorted
        self.random = random
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=False, batch_first=True)
        self.return_v = return_v
        self.use_direction = use_direction
    
    def forward(self, x):

        batch_size, V_n, Embed_dim = x.shape

        # the query is the input point itself, the shape of q is [BS, V_n, 1, Embed_dim]
        q = x.unsqueeze(-2)

        if self.random:

            # randomly select K points to the query            
            idx = torch.randint(0, V_n, (batch_size, V_n, self.K), device=x.device)

            q = x.unsqueeze(-2)
            k = knn_gather(x, idx)

            # compute the distances between the query and the randomly selected points
            distances = torch.linalg.norm(k - q, dim=3)
            # sort and gather the closest K points to the query
            k = knn_gather(x, distances.argsort())

        else:
            #input shape of x is [BS, V_n, Embed_dim]
            dists = knn_points(x, x, K=self.K, return_sorted=self.return_sorted)            
            # compute the key, the input shape is [BS, V_n, K, Embed_dim], it has the closest K points to the query
            k = knn_gather(x, dists.idx)
        #the value tensor contains the directions towards the closest points. 
        # the intuition here is that based on the query and key embeddings, the model will learn to predict
        # the best direction to move the new embedding, i.e., create a new point in the point cloud
        # the shape of v is [BS, V_n, K, Embed_dim]
        if self.use_direction:
            v = k - q
        else:
            v = k

        q = q.contiguous().view(batch_size * V_n, 1, Embed_dim) # Original point with dimension 1 added
        k = k.contiguous().view(batch_size * V_n, self.K, Embed_dim)
        v = v.contiguous().view(batch_size * V_n, self.K, Embed_dim)        

        v, x_w = self.attention(q, k, v)

        v = v.contiguous().view(batch_size, V_n, Embed_dim)
        x_w = x_w.contiguous().view(batch_size, V_n, self.K)

        x_w = torch.zeros(batch_size, V_n, device=x.device).scatter_add_(1, dists.idx.view(batch_size, -1), x_w.view(batch_size, -1))
        
        # The new predicted point is the sum of the input point and the weighted sum of the directions
        if self.use_direction:
            x = x + v
        else:
            x = v

        if self.return_v:
            if self.return_weights:
                return x, x_w, v
            return x, v
        
        if self.return_weights:
            return x, x_w
        return x

class MHA_KNN_V(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, return_weights=False, K=6, return_sorted=True, use_direction=True):
        super(MHA_KNN_V, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.return_weights = return_weights
        self.K = K
        self.return_sorted = return_sorted
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=False, batch_first=True)        
        self.use_direction = use_direction
    
    def forward(self, x, x_v, x_v_fixed=None):

        batch_size, V_n, Embed_dim = x.shape
        
        #input shape of x is [BS, V_n, Embed_dim]
        if x_v_fixed is None:
            x_v_fixed = x_v

        dists_idx = None
        K = self.K
        if isinstance(self.K, tuple):
            K = self.K[0]
            K_farthest = self.K[1]
            dists = knn_points(x_v_fixed, x_v, K=K, return_sorted=self.return_sorted) # The idx is of shape [BS, V_n, K]

            _, selected_indices = sample_farthest_points(x_v, K=K_farthest) # The idx is of shape [BS, K]
            selected_indices = selected_indices.unsqueeze(1).expand(-1, V_n, -1) # The idx is of shape [BS, V_n, K]
            dists_idx = torch.cat((dists.idx, selected_indices), dim=2)

            K = K + K_farthest
        else:            
            dists = knn_points(x_v_fixed, x_v, K=self.K, return_sorted=self.return_sorted)
            dists_idx = dists.idx
        # compute the key, the input shape is [BS, V_n, K, Embed_dim], it has the closest K points to the query. i.e, find the closest K points to each point in the point cloud
        k = knn_gather(x, dists_idx)

        #the value tensor contains the directions towards the closest points. 
        # the intuition here is that based on the query and key embeddings, the model will learn to predict
        # the best direction to move the new embedding, i.e., create a new point in the point cloud
        # the shape of v is [BS, V_n, K, Embed_dim]

        # the query is the input point itself, the shape of q is [BS, V_n, 1, Embed_dim]
        q = x.unsqueeze(-2)

        if self.use_direction:
            v = k - q
        else:
            v = k

        q = q.contiguous().view(batch_size * V_n, 1, Embed_dim) # Original point with dimension 1 added
        k = k.contiguous().view(batch_size * V_n, K, Embed_dim)
        v = v.contiguous().view(batch_size * V_n, K, Embed_dim)        

        v, x_w = self.attention(q, k, v)

        v = v.contiguous().view(batch_size, V_n, Embed_dim)
        x_w = x_w.contiguous().view(batch_size, V_n, K)
        
        # Based on the weights of the attention layer, we compute the new position of the points
        # Shape of x_w is [BS, V_n, K] and k_v (x_v after knn_gather) is [BS, V_n, K, 3]

        # x_w = torch.zeros(batch_size, V_n, device=x.device).scatter_add_(1, dists.idx.view(batch_size, -1), x_w.view(batch_size, -1))
        x_w = torch.zeros(batch_size, V_n, device=x.device).scatter_reduce_(dim=1, index=dists_idx.view(batch_size, -1), src=x_w.view(batch_size, -1), reduce="mean")
        x_w = x_w.unsqueeze(-1)
        
        # The new predicted point is the sum of the input point and the weighted sum of the directions
        if self.use_direction:
            x = x + v
        else:
            x = v

        if self.return_weights:
            return x, x_w
        return x
    
class KNN_Embedding_V(nn.Module):
    def __init__(self, input_dim, embed_dim, K=27, return_sorted=True):
        super(KNN_Embedding_V, self).__init__()
        self.input_dim = input_dim
        self.K = K
        self.return_sorted = return_sorted

        self.module = nn.Linear(self.input_dim*self.K, embed_dim)
    
    def forward(self, x, x_v):
        
        #input shape of x is [BS, V_n, Embed_dim]
        dists = knn_points(x_v, x_v, K=self.K, return_sorted=self.return_sorted)            
        # compute the key, the input shape is [BS, V_n, K, Embed_dim], it has the closest K points to the query
        x = knn_gather(x, dists.idx)
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        x = self.module(x)
        
        return x

class Residual(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x, *args, **kwargs):
        # Assume that the "query" tensor is given first, so we can compute the
        # residual.
        x_out = self.module(x, *args, **kwargs)
        if isinstance(x_out, tuple):
            return (x + x_out[0],) + x_out[1:]
        return x + x_out

class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)

class UnpoolSubDivision(nn.Module):
    def __init__(self, K=4):
        super(UnpoolSubDivision, self).__init__()
        self.K = K
        self.module = TimeDistributed(SubDivision())
    def __call__(self, x):
        
        dists = knn_points(x, x, K=self.K)
        
        x = knn_gather(x, dists.idx)
        x = self.module(x)
        x = x.view(x.shape[0], -1, x.shape[-1]).contiguous()

        return x

class SubDivision(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        self.mha = MHA(embed_dim, num_heads, dropout=dropout)
    def forward(self, x):

        attn_output, attn_output_weights = self.attention(x, x, x)
        # compute the mid point between the point at 0 and the other closest point
        mp = (x[:, 0:1] + x[:, 1:])/2.0
        # concatenate the points at 0th index and the interpolated points
        x = torch.cat((x[:, 0:1], mp), dim=1)
        return x
    

class UnpoolMHA(nn.Module):    
    def __call__(self, x):                
        x = x.view(x.shape[0], -1, x.shape[-1])
        return x
    
class SmoothAttention(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=64, K=4):
        super(SmoothAttention, self).__init__()
        self.embed_dim = embed_dim
        self.attn = SelfAttention(embed_dim, hidden_dim, dim=2)
        self.K = K
    
    def forward(self, x):

        # find closest points to self, i.e., each point in the sample finds the closest K points in the sample
        dists = knn_points(x, x, K=self.K)
        # gather the K closest points
        x = knn_gather(x, dists.idx)

        # apply self attention, i.e., weighted average of the K closest points
        x, x_s = self.attn(x, x)

        return x
    
class AttentionPooling(nn.Module):
    def __init__(self, embed_dim=128, pooling_factor=0.125, hidden_dim=64, K=4):
        super(AttentionPooling, self).__init__()
        
        self.embed_dim = embed_dim
        self.pooling_factor = pooling_factor
        # self.attn = SelfAttention(embed_dim, hidden_dim, dim=2)
        
        self.W1 = nn.Linear(embed_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        self.Tanh = nn.Tanh()
        self.Sigmoid = nn.Sigmoid()

        self.K = K
    
    def forward(self, x, x_v):


        x_s = self.Sigmoid(self.V(self.Tanh(self.W1(x))))

        # Grab the samples that have high score
        n_samples = int(x.shape[1]*self.pooling_factor)
        idx = torch.argsort(x_s, descending=True, dim=1)[:,:n_samples]
        
        x = knn_gather(x, idx).squeeze(2)
        x_s = knn_gather(x_s, idx).squeeze(2)
        x_v = knn_gather(x_v, idx).squeeze(2)
        
        return x, x_v, x_s
    
class Attention_V(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=64):
        super(Attention_V, self).__init__()
        
        self.embed_dim = embed_dim
        
        self.W1 = nn.Linear(embed_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        self.Tanh = nn.Tanh()
        self.Sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.Sigmoid(self.V(self.Tanh(self.W1(x))))
    
class Pooling_V(nn.Module):
    def __init__(self, pooling_factor=0.125):
        super(Pooling_V, self).__init__()
        self.pooling_factor = pooling_factor
    
    def forward(self, x, x_v, x_s):
        
        n_samples = int(x_s.shape[1]*self.pooling_factor)
        idx = torch.argsort(x_s, descending=True, dim=1)[:,:n_samples]
        
        x = knn_gather(x, idx).squeeze(2)
        x_s = knn_gather(x_s, idx).squeeze(2)
        x_v = knn_gather(x_v, idx).squeeze(2)
        
        return x, x_v, x_s
    
class AttentionPooling_V(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=64, K=27, pooling_factor=0.125, score_pooling=False):
        super(AttentionPooling_V, self).__init__()
        
        self.embed_dim = embed_dim
        self.pooling_factor = pooling_factor
        self.score_pooling = score_pooling

        if isinstance(K, tuple):
            self.K = K[0]
        else:
            self.K = K
        
        self.W1 = nn.Linear(embed_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        self.Tanh = nn.Tanh()
        self.Sigmoid = nn.Sigmoid()

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

    def get_pooling_idx(self, x_s, x_v, pf, K, x_v_fixed=None):

        # Find next level points
        if x_v_fixed is not None:
            n_samples = int(x_s.shape[1]*pf)
            x_v_fixed = x_v_fixed[:, :n_samples]

            dist = knn_points(x_v_fixed, x_v, K=1)
            x_v_next = knn_gather(x_v, dist.idx).squeeze(2)
        
        elif self.score_pooling:
            n_samples = int(x_s.shape[1]*pf)
            x_idx_next = torch.argsort(x_s, descending=True, dim=1)[:,:n_samples]
            x_v_next = knn_gather(x_v, x_idx_next).squeeze(2)
        else:
            # x_v_next, x_idx_next = self.sample_points(x_v, int(x_v.shape[1] * pf))
            x_v_next, _ = sample_farthest_points(x_v, K=int(x_v.shape[1] * pf)) # The idx is of shape [BS, K]

        # Find the closest points in the next level using the current level sampling, i.e., the output
        # dist.idx will have the indices of the points in the next level but dimension of the current level
        pooling = knn_points(x_v_next, x_v, K=K)
        unpooling = knn_points(x_v, x_v_next, K=K)

        pooling_idx = pooling.idx
        unpooling_idx = unpooling.idx

        
        return pooling_idx, unpooling_idx, x_v_next, x_v_fixed
        
    
    def forward(self, x, x_v, x_v_fixed=None):

        # find closest points to self, i.e., each point in the sample finds the closest K points in the sample
        x_s = self.Sigmoid(self.V(self.Tanh(self.W1(x))))
        
        pooling_idx, unpooling_idx, x_v, x_v_fixed = self.get_pooling_idx(x_s, x_v, pf=self.pooling_factor, K=self.K, x_v_fixed=x_v_fixed)
        
        x = knn_gather(x, pooling_idx)
        score = knn_gather(x_s, pooling_idx)
        
        attention_weights = score/torch.sum(score, dim=2, keepdim=True)

        x = attention_weights * x
        x = torch.sum(x, dim=2)

        if x_v_fixed is not None:
            return x, x_v, x_s, pooling_idx, unpooling_idx, x_v_fixed
        
        return x, x_v, x_s, pooling_idx, unpooling_idx
    
class AttentionPooling_Idx(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=64):
        super(AttentionPooling_Idx, self).__init__()
        
        self.embed_dim = embed_dim
        
        self.W1 = nn.Linear(embed_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        self.Tanh = nn.Tanh()
        self.Sigmoid = nn.Sigmoid()
    
    def forward(self, x, idx):
        # apply self attention, i.e., weighted average of the K closest points

        x_s = self.Sigmoid(self.V(self.Tanh(self.W1(x))))

        x = knn_gather(x, idx)
        score = knn_gather(x_s, idx)

        attention_weights = score/torch.sum(score, dim=2, keepdim=True)
        
        x = attention_weights * x
        x = torch.sum(x, dim=2)
        
        return x, x_s
    
class UnpoolMHA_KNN(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return torch.cat([x, self.module(x)], dim=1)
    
class SmoothAttention(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=64, K=4):
        super(SmoothAttention, self).__init__()
        self.embed_dim = embed_dim
        self.attn = SelfAttention(embed_dim, hidden_dim, dim=2)
        self.K = K
    
    def forward(self, x):

        # find closest points to self, i.e., each point in the sample finds the closest K points in the sample
        dists = knn_points(x, x, K=self.K)
        # gather the K closest points
        x = knn_gather(x, dists.idx)

        # apply self attention, i.e., weighted average of the K closest points
        x, x_s = self.attn(x, x)

        return x   
    
class SmoothMHA(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, K=12, dropout=0.1, dim=2):
        super(SmoothMHA, self).__init__()
        self.embed_dim = embed_dim
        self.attn_td = TimeDistributed(MHA(embed_dim, num_heads, dropout=dropout, return_weights=True))
        self.W = nn.Linear(K, 1)
        self.K = K
        self.dim = dim
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):

        # find closest points to self, i.e., each point in the sample finds the closest K points in the sample
        dists = knn_points(x, x, K=self.K)
        # gather the K closest points
        x = knn_gather(x, dists.idx)

        # apply self attention, i.e., weighted average of the K closest points
        x, x_s = self.attn_td(x)
        x_s = self.sigmoid(self.W(x_s))
        attention_weights = x_s/torch.sum(x_s, dim=self.dim, keepdim=True)
        x = attention_weights * x
        x = torch.sum(x, dim=self.dim)

        return x
    
class MHA_Idx(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, use_direction=True):
        super(MHA_Idx, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=False, batch_first=True)
        self.use_direction = use_direction
    
    def forward(self, x, idx):

        batch_size, V_n, Embed_dim = x.shape
        K = idx.shape[-1]
        
        k = knn_gather(x, idx)
        
        #the value tensor contains the directions towards the closest points. 
        # the intuition here is that based on the query and key embeddings, the model will learn to predict
        # the best direction to move the new embedding, i.e., create a new point in the point cloud
        # the shape of v is [BS, V_n, K, Embed_dim]

        # the query is the input point itself, the shape of q is [BS, V_n, 1, Embed_dim]
        q = x.unsqueeze(-2)

        if self.use_direction:
            v = k - q
        else:
            v = k

        q = q.contiguous().view(batch_size * V_n, 1, Embed_dim) # Original point with dimension 1 added
        k = k.contiguous().view(batch_size * V_n, K, Embed_dim)
        v = v.contiguous().view(batch_size * V_n, K, Embed_dim)        

        v, x_w = self.attention(q, k, v)

        v = v.contiguous().view(batch_size, V_n, Embed_dim)
        # x_w = x_w.contiguous().view(batch_size, V_n, K)
        
        # Based on the weights of the attention layer, we compute the new position of the points
        # Shape of x_w is [BS, V_n, K] and k_v (x_v after knn_gather) is [BS, V_n, K, 3]
        
        # The new predicted point is the sum of the input point and the weighted sum of the directions
        if self.use_direction:
            x = x + v
        else:
            x = v

        return x
    
class AttentionChunk(nn.Module):
    def __init__(self, input_dim, hidden_dim, chunks=16, permute_time_dim=True):
        super().__init__()
        
        self.attn = SelfAttention(input_dim, hidden_dim)
        self.chunks = chunks
        self.permute_time_dim = permute_time_dim

    def forward(self, x):

        # Shape of x is [BS, T, C, H, W] or [BS, C, T, H, W]
        assert len(x.shape) == 5

        if self.permute_time_dim:
            x = x.permute(0, 2, 1, 3, 4) # [BS, C, T, H, W] -> [BS, T, C, H, W]
        
        x_out = []
        x_shape = list(x.shape)
        x_shape[1] = self.chunks
        
        for ch in torch.chunk(x, chunks=self.chunks, dim=1): # Iterate in the time dimension and create chunks
            ch = ch.flatten(2) # Flatten the spatial dimensions
            ch, ch_s = self.attn(ch, ch) # Compute average attention for each chunk
            x_out.append(ch)
        x_out = torch.stack(x_out, dim=1).view(x_shape) # reshape to original shape but change the time dimension to the number of chunks
        
        if self.permute_time_dim:
            x_out = x_out.permute(0, 2, 1, 3, 4)
        return x_out

@torch.inference_mode()
def offset2bincount(offset):
    return torch.diff(
    offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )

@torch.inference_mode()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
    len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.inference_mode()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
    """

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return (self.norm(input.transpose(1, 2).contiguous()).transpose(1, 2).contiguous())
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError
    

class GroupedVectorAttention(nn.Module):
    def __init__(self,embed_channels,groups,attn_drop_rate=0.0,qkv_bias=True,pe_multiplier=False,pe_bias=True,):
        super(GroupedVectorAttention, self).__init__()
        self.embed_channels = embed_channels
        self.groups = groups
        assert embed_channels % groups == 0
        self.attn_drop_rate = attn_drop_rate
        self.qkv_bias = qkv_bias
        self.pe_multiplier = pe_multiplier
        self.pe_bias = pe_bias

        self.linear_q = nn.Sequential(nn.Linear(embed_channels,
                                                embed_channels, bias=qkv_bias),
                                                PointBatchNorm(embed_channels),
                                                nn.ReLU(inplace=True),
                                                )
        
        self.linear_k = nn.Sequential(nn.Linear(embed_channels, 
                                                embed_channels, bias=qkv_bias),
                                                PointBatchNorm(embed_channels),
                                                nn.ReLU(inplace=True),
                                                )

        self.linear_v = nn.Linear(embed_channels, embed_channels, bias=qkv_bias)

        if self.pe_multiplier:
            self.linear_p_multiplier = nn.Sequential(nn.Linear(3, embed_channels),
                                                     PointBatchNorm(embed_channels),
                                                     nn.ReLU(inplace=True),
                                                     nn.Linear(embed_channels, embed_channels),
                                                     )
        if self.pe_bias:
            self.linear_p_bias = nn.Sequential(nn.Linear(3, embed_channels),
                                               PointBatchNorm(embed_channels),
                                               nn.ReLU(inplace=True),
                                               nn.Linear(embed_channels, embed_channels),
                                               )
            
        self.weight_encoding = nn.Sequential(nn.Linear(embed_channels, groups),
                                             PointBatchNorm(groups),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(groups, groups),
                                             )
        
        self.softmax = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop_rate)

    def forward(self, feat, coord, reference_index):
        query, key, value = (self.linear_q(feat),self.linear_k(feat),self.linear_v(feat),)

        key = pointops.grouping(reference_index, key, coord, with_xyz=True)
        value = pointops.grouping(reference_index, value, coord, with_xyz=False)
        pos, key = key[:, :, 0:3], key[:, :, 3:]
        relation_qk = key - query.unsqueeze(1)
        if self.pe_multiplier:
            pem = self.linear_p_multiplier(pos)
            relation_qk = relation_qk * pem
        if self.pe_bias:
            peb = self.linear_p_bias(pos)
            relation_qk = relation_qk + peb
            value = value + peb

        weight = self.weight_encoding(relation_qk)
        weight = self.attn_drop(self.softmax(weight))

        mask = torch.sign(reference_index + 1)
        weight = torch.einsum("n s g, n s -> n s g", weight, mask)
        value = einops.rearrange(value, "n ns (g i) -> n ns g i", g=self.groups)
        feat = torch.einsum("n s g i, n s g -> n g i", value, weight)
        feat = einops.rearrange(feat, "n g i -> n (g i)")
        return feat
    
class Block(nn.Module):
    def __init__(self,embed_channels,groups,qkv_bias=True,pe_multiplier=False,pe_bias=True,attn_drop_rate=0.0,drop_path_rate=0.0,enable_checkpoint=False,):
        super(Block, self).__init__()
        self.attn = GroupedVectorAttention(embed_channels=embed_channels,
                                           groups=groups,
                                           qkv_bias=qkv_bias,
                                           attn_drop_rate=attn_drop_rate,
                                           pe_multiplier=pe_multiplier,
                                           pe_bias=pe_bias,
                                           )
        
        self.fc1 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.fc3 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.norm1 = PointBatchNorm(embed_channels)
        self.norm2 = PointBatchNorm(embed_channels)
        self.norm3 = PointBatchNorm(embed_channels)
        self.act = nn.ReLU(inplace=True)
        self.enable_checkpoint = enable_checkpoint
        self.drop_path = (DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity())

    def forward(self, points, reference_index):
        coord, feat, offset = points
        identity = feat
        feat = self.act(self.norm1(self.fc1(feat)))
        feat = (self.attn(feat, coord, reference_index) if not self.enable_checkpoint else checkpoint(self.attn, feat, coord, reference_index))
        feat = self.act(self.norm2(feat))
        feat = self.norm3(self.fc3(feat))
        feat = identity + self.drop_path(feat)
        feat = self.act(feat)
        return [coord, feat, offset]

class BlockSequence(nn.Module):
    def __init__( self, depth, embed_channels, groups, neighbours=16, qkv_bias=True, pe_multiplier=False, pe_bias=True, attn_drop_rate=0.0, drop_path_rate=0.0, enable_checkpoint=False,):
        super(BlockSequence, self).__init__()

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        elif isinstance(drop_path_rate, float):
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        else:
            drop_path_rates = [0.0 for _ in range(depth)]

        self.neighbours = neighbours
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(embed_channels=embed_channels,
                          groups=groups,
                          qkv_bias=qkv_bias,
                          pe_multiplier=pe_multiplier,
                          pe_bias=pe_bias,
                          attn_drop_rate=attn_drop_rate,
                          drop_path_rate=drop_path_rates[i],
                          enable_checkpoint=enable_checkpoint,
                          )
            
            self.blocks.append(block)

    def forward(self, points):
        coord, feat, offset = points
        # reference index query of neighbourhood attention
        # for windows attention, modify reference index query method
        reference_index, _ = pointops.knn_query(self.neighbours, coord, offset)
        for block in self.blocks:
            points = block(points, reference_index)
        return points

class GridPool(nn.Module):
    """
    Partition-based Pooling (Grid Pooling)
    """

    def __init__(self, in_channels, out_channels, grid_size, bias=False):
        super(GridPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size

        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = PointBatchNorm(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, points, start=None):
        coord, feat, offset = points
        batch = offset2batch(offset)
        feat = self.act(self.norm(self.fc(feat)))
        start = (
            segment_csr(coord,torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]),reduce="min",)
            if start is None
            else start
        )
        cluster = voxel_grid(pos=coord - start[batch], size=self.grid_size, batch=batch, start=0)
        unique, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)
        _, sorted_cluster_indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        coord = segment_csr(coord[sorted_cluster_indices], idx_ptr, reduce="mean")
        feat = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="max")
        batch = batch[idx_ptr[:-1]]
        offset = batch2offset(batch)
        return [coord, feat, offset], cluster
    
class Encoder(nn.Module):
    def __init__(self,depth,in_channels,embed_channels,groups,grid_size=None,neighbours=16,qkv_bias=True,pe_multiplier=False,
                            pe_bias=True,attn_drop_rate=None,drop_path_rate=None,enable_checkpoint=False):
        super(Encoder, self).__init__()

        self.down = GridPool(in_channels=in_channels,out_channels=embed_channels,grid_size=grid_size,)

        self.blocks = BlockSequence(depth=depth,
                                    embed_channels=embed_channels,
                                    groups=groups,
                                    neighbours=neighbours,
                                    qkv_bias=qkv_bias,
                                    pe_multiplier=pe_multiplier,
                                    pe_bias=pe_bias,
                                    attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.0,
                                    drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.0,
                                    enable_checkpoint=enable_checkpoint,
                                    )

    def forward(self, points):
        points, cluster = self.down(points)
        return self.blocks(points), cluster
    
class GVAPatchEmbed(nn.Module):
    def __init__(self,depth,in_channels,embed_channels,groups,neighbours=16,qkv_bias=True,pe_multiplier=False,pe_bias=True,attn_drop_rate=0.0,drop_path_rate=0.0,enable_checkpoint=False,):
    
        super(GVAPatchEmbed, self).__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.proj = nn.Sequential(nn.Linear(in_channels, embed_channels, bias=False),PointBatchNorm(embed_channels),nn.ReLU(inplace=True),)
        self.blocks = BlockSequence(depth=depth,
                                    embed_channels=embed_channels,
                                    groups=groups,
                                    neighbours=neighbours,
                                    qkv_bias=qkv_bias,
                                    pe_multiplier=pe_multiplier,
                                    pe_bias=pe_bias,
                                    attn_drop_rate=attn_drop_rate,
                                    drop_path_rate=drop_path_rate,
                                    enable_checkpoint=enable_checkpoint,
                                    )

    def forward(self, points):
        coord, feat, offset = points
        feat = self.proj(feat)
        return self.blocks([coord, feat, offset])