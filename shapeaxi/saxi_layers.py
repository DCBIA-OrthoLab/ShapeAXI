import torch
from torch import nn


from pytorch3d.ops import knn_points, knn_gather

# This file contains the definition of the IcosahedronConv2d, IcosahedronConv1d and IcosahedronLinear classes, which are used to perform convolution and linear operations on icosahedral meshes

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
    # Wrapper to apply a module to each time step of a sequence
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module
 
    def forward(self, input_seq):
        assert len(input_seq.size()) > 2
 
        # reshape input data --> (samples * timesteps, input_size)
        # squash timesteps

        size = input_seq.size()
        batch_size = size[0]
        time_steps = size[1]
        size_reshape = [batch_size*time_steps] + list(size[2:])
        reshaped_input = input_seq.contiguous().view(size_reshape)
        output = self.module(reshaped_input)

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
    
class ProjectionHead(nn.Module):
    # Projection MLP
    def __init__(self, input_dim=1280, hidden_dim=1280, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        )

    def forward(self, x):
        x_v = self.model(x)
        return x, x_v

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
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
    
    def forward(self, x):
        attn_output, attn_output_weights = self.attention(x, x, x)
        if self.return_weights:
            return attn_output, attn_output_weights
        return attn_output

class Residual(nn.Module):
    def __init__(self, module: nn.Module, dimension: int):
        super().__init__()
        self.module = module
        self.norm = nn.LayerNorm(dimension)

    def forward(self, x):
        # Assume that the "query" tensor is given first, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        return self.norm(x + self.module(x))

class FeedForward(nn.Module):
    def __init__(self, dimension: int, hidden_dimension: int, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, dimension),
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