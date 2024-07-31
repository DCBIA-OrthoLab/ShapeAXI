import torch

from pytorch3d.ops import (knn_points, 
                           knn_gather)

def saxi_point_triangle_distance(X, X_hat, K_triangle=1, ignore_first=False, randomize=False):
    """
    Compute the distance between a point and the nearest triangle.
    It uses the knn_points and knn_gather functions from PyTorch3D to find the nearest triangle.
    Args:
        X: (B, N0, 3) tensor of points
        X_hat: (B, N1, 3) tensor of points"""
    
    k_ignore = 0
    if ignore_first:
        k_ignore = 1

    dists = knn_points(X_hat, X, K=(3*K_triangle + k_ignore))
    start_idx = (3*(K_triangle-1)) + k_ignore

    if randomize:
        idx = dists.idx[:, :, torch.randperm(dists.idx.shape[2])]
    else:
        idx = dists.idx
    
    x = knn_gather(X, idx[:, :, start_idx:start_idx + 3])
    # Compute the normal of the triangle
    
    N = torch.cross(x[:, :, 1] - x[:, :, 0], x[:, :, 2] - x[:, :, 0], dim=-1)
    N = N / torch.norm(N, dim=1, keepdim=True)
    # Compute the vector from the point to the first vertex of the triangle
    X_v = (X_hat - x[:, :, 0]) 
    
    # return torch.sum(torch.abs(torch.einsum('ijk,ijk->ij', X_v, N))) + torch.sum(torch.square(X_v))
    return torch.sum(torch.abs(torch.einsum('ijk,ijk->ij', X_v, N)))