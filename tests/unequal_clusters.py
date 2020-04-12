from balanced_kmeans import kmeans
import pytest
import torch

configs = [
    (2, 8192, 30, 64)
]

@pytest.mark.parametrize("batch_size, N, dim, num_clusters", configs)
def test_gpu_speed(batch_size, N, dim, num_clusters):
    cluster_size = N // num_clusters
    X = torch.rand(batch_size, N, dim, device='cuda')
    choices, centers = kmeans(X, num_clusters=num_clusters)
