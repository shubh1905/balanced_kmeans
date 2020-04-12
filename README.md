# Balanced K-Means clustering in PyTorch

Balanced K-Means clustering in Pytorch with strong GPU acceleration.

**Disclaimer:** This project is heavily inspired by the project [kmeans_pytorch](https://github.com/subhadarship/kmeans_pytorch). Each part of the original implementation is combined with the appropriate attribution.

# Installation
As easy as:

`pip install balanced_kmeans`


# Getting started

First things first: Classical kmeans algorithm as easy as
```
from balanced_kmeans import kmeans
# experiment constants
N = 10000
batch_size = 10
num_clusters = 100
device = 'cuda'

cluster_size = N // num_clusters
X = torch.rand(batch_size, N, dim, device=device)
choices, centers = kmeans(X, num_clusters=num_clusters)
```

Now, if you want balanced kmeans you can run:

```
from balanced_kmeans import kmeans_equal
N = 10000
batch_size = 10
num_clusters = 100
device = 'cuda'

cluster_size = N // num_clusters
X = torch.rand(batch_size, N, dim, device=device)
choices, centers = kmeans_equal(X, num_clusters=num_clusters)
```

By default, forge initialization scheme is used for initial cluster centers.
However, you may change the initial cluster centers by providing the keyword
argument `initial_state` to either `kmeans` or `kmeans_equal`.

# Contributing
This is a pet project, so feel free to contribute if you want to add any extra
feature. For any bugs, please open a detailed issue.

# Credits
This implementation extends the package `kmeans_pytorch` which contains the
implementation of the original Lloyd's K-means algorithm in Pytorch. You can check (and star!)
the original package [here](https://github.com/subhadarship/kmeans_pytorch).


For licensing of this project, please refer to this repo as well as the `kmeans_pytorch` repo.
