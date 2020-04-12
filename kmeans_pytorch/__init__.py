import numpy as np
import torch
from tqdm import tqdm

def initialize(X, num_clusters):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = len(X)
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state


def kmeans(
        X,
        num_clusters,
        tol=1e-4):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    print(f'running k-means on {X.device}..')


    pairwise_distance_function = pairwise_distance

    # Step1: randomly group vectors to clusters
    initial_state = initialize(X, num_clusters)

    iteration = 0
    tqdm_meter = tqdm(desc='[running kmeans]')
    while True:
        dis = pairwise_distance_function(X, initial_state)
        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

            selected = torch.index_select(X, 0, selected)

            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increment iteration
        iteration = iteration + 1

        # update tqdm meter
        tqdm_meter.set_postfix(
            iteration=f'{iteration}',
            center_shift=f'{center_shift ** 2:0.6f}',
            tol=f'{tol:0.6f}'
        )
        tqdm_meter.update()
        if center_shift ** 2 < tol:
            break
    return choice_cluster, initial_state


def batch(fn):
    def do(X, *args, **kwargs):
        batch_size = X.shape[0]

        choices, centers = fn(X[0], *args, **kwargs)
        choices = choices.unsqueeze(0)
        centers = centers.unsqueeze(0)

        for i in range(1, batch_size):
            curr_choices, curr_centers = fn(X[i], *args, **kwargs)

            # concat in the batch dimension
            curr_choices = curr_choices.unsqueeze(0)
            curr_centers = curr_centers.unsqueeze(0)
            choices = torch.cat((choices, curr_choices), dim=0)
            centers = torch.cat((centers, curr_centers), dim=0)

        return choices, centers
    return do


@batch
def kmeans_equal(
        X,
        num_clusters,
        cluster_size,
        max_iters=100,
        initial_state=None,
        progress=False,
        tol=1e-4):
    """
    perform kmeans on equally sized clusters
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """

    print(f'running k-means for equal size clusters on {X.device}..')
    pairwise_distance_function = pairwise_distance

    if initial_state is None:
        # randomly group vectors to clusters (forgy initialization)
        initial_state = initialize(X, num_clusters)

    iteration = 0
    if progress:
        tqdm_meter = tqdm(desc='[running kmeans on equal size clusters]')

    while True:
        dis = pairwise_distance_function(X, initial_state)
        choices = torch.argsort(dis, dim=-1)
        initial_state_pre = initial_state.clone()
        for index in range(num_clusters):
            X_index, positions = torch.where(choices == index)
            selected_ind = X_index[torch.argsort(positions)[:cluster_size]]
            # mask out selected
            choices[selected_ind] = index

            # update cluster center
            selected = torch.index_select(X, 0, selected_ind)
            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increment iteration
        iteration = iteration + 1

        # update tqdm meter
        if progress:
            tqdm_meter.set_postfix(
                iteration=f'{iteration}',
                center_shift=f'{center_shift ** 2:0.6f}',
                tol=f'{tol:0.6f}'
            )
            tqdm_meter.update()

        if center_shift ** 2 < tol:
            break
        if iteration > max_iters:
            break

    return choices[:, 0], initial_state



def kmeans_predict(
        X,
        cluster_centers,
        distance='euclidean',
        device=torch.device('cpu')
):
    """
    predict using cluster centers
    :param X: (torch.tensor) matrix
    :param cluster_centers: (torch.tensor) cluster centers
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param device: (torch.device) device [default: 'cpu']
    :return: (torch.tensor) cluster ids
    """
    print(f'predicting on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    dis = pairwise_distance_function(X, cluster_centers)
    choice_cluster = torch.argmin(dis, dim=1)

    return choice_cluster.cpu()


def pairwise_distance(data1, data2):

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis
