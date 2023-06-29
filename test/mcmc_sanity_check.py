import torch
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, TSNE, Isomap

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

n_data = 50
path = ".//weights"
weight_path = sorted(glob(os.path.join(path, "model_11", "checkpoint_*")))


def print_model():
    weight1 = weight_path[0]
    for key in weight1.keys():
        print(key)


def vec_sample(weight_paths):
    weight = torch.load(weight_paths, map_location=device)
    vec = []

    for key in weight.keys():
        if 'weight' in key and 'conv' in key:
            w_np_flt = weight[key].cpu().numpy().flatten()
            vec.append(w_np_flt)
    out_vec = np.concatenate(vec).flatten()
    print(out_vec.shape)
    return out_vec


def vec_all():
    total_vec = []

    for i in range(len(weight_path)):
        print(f'checkpoints from epoch: {weight_path[i]} are vectorized!')

        sample_vec = vec_sample(weight_path[i])
        total_vec.append(sample_vec)
    total_vec = np.array(total_vec)
    # np.save(os.path.join(path, "total_vec.npy"), total_vec)
    print("shape of total weight", total_vec.shape)

    mds = MDS(n_components=2, metric=True, random_state=42)
    weight_reduced_mds = mds.fit_transform(total_vec)
    print(weight_reduced_mds.shape)

    plt.scatter(weight_reduced_mds[:, 0], weight_reduced_mds[:, 1])
    plt.title('MDS')
    plt.xlabel("$z_1$", fontsize=18)
    plt.ylabel("$z_2$", fontsize=18)
    plt.title('checkpoint_1-{}'.format(n_data))
    plt.savefig(os.path.join(path,  "mcmc11_sanity_checkpoint_1-{}.jpg".format(n_data)))
    plt.show()

    return total_vec


def plot_sanity():
    vec_weights = np.load(os.path.join(path, "total_vec.npy"))
    print(vec_weights.shape)
    mds = MDS(n_components=2, metric=True, random_state=42)
    weight_reduced_mds = mds.fit_transform(vec_weights)
    print(weight_reduced_mds.shape)

    plt.scatter(weight_reduced_mds[:, 0], weight_reduced_mds[:, 1])
    plt.title('MDS')
    plt.xlabel("$z_1$", fontsize=18)
    plt.ylabel("$z_2$", fontsize=18)
    plt.savefig(os.path.join(path, "mcmc_sanity_check.jpg"))
    plt.show()


if __name__ == '__main__':
    vec_all()
    # plot_sanity()
