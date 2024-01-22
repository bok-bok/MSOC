import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def plot_pca(features, labels, dataset_type, epoch=None):
    if dataset_type not in ["train", "test"]:
        raise ValueError("type must be either train or test")

    plt.figure(figsize=(10, 10))
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(features)
    # Identify unique classes and their colors
    classes = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))

    # Plot each class separately
    for cl, color in zip(classes, colors):
        indices = labels == cl
        plt.scatter(
            pca_features[indices, 0], pca_features[indices, 1], c=[color], label=f"Class {cl}"
        )

    plt.legend()

    save_dir = f"plots/{dataset_type}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(f"{save_dir}/{epoch}.png", dpi=300)
    plt.close()
