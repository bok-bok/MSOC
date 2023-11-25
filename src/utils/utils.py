import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def plot_pca(features, labels, epoch=None):
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

    plt.savefig(f"plots/pca_{epoch}.png", dpi=300)
