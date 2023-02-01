import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


class PCAanalysis:
    """Class to visualize the input data split per binary output class"""

    def __init__(self):
        self.fig_width = 20
        self.fig_height = 10
        self.pca = None
        self.list_features = None

    def apply_pca(self, dataset, ncomps, list_features=None, target=None):
        """Apply PCA algorithm in the scaled trained data and plot meaningful graphs"""
        self.list_features = list_features
        self.pca = PCA(n_components=ncomps)
        self.pca.fit(dataset)
        dataset_pca = self.pca.transform(dataset)
        print("X_train_scaled PCA type: {} and shape: {}".format(type(dataset_pca),
                                                                 dataset_pca.shape))
        if ncomps >= 2:
            if target is None:
                self.plot_first_second_pca(dataset_pca)
            else:
                dataset_pca_output1 = dataset_pca[target == 1, :]
                dataset_pca_output0 = dataset_pca[target == 0, :]
                print("X_train_scaled PCA output = 1 type: {} and shape: {}"
                      .format(type(dataset_pca_output1), dataset_pca_output1.shape))
                print("X_train_scaled PCA output = 0 type: {} and shape: {}"
                      .format(type(dataset_pca_output0), dataset_pca_output0.shape))
                self.plot_first_second_pca(dataset_pca_output1, dataset_pca_output0)
        self.plot_pca_breakdown()
        self.plot_pca_scree()
        print("PCA component shape: {} \n".format(self.pca.components_.shape))
        return dataset_pca

    def plot_pca_breakdown(self):
        """Plot the PCA breakdown per each feature"""
        _, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        plt.pcolormesh(self.pca.components_, cmap=plt.cm.cool)
        plt.colorbar()
        pca_yrange = [x + 0.5 for x in range(self.pca.components_.shape[0])]
        pca_xrange = [x + 0.5 for x in range(self.pca.components_.shape[1])]
        if self.list_features is None:
            str_xpca = []
            for i in range(len(pca_xrange)):
                str_xpca.append('Feature ' + str(i + 1))
            plt.xticks(pca_xrange, str_xpca, rotation=60, ha='center')
        else:
            plt.xticks(pca_xrange, self.list_features, rotation=60, ha='center')
        ax.xaxis.tick_top()
        str_ypca = []
        for i in range(self.pca.components_.shape[0]):
            str_ypca.append('Component ' + str(i + 1))
        plt.yticks(pca_yrange, str_ypca)
        plt.xlabel("Feature", weight='bold', fontsize=14)
        plt.ylabel("Principal components", weight='bold', fontsize=14)
        plt.savefig('PCA scaled breakdown.png', bbox_inches='tight')
        plt.clf()

    def plot_pca_scree(self):
        """Plot the scree plot of the PCA to understand the covered variance"""
        fig, ax1 = plt.subplots(figsize=(self.fig_width, self.fig_height))
        ax2 = ax1.twinx()
        label1 = ax1.plot(range(1, len(self.pca.components_) + 1), self.pca.explained_variance_ratio_,
                          'ro-', linewidth=2, label='Individual PCA variance')
        label2 = ax2.plot(range(1, len(self.pca.components_) + 1), np.cumsum(self.pca.explained_variance_ratio_),
                          'b^-', linewidth=2, label='Cumulative PCA variance')
        plt.title('Scree Plot', fontsize=20, fontweight='bold')
        ax1.set_xlabel('Principal Components', fontsize=14)
        ax1.set_ylabel('Proportion of Variance Explained', fontsize=14, color='r')
        ax2.set_ylabel('Cumulative Proportion of Variance Explained', fontsize=14, color='b')
        la = label1 + label2
        lb = [la[0].get_label(), la[1].get_label()]
        ax1.legend(la, lb, loc='upper center')
        ax1.grid(visible=True)
        ax2.grid(visible=True)
        plt.savefig('PCA scaled scree plot.png', bbox_inches='tight')
        plt.clf()

    def plot_first_second_pca(self, dataset_pca_output1, dataset_pca_output0=None):
        """Plot first vs second PCA component"""
        plt.subplots(figsize=(self.fig_width, self.fig_height))
        if dataset_pca_output0 is None:
            plt.scatter(dataset_pca_output1[:, 0], dataset_pca_output1[:, 1],
                        s=10, marker='^', c='red', label='output')
        else:
            plt.scatter(dataset_pca_output1[:, 0], dataset_pca_output1[:, 1],
                        s=10, marker='^', c='red', label='output=1')
            plt.scatter(dataset_pca_output0[:, 0], dataset_pca_output0[:, 1],
                        s=10, marker='o', c='blue', label='output=0')
        plt.title('Plot for first PCA component vs second PCA component', fontsize=20, fontweight='bold')
        plt.xlabel('First PCA', fontsize=14)
        plt.ylabel('Second PCA', fontsize=14)
        plt.legend()
        plt.grid()
        plt.savefig('PCA scaled first vs second.png', bbox_inches='tight')
        plt.clf()
