import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
import numpy as np


class DataPlot:
    """Class to plot data using visualization tools"""

    def __init__(self):
        self.fig_width = 20
        self.fig_height = 10
        self.bar_width = 0.25
        self.subplot_row = 2
        self.nfeat = 2

    def histogram(self, dataset, plot_name, ncolumns):
        """Plot histogram based on input dataset"""
        fig, axes = plt.subplots(math.ceil(dataset.shape[1] / ncolumns), ncolumns,
                                 figsize=(self.fig_width, self.fig_height))
        spare_axes = ncolumns - dataset.shape[1] % ncolumns
        if spare_axes == ncolumns:
            spare_axes = 0
        for axis in range(ncolumns - 1,  ncolumns - 1 - spare_axes, -1):
            fig.delaxes(axes[math.ceil(dataset.shape[1] / ncolumns) - 1, axis])
        ax = axes.ravel()
        for i in range(dataset.shape[1]):
            ax[i].hist(dataset.iloc[:, i], histtype='stepfilled', bins=25, alpha=0.25, color="#0000FF", lw=0)
            ax[i].set_title(dataset.keys()[i], fontsize=10, y=1.0, pad=-14, fontweight='bold')
            ax[i].grid(visible=True)
            ax[i].tick_params(axis='both', labelsize=8)
            ax[i].set_ylabel('Frequency', fontsize=8)
            ax[i].set_xlabel('Feature magnitude', fontsize=8)
        fig.suptitle('Histogram for life expectancy dataset features', fontsize=18, fontweight='bold')
        plt.subplots_adjust(top=0.85)
        fig.tight_layout()
        plt.savefig(plot_name + '.png', bbox_inches='tight')
        plt.clf()

    def target_vs_feature(self, dataset, target, plot_name, ncolumns):
        """Plot the target vs each feature"""
        fig, axes = plt.subplots(math.ceil(dataset.shape[1] / ncolumns), ncolumns,
                                 figsize=(self.fig_width, self.fig_height))
        spare_axes = ncolumns - dataset.shape[1] % ncolumns
        if spare_axes == ncolumns:
            spare_axes = 0
        for axis in range(ncolumns - 1,  ncolumns - 1 - spare_axes, -1):
            fig.delaxes(axes[math.ceil(dataset.shape[1] / ncolumns) - 1, axis])
        ax = axes.ravel()
        for i in range(dataset.shape[1]):
            ax[i].scatter(dataset[target], dataset.iloc[:, i],s=10, marker='o', c='blue')
            ax[i].grid(visible=True)
            ax[i].tick_params(axis='both', labelsize=8)
            ax[i].set_ylabel(dataset.keys()[i], fontsize=10)
            ax[i].set_xlabel(target, fontsize=10)
        fig.suptitle('Assessment between life expectancy and each feature', fontsize=24, fontweight='bold')
        plt.subplots_adjust(top=0.85)
        fig.tight_layout()
        plt.savefig(plot_name + '.png', bbox_inches='tight')
        plt.clf()

    def correlation_plot(self, dataset, target):
        """Plot the correlation matrix among features"""
        dataset = dataset.astype(float)
        # ALL FEATURES
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        plt.pcolormesh(dataset.corr(), cmap=plt.cm.cool)
        plt.colorbar()
        yrange = [x + 0.5 for x in range(dataset.corr().shape[0])]
        xrange = [x + 0.5 for x in range(dataset.corr().shape[1])]
        plt.xticks(xrange, dataset.keys(), rotation=75, ha='center')
        ax.xaxis.tick_top()
        plt.yticks(yrange, dataset.keys(), va='center')
        plt.xlabel("Features", weight='bold', fontsize=14)
        plt.ylabel("Features", weight='bold', fontsize=14)
        plt.title("Correlation matrix among all features", weight='bold', fontsize=24)
        fig.tight_layout()
        plt.savefig('Correlation matrix all features.png', bbox_inches='tight')
        plt.clf()
        # ONLY LIFE EXPECTANCY
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        dataset = dataset.astype(float)
        index = 0
        for i in range(len(dataset.keys())):
            if target.lower() == dataset.keys()[i].lower():
                index = i
                break
        corr_matrix = np.array(dataset.corr())
        plt.pcolormesh([corr_matrix[index, :]], cmap=plt.cm.cool)
        plt.colorbar()
        xrange = [x + 0.5 for x in range(dataset.corr().shape[1])]
        plt.xticks(xrange, dataset.keys(), rotation=75, ha='center')
        ax.xaxis.tick_top()
        plt.yticks([0.5], [target], va='center')
        plt.xlabel("Features", weight='bold', fontsize=14)
        plt.ylabel(target, weight='bold', fontsize=14)
        plt.title("Correlation matrix regarding target feature", weight='bold', fontsize=24)
        fig.tight_layout()
        plt.savefig('Correlation matrix target.png', bbox_inches='tight')
        plt.clf()

    def compare_regression_plot(self, ncolumns, algorithm, x, y):
        """Plot regression model vs real input for different algorithms"""
        nplots = len(algorithm)
        fig, axes = plt.subplots(math.ceil(nplots / ncolumns), ncolumns,
                                 figsize=(self.fig_width, self.fig_height))
        spare_axes = ncolumns - nplots % ncolumns
        if spare_axes == ncolumns:
            spare_axes = 0
        for axis in range(ncolumns - 1, ncolumns - 1 - spare_axes, -1):
            if (math.ceil(nplots / ncolumns) - 1) == 0:
                fig.delaxes(axes[axis])
            else:
                fig.delaxes(axes[math.ceil(nplots / ncolumns) - 1, axis])
        ax = axes.ravel()
        cmap = cm.get_cmap('tab10')
        colors = cmap.colors
        for i in range(nplots):
            ax[i].scatter(x, x, s=10, marker='o', c='black', label='Input data')
            ax[i].scatter(x, y[i], color=colors[i % len(colors)], s=10, marker='^', label=algorithm[i])
            ax[i].set_title(algorithm[i].upper() + ' model output vs input data', fontsize=18, fontweight='bold')
            ax[i].set_xlabel('Input data', fontsize=14, weight='bold')
            ax[i].set_ylabel('Model output', fontsize=14, weight='bold')
            ax[i].legend()
            ax[i].grid(visible=True)
        fig.suptitle('Regression comparison among different algorithms', fontsize=24, fontweight='bold')
        plt.subplots_adjust(top=0.85)
        fig.tight_layout()
        plt.savefig('Regression comparison.png', bbox_inches='tight')
        plt.clf()

    def plot_params_sweep(self, algorithm, test_values, fixed_params,
                          xtick='', ytick='', ztick='', xtag='', ytag='', ztag=''):
        """Plot parameter sweep for the cross validation grid search"""
        if len(ztick) == 0:
            fig, ax = plt.subplots(1, 1, figsize=(self.fig_width, self.fig_height))
            plt.pcolormesh(test_values, cmap=plt.cm.PuBuGn)
            plt.colorbar()
            ax.set_xlabel('Parameter sweep ' + xtag.upper(), fontsize=14)
            ax.set_ylabel('Parameter sweep ' + ytag.upper(), fontsize=14)
            ax.set_title('Test score sweep ' + ztag.upper() + ' with ' + algorithm.upper() +
                         '\n' + str(fixed_params), fontsize=24)
            ax.set_xticks(np.arange(0.5, len(xtick) + 0.5), labels=xtick, fontsize=14)
            plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
            ax.set_yticks(np.arange(0.5, len(ytick) + 0.5), labels=ytick, fontsize=14)
            ax.text(0.5, 0.5, str(round(test_values[0, 0], 4)),
                    ha="center", va="center", color="k", fontweight='bold', fontsize=12)
            for h in range(len(xtick)):
                ax.text(h + 0.5, 0.5, str(round(test_values[0, h], 4)),
                        ha="center", va="center", color="k", fontweight='bold', fontsize=12)
                for j in range(len(ytick)):
                    ax.text(h + 0.5, j + 0.5, str(round(test_values[j, h], 4)),
                            ha="center", va="center", color="k", fontweight='bold', fontsize=12)
        else:
            fig, axes = plt.subplots(round(math.ceil(len(ztick)) / self.subplot_row), self.subplot_row,
                                     figsize=(self.fig_width, self.fig_height))
            spare_axes = self.subplot_row - len(ztick) % self.subplot_row
            if spare_axes == self.subplot_row:
                spare_axes = 0
            for axis in range(self.subplot_row - 1, self.subplot_row - 1 - spare_axes, -1):
                if (math.ceil(len(ztick) / self.subplot_row) - 1) == 0:
                    fig.delaxes(axes[axis])
                else:
                    fig.delaxes(axes[math.ceil(len(ztick) / self.subplot_row) - 1, axis])
            ax = axes.ravel()
            for p in range(len(ax)):
                pcm = ax[p].pcolormesh(test_values[:, :, p], cmap=plt.cm.PuBuGn)
                divider = make_axes_locatable(ax[p])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(pcm, cax=cax, orientation='vertical')
                ax[p].set_xlabel('Parameter sweep ' + xtag.upper(), fontsize=16)
                ax[p].set_ylabel('Parameter sweep ' + ytag.upper(), fontsize=16)
                zstring = ''
                for m in range(len(ztag[p])):
                    zstring += r"$\bf{" + ztag[p][m].replace('_', ' ').upper() + '=' + str(ztick[p][m]) + '}$ '
                ax[p].set_title('Parameter ' + zstring, fontsize=18)
                ax[p].set_xticks(np.arange(0.5, len(xtick) + 0.5), labels=xtick, fontsize=14)
                plt.setp(ax[p].get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
                ax[p].set_yticks(np.arange(0.5, len(ytick) + 0.5), labels=ytick, fontsize=14)
                for h in range(len(xtick)):
                    for j in range(len(ytick)):
                        ax[p].text(h + 0.5, j + 0.5, str(round(test_values[j, h, p], 4)),
                                   ha="center", va="center", color="k", fontweight='bold', fontsize=12)
            fig.suptitle('Test score parameter sweep with ' + algorithm.upper() +
                         '\n' + str(fixed_params), fontsize=24)
            plt.subplots_adjust(top=0.85)
        fig.tight_layout()
        plt.savefig('Parameter sweep ' + algorithm.upper() + ' algorithm.png', bbox_inches='tight')
        plt.clf()
