import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import numpy as np
import math


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
        plt.close()

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
            ax[i].scatter(dataset[target], dataset.iloc[:, i], s=10, marker='o', c='blue')
            ax[i].grid(visible=True)
            ax[i].tick_params(axis='both', labelsize=8)
            ax[i].set_ylabel(dataset.keys()[i], fontsize=10)
            ax[i].set_xlabel(target, fontsize=10)
        fig.suptitle('Assessment between life expectancy and each feature', fontsize=24, fontweight='bold')
        plt.subplots_adjust(top=0.85)
        fig.tight_layout()
        plt.savefig(plot_name + '.png', bbox_inches='tight')
        plt.close()

    def correlation_plot(self, dataset, target):
        """Plot the correlation matrix among features"""
        dataset = dataset.astype(float)
        corr_matrix = np.array(dataset.corr())
        # ALL FEATURES
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        plt.pcolormesh(corr_matrix, cmap=plt.cm.cool)
        plt.colorbar()
        yrange = [x + 0.5 for x in range(corr_matrix.shape[0])]
        xrange = [x + 0.5 for x in range(corr_matrix.shape[1])]
        plt.xticks(xrange, dataset.keys(), rotation=75, ha='center')
        ax.xaxis.tick_top()
        plt.yticks(yrange, dataset.keys(), va='center')
        for i in range(len(xrange)):
            for j in range(len(yrange)):
                ax.text(xrange[i], yrange[j], str(round(corr_matrix[j, i], 1)),
                        ha="center", va="center", color="k", fontweight='bold', fontsize=12)
        plt.xlabel("Features", weight='bold', fontsize=14)
        plt.ylabel("Features", weight='bold', fontsize=14)
        plt.title("Correlation matrix among all features", weight='bold', fontsize=24)
        fig.tight_layout()
        plt.savefig('Correlation matrix all features.png', bbox_inches='tight')
        plt.close()
        # ONLY LIFE EXPECTANCY
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        index = 0
        for i in range(len(dataset.keys())):
            if target.lower() == dataset.keys()[i].lower():
                index = i
                break
        plt.pcolormesh([corr_matrix[index, :]], cmap=plt.cm.cool)
        plt.colorbar()
        xrange = [x + 0.5 for x in range(dataset.corr().shape[1])]
        plt.xticks(xrange, dataset.keys(), rotation=75, ha='center')
        ax.xaxis.tick_top()
        plt.yticks([0.5], [target], va='center')
        for i in range(len(xrange)):
            ax.text(xrange[i], 0.5, str(round(corr_matrix[index, i], 1)),
                    ha="center", va="center", color="k", fontweight='bold', fontsize=12)
        plt.xlabel("Features", weight='bold', fontsize=14)
        plt.ylabel(target, weight='bold', fontsize=14)
        plt.title("Correlation matrix regarding target feature", weight='bold', fontsize=24)
        fig.tight_layout()
        plt.savefig('Correlation matrix target.png', bbox_inches='tight')
        plt.close()

    def compare_regression_plot(self, ncolumns, algorithm, y_true, y_pred, tag):
        """Plot regression model vs real input for different algorithms"""
        nplots = len(algorithm)
        # Regression comparison y_test vs y_pred
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
            ax[i].scatter(y_true, y_true, s=10, marker='o', c='black', label='Input data')
            ax[i].scatter(y_true, y_pred[i], color=colors[i % len(colors)], s=10, marker='^', label=algorithm[i])
            ax[i].set_title(algorithm[i].upper() + ' model output vs input data', fontsize=18, fontweight='bold')
            ax[i].set_xlabel('Input data', fontsize=14, weight='bold')
            ax[i].set_ylabel('Model output', fontsize=14, weight='bold')
            ax[i].legend()
            ax[i].grid(visible=True)
        fig.suptitle('Regression comparison among different algorithms (' + tag + ')', fontsize=24, fontweight='bold')
        plt.subplots_adjust(top=0.85)
        fig.tight_layout()
        plt.savefig('Regression comparison ' + tag + '.png', bbox_inches='tight')
        plt.close()
        # Assess the worst deviation cases
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        nworst = 5
        matrix = np.zeros([nworst, nplots + 1])
        for i in range(nworst):
            matrix[i, 0] = i + 1
            for j in range(nplots):
                dev = abs(y_pred[j] - y_true)
                index = np.argsort(dev)[-(i + 1)]
                matrix[i, j + 1] = dev[index]
        label = []
        for i in range(nplots):
            label.append(algorithm[i].upper() + ' (mean: ' + str(np.round(np.mean(abs(y_pred[i] - y_true)), 2)) + ')')
        df = pd.DataFrame(matrix, columns=['X'] + label)
        df.plot(x='X', y=label, kind="bar", rot=0, ax=ax)
        plt.title('Worst test cases deviation assessment (' + tag + ')', fontsize=24)
        plt.xlabel('Worst test cases', fontweight='bold', fontsize=14)
        plt.ylabel('Max deviation', fontweight='bold', fontsize=14)
        plt.legend()
        plt.grid()
        plt.savefig('Regression max deviation ' + tag + '.png', bbox_inches='tight')
        plt.close()

    def compare_ensembled_models(self, tag, y_true, y_pred, weights_ini, labels_ini, metric=''):
        """Plot regression model vs real input for different algorithms"""
        # Regression comparison y_test vs y_pred
        if metric == '':
            metric = ['']
            fig, axes = plt.subplots(1, 2, figsize=(self.fig_width, self.fig_height))
        else:
            fig, axes = plt.subplots(2, len(metric), figsize=(self.fig_width, self.fig_height))
        cmap = cm.get_cmap('tab10')
        colors = cmap.colors
        ax = axes.ravel()
        for i in range(len(metric) * 2):
            if i < len(metric):
                ax[i].scatter(y_true, y_true, s=10, marker='o', c='black', label='Input data')
                if len(metric) == 1:
                    ax[i].scatter(y_true, y_pred, color=colors[i % len(colors)], s=10, marker='^', label='Prediction')
                    ax[i].set_title('Ensembled model', fontsize=20, fontweight='bold')
                else:
                    ax[i].scatter(y_true, y_pred[i], color=colors[i % len(colors)], s=10, marker='^', label=metric[i])
                    ax[i].set_title('Ensembled model optimizing ' + metric[i].upper(), fontsize=20, fontweight='bold')
                ax[i].set_xlabel('Input data', fontsize=14, weight='bold')
                ax[i].set_ylabel('Model output', fontsize=14, weight='bold')
                ax[i].legend()
                ax[i].grid(visible=True)
            else:
                # Rearrange data to avoid plot overlapping
                j = i - len(metric)
                index_to_remove = []
                labels = labels_ini.copy()
                if len(metric) == 1:
                    weights = weights_ini.copy()
                    for h in range(len(labels_ini)):
                        if weights_ini[h] < 0.00001:
                            index_to_remove.append(h)
                else:
                    weights = weights_ini[j].copy()
                    for h in range(len(labels_ini)):
                        if weights_ini[j][h] < 0.00001:
                            index_to_remove.append(h)
                index_to_remove.reverse()
                weights = weights.tolist()
                for h in range(len(index_to_remove)):
                    weights.pop(index_to_remove[h])
                    labels.pop(index_to_remove[h])
                explode = [0.1] * len(weights)
                ax[i].pie(x=weights, explode=explode, labels=labels, autopct='%1.1f%%',
                          shadow=True, textprops={'fontsize': 16})
                if len(metric) == 1:
                    ax[i].set_title('Ensembled model', fontsize=20, fontweight='bold')
                else:
                    ax[i].set_title('Ensembled model optimizing ' + metric[j].upper(), fontsize=20, fontweight='bold')
                ax[i].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=3)
        fig.suptitle('Regression comparison ensembled models for ' + tag.upper(), fontsize=24, fontweight='bold')
        plt.subplots_adjust(top=0.85)
        fig.tight_layout()
        plt.savefig('Regression comparison ensembled models ' + tag + '.png', bbox_inches='tight')
        plt.close()

    # @staticmethod
    # def make_autopct(values):
    #     def my_autopct(pct):
    #         val = int(round(pct * sum(values) / 100.0))
    #         return '{}\n{:.1f}%'.format(val, pct)
    #
    #     return my_autopct

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
            for p in range(len(ztick)):
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
        plt.close()

    def plot_regression(self, name, x_real, y_real, x_model, y_model, algorithm):
        """Plot regression model vs real target value"""
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        cmap = cm.get_cmap('tab10')
        colors = cmap.colors
        ax.scatter(x_real, y_real, s=10, marker='o', c='black', label='Input data')
        for i in range(len(y_model)):
            ax.plot(x_model, y_model[i], linewidth=2, color=colors[i % len(colors)], label=algorithm[i])
        plt.title('Regression ' + name.upper() + ' model performance vs real output data', fontsize=24)
        plt.xlabel('Input data', fontweight='bold', fontsize=14)
        plt.ylabel('Output data', fontweight='bold', fontsize=14)
        plt.legend()
        plt.grid()
        plt.savefig('Regression ' + name.upper() + ' analysis.png', bbox_inches='tight')
        plt.close()
