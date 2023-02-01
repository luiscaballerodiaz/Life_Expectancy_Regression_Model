import pandas as pd
import utils
from data_visualization import DataPlot
from pca_analysis import PCAanalysis


pd.set_option('display.max_columns', None)
column_target = 'Life expectancy '
visualization = DataPlot()
pca = PCAanalysis()

sourcedf = pd.read_csv('Life Expectancy Data.csv')
print("Full source data from CSV type: {} and shape: {}".format(type(sourcedf), sourcedf.shape))

# DATA SCRUBBING
visualization.histogram(dataset=sourcedf.iloc[:, 1:], plot_name='Original histogram', ncolumns=7)
visualization.target_vs_feature(dataset=sourcedf.iloc[:, 1:], target=column_target,
                                plot_name='Original target vs feature correlation', ncolumns=7)
column_target = 'Life expectancy '
feat_cat = ['Country', 'Status', 'Year']
sourcedf = utils.data_na_removal(sourcedf, column_target, feat_cat)
sourcedf = utils.data_outliers_removal(sourcedf, ['Population', '>', 1e9],
                                                 ['Measles ', '>', 100000],
                                                 ['GDP', '>', 100000],
                                                 ['under-five deaths ', '>', 1000],
                                                 [' thinness 5-9 years', '>', 25],
                                                 ['percentage expenditure', '>', 13000],
                                                 ['Schooling', '<', 0.1],
                                                 ['Income composition of resources', '<', 0.1])
print("Postprocessed data from CSV type: {} and shape: {}".format(type(sourcedf), sourcedf.shape))
visualization.histogram(dataset=sourcedf.iloc[:, 1:], plot_name='Postprocessed histogram', ncolumns=7)
visualization.target_vs_feature(dataset=sourcedf.iloc[:, 1:], target=column_target,
                                plot_name='Postprocessed target vs feature correlation', ncolumns=7)
visualization.correlation_plot(dataset=sourcedf.iloc[:, len(feat_cat):], target=column_target)
sourcedf, target, list_features = utils.data_one_hot_encoding(sourcedf, column_target, feat_cat)
print('Definitive feature shape: \n{}\n'.format(list_features.shape))

# DATA SPLIT AND SCALING
X_train, X_train_std, X_train_norm, y_train, X_test, X_test_std, X_test_norm, y_test = utils.data_split_scale(
    sourcedf, target, test_size=0.25)

# APPLY PCA TO ASSESS REGRESSION
train_pca = pca.apply_pca(X_train_std, 1)
utils.regression_analysis(visualization, train_pca, y_train,
                          neighbors=[1, 10, 100],
                          alpha_ridge=[500, 2500, 5000],
                          alpha_lasso=[1, 10, 100],
                          max_depth=[10, 50],
                          gamma=0.1, C=[0.01, 10, 1000],
                          activation='tanh', alpha_mlp=0.01, layers=[50, [100, 100], [1000, 1000]])

# APPLY OPTIMUM SETTINGS PER ALGORITHM
utils.optimum_tuning_analysis(visualization, X_train_std, X_train_norm, y_train, X_test_std, X_test_norm, y_test,
                              n_neighbors=3,
                              alpha_ridge=0.1,
                              alpha_lasso=0.001,
                              max_depth_tree=20,
                              n_estimators_random=1000, max_features=100, max_depth_random=24,
                              n_estimators_gradient=1000, learning_rate=0.1, max_depth_gradient=6,
                              gamma=0.00075, C=100,
                              activation='tanh', alpha_mlp=0.001, hidden_layer_sizes=[1000, 1000])

# GRID SEARCH AND MODEL OPTIMIZATION
scoring = 'neg_mean_absolute_error'
params = [
    {'preprocess1': [''], 'preprocess2': ['', 'std', 'norm'], 'estimator': ['knn reg'],
     'estimator__n_neighbors': [1, 3, 5, 8, 12, 20]},
    {'preprocess1': ['poly'], 'preprocess2': ['', 'std', 'norm'], 'estimator': ['linear reg'],
     'preprocess1__degree': [(1, 1), (1, 2)]},
    {'preprocess1': [''], 'preprocess2': ['', 'std', 'norm'], 'estimator': ['ridge'],
     'estimator__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]},
    {'preprocess1': [''], 'preprocess2': ['', 'std', 'norm'], 'estimator': ['lasso'],
     'estimator__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]},
    {'preprocess1': [''], 'preprocess2': [''], 'estimator': ['tree reg'],
     'estimator__max_depth': [5, 10, 15, 20, 26, 32, 40, 50, 100, 200]},
    {'preprocess1': [''], 'preprocess2': [''], 'estimator': ['random forest reg'],
     'estimator__n_estimators': [250, 500, 1000, 1500, 2000, 2500], 'estimator__max_depth': [16, 20, 24, 28],
     'estimator__max_features': [20, 40, 70, 100, 130, 160, 190, 220]},
    {'preprocess1': [''], 'preprocess2': [''], 'estimator': ['gradient boosting reg'],
     'estimator__n_estimators': [250, 500, 1000, 1500, 2000, 2500], 'estimator__max_depth': [4, 6, 8, 10],
     'estimator__learning_rate': [0.025, 0.05, 0.08, 0.1, 0.3, 0.5, 0.8]},
    {'preprocess1': [''], 'preprocess2': ['', 'std', 'norm'], 'estimator': ['svr'],
     'estimator__gamma': [0.0005, 0.00075, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.5],
     'estimator__C': [10, 20, 30, 50, 75, 100, 125, 150]},
    {'preprocess1': [''], 'preprocess2': ['std', 'norm'], 'estimator': ['mlp reg'],
     'estimator__alpha': [0.001, 0.01, 0.1, 1, 10],
     'estimator__activation': ['tanh', 'relu'],
     'estimator__hidden_layer_sizes': [250, 500, 1000, [250, 250], [500, 500], [1000, 1000]]}]

grid = utils.cross_grid_validation(params, X_train, y_train, X_test, y_test, scoring, 5)
pd_grid = pd.DataFrame(grid.cv_results_)
print(pd_grid)
utils.param_sweep_matrix(visualization, params=pd_grid['params'], test_score=pd_grid['mean_test_score'])
