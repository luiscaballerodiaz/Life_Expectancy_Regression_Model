import pandas as pd
import numpy as np
import utils
from data_visualization import DataPlot
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures


pd.set_option('display.max_columns', None)
column_target = 'Life expectancy '
visualization = DataPlot()
sourcedf = pd.read_csv('Life Expectancy Data.csv')
print("Full source data from CSV type: {} and shape: {}".format(type(sourcedf), sourcedf.shape))
visualization.histogram(dataset=sourcedf.iloc[:, 1:], plot_name='Original histogram', ncolumns=7)
visualization.target_vs_feature(dataset=sourcedf.iloc[:, 1:], target=column_target,
                                plot_name='Original target vs feature correlation', ncolumns=7)
sourcedf_na = sourcedf.isna()
print('Original null values: \n{}\n'.format(sourcedf_na.sum()))
for i in range(sourcedf.shape[0]):
    if sourcedf_na.iloc[i, sourcedf.columns.get_loc(column_target)]:
        sourcedf.drop(index=i, inplace=True)
imputer_num = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
feat_cat = ['Country', 'Status']
feat_num = [x for x in sourcedf.columns.values.tolist() if x not in feat_cat]
preprocessor = ColumnTransformer(transformers=[('imputer_cat', imputer_cat, feat_cat),
                                               ('imputer_num', imputer_num, feat_num)])
sourcedf = preprocessor.fit_transform(sourcedf)
sourcedf = pd.DataFrame(sourcedf, columns=feat_cat+feat_num)
print("Postprocessed data from CSV type: {} and shape: {}".format(type(sourcedf), sourcedf.shape))
print('Postprocessed null values: \n{}\n'.format(sourcedf.isna().sum()))
visualization.correlation_plot(dataset=sourcedf.iloc[:, len(feat_cat):], target=column_target)
visualization.histogram(dataset=sourcedf.iloc[:, 1:], plot_name='Postprocessed histogram', ncolumns=7)
visualization.target_vs_feature(dataset=sourcedf.iloc[:, 1:], target=column_target,
                                plot_name='Postprocessed target vs feature correlation', ncolumns=7)
target = sourcedf[column_target]
sourcedf.drop(column_target, axis=1, inplace=True)
sourcedf['Country'] = sourcedf['Country'].astype(str)
sourcedf['Status'] = sourcedf['Status'].astype(str)
sourcedf = pd.get_dummies(sourcedf, columns=['Country', 'Status'])
list_features = sourcedf.keys()
print('Definitive features list: \n{}\n'.format(list_features))
X_train, X_test, y_train, y_test = train_test_split(sourcedf, target, test_size=0.2,
                                                    shuffle=True, random_state=0)
print('\n')
print('X_train shape: {}'.format(X_train.shape))
print('y_train shape: {}'.format(y_train.shape))
print('X_test shape: {}'.format(X_test.shape))
print('y_test shape: {}'.format(y_test.shape))
print('\n')

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled[:, :19])
X_test_poly = poly.transform(X_test_scaled[:, :19])

reg = KNeighborsRegressor(n_neighbors=5)
reg.fit(X_train_scaled, y_train)
y_pred = reg.predict(X_test_scaled)
error = 0
for i in range(len(y_test)):
    error += abs(y_test[i] - y_pred[i])
error /= len(y_test)
print('KNeighbors Regressor: {}'.format(error))
y_knn = y_pred

reg = LinearRegression()
reg.fit(X_train_poly, y_train)
y_pred = reg.predict(X_test_poly)
error = 0
for i in range(len(y_test)):
    error += abs(y_test[i] - y_pred[i])
    if abs(y_test[i] - y_pred[i]) > 2000:
        print(i)
        print(X_test[i])
error /= len(y_test)
print('PolynomialRegression: {}'.format(error))
y_poly = y_pred

reg = LinearRegression()
reg.fit(X_train_scaled, y_train)
y_pred = reg.predict(X_test_scaled)
error = 0
for i in range(len(y_test)):
    error += abs(y_test[i] - y_pred[i])
error /= len(y_test)
print('LinearRegression: {}'.format(error))
y_linear = y_pred

reg = Ridge(alpha=10)
reg.fit(X_train_scaled, y_train)
y_pred = reg.predict(X_test_scaled)
error = 0
for i in range(len(y_test)):
    error += abs(y_test[i] - y_pred[i])
error /= len(y_test)
print('Ridge: {}'.format(error))
y_ridge = y_pred

reg = Lasso(alpha=1)
reg.fit(X_train_scaled, y_train)
y_pred = reg.predict(X_test_scaled)
error = 0
for i in range(len(y_test)):
    error += abs(y_test[i] - y_pred[i])
error /= len(y_test)
print('Lasso: {}'.format(error))
y_lasso = y_pred

reg = DecisionTreeRegressor(max_depth=14)
reg.fit(X_train_scaled, y_train)
y_pred = reg.predict(X_test_scaled)
error = 0
for i in range(len(y_test)):
    error += abs(y_test[i] - y_pred[i])
error /= len(y_test)
print('Decision Tree Regressor: {}'.format(error))
y_tree = y_pred

reg = RandomForestRegressor(n_estimators=200, max_features=8, max_depth=10)
reg.fit(X_train_scaled, y_train)
y_pred = reg.predict(X_test_scaled)
error = 0
for i in range(len(y_test)):
    error += abs(y_test[i] - y_pred[i])
error /= len(y_test)
print('Random Forest Regressor: {}'.format(error))
y_forest = y_pred

reg = GradientBoostingRegressor(n_estimators=50, learning_rate=0.5, max_depth=5)
reg.fit(X_train_scaled, y_train)
y_pred = reg.predict(X_test_scaled)
error = 0
for i in range(len(y_test)):
    error += abs(y_test[i] - y_pred[i])
error /= len(y_test)
print('Gradient Boosting Regressor: {}'.format(error))
y_gradient = y_pred

reg = SVR(gamma=1, C=1)
reg.fit(X_train_scaled, y_train)
y_pred = reg.predict(X_test_scaled)
error = 0
for i in range(len(y_test)):
    error += abs(y_test[i] - y_pred[i])
error /= len(y_test)
print('SVR: {}'.format(error))
y_svr = y_pred

reg = MLPRegressor(alpha=0.1, hidden_layer_sizes=300)
reg.fit(X_train_scaled, y_train)
y_pred = reg.predict(X_test_scaled)
error = 0
for i in range(len(y_test)):
    error += abs(y_test[i] - y_pred[i])
error /= len(y_test)
print('Multilayer Perceptron Regressor: {}'.format(error))
y_mlp = y_pred

visualization.compare_regression_plot(ncolumns=2, algorithm=['knn', 'polynomial', 'linear', 'ridge', 'lasso', 'tree',
                                                             'random forest', 'gradient boosting', 'svr', 'mlp'],
                                      x=y_test, y=[y_knn, y_poly, y_linear, y_ridge, y_lasso, y_tree, y_forest,
                                                   y_gradient, y_svr, y_mlp])

# Grid search and model optimization
scoring = 'neg_mean_absolute_error'
params = [
    {'preprocess1': [''], 'preprocess2': ['', 'std', 'norm'], 'estimator': ['knn reg'],
     'estimator__n_neighbors': [3, 25, 50, 100]},
    {'preprocess1': [''], 'preprocess2': ['', 'std', 'norm'], 'estimator': ['linear reg']},
    # {'preprocess1': ['poly'], 'preprocess2': ['', 'std', 'norm'], 'estimator': ['linear reg'],
    #  'preprocess1__degree': [2, 3, 4]},
    {'preprocess1': [''], 'preprocess2': ['', 'std', 'norm'], 'estimator': ['ridge'],
     'estimator__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]},
    {'preprocess1': [''], 'preprocess2': ['', 'std', 'norm'], 'estimator': ['lasso'],
     'estimator__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]},
    {'preprocess1': [''], 'preprocess2': ['', 'std', 'norm'], 'estimator': ['tree reg'],
     'estimator__max_depth': [5, 15, 25, 50, 100, 150, 200]}]
    # {'preprocess1': [''], 'preprocess2': ['', 'std', 'norm'], 'estimator': ['random forest reg'],
    #  'estimator__n_estimators': [15, 25, 50, 100, 150, 200], 'estimator__max_depth': [5, 15, 25, 50, 100, 150, 200],
    #  'estimator__max_features': [5, 15, 25, 50, 100, 150, 200]},
    # {'preprocess1': [''], 'preprocess2': ['', 'std', 'norm'], 'estimator': ['gradient boosting reg'],
    #  'estimator__n_estimators': [15, 25, 50, 100, 150, 200], 'estimator__max_depth': [1, 3, 5, 10, 15],
    #  'estimator__learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2]},
    # {'preprocess1': [''], 'preprocess2': ['', 'std', 'norm'], 'estimator': ['svr'],
    #  'estimator__gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    #  'estimator__C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]},
    # {'preprocess1': [''], 'preprocess2': ['', 'std', 'norm'], 'estimator': ['mlp reg'],
    #  'estimator__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    #  'estimator__activation': ['tanh', 'relu', 'logistic'],
    #  'estimator__hidden_layers': [250, 500, 1000, [250, 250], [350, 350], [500, 500]]}]

grid = utils.cross_grid_validation(params, X_train, y_train, X_test, y_test, scoring, 5)
pd_grid = pd.DataFrame(grid.cv_results_)
print(pd_grid)
utils.param_sweep_matrix(visualization, params=pd_grid['params'], test_score=pd_grid['mean_test_score'])





