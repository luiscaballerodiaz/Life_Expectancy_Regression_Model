import time
import math
import numpy as np
import pandas as pd
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize


def cross_grid_validation(param_grid, X_train, y_train, X_test, y_test, scoring, nfolds=5):
    time0 = time.time()
    model = []
    preprocess = []
    for i in range(len(param_grid)):
        model = []
        for j in range(len(param_grid[i]['estimator'])):
            model.append(create_model(param_grid[i]['estimator'][j]))
        param_grid[i]['estimator'] = model
        preproc = []
        for j in range(len(param_grid[i]['preprocess1'])):
            preproc.append(create_preprocess(param_grid[i]['preprocess1'][j]))
        param_grid[i]['preprocess1'] = preproc
        preproc = []
        for j in range(len(param_grid[i]['preprocess2'])):
            preproc.append(create_preprocess(param_grid[i]['preprocess2'][j]))
        param_grid[i]['preprocess2'] = preproc
    pipe = Pipeline([('preprocess1', preprocess), ('preprocess2', preprocess), ('estimator', model)])
    grid_search = GridSearchCV(pipe, param_grid, cv=nfolds, scoring=scoring)
    grid_search.fit(X_train, y_train)
    print('Best estimator pre-preprocessing: {}'.format(str(grid_search.best_estimator_.named_steps['preprocess1'])))
    print('Best estimator preprocessing: {}'.format(str(grid_search.best_estimator_.named_steps['preprocess2'])))
    print('Best estimator estimator: {}\n'.format(str(grid_search.best_estimator_.named_steps['estimator'])))
    print('Best parameters: {}'.format(grid_search.best_params_))
    print('Best cross-validation score: {:.4f}'.format(grid_search.best_score_))
    print('Test set score: {:.4f}'.format(grid_search.score(X_test, y_test)))
    print('Grid search time: {:.1f}\n'.format(time.time() - time0))
    return grid_search


def create_preprocess(pre):
    if 'norm' in pre.lower():
        preprocess = MinMaxScaler()
    elif 'std' in pre.lower() or 'standard' in pre.lower():
        preprocess = StandardScaler()
    elif 'count' in pre.lower():
        preprocess = CountVectorizer()
    elif 'tfidf' in pre.lower():
        preprocess = TfidfVectorizer(norm=None)
    elif 'poly' in pre.lower():
        preprocess = PolynomialFeatures()
    else:
        preprocess = None
        print('WARNING: no preprocessor was selected\n')
    return preprocess


def create_model(algorithm):
    if 'knn class' in algorithm.lower():
        model = KNeighborsClassifier()
    elif 'logistic' in algorithm.lower() or 'logreg' in algorithm.lower():
        model = LogisticRegression(random_state=0)
    elif 'linearsvc' in algorithm.lower() or 'linear svc' in algorithm.lower():
        model = LinearSVC(random_state=0, dual=False)
    elif 'gaussian' in algorithm.lower():
        model = GaussianNB()
    elif 'multinomial' in algorithm.lower():
        model = MultinomialNB()
    elif 'tree class' in algorithm.lower():
        model = DecisionTreeClassifier(random_state=0)
    elif 'forest class' in algorithm.lower() or 'random class' in algorithm.lower():
        model = RandomForestClassifier(random_state=0)
    elif 'gradient class' in algorithm.lower() or 'boosting class' in algorithm.lower():
        model = GradientBoostingClassifier(random_state=0)
    elif 'svm' in algorithm.lower():
        model = SVC(random_state=0)
    elif 'mlp class' in algorithm.lower():
        model = MLPClassifier(random_state=0)
    elif 'knn reg' in algorithm.lower():
        model = KNeighborsRegressor()
    elif 'linear reg' in algorithm.lower() or 'linear regression' in algorithm.lower():
        model = LinearRegression()
    elif 'ridge' in algorithm.lower():
        model = Ridge(random_state=0)
    elif 'lasso' in algorithm.lower():
        model = Lasso(random_state=0)
    elif 'tree reg' in algorithm.lower():
        model = DecisionTreeRegressor(random_state=0)
    elif 'forest reg' in algorithm.lower() or 'random reg' in algorithm.lower():
        model = RandomForestRegressor(random_state=0)
    elif 'gradient reg' in algorithm.lower() or 'boosting reg' in algorithm.lower():
        model = GradientBoostingRegressor(random_state=0)
    elif 'svr' in algorithm.lower():
        model = SVR()
    elif 'mlp reg' in algorithm.lower():
        model = MLPRegressor(random_state=0)
    else:
        print('\nERROR: Algorithm was NOT provided. Note the type must be a list.\n')
        model = None
    return model


def param_sweep_matrix(dataplot, params, test_score):
    """Postprocess the cross validation grid search results and plot the parameter sweep"""
    models = []
    model = ''
    for i in range(len(params)):
        string = str(params[i]['estimator'])
        for k in range(len(string)):
            if string[k] == '(':
                model = string[:k]
                break
        if model not in models:
            models.append(model)
    for m in range(len(models)):
        test = []
        feat_name = []
        feat = []
        for i in range(len(params)):
            string = str(params[i]['estimator'])
            for k in range(len(string)):
                if string[k] == '(':
                    model = string[:k]
                    break
            if model == models[m]:
                test.append(test_score[i])
                for key, value in params[i].items():
                    if key == 'preprocess1':
                        if key not in feat_name:
                            feat_name.append(key)
                            feat.append([])
                        for k in range(len(feat_name)):
                            if feat_name[k] == key:
                                if value is not None:
                                    value = str(value)[:str(value).index('(')]
                                feat[k].append(value)
                                break
                    if key == 'preprocess2':
                        if key not in feat_name:
                            feat_name.append(key)
                            feat.append([])
                        for k in range(len(feat_name)):
                            if feat_name[k] == key:
                                if value is not None:
                                    value = str(value)[:str(value).index('(')]
                                feat[k].append(value)
                                break
                    if 'estimator__' in key:
                        key = key.replace('estimator__', '')
                        if key not in feat_name:
                            feat_name.append(key)
                            feat.append([])
                        for k in range(len(feat_name)):
                            if feat_name[k] == key:
                                feat[k].append(value)
                                break
                    if 'preprocess1__' in key:
                        key = key.replace('preprocess1__', '')
                        if key not in feat_name:
                            feat_name.append(key)
                            feat.append([])
                        for k in range(len(feat_name)):
                            if feat_name[k] == key:
                                feat[k].append(value)
                                break
                    if 'preprocess2__' in key:
                        key = key.replace('preprocess2__', '')
                        if key not in feat_name:
                            feat_name.append(key)
                            feat.append([])
                        for k in range(len(feat_name)):
                            if feat_name[k] == key:
                                feat[k].append(value)
                                break
        feat_name_unique = {}
        feat_name_sweep = []
        feat_sweep = []
        feat_index = []
        for i in range(len(feat)):
            unique = []
            [unique.append(feat[i][x]) for x in range(len(feat[i])) if feat[i][x] not in unique]
            if len(unique) == 1:
                feat_name_unique[feat_name[i]] = unique[0]
            elif len(unique) > 1:
                feat_name_sweep.append(feat_name[i])
                feat_sweep.append(unique)
                feat_index.append(i)
        if len(feat_sweep) == 0:
            test_matrix = np.array([test])
            dataplot.plot_params_sweep(models[m], test_matrix, feat_name_unique)
        elif len(feat_sweep) == 1:
            test_matrix = np.array([test])
            dataplot.plot_params_sweep(models[m], test_matrix, feat_name_unique,
                                       xtick=feat_sweep[0], xtag=feat_name_sweep[0])
        elif len(feat_sweep) == 2:
            test_matrix = np.zeros([len(feat_sweep[0]), len(feat_sweep[1])])
            for j in range(len(feat_sweep[0])):
                for h in range(len(feat_sweep[1])):
                    test_index = 0
                    for r in range(len(test)):
                        if feat_sweep[0][j] == feat[feat_index[0]][r] and feat_sweep[1][h] == feat[feat_index[1]][r]:
                            test_index = r
                            break
                    test_matrix[j, h] = test[test_index]
            dataplot.plot_params_sweep(models[m], test_matrix, feat_name_unique,
                                       xtick=feat_sweep[1], xtag=feat_name_sweep[1],
                                       ytick=feat_sweep[0], ytag=feat_name_sweep[0])
        elif len(feat_sweep) >= 3:
            feat_sweep_bu = feat_sweep.copy()
            feat_name_sweep_bu = feat_name_sweep.copy()
            feat_index_bu = feat_index.copy()
            dims = []
            for i in range(len(feat_sweep)):
                dims.append(len(feat_sweep[i]))
            for i in range(2):
                index_max = dims.index(max(dims))
                dims[index_max] = 0
                feat_sweep[i] = feat_sweep_bu[index_max]
                feat_name_sweep[i] = feat_name_sweep_bu[index_max]
                feat_index[i] = feat_index_bu[index_max]
                feat_sweep[index_max] = feat_sweep_bu[i]
                feat_name_sweep[index_max] = feat_name_sweep_bu[i]
                feat_index[index_max] = feat_index_bu[i]
                feat_sweep_bu = feat_sweep.copy()
                feat_name_sweep_bu = feat_name_sweep.copy()
                feat_index_bu = feat_index.copy()
            depth = 1
            for i in range(2, len(feat_sweep)):
                depth *= len(feat_sweep[i])
            zfeat = [[] for _ in range(depth)]
            ztag = [[] for _ in range(depth)]
            for i in range(depth):
                for j in range(2, len(feat_sweep)):
                    ztag[i].append(feat_name_sweep[j])
            for j in range(2, len(feat_sweep)):
                ind = 0
                ind_split = 0
                cum_depth = 0
                for h in range(j, len(feat_sweep)):
                    cum_depth += len(feat_sweep[h])
                for i in range(depth):
                    split = math.ceil((cum_depth / len(feat_sweep[j])))
                    if ind_split == split:
                        ind_split = 0
                        ind += 1
                        if ind == len(feat_sweep[j]):
                            ind = 0
                    zfeat[i].append(feat_sweep[j][ind])
                    ind_split += 1
            test_matrix = np.zeros([len(feat_sweep[0]), len(feat_sweep[1]), depth])
            for p in range(depth):
                for j in range(len(feat_sweep[0])):
                    for h in range(len(feat_sweep[1])):
                        test_index = 0
                        for r in range(len(test)):
                            if feat_sweep[1][h] == feat[feat_index[1]][r] \
                                    and feat_sweep[0][j] == feat[feat_index[0]][r]:
                                for t in range(2, len(feat_sweep)):
                                    if zfeat[p][t - 2] == feat[feat_index[t]][r]:
                                        if t == (len(feat_sweep) - 1):
                                            test_index = r
                                        else:
                                            continue
                                    else:
                                        break
                        test_matrix[j, h, p] = test[test_index]
            dataplot.plot_params_sweep(models[m], test_matrix, feat_name_unique,
                                       xtick=feat_sweep[1], xtag=feat_name_sweep[1],
                                       ytick=feat_sweep[0], ytag=feat_name_sweep[0],
                                       ztick=zfeat, ztag=ztag)


def data_na_removal(sourcedf, column_target, feat_cat):
    """Remove na values from the column_target column"""
    sourcedf_na = sourcedf.isna()
    print('Original null values: \n{}\n'.format(sourcedf_na.sum()))
    for i in range(sourcedf.shape[0]):
        if sourcedf_na.iloc[i, sourcedf.columns.get_loc(column_target)]:
            sourcedf.drop(index=i, inplace=True)
    imputer_num = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    feat_num = [x for x in sourcedf.columns.values.tolist() if x not in feat_cat]
    preprocessor = ColumnTransformer(transformers=[('imputer_cat', imputer_cat, feat_cat),
                                                   ('imputer_num', imputer_num, feat_num)])
    sourcedf = preprocessor.fit_transform(sourcedf)
    sourcedf = pd.DataFrame(sourcedf, columns=feat_cat + feat_num)
    print('Postprocessed null values: \n{}\n'.format(sourcedf.isna().sum()))
    return sourcedf


def data_outliers_removal(df, *outliers):
    """Remove sample containing outliers"""
    for i in range(len(outliers)):
        if outliers[i][1] == '>':
            df = df.loc[df[outliers[i][0]] < outliers[i][2], :]
        elif outliers[i][1] == '>=':
            df = df.loc[df[outliers[i][0]] <= outliers[i][2], :]
        elif outliers[i][1] == '=':
            df = df.loc[df[outliers[i][0]] == outliers[i][2], :]
        elif outliers[i][1] == '<':
            df = df.loc[df[outliers[i][0]] > outliers[i][2], :]
        elif outliers[i][1] == '<=':
            df = df.loc[df[outliers[i][0]] >= outliers[i][2], :]
    return df


def data_one_hot_encoding(df, column_target, feat_cat):
    """Apply one hot encoding the categorical features and output feature dataset and target column separately"""
    target = df[column_target]
    df.drop(column_target, axis=1, inplace=True)
    for i in range(len(feat_cat)):
        df[feat_cat[i]] = df[feat_cat[i]].astype(str)
    df = pd.get_dummies(df, columns=feat_cat)
    list_features = df.keys()
    return df, target, list_features


def data_split_scale(df, target, test_size=0.2):
    x_train, x_test, y_train, y_test = train_test_split(df, target, test_size=test_size, shuffle=True, random_state=0)
    print('\n')
    print('X_train shape: {}'.format(x_train.shape))
    print('y_train shape: {}'.format(y_train.shape))
    print('X_test shape: {}'.format(x_test.shape))
    print('y_test shape: {}'.format(y_test.shape))
    print('\n')

    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_std = scaler.transform(x_train)
    x_test_std = scaler.transform(x_test)

    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train_norm = scaler.transform(x_train)
    x_test_norm = scaler.transform(x_test)
    return x_train, x_train_std, x_train_norm, y_train, x_test, x_test_std, x_test_norm, y_test


def regression_analysis(dataplot, x, y, neighbors, alpha_ridge, alpha_lasso, max_depth, gamma, C,
                        activation, alpha_mlp, layers):
    """Compare regression model features vs the parametrization"""
    x_model = np.linspace(min(x), max(x), 1000)
    for p in range(6):
        name = ''
        limit = 0
        y_model = []
        algorithm = []
        reg = LinearRegression()
        reg.fit(x, y)
        y_model.append(reg.predict(x_model))
        algorithm.append('Linear Regression')
        if p == 0:
            limit = len(neighbors)
        elif p == 1:
            limit = len(alpha_ridge)
        elif p == 2:
            limit = len(alpha_lasso)
        elif p == 3:
            limit = len(max_depth)
        elif p == 4:
            limit = len(C)
        elif p == 5:
            limit = len(layers)
        for i in range(limit):
            if p == 0:
                reg = KNeighborsRegressor(n_neighbors=neighbors[i])
                name = 'KNN'
                algorithm.append(name + ' Regression neighbors=' + str(neighbors[i]))
            elif p == 1:
                reg = Ridge(random_state=0, alpha=alpha_ridge[i])
                name = 'Ridge'
                algorithm.append(name + ' Regression alpha=' + str(alpha_ridge[i]))
            elif p == 2:
                reg = Lasso(random_state=0, alpha=alpha_lasso[i])
                name = 'Lasso'
                algorithm.append(name + ' Regression alpha=' + str(alpha_lasso[i]))
            elif p == 3:
                reg = DecisionTreeRegressor(random_state=0, max_depth=max_depth[i])
                name = 'Decision Tree'
                algorithm.append(name + ' Regression max_depth=' + str(max_depth[i]))
            elif p == 4:
                reg = SVR(gamma=gamma, C=C[i])
                name = 'SVR'
                algorithm.append(name + ' Regression gamma=' + str(gamma) + ' and C=' + str(C[i]))
            elif p == 5:
                reg = MLPRegressor(random_state=0, activation=activation, alpha=alpha_mlp, hidden_layer_sizes=layers[i])
                name = 'MLP Neural Network'
                algorithm.append(name + ' Regression alpha=' + str(alpha_mlp) + ' and layers=' + str(layers[i]))
            reg.fit(x, y)
            y_model.append(reg.predict(x_model))
        dataplot.plot_regression(name, x, y, x_model, y_model, algorithm)


def optimal_tuning_and_ensemble(dataplot, X_train, X_train2, y_train, X_test, X_test2, y_test, n_neighbors, alpha_ridge,
                                alpha_lasso, max_depth_tree, n_estimators_random, max_features, max_depth_random,
                                n_estimators_gradient, learning_rate, max_depth_gradient, gamma, C, activation,
                                alpha_mlp, hidden_layer_sizes):
    """Create models per each algorithm based on the optimum tuning for comparison purposes"""
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    y_pred_train = reg.predict(X_train)
    print('KNeighbors Regressor MAE train score: {}'.format(round(mean_absolute_error(y_train, y_pred_train), 4)))
    print('KNeighbors Regressor MAE test score: {}'.format(round(mean_absolute_error(y_test, y_pred), 4)))
    print('KNeighbors Regressor R2 train score: {}'.format(round(r2_score(y_train, y_pred_train), 4)))
    print('KNeighbors Regressor R2 test score: {}'.format(round(r2_score(y_test, y_pred), 4)))
    print('KNeighbors Regressor MSE train score: {}'.format(round(mean_squared_error(y_train, y_pred_train), 4)))
    print('KNeighbors Regressor MSE test score: {}\n'.format(round(mean_squared_error(y_test, y_pred), 4)))
    y_knn = y_pred

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    y_pred_train = reg.predict(X_train)
    print('Lineal Regression MAE train score: {}'.format(round(mean_absolute_error(y_train, y_pred_train), 4)))
    print('Lineal Regression MAE test score: {}'.format(round(mean_absolute_error(y_test, y_pred), 4)))
    print('Lineal Regression R2 train score: {}'.format(round(r2_score(y_train, y_pred_train), 4)))
    print('Lineal Regression R2 test score: {}'.format(round(r2_score(y_test, y_pred), 4)))
    print('Lineal Regression MSE train score: {}'.format(round(mean_squared_error(y_train, y_pred_train), 4)))
    print('Lineal Regression MSE test score: {}\n'.format(round(mean_squared_error(y_test, y_pred), 4)))
    y_linear = y_pred

    reg = Ridge(random_state=0, alpha=alpha_ridge)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    y_pred_train = reg.predict(X_train)
    print('Ridge Regression MAE train score: {}'.format(round(mean_absolute_error(y_train, y_pred_train), 4)))
    print('Ridge Regression MAE test score: {}'.format(round(mean_absolute_error(y_test, y_pred), 4)))
    print('Ridge Regression R2 train score: {}'.format(round(r2_score(y_train, y_pred_train), 4)))
    print('Ridge Regression R2 test score: {}'.format(round(r2_score(y_test, y_pred), 4)))
    print('Ridge Regression MSE train score: {}'.format(round(mean_squared_error(y_train, y_pred_train), 4)))
    print('Ridge Regression MSE test score: {}\n'.format(round(mean_squared_error(y_test, y_pred), 4)))
    y_ridge = y_pred

    reg = Lasso(random_state=0, alpha=alpha_lasso)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    y_pred_train = reg.predict(X_train)
    print('Lasso Regression MAE train score: {}'.format(round(mean_absolute_error(y_train, y_pred_train), 4)))
    print('Lasso Regression MAE test score: {}'.format(round(mean_absolute_error(y_test, y_pred), 4)))
    print('Lasso Regression R2 train score: {}'.format(round(r2_score(y_train, y_pred_train), 4)))
    print('Lasso Regression R2 test score: {}'.format(round(r2_score(y_test, y_pred), 4)))
    print('Lasso Regression MSE train score: {}'.format(round(mean_squared_error(y_train, y_pred_train), 4)))
    print('Lasso Regression MSE test score: {}\n'.format(round(mean_squared_error(y_test, y_pred), 4)))
    y_lasso = y_pred

    reg = DecisionTreeRegressor(random_state=0, max_depth=max_depth_tree)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    y_pred_train = reg.predict(X_train)
    print('Decision Tree Regressor MAE train score: {}'.format(round(mean_absolute_error(y_train, y_pred_train), 4)))
    print('Decision Tree Regressor MAE test score: {}'.format(round(mean_absolute_error(y_test, y_pred), 4)))
    print('Decision Tree Regressor R2 train score: {}'.format(round(r2_score(y_train, y_pred_train), 4)))
    print('Decision Tree Regressor R2 test score: {}'.format(round(r2_score(y_test, y_pred), 4)))
    print('Decision Tree Regressor MSE train score: {}'.format(round(mean_squared_error(y_train, y_pred_train), 4)))
    print('Decision Tree Regressor MSE test score: {}\n'.format(round(mean_squared_error(y_test, y_pred), 4)))
    y_tree = y_pred

    reg = RandomForestRegressor(random_state=0, n_estimators=n_estimators_random, max_features=max_features,
                                max_depth=max_depth_random)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    y_pred_train = reg.predict(X_train)
    print('Random Forest Regressor MAE train score: {}'.format(round(mean_absolute_error(y_train, y_pred_train), 4)))
    print('Random Forest Regressor MAE test score: {}'.format(round(mean_absolute_error(y_test, y_pred), 4)))
    print('Random Forest Regressor R2 train score: {}'.format(round(r2_score(y_train, y_pred_train), 4)))
    print('Random Forest Regressor R2 test score: {}'.format(round(r2_score(y_test, y_pred), 4)))
    print('Random Forest Regressor MSE train score: {}'.format(round(mean_squared_error(y_train, y_pred_train), 4)))
    print('Random Forest Regressor MSE test score: {}\n'.format(round(mean_squared_error(y_test, y_pred), 4)))
    y_forest = y_pred

    reg = GradientBoostingRegressor(random_state=0, n_estimators=n_estimators_gradient, learning_rate=learning_rate,
                                    max_depth=max_depth_gradient)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    y_pred_train = reg.predict(X_train)
    print(
        'Gradient Boosting Regressor MAE train score: {}'.format(round(mean_absolute_error(y_train, y_pred_train), 4)))
    print('Gradient Boosting Regressor MAE test score: {}'.format(round(mean_absolute_error(y_test, y_pred), 4)))
    print('Gradient Boosting Regressor R2 train score: {}'.format(round(r2_score(y_train, y_pred_train), 4)))
    print('Gradient Boosting Regressor R2 test score: {}'.format(round(r2_score(y_test, y_pred), 4)))
    print('Gradient Boosting Regressor MSE train score: {}'.format(round(mean_squared_error(y_train, y_pred_train), 4)))
    print('Gradient Boosting Regressor MSE test score: {}\n'.format(round(mean_squared_error(y_test, y_pred), 4)))
    y_gradient = y_pred

    reg = SVR(gamma=gamma, C=C)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    y_pred_train = reg.predict(X_train)
    print('SVR MAE train score: {}'.format(round(mean_absolute_error(y_train, y_pred_train), 4)))
    print('SVR MAE test score: {}'.format(round(mean_absolute_error(y_test, y_pred), 4)))
    print('SVR R2 train score: {}'.format(round(r2_score(y_train, y_pred_train), 4)))
    print('SVR R2 test score: {}'.format(round(r2_score(y_test, y_pred), 4)))
    print('SVR MSE train score: {}'.format(round(mean_squared_error(y_train, y_pred_train), 4)))
    print('SVR MSE test score: {}\n'.format(round(mean_squared_error(y_test, y_pred), 4)))
    y_svr = y_pred

    reg = MLPRegressor(random_state=0, activation=activation, alpha=alpha_mlp, hidden_layer_sizes=hidden_layer_sizes)
    reg.fit(X_train2, y_train)
    y_pred = reg.predict(X_test2)
    y_pred_train = reg.predict(X_train2)
    print('MLP MAE train score: {}'.format(round(mean_absolute_error(y_train, y_pred_train), 4)))
    print('MLP MAE test score: {}'.format(round(mean_absolute_error(y_test, y_pred), 4)))
    print('MLP R2 train score: {}'.format(round(r2_score(y_train, y_pred_train), 4)))
    print('MLP R2 test score: {}'.format(round(r2_score(y_test, y_pred), 4)))
    print('MLP MSE train score: {}'.format(round(mean_squared_error(y_train, y_pred_train), 4)))
    print('MLP MSE test score: {}\n'.format(round(mean_squared_error(y_test, y_pred), 4)))
    y_mlp = y_pred

    y_pred_models = [y_knn, y_linear, y_ridge, y_lasso, y_tree, y_forest, y_gradient, y_svr, y_mlp]
    dataplot.compare_regression_plot(ncolumns=3, algorithm=['KNN', 'LINEAR', 'RIDGE', 'LASSO', 'TREE', 'RANDOM FOREST',
                                                            'GRADIENT BOOSTING', 'SVR', 'MLP'],
                                     y_true=y_test, y_pred=y_pred_models, tag='optimal tuning')

    y_test = np.array(y_test)
    y_pred_models = np.array(y_pred_models).transpose()
    weights_ini = np.array([1 / y_pred_models.shape[1]] * y_pred_models.shape[1])
    mae = minimize(fun=minimize_mae,
                   x0=weights_ini,
                   method='SLSQP',
                   args=(y_test, y_pred_models),
                   bounds=[(0, 1)] * y_pred_models.shape[1],
                   options={'disp': True, 'maxiter': 10000, 'eps': 1e-10, 'ftol': 1e-8},
                   constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1})
    y_mae = np.dot(y_pred_models, mae.x)
    print('MAE test score: {}'.format(round(mean_absolute_error(y_test, y_mae), 4)))
    print('R2 test score: {}'.format(round(r2_score(y_test, y_mae), 4)))
    print('MSE test score: {}\n'.format(round(mean_squared_error(y_test, y_mae), 4)))

    mse = minimize(fun=minimize_mse,
                   x0=weights_ini,
                   method='SLSQP',
                   args=(y_test, y_pred_models),
                   bounds=[(0, 1)] * y_pred_models.shape[1],
                   options={'disp': True, 'maxiter': 10000, 'eps': 1e-10, 'ftol': 1e-8},
                   constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1})
    y_mse = np.dot(y_pred_models, mse.x)
    print('MAE test score: {}'.format(round(mean_absolute_error(y_test, y_mse), 4)))
    print('R2 test score: {}'.format(round(r2_score(y_test, y_mse), 4)))
    print('MSE test score: {}\n'.format(round(mean_squared_error(y_test, y_mse), 4)))

    r2 = minimize(fun=minimize_r2,
                  x0=weights_ini,
                  method='SLSQP',
                  args=(y_test, y_pred_models),
                  bounds=[(0, 1)] * y_pred_models.shape[1],
                  options={'disp': True, 'maxiter': 10000, 'eps': 1e-10, 'ftol': 1e-8},
                  constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1})
    y_r2 = np.dot(y_pred_models, r2.x)
    print('MAE test score: {}'.format(round(mean_absolute_error(y_test, y_r2), 4)))
    print('R2 test score: {}'.format(round(r2_score(y_test, y_r2), 4)))
    print('MSE test score: {}\n'.format(round(mean_squared_error(y_test, y_r2), 4)))

    dataplot.compare_ensembled_models(metric=['MAE', 'MSE', 'R2'], y_true=y_test, y_pred=[y_mae, y_mse, y_r2],
                                      weights_ini=[mae.x, mse.x, r2.x], labels_ini=['KNN', 'LINEAR', 'RIDGE', 'LASSO',
                                                                                    'TREE', 'RANDOM FOREST',
                                                                                    'GRADIENT BOOSTING', 'SVR', 'MLP'])


def minimize_mae(weights, y_test, y_pred_models):
    """ Calculate the score of a weighted model predictions"""
    return (np.sum(np.absolute(y_test - np.dot(y_pred_models, weights)))) / y_test.shape[0]


def minimize_mse(weights, y_test, y_pred_models):
    """ Calculate the score of a weighted model predictions"""
    return (np.sum(np.square(y_test - np.dot(y_pred_models, weights)))) / y_test.shape[0]


def minimize_r2(weights, y_test, y_pred_models):
    """ Calculate the score of a weighted model predictions"""
    return -r2_score(y_test, np.dot(y_pred_models, weights))
