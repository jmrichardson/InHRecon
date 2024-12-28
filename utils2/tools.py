from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import random
import torch.nn as nn
import torch.utils.data as Data
from scipy.special import expit
from sklearn import linear_model
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.svm import LinearSVC
from sklearn.inspection import partial_dependence
import itertools
from statistics import mode


from .logger import error, info


from sklearn.model_selection import GroupKFold
import numpy as np


class GroupGapSingle(GroupKFold):
    def __init__(self, test_size=0.3, gap=5):
        self.n_splits = 1
        self.test_size = test_size
        self.gap = gap

    def split(self, X, y=None):
        if 'era' not in X.columns:
            raise ValueError("The input X must have an 'era' column to define groups.")

        groups = X['era'].values
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        if isinstance(self.test_size, float) and self.test_size < 1.0:
            test_size_groups = max(1, int(self.test_size * n_groups))
        elif isinstance(self.test_size, int):
            test_size_groups = self.test_size
        else:
            raise ValueError("Test size must be a float < 1.0 or an integer.")

        if test_size_groups >= n_groups:
            raise ValueError("Test size must be smaller than the number of unique groups.")

        test_start_idx = n_groups - test_size_groups
        train_end_idx = max(0, test_start_idx - self.gap)

        train_groups = unique_groups[:train_end_idx]
        test_groups = unique_groups[test_start_idx:]

        train_idx = np.where(np.isin(groups, train_groups))[0]
        test_idx = np.where(np.isin(groups, test_groups))[0]

        # Yield only a single split to satisfy the single split requirement
        yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None):
        return self.n_splits


def cube(x):
    return x ** 3


def justify_operation_type(o):
    if o == 'sqrt':
        o = np.sqrt
    elif o == 'square':
        o = np.square
    elif o == 'sin':
        o = np.sin
    elif o == 'cos':
        o = np.cos
    elif o == 'tanh':
        o = np.tanh
    elif o == 'reciprocal':
        o = np.reciprocal
    elif o == '+':
        o = np.add
    elif o == '-':
        o = np.subtract
    elif o == '/':
        o = np.divide
    elif o == '*':
        o = np.multiply
    elif o == 'stand_scaler':
        o = StandardScaler()
    elif o == 'minmax_scaler':
        o = MinMaxScaler(feature_range=(-1, 1))
    elif o == 'quan_trans':
        o = QuantileTransformer(random_state=0)
    elif o == 'exp':
        o = np.exp
    elif o == 'cube':
        o = cube
    elif o == 'sigmoid':
        o = expit
    elif o == 'log':
        o = np.log
    else:
        print('yeah maybe some other op')
    return o


def mi_feature_distance(features, y):
    dis_mat = []
    for i in range(features.shape[1]):
        tmp = []
        for j in range(features.shape[1]):
            tmp.append(np.abs(mutual_info_regression(features[:, i].reshape
                                                     (-1, 1), y) - mutual_info_regression(features[:, j].reshape
                                                                                          (-1, 1), y))[0] / (
                               mutual_info_regression(features[:, i].
                                                      reshape(-1, 1), features[:, j].reshape(-1, 1))[
                                   0] + 1e-05))
        dis_mat.append(np.array(tmp))
    dis_mat = np.array(dis_mat)
    return dis_mat

def dict_order(df):
    cols = list(df.columns)
    dict_order = {col: 1 for col in cols}
    return dict_order

def feature_distance(feature, y):
    return mi_feature_distance(feature, y)

'''
for ablation study
if mode == c then don't do cluster
'''
def cluster_features(features, y, cluster_num=2, mode=''):
    if mode == 'c':
        return _wocluster_features(features, y, cluster_num)
    else:
        return _cluster_features(features, y, cluster_num)

# def _cluster_features(features, y, cluster_num=2):
#     k = int(np.sqrt(features.shape[1]))
#     features = feature_distance(features, y)
#     features = features.reshape(features.shape[0], -1)
#     clustering = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='single').fit(features)
#     labels = clustering.labels_
#     clusters = defaultdict(list)
#     for ind, item in enumerate(labels):
#         clusters[item].append(ind)
#     print(clusters)
#     return clusters

'''
return single column as cluster
'''

def _cluster_features(features, y, cluster_num=2):
    clusters = defaultdict(list)
    for ind, item in enumerate(range(features.shape[1])):
        #clusters[item].append(ind)
        clusters[item] = ind
    return clusters

def _wocluster_features(features, y, cluster_num=2):
    clusters = defaultdict(list)
    for ind, item in enumerate(range(features.shape[1])):
        #clusters[item].append(ind)
        clusters[item] = ind
    return clusters



SUPPORT_STATE_METHOD = {
    'ds'
}

# def change_feature_type(df):
#     for column in df:
#         n = df[column].nunique()
#         if n<26:
#             df[column] = df[column].astype('category')
#     return df

def change_feature_type(df):
    for column in df:
        n = df[column].nunique()
        if n < 26:
            df[column] = df[column].astype('category')
        else:
            df[column] = pd.to_numeric(df[column], errors='coerce')
    return df

def dataframe_scaler(df):
    if df.iloc[:, -1].dtype == 'object':
        df.iloc[:, -1] = pd.to_numeric(df.iloc[:, -1], errors='coerce')
    df[df.columns[:-1]] = change_feature_type(df[df.columns[:-1]])
    scaler = MinMaxScaler(feature_range=(-1,1))
    dg = df.iloc[:, :-1]
    num_df = dg.select_dtypes(include=['number'])
    if num_df.shape[1]>0:
        df[num_df.columns] = scaler.fit_transform(df[num_df.columns].values)
    return df


def feature_state_generation(X, counter):
    return _feature_state_generation_des(X, counter)

def add_noise_to_list(lst, counter):
    noisy_list = lst.copy()
    n = random.randint(1, len(lst))  # Number of elements to add noise to

    # Randomly select n indices from the list
    indices = random.sample(range(len(lst)), n)

    for idx in indices:
        element = lst[idx]
        noise_range = abs(element) / (2*counter)
        noise = random.uniform(-noise_range, noise_range)
        noisy_element = element + noise
        noisy_list[idx] = noisy_element

    return noisy_list

def _feature_state_generation_des(X, counter):
    feature_matrix = []
    for i in range(8):
        feature_matrix = feature_matrix + list(X.astype(np.float64).
                                               describe().iloc[i, :].describe().fillna(0).values)
    feature_matrix = add_noise_to_list(feature_matrix, counter)
    return feature_matrix

def parent_match_calc(a, b, c):
    set_a, set_b, set_c = set(a), set(b), set(c)
    match_1 = len(set_a.intersection(set_b))
    match_2 = len(set_a.intersection(set_c))
    return match_1+match_2

def relative_absolute_error(y_test, y_predict):
    y_test = np.array(y_test)
    y_predict = np.array(y_predict)
    error = np.sum(np.abs(y_test - y_predict)) / np.sum(np.abs(np.mean(
        y_test) - y_test))
    return error


def downstream_task_new(data, task_type):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].astype(float)
    if task_type == 'cls':
        clf = RandomForestClassifier(random_state=0)
        f1_list = []
        skf = GroupGapSingle(test_size=.3)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
        return clf, np.mean(f1_list)
    elif task_type == 'reg':
        kf = GroupGapSingle(test_size=.3)
        reg = RandomForestRegressor(random_state=0)
        rae_list = []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            reg.fit(X_train, y_train)
            y_predict = reg.predict(X_test)
            rae_list.append(1 - relative_absolute_error(y_test, y_predict))
        return reg, np.mean(rae_list)
    else:
        return -1


# def downstream_task_new(data, task_type, state_num=10):
#     X = data.iloc[:, :-1]
#     y = data.iloc[:, -1].astype(int)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
#                                                         random_state=state_num, shuffle=True)
#     if task_type == 'cls':
#         clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
#         y_predict = clf.predict(X_test)
#         return clf, f1_score(y_test, y_predict, average='weighted')
#     if task_type == 'reg':
#         reg = RandomForestRegressor(random_state=0).fit(X_train, y_train)
#         y_predict = reg.predict(X_test)
#         return reg, 1 - relative_absolute_error(y_test, y_predict)

        # if metric_type == 'acc':
        #     return accuracy_score(y_test, y_predict)
        # elif metric_type == 'pre':
        #     return precision_score(y_test, y_predict)
        # elif metric_type == 'rec':
        #     return recall_score(y_test, y_predict)
        # elif metric_type == 'f1':
        #     return clf, f1_score(y_test, y_predict, average='weighted')
        # if metric_type == 'mae':
        #     return mean_absolute_error(y_test, y_predict)
        # elif metric_type == 'mse':
        #     return mean_squared_error(y_test, y_predict)
        # elif metric_type == 'rae':
        #     return reg, 1 - relative_absolute_error(y_test, y_predict)

def insert_generated_feature_to_original_feas(feas, f):
    y_label = pd.DataFrame(feas[feas.columns[len(feas.columns) - 1]])
    y_label.columns = [feas.columns[len(feas.columns) - 1]]
    feas = feas.drop(columns=feas.columns[len(feas.columns) - 1])
    final_data = pd.concat([feas, f, y_label], axis=1)
    return final_data

def downstream_task_cross_validataion(data, task_type):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].astype(float)
    if task_type == 'cls':
        clf = RandomForestClassifier(random_state=0)
        scores = cross_val_score(clf, X, y, cv=5, scoring='f1_weighted')
        print(scores)
    if task_type == 'reg':
        reg = RandomForestRegressor(random_state=0)
        scores = 1 - cross_val_score(reg, X, y, cv=5, scoring=make_scorer(
            relative_absolute_error))
        print(scores)


def test_task_new(Dg, task='cls', state_num = 10):
    X = Dg.iloc[:, :-1]
    y = Dg.iloc[:, -1].astype(float)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
    #                                                     random_state = state_num, shuffle=True)
    # if task == 'cls':
    #     clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
    #     y_predict = clf.predict(X_test)
    #     precision = precision_score(y_test, y_predict,  average='weighted')
    #     recall = recall_score(y_test, y_predict,  average='weighted')
    #     f1 = f1_score(y_test, y_predict, average='weighted')
    #     return precision, recall, f1
    # elif task == 'reg':
    #     reg = RandomForestRegressor(random_state=0).fit(X_train, y_train)
    #     y_predict = reg.predict(X_test)
    #     mae =  1 - mean_absolute_error(y_test, y_predict)
    #     mse = 1 - mean_squared_error(y_test, y_predict)
    #     rae = 1 - relative_absolute_error(y_test, y_predict)
    #     return mae, mse, rae
    # else:
    #     return -1
    if task == 'cls':
        clf = RandomForestClassifier(random_state=0)
        pre_list, rec_list, f1_list = [], [], []
        skf = GroupGapSingle(test_size=.3)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            pre_list.append(precision_score(y_test, y_predict, average=
            'weighted'))
            rec_list.append(recall_score(y_test, y_predict, average='weighted')
                            )
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
        return np.mean(pre_list), np.mean(rec_list), np.mean(f1_list)
    elif task == 'reg':
        kf = GroupGapSingle(test_size=.3)
        reg = RandomForestRegressor(random_state=0)
        mae_list, mse_list, rae_list = [], [], []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            reg.fit(X_train, y_train)
            y_predict = reg.predict(X_test)
            mae_list.append(mean_absolute_error(y_test, y_predict))
            mse_list.append(mean_squared_error(y_test, y_predict))
            rae_list.append(1 - relative_absolute_error(y_test, y_predict))
        return np.mean(mae_list), np.mean(mse_list), np.mean(rae_list)
    else:
        return -1


def overall_feature_selection(best_features, task_type):
    if task_type == 'reg':
        data = pd.concat([fea for fea in best_features], axis=1)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1].astype(float)
        reg = linear_model.Lasso(alpha=0.1).fit(X, y)
        model = SelectFromModel(reg, prefit=True)
        X = X.loc[:, model.get_support()]
        new_data = pd.concat([X, y], axis=1)
        mae, mse, rae = test_task_new(new_data, task_type)
        info('mae: {:.3f}, mse: {:.3f}, 1-rae: {:.3f}'.format(mae, mse, 1 -
                                                              rae))
    elif task_type == 'cls':
        data = pd.concat([fea for fea in best_features], axis=1)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1].astype(float)
        clf = LinearSVC(C=0.01, penalty='l1', dual=False).fit(X, y)
        model = SelectFromModel(clf, prefit=True)
        X = X.loc[:, model.get_support()]
        new_data = pd.concat([X, y], axis=1)
        acc, pre, rec, f1 = test_task_new(new_data, task_type)
        info('acc: {:.3f}, pre: {:.3f}, rec: {:.3f}, f1: {:.3f}'.format(acc,
                                                                        pre, rec, f1))
    return new_data


def calc_h_stat(task_type, model, X, feats):
    h_stat = 0
    if task_type == 'reg':
        h_stat = calc_h_stat_reg(model, X, feats)
    else:
        h_stat = calc_h_stat_cls (model, X, feats)
    return h_stat

def compute_h_val(f_vals, selectedfeatures):
    numer_els = f_vals[tuple(selectedfeatures)].copy()
    denom_els = f_vals[tuple(selectedfeatures)].copy()
    sign = -1.0
    for n in range(len(selectedfeatures)-1, 0, -1):
        for subfeatures in itertools.combinations(selectedfeatures, n):
            numer_els += sign * f_vals[tuple(subfeatures)]
        sign *= -1.0
    numer = np.sum(numer_els**2)
    denom = np.sum(denom_els**2)
    return np.sqrt(numer/denom)

def calc_h_stat_reg(model, X, feats):
    def center(arr): 
        return arr - np.mean(arr)

    def cartesian_product(*arrays):
        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[...,i] = a
        return arr.reshape(-1, la)

    def compute_f_vals_sklearn(model, X, feats=None, grid_resolution=10):

        def _pd_to_df(pde, feature_names):
            a = pde['values']
            df = pd.DataFrame(cartesian_product(*a))
            rename = {i: feature_names[i] for i in range(len(feature_names))}
            df.rename(columns=rename, inplace=True)
            df['preds'] = pde['average'].flatten()
            return df

        def _get_feat_idxs(feats):
            return [tuple(list(X.columns).index(f) for f in feats)]

        f_vals = {}
        if feats is None:
            feats = list(X.columns)

        # Calculate partial dependencies for full feature set
        pd_full = partial_dependence(
            model, X, _get_feat_idxs(feats), 
            grid_resolution=grid_resolution
        )

        # Establish the grid
        df_full = _pd_to_df(pd_full, feats)
        grid = df_full.drop('preds', axis=1)

        # Store
        f_vals[tuple(feats)] = center(df_full.preds.values)

        # Calculate partial dependencies for [1..SFL-1]
        for n in range(1, len(feats)):
            for subset in itertools.combinations(feats, n):
                pd_part = partial_dependence(
                    model, X, _get_feat_idxs(subset),
                    grid_resolution=grid_resolution
                )
                df_part = _pd_to_df(pd_part, subset)
                joined = pd.merge(grid, df_part, how='left')
                f_vals[tuple(subset)] = center(joined.preds.values)
        return f_vals

    f_val = compute_f_vals_sklearn(model, X, feats)
    h_val = compute_h_val(f_val, feats)
    return h_val

def calc_h_stat_cls(model, X, feats):
    def center(arr): 
        return arr - np.mean(arr)
    def compute_f_vals_manual(model, X, feats=None):

        def _partial_dependence(model, X, feats):
            P = X.copy()
            for f in P.columns:
                if f in feats: continue
                if P[f].dtypes== 'category':
                    P.loc[:,f] = mode(P[f])
                else:
                    P.loc[:,f] = np.mean(P[f])
            # Assumes a regressor here, use return model.predict_proba(P)[:,1] for binary classification
            return model.predict_proba(P)[:,1]

        f_vals = {}
        if feats is None:
            feats = list(X.columns)

        # Calculate partial dependencies for full feature set
        full_preds = _partial_dependence(model, X, feats)
        f_vals[tuple(feats)] = center(full_preds)

        # Calculate partial dependencies for [1..SFL-1]
        for n in range(1, len(feats)):
            for subset in itertools.combinations(feats, n):
                pd_part = _partial_dependence(model, X, subset)
                f_vals[tuple(subset)] = center(pd_part)

        return f_vals
    
    f_vals = compute_f_vals_manual(model, X, feats)
    h_vals = compute_h_val(f_vals, feats)

    return h_vals
