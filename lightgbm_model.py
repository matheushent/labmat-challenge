from __future__ import print_function

from optparse import OptionParser
import pandas as pd
import numpy as np
import time
import os

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

from utils.process_data import tfidf_transformer

parser = OptionParser()

parser.add_option("-n", "--name", dest="csv_name", help="Name of the csv file containing the classification report.")
(options, args) = parser.parse_args()

if not options.csv_name:
    parser.error('Error: csv file name must be specified. Pass -n via command line')

init = time.time()

data = pd.read_csv('dataset.csv')
data = data.drop(labels='Source.1', axis=1)
data.dropna(axis=0, inplace=True)
data['Business line'] = data['Business line'].replace({
    'Commercial': 'Comercial'
})

features = [c for c in data.columns if c not in ['Business line', 'Topic']]
target_1 = data['Business line']
target_2 = data['Topic']

# x = pd.DataFrame()
x = tfidf_transformer(data['Summary'])
# x['Source'] = data['Source']

# x_train, x_test, y_train, y_test = train_test_split(x, target_1, test_size=0.2)

param = {
    'bagging_freq': 5,
    'bagging_fraction': 0.4,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.05,
    'learning_rate': 0.01,
    'max_depth': -1,  
    'metric':'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'multiclass', 
    'verbosity': 1
}

folds = StratifiedKFold(n_splits=10, shuffle=False, random_state=44000)
oof = np.zeros((x.shape[0], 1))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(x.values, target_1.values)):
    print("Fold {}".format(fold_))
    trn_data = lgb.Dataset(x.iloc[trn_idx][features], label=target_1.iloc[trn_idx])
    val_data = lgb.Dataset(x.iloc[val_idx][features], label=target_1.iloc[val_idx])

    num_round = 1000000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 3000)
    oof[val_idx] = clf.predict(x.iloc[val_idx][features], num_iteration=clf.best_iteration)

print("CV score (1): {:<8.5f}".format(roc_auc_score(target_1, oof)))
report = classification_report(target_1, oof)
classification_1 = pd.DataFrame(report).transpose()
out_path = os.path.join('reports', '1_' + options.csv_name)
classification_1.to_csv(out_path, encoding='utf-8')

oof = np.zeros((x.shape[0], 1))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(x.values, target_2.values)):
    print("Fold {}".format(fold_))
    trn_data = lgb.Dataset(x.iloc[trn_idx][features], label=target_2.iloc[trn_idx])
    val_data = lgb.Dataset(x.iloc[val_idx][features], label=target_2.iloc[val_idx])

    num_round = 1000000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 3000)
    oof[val_idx] = clf.predict(x.iloc[val_idx][features], num_iteration=clf.best_iteration)

print("CV score (2): {:<8.5f}".format(roc_auc_score(target_2, oof)))
report = classification_report(target_2, oof)
classification_2 = pd.DataFrame(report).transpose()
out_path = os.path.join('reports', '2_' + options.csv_name)
classification_2.to_csv(out_path, encoding='utf-8')