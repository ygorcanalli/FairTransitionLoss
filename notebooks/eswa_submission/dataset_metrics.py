from dataset_readers import *
import numpy as np
import pandas as pd
import scipy

from aif360.metrics import ClassificationMetric
from models import SimpleMLP
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef as mcc
#%%
datasets = [
    adult_dataset_reader,
    bank_dataset_reader,
    compas_dataset_reader,
    german_dataset_reader
]
#%%
def model_initializer(sens_attr):

    model = RandomForestClassifier(n_estimators=1000)

    return model
#%%
def eval(dataset_train, dataset_test, unprivileged_groups, privileged_groups, sens_attr):
    model = model_initializer(sens_attr)

    scaler = StandardScaler()
    dataset_train.features = scaler.fit_transform(dataset_train.features)
    dataset_test.features = scaler.transform(dataset_test.features)

    model = model.fit(dataset_train.features, dataset_train.labels.ravel())
    y_pred_prob = model.predict_proba(dataset_test.features)
    pos_ind = np.where(model.classes_ == dataset_test.favorable_label)[0][0]
    y_pred = (y_pred_prob[:, pos_ind] > 0.5).astype(np.float64)

    dataset_pred = dataset_test.copy()
    dataset_pred.labels = y_pred
    metric = ClassificationMetric(
            dataset_test, dataset_pred,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)

    metrics = dict()
    metrics['overall_acc'] = abs(metric.accuracy())
    metrics['bal_acc'] = abs((metric.true_positive_rate()
                                 + metric.true_negative_rate()) / 2)
    metrics['avg_odds_diff'] = metric.average_abs_odds_difference()
    metrics['disp_imp'] = abs(metric.disparate_impact())
    metrics['stat_par_diff'] = abs(metric.statistical_parity_difference())
    metrics['eq_opp_diff'] = abs(metric.equal_opportunity_difference())
    metrics['theil_ind'] = abs(metric.theil_index())
    metrics['f1_score'] = metric.f1_score()
    metrics.update(metric.performance_measures())
    return metrics
#%%
def eval_rcc(dataset_train, dataset_test, sens_attr):
    dataset_train_cp = dataset_train.copy()
    dataset_test_cp = dataset_test.copy()

    sensitive_index_features = dataset_train_cp.feature_names.index(sens_attr)
    sensitive_index_proteced = dataset_train_cp.protected_attribute_names.index(sens_attr)

    # switch protected attributes and labels
    temp = dataset_train_cp.protected_attributes[:,sensitive_index_proteced]
    dataset_train_cp.protected_attributes = dataset_train_cp.labels.reshape(dataset_train_cp.labels.shape[0])
    dataset_train_cp.features[:,sensitive_index_features] = dataset_train_cp.labels.reshape(dataset_train_cp.labels.shape[0])
    dataset_train_cp.labels = temp

    temp = dataset_test_cp.protected_attributes[:,sensitive_index_proteced]
    dataset_test_cp.protected_attributes = dataset_test_cp.labels.reshape(dataset_test_cp.labels.shape[0])
    dataset_test_cp.features[:,sensitive_index_features] = dataset_test_cp.labels.reshape(dataset_test_cp.labels.shape[0])
    dataset_test_cp.labels = temp

    model = model_initializer(sens_attr)

    scaler = StandardScaler()
    dataset_train_cp.features = scaler.fit_transform(dataset_train_cp.features)
    dataset_test_cp.features = scaler.transform(dataset_test_cp.features)

    model = model.fit(dataset_train_cp.features, dataset_train.labels.ravel())

    y_pred_prob = model.predict_proba(dataset_test_cp.features)
    pos_ind = np.where(model.classes_ == dataset_test_cp.favorable_label)[0][0]
    y_pred = (y_pred_prob[:, pos_ind] > 0.5).astype(np.float64)
    y_true = dataset_test_cp.labels.reshape(dataset_test_cp.labels.shape[0])

    rcc = mcc(y_true, y_pred)

    return rcc
#%%
def get_n_cov_corr_mean(correlation, sensitive_feature_index, num_mean):
    correlation_array = []
    for i in np.argsort(np.abs(correlation[sensitive_feature_index]))[::-1]:
        if i != sensitive_feature_index and len(correlation_array) < num_mean and not np.isnan(correlation[sensitive_feature_index,i]):
            correlation_array.append(np.abs(correlation[sensitive_feature_index,i]))
    return np.mean(np.array(correlation_array))


#%%
dataset_info = []
for dataset_reader in datasets:
    dataset_expanded_train, dataset_train, dataset_val, dataset_test, unprivileged_groups, privileged_groups, sens_attr = dataset_reader()

    num_instances = dataset_expanded_train.features.shape[0] + dataset_test.features.shape[0]
    num_encoded_features = dataset_expanded_train.features.shape[1]
    positive_instances = np.vstack([dataset_expanded_train.labels == dataset_expanded_train.favorable_label, dataset_test.labels == dataset_test.favorable_label]).reshape(num_instances)
    negative_instances = np.vstack([dataset_expanded_train.labels == dataset_expanded_train.unfavorable_label, dataset_test.labels == dataset_test.unfavorable_label]).reshape(num_instances)
    sensitive_index = dataset_expanded_train.protected_attribute_names.index(sens_attr)

    privileged_instances = np.hstack([dataset_expanded_train.protected_attributes[:,sensitive_index] == dataset_expanded_train.privileged_protected_attributes[sensitive_index], dataset_test.protected_attributes[:,sensitive_index] == dataset_test.privileged_protected_attributes[sensitive_index]]).reshape(num_instances)

    unprivileged_instances = np.hstack([dataset_expanded_train.protected_attributes[:,sensitive_index] == dataset_expanded_train.unprivileged_protected_attributes[sensitive_index], dataset_test.protected_attributes[:,sensitive_index] == dataset_test.unprivileged_protected_attributes[sensitive_index]]).reshape(num_instances)

    positive_privileged_instances = np.logical_and(positive_instances, privileged_instances)
    positive_unprivileged_instances = np.logical_and(positive_instances, unprivileged_instances)
    negative_privileged_instances = np.logical_and(negative_instances, privileged_instances)
    negative_unprivileged_instances = np.logical_and(negative_instances, unprivileged_instances)

    metrics = eval(dataset_expanded_train, dataset_test, unprivileged_groups, privileged_groups, sens_attr)

    covariance = np.cov(dataset_expanded_train.features.T)
    max_cov = get_n_cov_corr_mean(covariance, sensitive_index, 1)
    three_cov_mean = get_n_cov_corr_mean(covariance, sensitive_index, 3)

    pearson = np.corrcoef(dataset_expanded_train.features.T)
    max_pearson = get_n_cov_corr_mean(pearson, sensitive_index, 1)
    three_pearson_mean = get_n_cov_corr_mean(pearson, sensitive_index, 3)

    #rcc = eval_rcc(dataset_expanded_train, dataset_test, sens_attr)

    info = {
        'Dataset': dataset_reader.__name__,
        '\# Features' : num_encoded_features,
        '\# Instances': num_instances,
        'Sens. Attr.': sens_attr,
        'Positives': "{:.2%}".format(np.sum(positive_instances)/num_instances),
        'Negatives': "{:.2%}".format(np.sum(negative_instances)/num_instances),
        'Privileged': "{:.2%}".format(np.sum(privileged_instances)/num_instances),
        'Unprivileged': "{:.2%}".format(np.sum(unprivileged_instances)/num_instances),
        'Acc': "{:.3f}".format(metrics['ACC']),
        'MCC': "{:.3f}".format(metrics['MCC']),
        'Stat. Parity.': "{:.3f}".format(metrics['stat_par_diff']),
        'Eq. Opp.': "{:.3f}".format(metrics['eq_opp_diff']),
        'Eq. Odds': "{:.3f}".format(metrics['avg_odds_diff']),
        'Max. Cov.': "{:.3f}".format(max_cov),
        '3-Cov. Mean': "{:.3f}".format(three_cov_mean),
        'Max. Correlation': "{:.3f}".format(max_pearson),
        '3-Pearson Mean': "{:.3f}".format(three_pearson_mean)
        #'RCC': "{:.3f}".format(rcc)

    }
    dataset_info.append(info)
#%%
dataset_info_df = pd.DataFrame(dataset_info).set_index(['Dataset'])
dataset_info_df_transpose = dataset_info_df.transpose()
dataset_info_df_transpose.to_latex(('tables/dataset_info.tex'))