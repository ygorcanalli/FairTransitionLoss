
from multiprocessing import Pool

import pandas as pd
from pprint import pprint
from datetime import datetime
from aif360.datasets import AdultDataset
from aif360.metrics import ClassificationMetric
from models import FairTransitionLossMLP, SimpleMLP
from aif360.algorithms.inprocessing import PrejudiceRemover, AdversarialDebiasing, MetaFairClassifier, GerryFairClassifier
from fitness_rules import *
from sklearn.preprocessing import StandardScaler
import tensorflow.compat.v1 as tf_old
tf_old.disable_eager_execution()
import tensorflow as tf
import numpy as np
import pygad

def adult_dataset_reader():
    label_map = {1.0: '>50K', 0.0: '<=50K'}
    protected_attribute_maps = [{1.0: 'Male', 0.0: 'Female'}]
    ad = AdultDataset(protected_attribute_names=['sex'],
                      categorical_features=['workclass', 'education', 'marital-status',
                                            'occupation', 'relationship', 'native-country', 'race'],
                      privileged_classes=[['Male']], metadata={'label_map': label_map,
                                                               'protected_attribute_maps': protected_attribute_maps})
    (dataset_train,
     dataset_test) = AdultDataset().split([0.8], shuffle=True)

    sens_ind = 1
    sens_attr = dataset_train.protected_attribute_names[sens_ind]

    unprivileged_groups = [{sens_attr: v} for v in
                           dataset_train.unprivileged_protected_attributes[sens_ind]]
    privileged_groups = [{sens_attr: v} for v in
                         dataset_train.privileged_protected_attributes[sens_ind]]

    return dataset_train, dataset_test, unprivileged_groups, privileged_groups, sens_attr

def ftl_mlp_initializer(sens_attr, d_0, p_0, d_1, p_1):
    model = FairTransitionLossMLP(sensitive_attr=sens_attr,
                                  hidden_sizes=[128],
                                  dropout=0.1,
                                  batch_size=32,
                                  privileged_demotion=d_0, privileged_promotion=p_0,
                                  protected_demotion=d_1, protected_promotion=p_1)

    return model


def eval_metrics(model, dataset, unprivileged_groups, privileged_groups):
    try:
        # sklearn classifier
        y_pred_prob = model.predict_proba(dataset.features)
        pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
    except AttributeError:
        # aif360 inprocessing algorithm
        y_pred_prob = model.predict(dataset).scores
        pos_ind = 0

    y_val_pred = (y_pred_prob[:, pos_ind] > 0.5).astype(np.float64)

    dataset_pred = dataset.copy()
    dataset_pred.labels = y_val_pred
    metric = ClassificationMetric(
            dataset, dataset_pred,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)

    metrics = dict()
    metrics['overall_acc'] = abs(metric.accuracy())
    metrics['bal_acc'] = abs((metric.true_positive_rate()
                                 + metric.true_negative_rate()) / 2)
    metrics['avg_odds_diff'] = abs(metric.average_odds_difference())
    metrics['disp_imp'] = abs(metric.disparate_impact())
    metrics['stat_par_diff'] = abs(metric.statistical_parity_difference())
    metrics['eq_opp_diff'] = abs(metric.equal_opportunity_difference())
    metrics['theil_ind'] = abs(metric.theil_index())
    metrics['acc_parity'] = metrics['overall_acc'] - metrics['stat_par_diff']
    metrics['acc_odds'] = metrics['overall_acc'] - metrics['avg_odds_diff']
    metrics['acc_opportunity'] = metrics['overall_acc'] - metrics['eq_opp_diff']

    return metrics

def eval_model(d_0, p_0, d_1, p_1):
    dataset_train, dataset_test, unprivileged_groups, privileged_groups, sens_attr = adult_dataset_reader()
    # training
    model = ftl_mlp_initializer(sens_attr, d_0, p_0, d_1, p_1)

    scaler = StandardScaler()

    scaled_train = dataset_train.copy()
    scaled_train.features = scaler.fit_transform(scaled_train.features)

    scaled_test = dataset_test.copy()
    scaled_test.features = scaler.transform(scaled_test.features)

    model = model.fit(scaled_test)

    test_metrics = eval_metrics(model, scaled_test, unprivileged_groups, privileged_groups)
    test_metrics['d_0'] = d_0
    test_metrics['p_0'] = p_0
    test_metrics['d_1'] = d_1
    test_metrics['p_1'] = p_1
    return test_metrics

start_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')

d_0s = [0.0, 0.1, 0.2, 0.3]
p_0s = [0.0, 0.1, 0.2, 0.3]
d_1s = [0.0, 0.1, 0.2, 0.3]
p_1s = [0.0, 0.1, 0.2, 0.3]
results = []

for d_0 in d_0s:
    for p_0 in p_0s:
        for d_1 in d_1s:
            for p_1 in p_1s:
                result = eval_model(d_0, p_0, d_1, p_1)
                results.append(result)
                results_df = pd.DataFrame(results)
                pprint(results_df)
                results_df.to_csv('results_%s.csv' % start_time)

