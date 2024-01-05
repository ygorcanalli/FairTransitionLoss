import pandas as pd
from datetime import datetime
from aif360.metrics import ClassificationMetric
from aif360.algorithms.inprocessing import PrejudiceRemover, AdversarialDebiasing, MetaFairClassifier, GerryFairClassifier
from models import (FairTransitionLossMLP, SimpleMLP, describe_metrics, AdaptativePriorityReweightingDP,
                    AdaptativePriorityReweightingEOD, AdaptativePriorityReweightingEOP)
from fitness_rules import *
from dataset_readers import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow.compat.v1 as tf_old
tf_old.disable_eager_execution()
import numpy as np
import gc
import json
import os
import optuna
from util import mathew_correlation_coefficient, f1_score
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from flatten_json import flatten

start_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
def eval(model, dataset, unprivileged_groups, privileged_groups, hyperparameters):
    try:
        # sklearn classifier
        y_pred_prob = model.predict_proba(dataset.features)
        pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
        y_pred = (y_pred_prob[:, 1] > 0.5).astype(np.float64)

        y_pred_mapped = y_pred.copy()
        # Map the dataset labels to back to their original values.
        y_pred_mapped[y_pred == 0] = dataset.unfavorable_label
        y_pred_mapped[y_pred == 1] = dataset.favorable_label

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_pred_mapped

    except AttributeError:
        # aif360 inprocessing algorithm
        y_pred = model.predict(dataset).labels

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_pred

        # Map the dataset labels to back to their original values.
        temp_labels = dataset_pred.labels.copy()

        temp_labels[(dataset_pred.labels == 1.0).ravel(), 0] = dataset.favorable_label
        temp_labels[(dataset_pred.labels == 0.0).ravel(), 0] = dataset.unfavorable_label

        dataset_pred.labels = temp_labels.copy()
    metric = ClassificationMetric(
            dataset, dataset_pred,
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
    metrics['protected_metrics'] = metric.performance_measures(False)
    metrics['privileged_metrics'] = metric.performance_measures(True)
    metrics.update(metric.performance_measures())
    metrics['MCC'] = mathew_correlation_coefficient(metrics)
    metrics['f1_score'] = f1_score(metrics)
    metrics.update(hyperparameters)

    return flatten(metrics)

def train_model(dataset_reader, hyperparameters):
    dataset_expanded_train, dataset_train, dataset_val, dataset_test, unprivileged_groups, privileged_groups, sens_attr = dataset_reader(shuffle=False)

    scaler = StandardScaler()
    dataset_train.features = scaler.fit_transform(dataset_train.features)
    dataset_val.features = scaler.transform(dataset_val.features)

    scaler = StandardScaler()
    dataset_expanded_train.features = scaler.fit_transform(dataset_expanded_train.features)
    dataset_test.features = scaler.transform(dataset_test.features)

    model = ftl_mlp_initializer(sens_attr, unprivileged_groups, privileged_groups, hyperparameters=hyperparameters)

    model = model.fit(dataset_expanded_train)
    result = eval(model, dataset_test, unprivileged_groups, privileged_groups, hyperparameters=hyperparameters)

    print('-----------------------------------')
    describe_metrics(result)
    result['method'] = ftl_mlp_initializer.__name__
    result['dataset'] = dataset_reader.__name__
    print('-----------------------------------')

    result['solution_tf_history'] = model.history.history

    return result

def ftl_mlp_initializer(sens_attr, unprivileged_groups, privileged_groups, hyperparameters=None, fitness_rule=None):
    hidden_sizes = [100,100]
    dropout = 0.2
    privileged_demotion = hyperparameters['privileged_demotion']
    privileged_promotion = hyperparameters['privileged_promotion']
    protected_demotion = hyperparameters['protected_demotion']
    protected_promotion = hyperparameters['protected_promotion']

    model = FairTransitionLossMLP(sensitive_attr=sens_attr,
                                  hidden_sizes=hidden_sizes,
                                  dropout=dropout,
                                  batch_size=64,
                                  privileged_demotion=privileged_demotion,
                                  privileged_promotion=privileged_promotion,
                                  protected_demotion=protected_demotion,
                                  protected_promotion=protected_promotion)

    return model

datasets = [
    #adult_dataset_reader,
    bank_dataset_reader
    #compas_dataset_reader,
    #german_dataset_reader
]

results = []

for dataset_reader in datasets:
    for protected_promotion in np.arange(0.0, 1.0, 0.05):
        hyperparameters = {'privileged_demotion': 0.0,
                           'privileged_promotion': 0.0,
                           'protected_demotion': 0.0,
                           'protected_promotion': protected_promotion}
        result = train_model(dataset_reader, hyperparameters=hyperparameters)
        print('Best metrics')
        print('Dataset:', dataset_reader.__name__)
        print('Method:', ftl_mlp_initializer.__name__)
        describe_metrics(result)
        results.append(result)
        results_df = pd.DataFrame(results)
        results_df.to_csv('raw_results/results_%s.csv' % start_time)
        with open('raw_results/results_%s.json' % start_time, 'w') as file:
            json.dump(results, file)
        gc.collect()
