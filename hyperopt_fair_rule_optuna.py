import pandas as pd
from datetime import datetime
from aif360.metrics import ClassificationMetric
from aif360.algorithms.inprocessing import PrejudiceRemover, AdversarialDebiasing, MetaFairClassifier, GerryFairClassifier
from models import FairTransitionLossMLP, SimpleMLP, describe_metrics, APW_DP
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

N_TRIALS = 5
N_JOBS = 5
SAMPLER = TPESampler
PRUNER = HyperbandPruner
CONNECTION_STRING = os.environ.get('CONNECTION_STRING')
if CONNECTION_STRING is None:
    CONNECTION_STRING = 'mysql+pymysql://optuna:optuna@localhost:3306/optuna'
start_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')

tf_old.compat.v1.disable_eager_execution()

def get_sampler():
    return TPESampler()

def get_pruner():
    return HyperbandPruner()

def eval(model, dataset, unprivileged_groups, privileged_groups, fitness_rule, hyperparameters):
    try:
        # sklearn classifier
        y_pred_prob = model.predict_proba(dataset.features)
        pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
        y_pred = (y_pred_prob[:, 1] > 0.5).astype(np.float64)

        # Map the dataset labels to back to their original values.
        y_pred[y_pred == 0] = dataset.unfavorable_label
        y_pred[y_pred == 1] = dataset.favorable_label

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_pred

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
    metrics.update(metric.performance_measures())
    metrics['MCC'] = mathew_correlation_coefficient(metrics)
    metrics['f1_score'] = f1_score(metrics)
    metrics['fitness'] = fitness_rule(metrics)
    if type(hyperparameters) is not dict:
        metrics['solution'] = hyperparameters.params
    else:
        metrics['solution'] = hyperparameters

    return metrics


def tune_model(dataset_reader, model_initializer, fitness_rule):
    tune_results_history = []
    dataset_expanded_train, dataset_train, dataset_val, dataset_test, unprivileged_groups, privileged_groups, sens_attr = dataset_reader()

    scaler = StandardScaler()
    dataset_train.features = scaler.fit_transform(dataset_train.features)
    dataset_val.features = scaler.transform(dataset_val.features)

    scaler = StandardScaler()
    dataset_expanded_train.features = scaler.fit_transform(dataset_expanded_train.features)
    dataset_test.features = scaler.transform(dataset_test.features)

    def objective(trial):
        # training
        trial_model = model_initializer(sens_attr, unprivileged_groups, privileged_groups, trial)
        trial_model = trial_model.fit(dataset_train.copy())
        result = eval(trial_model, dataset_val.copy(), unprivileged_groups, privileged_groups, fitness_rule, trial)
        tune_results_history.append(result)
        return result['fitness']

    if fitness_rule is not None:
        # best solution
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        study = optuna.create_study(direction='maximize',
                                    study_name="{0}_{1}_{2}".format(fitness_rule.__name__,model_initializer.__name__,now) ,
                                    pruner=get_pruner(),
                                    sampler=get_sampler())

        study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS)

        # eval on test set
        model = model_initializer(sens_attr, unprivileged_groups, privileged_groups, study.best_params)
    else:
        model = model_initializer(sens_attr, unprivileged_groups, privileged_groups)

    model = model.fit(dataset_expanded_train)
    best_result = eval(model, dataset_test, unprivileged_groups, privileged_groups, fitness_rule, study.best_params)

    best_result['tune_results_history'] = tune_results_history
    if fitness_rule is not None:
        best_result['fitness_rule'] = fitness_rule.__name__
    else:
        best_result['fitness_rule'] = 'No optimization'

    print('-----------------------------------')
    describe_metrics(best_result)
    best_result['method'] = model_initializer.__name__
    best_result['dataset'] = dataset_reader.__name__
    print('-----------------------------------')

    try:
        # tk classifier
        best_result['best_solution_tf_history'] = model.history.history
    except AttributeError:
        # aif360 inprocessing algorithm
        best_result['best_solution_tf_history'] = None

    return best_result

def ftl_mlp_initializer(sens_attr, unprivileged_groups, privileged_groups, hyperparameters=None):
    hidden_sizes = [100,100]
    if type(hyperparameters) is not dict:
        dropout = hyperparameters.suggest_float('dropout', 0.0, 0.2)
    else:
        dropout = hyperparameters['dropout']
    if type(hyperparameters) is not dict:
        privileged_demotion = hyperparameters.suggest_float('privileged_demotion', 0.0, 1.0)
        privileged_promotion = hyperparameters.suggest_float('privileged_promotion', 0.0, 1.0)
        protected_demotion = hyperparameters.suggest_float('protected_demotion', 0.0, 1.0)
        protected_promotion = hyperparameters.suggest_float('protected_promotion', 0.0, 1.0)
    else:
        privileged_demotion = hyperparameters['privileged_demotion']
        privileged_promotion = hyperparameters['privileged_promotion']
        protected_demotion = hyperparameters['protected_demotion']
        protected_promotion = hyperparameters['protected_promotion']

    if hyperparameters is not None:
        model = FairTransitionLossMLP(sensitive_attr=sens_attr,
                                      hidden_sizes=hidden_sizes,
                                      dropout=dropout,
                                      batch_size=64,
                                      privileged_demotion=privileged_demotion,
                                      privileged_promotion=privileged_promotion,
                                      protected_demotion=protected_demotion,
                                      protected_promotion=protected_promotion)
    else:
        model = FairTransitionLossMLP(sensitive_attr=sens_attr,
                                      hidden_sizes=[32],
                                      dropout=0.1,
                                      batch_size=64)
    return model

def simple_mlp_initializer(sens_attr, unprivileged_groups, privileged_groups, hyperparameters=None):
    hidden_sizes = [100, 100]
    if type(hyperparameters) is not dict:
        dropout = hyperparameters.suggest_float('dropout', 0.0, 0.2)
    else:
        dropout = hyperparameters['dropout']
    if hyperparameters is not None:
        model = SimpleMLP(sensitive_attr=sens_attr,
                        hidden_sizes=hidden_sizes,
                        dropout=dropout,
                        batch_size=64)
    else:
        model = SimpleMLP(sensitive_attr=sens_attr,
                        hidden_sizes=[32],
                        dropout=0.1,
                        batch_size=64)
    return model

def prejudice_remover_initializer(sens_attr, unprivileged_groups, privileged_groups, hyperparameters=None):
    if type(hyperparameters) is not dict:
        eta = hyperparameters.suggest_float('eta', 0.0, 50.0)
    else:
        eta = hyperparameters['eta']
    if hyperparameters is not None:

        model = PrejudiceRemover(eta=eta, sensitive_attr=sens_attr)
    else:
        model = PrejudiceRemover(sensitive_attr=sens_attr)
    return model

def meta_fair_classifier_sr_initializer(sens_attr, unprivileged_groups, privileged_groups, hyperparameters=None):
    if type(hyperparameters) is not dict:
        tau = hyperparameters.suggest_float('tau', 0.0, 2.0)
    else:
        tau = hyperparameters['tau']
    if hyperparameters is not None:
        model = MetaFairClassifier(tau=tau, sensitive_attr=sens_attr, type='sr')
    else:
        model = MetaFairClassifier(sensitive_attr=sens_attr, type='sr')
    return model

def meta_fair_classifier_fdr_initializer(sens_attr, unprivileged_groups, privileged_groups, hyperparameters=None):
    if type(hyperparameters) is not dict:
        tau = hyperparameters.suggest_float('tau', 0.0, 2.0)
    else:
        tau = hyperparameters['tau']
    if hyperparameters is not None:
        model = MetaFairClassifier(tau=tau, sensitive_attr=sens_attr, type='fdr')
    else:

        model = MetaFairClassifier(sensitive_attr=sens_attr, type='sr')
    return model



def adversarial_debiasing_initializer(sens_attr, unprivileged_groups, privileged_groups, hyperparameters=None):
    classifier_num_hidden_units = 100
    if type(hyperparameters) is not dict:
        adversary_loss_weight = hyperparameters.suggest_float('adversary_loss_weight', 0.0, 1.0)
    else:
        adversary_loss_weight = hyperparameters['adversary_loss_weight']
    if hyperparameters is not None:
        model = AdversarialDebiasing(unprivileged_groups, privileged_groups,
                                     'adv_debias_' + str(datetime.now().timestamp()), tf_old.Session(),
                                     classifier_num_hidden_units=classifier_num_hidden_units,
                                     adversary_loss_weight=adversary_loss_weight)
    else:
        model = AdversarialDebiasing(unprivileged_groups, privileged_groups,
                                     'adv_debias_' + str(datetime.now().timestamp()), tf_old.Session())
    return model


def adversarial_debiasing_initializer(sens_attr, unprivileged_groups, privileged_groups, hyperparameters=None):
    classifier_num_hidden_units = 100
    if type(hyperparameters) is not dict:
        adversary_loss_weight = hyperparameters.suggest_float('adversary_loss_weight', 0.0, 1.0)
    else:
        adversary_loss_weight = hyperparameters['adversary_loss_weight']
    if hyperparameters is not None:
        model = AdversarialDebiasing(unprivileged_groups, privileged_groups,
                                     'adv_debias_' + str(datetime.now().timestamp()), tf_old.Session(),
                                     classifier_num_hidden_units=classifier_num_hidden_units,
                                     adversary_loss_weight=adversary_loss_weight)
    else:
        model = AdversarialDebiasing(unprivileged_groups, privileged_groups,
                                     'adv_debias_' + str(datetime.now().timestamp()), tf_old.Session())
    return model

def gerry_fair_classifier_initializer(sens_attr, unprivileged_groups, privileged_groups, hyperparameters=None):
    if type(hyperparameters) is not dict:
        C = hyperparameters.suggest_float('C', 0.0, 20.0)
        gamma = hyperparameters.suggest_categorical('gamma', [0.1, 0.01, 0.001])
    else:
        C = hyperparameters['C']
        gamma = hyperparameters['gamma']
    if hyperparameters is not None:
        model = GerryFairClassifier(C=C, gamma=gamma, fairness_def='FN')
    else:
        model = GerryFairClassifier(fairness_def='FN')
    return model

def adaptative_priority_reweighting_classifier_initializer(sens_attr, unprivileged_groups, privileged_groups, hyperparameters=None):
    if type(hyperparameters) is not dict:
        alpha = hyperparameters.suggest_float('alpha', 0.0, 10000.0)
        eta = hyperparameters.suggest_float('eta', 0.5, 3.0)
    else:
        alpha = hyperparameters['alpha']
        eta = hyperparameters['eta']
    if hyperparameters is not None:
        model = APW_DP(sensitive_attr=sens_attr, alpha=alpha, eta=eta)
    else:
        model = APW_DP(sensitive_attr=sens_attr)
    return model


datasets = [
    adult_dataset_reader,
    #bank_dataset_reader,
    #compas_dataset_reader
    #german_dataset_reader
]

rules = [
    mcc_parity,
    #mcc_odds,
    #mcc_opportunity,
    acc_parity,
    #acc_odds,
    #acc_opportunity
]

methods = [
    adaptative_priority_reweighting_classifier_initializer
    #meta_fair_classifier_sr_initializer,
    #gerry_fair_classifier_initializer,
    #simple_mlp_initializer,
    #ftl_mlp_initializer,
    #adversarial_debiasing_initializer
    #prejudice_remover_initializer
]

results = []

for dataset_reader in datasets:
    for fitness_rule in rules:
        for model_initializer in methods:
            result = tune_model(dataset_reader, model_initializer, fitness_rule)
            print('Best metrics')
            print('Dataset:', dataset_reader.__name__)
            print('Method:', model_initializer.__name__)
            if fitness_rule is not None:
                print('Fitness rule:', fitness_rule.__name__)
            else:
                print('Fitness rule: None')
            describe_metrics(result)
            results.append(result)
            results_df = pd.DataFrame(results)
            results_df.to_csv('raw_results/results_%s.csv' % start_time)
            with open('raw_results/results_%s.json' % start_time, 'w') as file:
                json.dump(results, file)
            gc.collect()
