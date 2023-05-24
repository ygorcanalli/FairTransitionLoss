import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from matplotlib import cm
from multiprocessing import Pool
import os

from aif360.metrics import ClassificationMetric
from aif360.datasets import AdultDataset
from util import describe, describe_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# Bias mitigation techniques
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import PrejudiceRemover, AdversarialDebiasing, MetaFairClassifier
from models import FairTransitionLossMLP

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

N_TRIALS = 50
SAMPLER = TPESampler
PRUNER = 'default'#HyperbandPruner

def get_sampler():
    return TPESampler()

def get_pruner():
    return HyperbandPruner()


import time
from multiprocessing import Pool

import tensorflow.compat.v1 as tf_old
tf_old.disable_eager_execution()


label_map = {1.0: '>50K', 0.0: '<=50K'}
protected_attribute_maps = [{1.0: 'Male', 0.0: 'Female'}]
ad = AdultDataset(protected_attribute_names=['sex'],
                  categorical_features=['workclass', 'education', 'marital-status',
                                        'occupation', 'relationship', 'native-country', 'race'],
                  privileged_classes=[['Male']], metadata={'label_map': label_map,
                                                           'protected_attribute_maps': protected_attribute_maps})
(dataset_train,
 dataset_val,
 dataset_test) = AdultDataset().split([0.5, 0.8], shuffle=True)

sens_ind = 1
sens_attr = dataset_train.protected_attribute_names[sens_ind]

unprivileged_groups = [{sens_attr: v} for v in
                       dataset_train.unprivileged_protected_attributes[sens_ind]]
privileged_groups = [{sens_attr: v} for v in
                     dataset_train.privileged_protected_attributes[sens_ind]]

describe(dataset_train, dataset_val, dataset_test)

CONNECTION_STRING = os.environ.get('CONNECTION_STRING')
def eval_model(model, dataset, unprivileged_groups, privileged_groups):
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

    return metrics


def meta_fair_classifier_sr(train, val, test, unprivileged_groups, privileged_groups, fitness_rule=None):
    return meta_fair_classifier(train, val, test, unprivileged_groups, privileged_groups, "sr", fitness_rule=fitness_rule)

def meta_fair_classifier_fdr(train, val, test, unprivileged_groups, privileged_groups, fitness_rule=None):
    return meta_fair_classifier(train, val, test, unprivileged_groups, privileged_groups, "fdr", fitness_rule=fitness_rule)

def meta_fair_classifier(train, val, test, unprivileged_groups, privileged_groups, type, fitness_rule=None):
    # training

    def objective(trial):
        tau = trial.suggest_float('tau', 0.01, 2.0)

        model = MetaFairClassifier(tau=tau, sensitive_attr=sens_attr, type=type)
        scaler = StandardScaler()

        scaled_train = train.copy()
        scaled_train.features = scaler.fit_transform(scaled_train.features)

        scaled_val = val.copy()
        scaled_val.features = scaler.transform(scaled_val.features)

        model = model.fit(scaled_train)

        val_metrics = eval_model(model, scaled_val, unprivileged_groups, privileged_groups)
        return fitness_rule(val_metrics)

    if fitness_rule is not None:
        # best solution
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        study = optuna.create_study(direction='maximize',
                                    study_name="meta_fair_classifier_" + now,
                                    storage=CONNECTION_STRING,
                                    pruner=get_pruner(),
                                    sampler=get_sampler())

        study.optimize(objective, n_trials=N_TRIALS, n_jobs=6)
        # eval on test set
        model = MetaFairClassifier(tau=study.best_params['tau'], sensitive_attr=sens_attr, type=type)
    else:
        model = MetaFairClassifier(sensitive_attr=sens_attr, type=type)

    scaler = StandardScaler()
    scaled_train = train.copy()
    scaled_train.features = scaler.fit_transform(scaled_train.features)

    scaled_val = val.copy()
    scaled_val.features = scaler.fit_transform(scaled_val.features)

    model = model.fit(scaled_train)
    model = model.fit(scaled_val)

    scaled_test = test.copy()
    scaled_test.features = scaler.transform(scaled_test.features)

    test_metrics = eval_model(model, scaled_test, unprivileged_groups, privileged_groups)

    if fitness_rule is not None:
        test_metrics['fitness_rule'] = fitness_rule.__name__
    else:
        test_metrics['fitness_rule'] = 'No optimization'

    print('-----------------------------------')
    print('Meta Fair Classifier - Test metrics')
    print('-----------------------------------')
    describe_metrics(test_metrics)
    test_metrics['method'] = 'meta_fair_classifier'
    test_metrics['method+fitness_rule'] = '%s+%s' % (test_metrics['method'], test_metrics['fitness_rule'])
    print('-----------------------------------')

    return test_metrics

def fair_mlp(train, val, test, unprivileged_groups, privileged_groups, fitness_rule=None):
    # training

    def objective(trial):
        privileged_demotion = trial.suggest_float('privileged_demotion', 0.0, 1.0)
        privileged_promotion = trial.suggest_float('privileged_promotion', 0.0, 1.0)
        protected_demotion = trial.suggest_float('protected_demotion', 0.0, 1.0)
        protected_promotion = trial.suggest_float('protected_promotion', 0.0, 1.0)

        model = FairTransitionLossMLP(sensitive_attr=sens_attr,
                                      hidden_sizes=[16, 32],
                                      batch_size=32,
                                      privileged_demotion=privileged_demotion, privileged_promotion=privileged_promotion,
                                      protected_demotion=protected_demotion, protected_promotion=protected_promotion)
        scaler = StandardScaler()

        scaled_train = train.copy()
        scaled_train.features = scaler.fit_transform(scaled_train.features)

        scaled_val = val.copy()
        scaled_val.features = scaler.transform(scaled_val.features)

        model = model.fit(scaled_train)

        val_metrics = eval_model(model, scaled_val, unprivileged_groups, privileged_groups)
        return fitness_rule(val_metrics)

    if fitness_rule is not None:
        # best solution
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        study = optuna.create_study(direction='maximize',
                                    study_name="fair_mlp_" + now,
                                    storage=CONNECTION_STRING,
                                    pruner=get_pruner(),
                                    sampler=get_sampler())

        study.optimize(objective, n_trials=N_TRIALS, n_jobs=6)
        # eval on test set
        model = FairTransitionLossMLP(sensitive_attr=sens_attr,
                                      hidden_sizes=[16, 32],
                                      batch_size=32,
                                      privileged_demotion=study.best_params['privileged_demotion'],
                                      privileged_promotion=study.best_params['privileged_promotion'],
                                      protected_demotion=study.best_params['protected_demotion'],
                                      protected_promotion=study.best_params['protected_promotion'])
    else:
        model = FairTransitionLossMLP(sensitive_attr=sens_attr,
                                      hidden_sizes=[16, 32],
                                      batch_size=32)

    scaler = StandardScaler()
    scaled_train = train.copy()
    scaled_train.features = scaler.fit_transform(scaled_train.features)

    scaled_val = val.copy()
    scaled_val.features = scaler.fit_transform(scaled_val.features)

    model = model.fit(scaled_train)
    model = model.fit(scaled_val)

    scaled_test = test.copy()
    scaled_test.features = scaler.transform(scaled_test.features)

    test_metrics = eval_model(model, scaled_test, unprivileged_groups, privileged_groups)

    if fitness_rule is not None:
        test_metrics['fitness_rule'] = fitness_rule.__name__
    else:
        test_metrics['fitness_rule'] = 'No optimization'

    print('-----------------------------------')
    print('Fair MLP - Test metrics')
    print('-----------------------------------')
    describe_metrics(test_metrics)
    test_metrics['method'] = 'fair_mlp'
    test_metrics['method+fitness_rule'] = '%s+%s' % (test_metrics['method'], test_metrics['fitness_rule'])
    print('-----------------------------------')

    return test_metrics



def prejudice_remover(train, val, test, unprivileged_groups, privileged_groups, fitness_rule=None):
    # training

    def objective(trial):
        eta = trial.suggest_float('eta', 0.01, 50.0)

        model = PrejudiceRemover(eta=eta, sensitive_attr=sens_attr)
        scaler = StandardScaler()

        scaled_train = train.copy()
        scaled_train.features = scaler.fit_transform(scaled_train.features)

        scaled_val = val.copy()
        scaled_val.features = scaler.transform(scaled_val.features)

        model = model.fit(scaled_train)

        val_metrics = eval_model(model, scaled_val, unprivileged_groups, privileged_groups)
        return fitness_rule(val_metrics)

    if fitness_rule is not None:
        # best solution
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        study = optuna.create_study(direction='maximize',
                                    study_name="prejudice_remover_" + now,
                                    storage=CONNECTION_STRING,
                                    sampler=get_sampler(),
                                    pruner=get_pruner())

        study.optimize(objective, n_trials=N_TRIALS, n_jobs=6)
        # eval on test set
        model = PrejudiceRemover(eta=study.best_params['eta'], sensitive_attr=sens_attr)
    else:
        model = PrejudiceRemover(sensitive_attr=sens_attr)

    scaler = StandardScaler()
    scaled_train = train.copy()
    scaled_train.features = scaler.fit_transform(scaled_train.features)

    scaled_val = val.copy()
    scaled_val.features = scaler.fit_transform(scaled_val.features)

    model = model.fit(scaled_train)
    model = model.fit(scaled_val)

    scaled_test = test.copy()
    scaled_test.features = scaler.transform(scaled_test.features)

    test_metrics = eval_model(model, scaled_test, unprivileged_groups, privileged_groups)

    if fitness_rule is not None:
        test_metrics['fitness_rule'] = fitness_rule.__name__
    else:
        test_metrics['fitness_rule'] = 'No optimization'

    print('-----------------------------------')
    print('Prejudice Remover - Test metrics')
    print('-----------------------------------')
    describe_metrics(test_metrics)
    test_metrics['method'] = 'prejudice_remover'
    test_metrics['method+fitness_rule'] = '%s+%s' % (test_metrics['method'], test_metrics['fitness_rule'])
    print('-----------------------------------')

    return test_metrics

def adversarial_debiasing(train, val, test, unprivileged_groups, privileged_groups, fitness_rule=None):
    # training

    def objective(trial):
        classifier_num_hidden_units = trial.suggest_categorical('classifier_num_hidden_units', [128, 256, 512])
        adversary_loss_weight = trial.suggest_float('adversary_loss_weight', 0.01, 0.2)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])

        model = AdversarialDebiasing(unprivileged_groups=unprivileged_groups,
                                     privileged_groups=privileged_groups,
                                     scope_name='adv_debias_' + str(datetime.now().timestamp()),
                                     sess=tf_old.Session(),
                                     classifier_num_hidden_units=classifier_num_hidden_units,
                                     adversary_loss_weight=adversary_loss_weight,
                                     batch_size=batch_size
                                     )
        scaler = StandardScaler()

        scaled_train = train.copy()
        scaled_train.features = scaler.fit_transform(scaled_train.features)

        scaled_val = val.copy()
        scaled_val.features = scaler.transform(scaled_val.features)

        model = model.fit(scaled_train)

        val_metrics = eval_model(model, scaled_val, unprivileged_groups, privileged_groups)
        return fitness_rule(val_metrics)

    if fitness_rule is not None:
        # best solution
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        study = optuna.create_study(direction='maximize',
                                    study_name="adversarial_debiasing_" + now,
                                    storage=CONNECTION_STRING,
                                    pruner=get_pruner(),
                                    sampler=get_sampler())

        study.optimize(objective, n_trials=N_TRIALS, n_jobs=6)
        # eval on test set
        model = AdversarialDebiasing(unprivileged_groups=unprivileged_groups,
                                     privileged_groups=privileged_groups,
                                     scope_name='adv_debias_' + str(datetime.now().timestamp()),
                                     sess=tf_old.Session(),
                                     classifier_num_hidden_units=study.best_params['classifier_num_hidden_units'],
                                     adversary_loss_weight=study.best_params['adversary_loss_weight'],
                                     batch_size=study.best_params['batch_size']
                                     )
    else:
        model = AdversarialDebiasing(unprivileged_groups=unprivileged_groups,
                                     privileged_groups=privileged_groups,
                                     scope_name='adv_debias_' + str(datetime.now().timestamp()),
                                     sess=tf_old.Session(),
                                     )

    scaler = StandardScaler()
    scaled_train = train.copy()
    scaled_train.features = scaler.fit_transform(scaled_train.features)

    scaled_val = val.copy()
    scaled_val.features = scaler.fit_transform(scaled_val.features)

    model = model.fit(scaled_train)
    model = model.fit(scaled_val)

    scaled_test = test.copy()
    scaled_test.features = scaler.transform(scaled_test.features)

    test_metrics = eval_model(model, scaled_test, unprivileged_groups, privileged_groups)

    if fitness_rule is not None:
        test_metrics['fitness_rule'] = fitness_rule.__name__
    else:
        test_metrics['fitness_rule'] = 'No optimization'

    print('-----------------------------------')
    print('Adversarial Debiasing - Test metrics')
    print('-----------------------------------')
    describe_metrics(test_metrics)
    test_metrics['method'] = 'adversarial_debiasing'
    test_metrics['method+fitness_rule'] = '%s+%s' % (test_metrics['method'], test_metrics['fitness_rule'])
    print('-----------------------------------')

    return test_metrics


def logistic_regression(train, val, test, unprivileged_groups, privileged_groups, fitness_rule=None):
    # training
    def objective(trial):

        solver = trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
        tol = trial.suggest_float('tol', 0.00001, 0.01, log=True)
        C = trial.suggest_float('C', 0.1, 10.0, log=True)


        model = make_pipeline(StandardScaler(),
                              LogisticRegression(solver=solver,
                                                 tol=tol, C=C,
                                                 random_state=1))
        fit_params = {'logisticregression__sample_weight': train.instance_weights}

        model = model.fit(train.features, train.labels.ravel(), **fit_params)

        val_metrics = eval_model(model, val, unprivileged_groups, privileged_groups)
        return fitness_rule(val_metrics)

    if fitness_rule is not None:

        # best solution
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        study = optuna.create_study(direction='maximize',
                                    study_name="logistic_regression_" + now,
                                    storage=CONNECTION_STRING,
                                    pruner=get_pruner(),
                                    sampler=get_sampler())

        study.optimize(objective, n_trials=N_TRIALS, n_jobs=6)

        # eval on test set
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(solver=study.best_params['solver'],
                                                 tol=study.best_params['tol'],
                                                 C=study.best_params['C'],
                                                 random_state=1))
    else:
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(random_state=1))

    fit_params = {'logisticregression__sample_weight': train.instance_weights}
    model = model.fit(train.features, train.labels.ravel(), **fit_params)
    fit_params = {'logisticregression__sample_weight': val.instance_weights}
    model = model.fit(val.features, val.labels.ravel(), **fit_params)

    test_metrics = eval_model(model, test, unprivileged_groups, privileged_groups)

    if fitness_rule is not None:
        test_metrics['fitness_rule'] = fitness_rule.__name__
    else:
        test_metrics['fitness_rule'] = 'No optimization'


    print('-----------------------------------')
    print('Logistc Regression - Test metrics')
    print('-----------------------------------')
    describe_metrics(test_metrics)
    test_metrics['method'] = 'logistic_regression'
    test_metrics['method+fitness_rule'] = '%s+%s' % (test_metrics['method'], test_metrics['fitness_rule'])
    print('-----------------------------------')
    return test_metrics

def plot_comparison(results, name):
    results_df = pd.DataFrame(results)
    my_cmap = plt.cm.get_cmap('Dark2')
    n_bars = len(results_df.keys()) - 1
    colors = my_cmap([x for x in range(n_bars)])
    print(results_df)
    fig, axis = plt.subplots(3, 1, figsize=(14, 12), sharex=True,
                             gridspec_kw={'height_ratios': [1, 2, 2]})
    fig.suptitle("Comparação de Regras de Otimização em Fairness\n%s" % name, fontsize=16)
    axis[0].axis('off')
    axis[1] = results_df[['overall_acc', 'bal_acc', 'fitness_rule']] \
        .plot.bar(x='fitness_rule', rot=0, ax=axis[1], grid=True, legend=False, color=colors[:3])
    axis[2] = results_df[['avg_odds_diff', 'stat_par_diff', 'eq_opp_diff', 'fitness_rule']] \
        .plot.bar(x='fitness_rule', rot=0, ax=axis[2], grid=True, legend=False, color=colors[3:])
    bars_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    bars, labels = [sum(lol, []) for lol in zip(*bars_labels)]
    metric_explain = get_metric_explain()
    labels = [metric_explain[label] for label in labels]
    fig.legend(bars, labels,
               loc='lower left', bbox_to_anchor=(0.03, 0.78),
               fancybox=True, shadow=False, ncol=1,
               fontsize='x-large')
    fig.tight_layout()
    fig.savefig('%s_plot.png' % name, dpi=300)

def smooth_parity(metrics):
    acc = metrics['overall_acc']
    par = metrics['stat_par_diff']
    return -par - (acc-1)**2

def smooth_odds(metrics):
    acc = metrics['overall_acc']
    odds = metrics['avg_odds_diff']
    return -odds - (acc-1)**2

def smooth_opportunity(metrics):
    acc = metrics['overall_acc']
    opp = metrics['eq_opp_diff']
    return -opp - (acc-1)**2

def linear_parity(metrics):
    acc = metrics['overall_acc']
    par = metrics['stat_par_diff']
    return acc - abs(par)

def linear_odds(metrics):
    acc = metrics['overall_acc']
    odds = metrics['avg_odds_diff']
    return acc - abs(odds)

def linear_opportunity(metrics):
    acc = metrics['overall_acc']
    opp = metrics['eq_opp_diff']
    return acc - abs(opp)

def get_metric_explain():
    return {
        'overall_acc': 'Accuracy - acurácia padrão',
        'bal_acc': 'Balanced Accuracy - média entre taxa de verdadeiros positivos e verdadeiros negativos',
        'avg_odds_diff': 'Equalized Odds - diferença de verdadeiros e falsos positivos dentre os grupos',
        'disp_imp': 'Disparate Impact - razão de previsões positivas dentre os grupos',
        'stat_par_diff': 'Statistical Parity - diferença de previsões positivas dentre os grupos',
        'eq_opp_diff': 'Equal Opportunity - diferença de falsos negativos dentre os grupos'
    }

def describe_metrics(metrics):
    best_ind = np.argmax(metrics['bal_acc'])

    print("Best balanced accuracy: {:6.4f}".format(metrics['bal_acc']))
    print("overall accuracy: {:6.4f}".format(metrics['overall_acc']))
#     disp_imp_at_best_ind = np.abs(1 - np.array(metrics['disp_imp']))
    disp_imp_at_best_ind = 1 - min(metrics['disp_imp'], 1/metrics['disp_imp'])
    print("Corresponding 1-min(DI, 1/DI) value: {:6.4f}".format(disp_imp_at_best_ind))
    print("Corresponding average odds difference value: {:6.4f}".format(metrics['avg_odds_diff']))
    print("Corresponding statistical parity difference value: {:6.4f}".format(metrics['stat_par_diff']))
    print("Corresponding equal opportunity difference value: {:6.4f}".format(metrics['eq_opp_diff']))
    print("Corresponding Theil index value: {:6.4f}".format(metrics['theil_ind']))

models = [
    fair_mlp,
    prejudice_remover,
    meta_fair_classifier_sr,
    meta_fair_classifier_fdr
]

fitness_rules = [
    #None,
    smooth_parity,
    smooth_odds,
    smooth_opportunity,
    linear_parity,
    linear_odds,
    linear_opportunity
]



for model in models:
    results = []
    name = model.__name__
    for fitness_rule in fitness_rules:
        result = model(dataset_train, dataset_val, dataset_test, unprivileged_groups, privileged_groups, fitness_rule)
        results.append(result)
        plot_comparison(results, name)
        pd_results = pd.DataFrame(results)
        pd_results.to_csv('%s_results.csv' % name)


