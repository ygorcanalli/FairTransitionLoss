import numpy as np
from matplotlib import cm
from collections import defaultdict
from multiprocessing import Pool
import os


from aif360.datasets import AdultDataset
from util import describe, describe_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# Bias mitigation techniques
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import PrejudiceRemover, AdversarialDebiasing, MetaFairClassifier
# Fair loss
from models import FairMLP, SimpleMLP
from util import eval_model, plot_comparison, get_ga_instance, fitness_rule_a, fitness_rule_b, fitness_rule_c


import time
from multiprocessing import Pool

device = 'cpu'
if device == 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import tensorflow.compat.v1 as tf_old

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


def eval_logistic_regression(train, val, test, unprivileged_groups, privileged_groups):
    # training
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(solver='liblinear', random_state=1))
    fit_params = {'logisticregression__sample_weight': train.instance_weights}

    model = model.fit(train.features, train.labels.ravel(), **fit_params)

    # hyperparameter tunning in validation set
    thresh_arr = np.linspace(0.01, 0.5, 50)
    val_metrics = eval_model(model, val, thresh_arr, unprivileged_groups, privileged_groups)
    # best solution
    best_ind = np.argmax(val_metrics['bal_acc'])
    # eval 0n test set
    test_metrics = eval_model(model, test, [thresh_arr[best_ind]], unprivileged_groups, privileged_groups)

    print('-----------------------------------')
    print('Logistc Regression - Test metrics')
    print('-----------------------------------')
    best_metrics = describe_metrics(test_metrics, [thresh_arr[best_ind]])
    best_metrics['method'] = 'logistic_regression'
    print('-----------------------------------')
    return best_metrics


def eval_random_forest(train, val, test, unprivileged_groups, privileged_groups):
    # training
    model = make_pipeline(StandardScaler(),
                          RandomForestClassifier(n_estimators=500, min_samples_leaf=25))
    fit_params = {'randomforestclassifier__sample_weight': train.instance_weights}

    model = model.fit(train.features, train.labels.ravel(), **fit_params)

    # hyperparameter tunning in validation set
    thresh_arr = np.linspace(0.01, 0.5, 50)
    val_metrics = eval_model(model, val, thresh_arr, unprivileged_groups, privileged_groups)
    # best solution
    best_ind = np.argmax(val_metrics['bal_acc'])
    # eval 0n test set
    test_metrics = eval_model(model, test, [thresh_arr[best_ind]], unprivileged_groups, privileged_groups)

    print('-----------------------------------')
    print('Random Forest - Test metrics')
    print('-----------------------------------')
    best_metrics = describe_metrics(test_metrics, [thresh_arr[best_ind]])
    best_metrics['method'] = 'random_forest'
    print('-----------------------------------')
    return best_metrics


def eval_logistic_regression_reweighting(train, val, test, unprivileged_groups, privileged_groups):
    # training
    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)
    rw_train = RW.fit_transform(train.copy())

    model = make_pipeline(StandardScaler(),
                          LogisticRegression(solver='liblinear', random_state=1))
    fit_params = {'logisticregression__sample_weight': rw_train.instance_weights}

    model = model.fit(rw_train.features, rw_train.labels.ravel(), **fit_params)

    # hyperparameter tunning in validation set
    thresh_arr = np.linspace(0.01, 0.5, 50)
    val_metrics = eval_model(model, val, thresh_arr, unprivileged_groups, privileged_groups)
    # best solution
    best_ind = np.argmax(val_metrics['bal_acc'])
    # eval 0n test set
    test_metrics = eval_model(model, test, [thresh_arr[best_ind]], unprivileged_groups, privileged_groups)

    print('-----------------------------------')
    print('Logistic Regression + Reweighing - Test metrics')
    print('-----------------------------------')
    best_metrics = describe_metrics(test_metrics, [thresh_arr[best_ind]])
    best_metrics['method'] = 'logistic_regression_rw'
    print('-----------------------------------')
    return best_metrics


def eval_random_forest_reweighting(train, val, test, unprivileged_groups, privileged_groups):
    # training
    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)
    rw_train = RW.fit_transform(train.copy())

    model = make_pipeline(StandardScaler(),
                          RandomForestClassifier(n_estimators=500, min_samples_leaf=25))
    fit_params = {'randomforestclassifier__sample_weight': rw_train.instance_weights}

    model = model.fit(rw_train.features, rw_train.labels.ravel(), **fit_params)

    # hyperparameter tunning in validation set
    thresh_arr = np.linspace(0.01, 0.5, 50)
    val_metrics = eval_model(model, val, thresh_arr, unprivileged_groups, privileged_groups)
    # best solution
    best_ind = np.argmax(val_metrics['bal_acc'])
    # eval 0n test set
    test_metrics = eval_model(model, test, [thresh_arr[best_ind]], unprivileged_groups, privileged_groups)

    print('-----------------------------------')
    print('Random Forest + Reweighing - Test metrics')
    print('-----------------------------------')
    best_metrics = describe_metrics(test_metrics, [thresh_arr[best_ind]])
    best_metrics['method'] = 'random_forest_rw'
    print('-----------------------------------')
    return best_metrics


def eval_prejudice_remover(train, val, test, unprivileged_groups, privileged_groups):
    # training
    model = PrejudiceRemover(sensitive_attr=sens_attr, eta=25.0)
    scaler = StandardScaler()

    scaled_train = train.copy()
    scaled_train.features = scaler.fit_transform(scaled_train.features)

    scaled_val = val.copy()
    scaled_test = test.copy()
    scaled_val.features = scaler.fit_transform(scaled_val.features)
    scaled_test.features = scaler.fit_transform(scaled_test.features)

    model = model.fit(scaled_train)

    # hyperparameter tunning in validation set
    thresh_arr = np.linspace(0.01, 0.5, 50)
    val_metrics = eval_model(model, scaled_val, thresh_arr, unprivileged_groups, privileged_groups)
    # best solution
    best_ind = np.argmax(val_metrics['bal_acc'])
    # eval 0n test set
    test_metrics = eval_model(model, scaled_test, [thresh_arr[best_ind]], unprivileged_groups, privileged_groups)

    print('-----------------------------------')
    print('Prejudice Remover - Test metrics')
    print('-----------------------------------')
    best_metrics = describe_metrics(test_metrics, [thresh_arr[best_ind]])
    best_metrics['method'] = 'prejudice_remover'
    print('-----------------------------------')
    return best_metrics

def eval_meta_fair_classifier(train, val, test, unprivileged_groups, privileged_groups):
    # training
    model = MetaFairClassifier(sensitive_attr=sens_attr, type="sr")
    scaler = StandardScaler()

    scaled_train = train.copy()
    scaled_train.features = scaler.fit_transform(scaled_train.features)

    scaled_val = val.copy()
    scaled_test = test.copy()
    scaled_val.features = scaler.fit_transform(scaled_val.features)
    scaled_test.features = scaler.fit_transform(scaled_test.features)

    model = model.fit(scaled_train)

    # hyperparameter tunning in validation set
    thresh_arr = np.linspace(0.01, 0.5, 50)
    val_metrics = eval_model(model, scaled_val, thresh_arr, unprivileged_groups, privileged_groups)
    # best solution
    best_ind = np.argmax(val_metrics['bal_acc'])
    # eval 0n test set
    test_metrics = eval_model(model, scaled_test, [thresh_arr[best_ind]], unprivileged_groups, privileged_groups)

    print('-----------------------------------')
    print('Meta Fair Classifier - Test metrics')
    print('-----------------------------------')
    best_metrics = describe_metrics(test_metrics, [thresh_arr[best_ind]])
    best_metrics['method'] = 'meta_fair_classifier'
    print('-----------------------------------')
    return best_metrics


def eval_adversarial_debiasing(train, val, test, unprivileged_groups, privileged_groups):
    # training
    tf_old.disable_eager_execution()
    model = AdversarialDebiasing(unprivileged_groups=unprivileged_groups,
                                 privileged_groups=privileged_groups,
                                 scope_name='adv_debias',
                                 sess=tf_old.Session())
    scaler = StandardScaler()

    scaled_train = train.copy()
    scaled_train.features = scaler.fit_transform(scaled_train.features)

    scaled_val = val.copy()
    scaled_test = test.copy()
    scaled_val.features = scaler.fit_transform(scaled_val.features)
    scaled_test.features = scaler.fit_transform(scaled_test.features)

    model = model.fit(scaled_train)

    # hyperparameter tunning in validation set
    thresh_arr = np.linspace(0.01, 0.5, 50)
    val_metrics = eval_model(model, scaled_val, thresh_arr, unprivileged_groups, privileged_groups)
    # best solution
    best_ind = np.argmax(val_metrics['bal_acc'])
    # eval 0n test set
    test_metrics = eval_model(model, scaled_test, [thresh_arr[best_ind]], unprivileged_groups, privileged_groups)

    print('-----------------------------------')
    print('Adversarial Debiasing - Test metrics')
    print('-----------------------------------')
    best_metrics = describe_metrics(test_metrics, [thresh_arr[best_ind]])
    best_metrics['method'] = 'adversarial_debiasing'
    print('-----------------------------------')
    return best_metrics

def eval_fair_loss_mlp(train, val, test, unprivileged_groups, privileged_groups):
    # training

    scaler = StandardScaler()
    scaled_train = train.copy()
    scaled_train.features = scaler.fit_transform(scaled_train.features)

    scaled_val = val.copy()
    scaled_test = test.copy()

    scaled_val.features = scaler.fit_transform(scaled_val.features)
    scaled_test.features = scaler.fit_transform(scaled_test.features)
    thresh_arr = [0.5]


    def fitness_function(solution, solution_idx):
        model = FairMLP(sensitive_attr=sens_attr,
                        hidden_sizes=[16, 32],
                        batch_size=32,
                        privileged_demotion=solution[0], privileged_promotion=solution[1],
                        protected_demotion=solution[2], protected_promotion=solution[3])

        model = model.fit(scaled_train)
        val_metrics = eval_model(model, scaled_val, thresh_arr, unprivileged_groups, privileged_groups)
        best_metrics = describe_metrics(val_metrics, thresh_arr)

        fitness = fitness_rule_c(best_metrics)
        print('Fitness', fitness)
        print('-----------------------')
        return fitness

    ga_instance = get_ga_instance(fitness_function)
    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=1 / solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

    # After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
    ga_instance.plot_result(title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)

    best_solution = ga_instance.best_solution
    best_model = FairMLP(sensitive_attr=sens_attr,
                    hidden_sizes=[16, 32],
                    batch_size=32,
                    privileged_demotion=solution[0], privileged_promotion=solution[1],
                    protected_demotion=solution[2], protected_promotion=solution[3])

    best_model = best_model.fit(scaled_train)
    best_model = best_model.fit(scaled_val)

    test_metrics = eval_model(best_model, scaled_test, thresh_arr, unprivileged_groups, privileged_groups)
    print('-----------------------------------')
    print('Fair Loss MLP - Test metrics')
    print('-----------------------------------')
    best_metrics = describe_metrics(test_metrics, thresh_arr)
    best_metrics['method'] = 'fair_loss_mlp'
    print('-----------------------------------')
    return best_metrics


def eval_simple_mlp(train, val, test, unprivileged_groups, privileged_groups):
    # training
    model = SimpleMLP(sensitive_attr=sens_attr,
                      hidden_sizes=[16, 32],
                      batch_size=32)
    scaler = StandardScaler()

    scaled_train = train.copy()
    scaled_train.features = scaler.fit_transform(scaled_train.features)

    scaled_val = val.copy()
    scaled_test = test.copy()
    scaled_val.features = scaler.fit_transform(scaled_val.features)
    scaled_test.features = scaler.fit_transform(scaled_test.features)

    model = model.fit(scaled_train)

    # hyperparameter tunning in validation set
    thresh_arr = np.linspace(0.01, 0.5, 50)
    val_metrics = eval_model(model, scaled_val, thresh_arr, unprivileged_groups, privileged_groups)
    # best solution
    best_ind = np.argmax(val_metrics['bal_acc'])
    # eval 0n test set
    test_metrics = eval_model(model, scaled_test, [thresh_arr[best_ind]], unprivileged_groups, privileged_groups)

    print('-----------------------------------')
    print('Simple MLP - Test metrics')
    print('-----------------------------------')
    best_metrics = describe_metrics(test_metrics, [thresh_arr[best_ind]])
    best_metrics['method'] = 'mlp'
    print('-----------------------------------')
    return best_metrics


functions = [
    eval_fair_loss_mlp,
    #eval_simple_mlp,
    #eval_logistic_regression,
    #eval_random_forest,
    #eval_logistic_regression_reweighting,
    #eval_random_forest_reweighting,
    #eval_prejudice_remover,
    #eval_adversarial_debiasing,
    eval_meta_fair_classifier
]

results = []

for eval in functions:
    results.append(eval(dataset_train, dataset_val, dataset_test, unprivileged_groups, privileged_groups))
    plot_comparison(results)


