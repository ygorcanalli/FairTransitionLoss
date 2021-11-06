import numpy as np
from collections import defaultdict
from aif360.datasets import AdultDataset
from util import describe, describe_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from aif360.metrics import ClassificationMetric
# Bias mitigation techniques
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import PrejudiceRemover
# Fair loss
from models import FairMLP, SimpleMLP
from util import eval_model

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
    describe_metrics(test_metrics, [thresh_arr[best_ind]])
    print('-----------------------------------')


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
    describe_metrics(test_metrics, [thresh_arr[best_ind]])
    print('-----------------------------------')


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
    describe_metrics(test_metrics, [thresh_arr[best_ind]])
    print('-----------------------------------')


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
    describe_metrics(test_metrics, [thresh_arr[best_ind]])
    print('-----------------------------------')


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
    describe_metrics(test_metrics, [thresh_arr[best_ind]])
    print('-----------------------------------')


def eval_fair_loss_mlp(train, val, test, unprivileged_groups, privileged_groups):
    # training
    model = FairMLP(sensitive_attr=sens_attr,
                    hidden_sizes=[16, 32],
                    batch_size=32,
                    privileged_demotion=0.77, privileged_promotion=0.22,
                    protected_demotion=0.32, protected_promotion=0.03)

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
    #test_metrics = eval_model(model, scaled_test, [thresh_arr[best_ind]], unprivileged_groups, privileged_groups)
    test_metrics = eval_model(model, scaled_test, [0.5], unprivileged_groups, privileged_groups)
    print('-----------------------------------')
    print('Fair Loss MLP - Test metrics')
    print('-----------------------------------')
    describe_metrics(test_metrics, [thresh_arr[best_ind]])
    print('-----------------------------------')


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
    describe_metrics(test_metrics, [thresh_arr[best_ind]])
    print('-----------------------------------')

eval_fair_loss_mlp(dataset_train, dataset_val, dataset_test, unprivileged_groups, privileged_groups)
eval_simple_mlp(dataset_train, dataset_val, dataset_test, unprivileged_groups, privileged_groups)
eval_logistic_regression(dataset_train, dataset_val, dataset_test, unprivileged_groups, privileged_groups)
eval_random_forest(dataset_train, dataset_val, dataset_test, unprivileged_groups, privileged_groups)
eval_logistic_regression_reweighting(dataset_train, dataset_val, dataset_test, unprivileged_groups, privileged_groups)
eval_random_forest_reweighting(dataset_train, dataset_val, dataset_test, unprivileged_groups, privileged_groups)
eval_prejudice_remover(dataset_train, dataset_val, dataset_test, unprivileged_groups, privileged_groups)
