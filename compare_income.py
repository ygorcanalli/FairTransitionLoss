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
from forward import FairMLP


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
    y_val_pred_prob = model.predict_proba(val.features)
    val_pos_ind = np.where(model.classes_ == val.favorable_label)[0][0]

    val_metrics = defaultdict(list)
    for thresh in thresh_arr:
        y_val_pred = (y_val_pred_prob[:, val_pos_ind] > thresh).astype(np.float64)

        dataset_pred = val.copy()
        dataset_pred.labels = y_val_pred
        metric = ClassificationMetric(
            val, dataset_pred,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)

        val_metrics['bal_acc'].append((metric.true_positive_rate()
                                       + metric.true_negative_rate()) / 2)
        val_metrics['avg_odds_diff'].append(metric.average_odds_difference())
        val_metrics['disp_imp'].append(metric.disparate_impact())
        val_metrics['stat_par_diff'].append(metric.statistical_parity_difference())
        val_metrics['eq_opp_diff'].append(metric.equal_opportunity_difference())
        val_metrics['theil_ind'].append(metric.theil_index())

    # best solution
    best_ind = np.argmax(val_metrics['bal_acc'])

    # evalutate on test set
    y_test_pred_prob = model.predict_proba(test.features)
    test_pos_ind = np.where(model.classes_ == test.favorable_label)[0][0]

    best_thresh = thresh_arr[best_ind]
    y_test_pred = (y_test_pred_prob[:, test_pos_ind] > best_thresh).astype(np.float64)

    test_metrics = defaultdict(list)
    dataset_pred = test.copy()
    dataset_pred.labels = y_test_pred
    metric = ClassificationMetric(
        test, dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)

    test_metrics['bal_acc'].append((metric.true_positive_rate()
                                   + metric.true_negative_rate()) / 2)
    test_metrics['avg_odds_diff'].append(metric.average_odds_difference())
    test_metrics['disp_imp'].append(metric.disparate_impact())
    test_metrics['stat_par_diff'].append(metric.statistical_parity_difference())
    test_metrics['eq_opp_diff'].append(metric.equal_opportunity_difference())
    test_metrics['theil_ind'].append(metric.theil_index())

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
    y_val_pred_prob = model.predict_proba(val.features)
    val_pos_ind = np.where(model.classes_ == val.favorable_label)[0][0]

    val_metrics = defaultdict(list)
    for thresh in thresh_arr:
        y_val_pred = (y_val_pred_prob[:, val_pos_ind] > thresh).astype(np.float64)

        dataset_pred = val.copy()
        dataset_pred.labels = y_val_pred
        metric = ClassificationMetric(
            val, dataset_pred,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)

        val_metrics['bal_acc'].append((metric.true_positive_rate()
                                       + metric.true_negative_rate()) / 2)
        val_metrics['avg_odds_diff'].append(metric.average_odds_difference())
        val_metrics['disp_imp'].append(metric.disparate_impact())
        val_metrics['stat_par_diff'].append(metric.statistical_parity_difference())
        val_metrics['eq_opp_diff'].append(metric.equal_opportunity_difference())
        val_metrics['theil_ind'].append(metric.theil_index())

    # best solution
    best_ind = np.argmax(val_metrics['bal_acc'])

    # evalutate on test set
    y_test_pred_prob = model.predict_proba(test.features)
    test_pos_ind = np.where(model.classes_ == test.favorable_label)[0][0]

    best_thresh = thresh_arr[best_ind]
    y_test_pred = (y_test_pred_prob[:, test_pos_ind] > best_thresh).astype(np.float64)

    test_metrics = defaultdict(list)
    dataset_pred = test.copy()
    dataset_pred.labels = y_test_pred
    metric = ClassificationMetric(
        test, dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)

    test_metrics['bal_acc'].append((metric.true_positive_rate()
                                   + metric.true_negative_rate()) / 2)
    test_metrics['avg_odds_diff'].append(metric.average_odds_difference())
    test_metrics['disp_imp'].append(metric.disparate_impact())
    test_metrics['stat_par_diff'].append(metric.statistical_parity_difference())
    test_metrics['eq_opp_diff'].append(metric.equal_opportunity_difference())
    test_metrics['theil_ind'].append(metric.theil_index())

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
    y_val_pred_prob = model.predict_proba(val.features)
    val_pos_ind = np.where(model.classes_ == val.favorable_label)[0][0]

    val_metrics = defaultdict(list)
    for thresh in thresh_arr:
        y_val_pred = (y_val_pred_prob[:, val_pos_ind] > thresh).astype(np.float64)

        dataset_pred = val.copy()
        dataset_pred.labels = y_val_pred
        metric = ClassificationMetric(
            val, dataset_pred,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)

        val_metrics['bal_acc'].append((metric.true_positive_rate()
                                       + metric.true_negative_rate()) / 2)
        val_metrics['avg_odds_diff'].append(metric.average_odds_difference())
        val_metrics['disp_imp'].append(metric.disparate_impact())
        val_metrics['stat_par_diff'].append(metric.statistical_parity_difference())
        val_metrics['eq_opp_diff'].append(metric.equal_opportunity_difference())
        val_metrics['theil_ind'].append(metric.theil_index())

    # best solution
    best_ind = np.argmax(val_metrics['bal_acc'])

    # evalutate on test set
    y_test_pred_prob = model.predict_proba(test.features)
    test_pos_ind = np.where(model.classes_ == test.favorable_label)[0][0]

    best_thresh = thresh_arr[best_ind]
    y_test_pred = (y_test_pred_prob[:, test_pos_ind] > best_thresh).astype(np.float64)

    test_metrics = defaultdict(list)
    dataset_pred = test.copy()
    dataset_pred.labels = y_test_pred
    metric = ClassificationMetric(
        test, dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)

    test_metrics['bal_acc'].append((metric.true_positive_rate()
                                   + metric.true_negative_rate()) / 2)
    test_metrics['avg_odds_diff'].append(metric.average_odds_difference())
    test_metrics['disp_imp'].append(metric.disparate_impact())
    test_metrics['stat_par_diff'].append(metric.statistical_parity_difference())
    test_metrics['eq_opp_diff'].append(metric.equal_opportunity_difference())
    test_metrics['theil_ind'].append(metric.theil_index())

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
    y_val_pred_prob = model.predict_proba(val.features)
    val_pos_ind = np.where(model.classes_ == val.favorable_label)[0][0]

    val_metrics = defaultdict(list)
    for thresh in thresh_arr:
        y_val_pred = (y_val_pred_prob[:, val_pos_ind] > thresh).astype(np.float64)

        dataset_pred = val.copy()
        dataset_pred.labels = y_val_pred
        metric = ClassificationMetric(
            val, dataset_pred,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)

        val_metrics['bal_acc'].append((metric.true_positive_rate()
                                       + metric.true_negative_rate()) / 2)
        val_metrics['avg_odds_diff'].append(metric.average_odds_difference())
        val_metrics['disp_imp'].append(metric.disparate_impact())
        val_metrics['stat_par_diff'].append(metric.statistical_parity_difference())
        val_metrics['eq_opp_diff'].append(metric.equal_opportunity_difference())
        val_metrics['theil_ind'].append(metric.theil_index())

    # best solution
    best_ind = np.argmax(val_metrics['bal_acc'])

    # evalutate on test set
    y_test_pred_prob = model.predict_proba(test.features)
    test_pos_ind = np.where(model.classes_ == test.favorable_label)[0][0]

    best_thresh = thresh_arr[best_ind]
    y_test_pred = (y_test_pred_prob[:, test_pos_ind] > best_thresh).astype(np.float64)

    test_metrics = defaultdict(list)
    dataset_pred = test.copy()
    dataset_pred.labels = y_test_pred
    metric = ClassificationMetric(
        test, dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)

    test_metrics['bal_acc'].append((metric.true_positive_rate()
                                   + metric.true_negative_rate()) / 2)
    test_metrics['avg_odds_diff'].append(metric.average_odds_difference())
    test_metrics['disp_imp'].append(metric.disparate_impact())
    test_metrics['stat_par_diff'].append(metric.statistical_parity_difference())
    test_metrics['eq_opp_diff'].append(metric.equal_opportunity_difference())
    test_metrics['theil_ind'].append(metric.theil_index())

    print('-----------------------------------')
    print('Random Forest + Reweighing - Test metrics')
    print('-----------------------------------')
    describe_metrics(test_metrics, [thresh_arr[best_ind]])
    print('-----------------------------------')
def eval_prejudice_remover(train, val, test, unprivileged_groups, privileged_groups):
    # training
    model = PrejudiceRemover(sensitive_attr=sens_attr, eta=25.0)
    pr_orig_scaler = StandardScaler()

    dataset = train.copy()
    dataset.features = pr_orig_scaler.fit_transform(dataset.features)

    model = model.fit(dataset)

    # hyperparameter tunning in validation set
    thresh_arr = np.linspace(0.01, 0.5, 50)
    dataset = val.copy()
    y_val_pred_prob = model.predict(dataset).scores
    val_pos_ind = 0

    val_metrics = defaultdict(list)
    for thresh in thresh_arr:
        y_val_pred = (y_val_pred_prob[:, val_pos_ind] > thresh).astype(np.float64)

        dataset_pred = val.copy()
        dataset_pred.labels = y_val_pred
        metric = ClassificationMetric(
            val, dataset_pred,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)

        val_metrics['bal_acc'].append((metric.true_positive_rate()
                                       + metric.true_negative_rate()) / 2)
        val_metrics['avg_odds_diff'].append(metric.average_odds_difference())
        val_metrics['disp_imp'].append(metric.disparate_impact())
        val_metrics['stat_par_diff'].append(metric.statistical_parity_difference())
        val_metrics['eq_opp_diff'].append(metric.equal_opportunity_difference())
        val_metrics['theil_ind'].append(metric.theil_index())

    # best solution
    best_ind = np.argmax(val_metrics['bal_acc'])

    # evalutate on test set
    dataset = test.copy()
    dataset.features = pr_orig_scaler.transform(dataset.features)
    y_test_pred_prob = model.predict(dataset).scores
    test_pos_ind = 0

    best_thresh = thresh_arr[best_ind]
    y_test_pred = (y_test_pred_prob[:, test_pos_ind] > best_thresh).astype(np.float64)

    test_metrics = defaultdict(list)
    dataset_pred = test.copy()
    dataset_pred.labels = y_test_pred
    metric = ClassificationMetric(
        test, dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)

    test_metrics['bal_acc'].append((metric.true_positive_rate()
                                   + metric.true_negative_rate()) / 2)
    test_metrics['avg_odds_diff'].append(metric.average_odds_difference())
    test_metrics['disp_imp'].append(metric.disparate_impact())
    test_metrics['stat_par_diff'].append(metric.statistical_parity_difference())
    test_metrics['eq_opp_diff'].append(metric.equal_opportunity_difference())
    test_metrics['theil_ind'].append(metric.theil_index())

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
                    privileged_demotion=0.0, privileged_promotion=0.0,
                    protected_demotion=0.0, protected_promotion=0.0,)
    pr_orig_scaler = StandardScaler()

    dataset = train.copy()
    dataset.features = pr_orig_scaler.fit_transform(dataset.features)

    model.fit(dataset)

    # hyperparameter tunning in validation set
    thresh_arr = np.linspace(0.01, 0.5, 50)
    dataset = val.copy()
    y_val_pred_prob = model.predict_proba(dataset.features)
    val_pos_ind = 1

    val_metrics = defaultdict(list)
    for thresh in thresh_arr:
        y_val_pred = (y_val_pred_prob[:, val_pos_ind] > thresh).astype(np.float64)

        dataset_pred = val.copy()
        dataset_pred.labels = y_val_pred
        metric = ClassificationMetric(
            val, dataset_pred,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)

        val_metrics['bal_acc'].append((metric.true_positive_rate()
                                       + metric.true_negative_rate()) / 2)
        val_metrics['avg_odds_diff'].append(metric.average_odds_difference())
        val_metrics['disp_imp'].append(metric.disparate_impact())
        val_metrics['stat_par_diff'].append(metric.statistical_parity_difference())
        val_metrics['eq_opp_diff'].append(metric.equal_opportunity_difference())
        val_metrics['theil_ind'].append(metric.theil_index())

    # best solution
    best_ind = np.argmax(val_metrics['bal_acc'])

    # evalutate on test set
    dataset = test.copy()
    dataset.features = pr_orig_scaler.transform(dataset.features)
    y_test_pred_prob = model.predict_proba(dataset.features)
    test_pos_ind = 1

    best_thresh = thresh_arr[best_ind]
    y_test_pred = (y_test_pred_prob[:, test_pos_ind] > best_thresh).astype(np.float64)

    test_metrics = defaultdict(list)
    dataset_pred = test.copy()
    dataset_pred.labels = y_test_pred
    metric = ClassificationMetric(
        test, dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)

    test_metrics['bal_acc'].append((metric.true_positive_rate()
                                   + metric.true_negative_rate()) / 2)
    test_metrics['avg_odds_diff'].append(metric.average_odds_difference())
    test_metrics['disp_imp'].append(metric.disparate_impact())
    test_metrics['stat_par_diff'].append(metric.statistical_parity_difference())
    test_metrics['eq_opp_diff'].append(metric.equal_opportunity_difference())
    test_metrics['theil_ind'].append(metric.theil_index())

    print('-----------------------------------')
    print('Fair Loss MLP - Test metrics')
    print('-----------------------------------')
    describe_metrics(test_metrics, [thresh_arr[best_ind]])
    print('-----------------------------------')
eval_fair_loss_mlp(dataset_train, dataset_val, dataset_test, unprivileged_groups, privileged_groups)
eval_logistic_regression(dataset_train, dataset_val, dataset_test, unprivileged_groups, privileged_groups)
eval_random_forest(dataset_train, dataset_val, dataset_test, unprivileged_groups, privileged_groups)
eval_logistic_regression_reweighting(dataset_train, dataset_val, dataset_test, unprivileged_groups, privileged_groups)
eval_random_forest_reweighting(dataset_train, dataset_val, dataset_test, unprivileged_groups, privileged_groups)
eval_prejudice_remover(dataset_train, dataset_val, dataset_test, unprivileged_groups, privileged_groups)



