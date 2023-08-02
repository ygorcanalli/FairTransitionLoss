


def eval(model, dataset, unprivileged_groups, privileged_groups, fitness_rule):
    try:
        # sklearn classifier
        y_pred_prob = model.predict_proba(dataset.features)
        pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
        y_pred = (y_pred_prob[:, pos_ind] > 0.5).astype(np.float64)
    except AttributeError:
        # aif360 inprocessing algorithm
        y_pred = model.predict(dataset).labels

    dataset_pred = dataset.copy()
    dataset_pred.labels = y_pred
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
    metrics['f1_score'] = metric.f1_score()
    metrics.update(metric.performance_measures())
    metrics['fitness'] = fitness_rule(metrics)
    return metrics

data = CompasDataset()
(dataset_expanded_train,
 dataset_test) = data.split([0.8], shuffle=True)

(dataset_train,
 dataset_val) = dataset_expanded_train.split([0.8], shuffle=True)
sens_ind = 1
sens_attr = dataset_train.protected_attribute_names[sens_ind]

unprivileged_groups = [{sens_attr: v} for v in
                       dataset_train.unprivileged_protected_attributes[sens_ind]]
privileged_groups = [{sens_attr: v} for v in
                     dataset_train.privileged_protected_attributes[sens_ind]]

describe_dataset(dataset_train, dataset_val, dataset_test)

(dataset_train,
 dataset_val) = dataset_expanded_train.split([0.8])
sens_ind = 0
sens_attr = dataset_train.protected_attribute_names[sens_ind]

unprivileged_groups = [{sens_attr: v} for v in
                       dataset_train.unprivileged_protected_attributes[sens_ind]]
privileged_groups = [{sens_attr: v} for v in
                     dataset_train.privileged_protected_attributes[sens_ind]]

#model = FairTransitionLossMLP(sensitive_attr=sens_attr, dropout=0.2,
                 #num_epochs=30, batch_size=32, hidden_sizes=[100,100],
                 #privileged_demotion=0, privileged_promotion=0,
                 #protected_demotion=0, protected_promotion=0)
#model = MetaFairClassifier(sensitive_attr=sens_attr, type='sr')
#model = PrejudiceRemover(sensitive_attr=sens_attr)
model = SimpleMLP(sensitive_attr=sens_attr)
model = model.fit(dataset_train,verbose=True)

val_metrics = eval(model, dataset_val, unprivileged_groups, privileged_groups, mcc_odds)
test_metrics = eval(model, dataset_test, unprivileged_groups, privileged_groups, mcc_odds)
describe_metrics(val_metrics)
describe_metrics(test_metrics)
