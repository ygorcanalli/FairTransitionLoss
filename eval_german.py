from aif360.datasets import GermanDataset
from aif360.algorithms.inprocessing import PrejudiceRemover, MetaFairClassifier
from models import describe_metrics, SimpleMLP, FairTransitionLossMLP
import fitness_rules
from fitness_rules import *
from aif360.metrics import ClassificationMetric
from pprint import pprint

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
    metrics['fitness'] = fitness_rules.linear_odds(metrics)

    return metrics

#label_map = {1: 'Good Credit', 2: 'Bad Credit'}
#protected_attribute_maps = [{1: 'Male', 0: 'Female'}]
#data = GermanDataset(protected_attribute_names=['sex'],
#privileged_classes=[['Male']], metadata={'label_map': label_map,
#                    'protected_attribute_maps': protected_attribute_maps})
data = GermanDataset()
data.instance_weights[data.labels[:,0] == 2.0] = 5.0
(dataset_expanded_train,
 dataset_test) = data.split([0.8])

(dataset_train,
 dataset_val) = dataset_expanded_train.split([0.8])
sens_ind = 0
sens_attr = dataset_train.protected_attribute_names[sens_ind]

unprivileged_groups = [{sens_attr: v} for v in
                       dataset_train.unprivileged_protected_attributes[sens_ind]]
privileged_groups = [{sens_attr: v} for v in
                     dataset_train.privileged_protected_attributes[sens_ind]]

model = FairTransitionLossMLP(sensitive_attr=sens_attr, dropout=0.2,
                 num_epochs=30, batch_size=32, hidden_sizes=[100,100],
                 privileged_demotion=0, privileged_promotion=0,
                 protected_demotion=0, protected_promotion=0)
#model = MetaFairClassifier(sensitive_attr=sens_attr, type='sr')
#model = PrejudiceRemover(sensitive_attr=sens_attr)

model = model.fit(dataset_train)

val_metrics = eval_metrics(model, dataset_val, unprivileged_groups, privileged_groups)
test_metrics = eval_metrics(model, dataset_test, unprivileged_groups, privileged_groups)
describe_metrics(val_metrics)
describe_metrics(test_metrics)
