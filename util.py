from collections import defaultdict
from aif360.metrics import ClassificationMetric
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

def describe(train=None, val=None, test=None):
    if train is not None:
        print("#### Training Dataset shape")
        print(train.features.shape)
    if val is not None:
        print("#### Validation Dataset shape")
        print(val.features.shape)
    print("#### Test Dataset shape")
    print(test.features.shape)
    print("#### Favorable and unfavorable labels")
    print(test.favorable_label, test.unfavorable_label)
    print("#### Protected attribute names")
    print(test.protected_attribute_names)
    print("#### Privileged and unprivileged protected attribute values")
    print(test.privileged_protected_attributes,
          test.unprivileged_protected_attributes)
    print("#### Dataset feature names")
    print(test.feature_names)

def describe_metrics(metrics, thresh_arr):
    best_ind = np.argmax(metrics['bal_acc'])

    print("Threshold corresponding to Best balanced accuracy: {:6.4f}".format(thresh_arr[best_ind]))
    print("Best balanced accuracy: {:6.4f}".format(metrics['bal_acc'][best_ind]))
    print("overall accuracy: {:6.4f}".format(metrics['overall_acc'][best_ind]))
#     disp_imp_at_best_ind = np.abs(1 - np.array(metrics['disp_imp']))[best_ind]
    disp_imp_at_best_ind = 1 - min(metrics['disp_imp'][best_ind], 1/metrics['disp_imp'][best_ind])
    print("Corresponding 1-min(DI, 1/DI) value: {:6.4f}".format(disp_imp_at_best_ind))
    print("Corresponding average odds difference value: {:6.4f}".format(metrics['avg_odds_diff'][best_ind]))
    print("Corresponding statistical parity difference value: {:6.4f}".format(metrics['stat_par_diff'][best_ind]))
    print("Corresponding equal opportunity difference value: {:6.4f}".format(metrics['eq_opp_diff'][best_ind]))
    print("Corresponding Theil index value: {:6.4f}".format(metrics['theil_ind'][best_ind]))

    best_metrics = {
        'overall_acc': abs(metrics['overall_acc'][best_ind]),
        'bal_acc': abs(metrics['bal_acc'][best_ind]),
        'disp_imp': abs(disp_imp_at_best_ind),
        #'disp_imp': abs(metrics['disp_imp'][best_ind]),
        'avg_odds_diff': abs(metrics['avg_odds_diff'][best_ind]),
        'stat_par_diff': abs(metrics['stat_par_diff'][best_ind]),
        'eq_opp_diff': abs(metrics['eq_opp_diff'][best_ind])
        #'theil_ind': abs(metrics['theil_ind'][best_ind])
    }

    return best_metrics


def eval_model(model, dataset, thresh_arr, unprivileged_groups, privileged_groups):
    try:
        # sklearn classifier
        y_pred_prob = model.predict_proba(dataset.features)
        pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
    except AttributeError:
        # aif360 inprocessing algorithm
        y_pred_prob = model.predict(dataset).scores
        pos_ind = 0

    metric_arrs = defaultdict(list)
    for thresh in thresh_arr:
        y_val_pred = (y_pred_prob[:, pos_ind] > thresh).astype(np.float64)

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_val_pred
        metric = ClassificationMetric(
                dataset, dataset_pred,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)

        metric_arrs['overall_acc'].append(  metric.accuracy() )
        metric_arrs['bal_acc'].append((metric.true_positive_rate()
                                     + metric.true_negative_rate()) / 2)
        metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
        metric_arrs['disp_imp'].append(metric.disparate_impact())
        metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
        metric_arrs['eq_opp_diff'].append(metric.equal_opportunity_difference())
        metric_arrs['theil_ind'].append(metric.theil_index())

    return metric_arrs

def plot_comparison(results):
    results_df = pd.DataFrame(results)
    my_cmap = plt.cm.get_cmap('Dark2')
    n_bars = len(results_df.keys()) - 1
    colors = my_cmap([x for x in range(n_bars)])
    print(results_df)
    fig, axis = plt.subplots(3, 1, figsize=(14, 12), sharex=True,
                             gridspec_kw={'height_ratios': [1, 2, 2]})
    fig.suptitle("Comparação de Métricas em Fairness", fontsize=16)
    axis[0].axis('off')
    axis[1] = results_df[['overall_acc', 'bal_acc', 'disp_imp', 'method']] \
        .plot.bar(x='method', rot=0, ax=axis[1], grid=True, legend=False, color=colors[:3])
    axis[2] = results_df[['avg_odds_diff', 'stat_par_diff', 'eq_opp_diff', 'method']] \
        .plot.bar(x='method', rot=0, ax=axis[2], grid=True, legend=False, color=colors[3:])
    bars_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    bars, labels = [sum(lol, []) for lol in zip(*bars_labels)]
    metric_explain = get_metric_explain()
    labels = [metric_explain[label] for label in labels]
    fig.legend(bars, labels,
               loc='lower left', bbox_to_anchor=(0.03, 0.78),
               fancybox=True, shadow=False, ncol=1,
               fontsize='x-large')
    fig.tight_layout()
    fig.show()

def get_metric_explain():
    return {
        'overall_acc': 'Accuracy - acurácia padrão',
        'bal_acc': 'Balanced Accuracy - média entre taxa de verdadeiros positivos e verdadeiros negativos',
        'avg_odds_diff': 'Average Odds Difference - diferença de verdadeiros e falsos positivos dentre os grupos',
        'disp_imp': 'Disparate Impact - razão de previsões positivas dentre os grupos',
        'stat_par_diff': 'Statistical Parity Difference - diferença de previsões positivas dentre os grupos',
        'eq_opp_diff': 'Equalized Odds Difference - diferença de verdadeiros positivos dentre os grupos'
    }