import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import textwrap
def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0)

def plot_comparison(results, dataset_name, fairness_metric):
    my_cmap = plt.cm.get_cmap('Set2')
    colors = my_cmap([x for x in range(6)])
    print(results)
    fig, axis = plt.subplots(1, 2, figsize=(16, 6))
    #fig.suptitle("Minimizing %s Difference on %s\n" % (fairness_metric,dataset_name), fontsize=18)
    axis[1].tick_params(axis='x', which='major', labelsize=14)
    axis[1].tick_params(axis='y', which='major', labelsize=14)
    axis[0].tick_params(axis='x', which='major', labelsize=14)
    axis[0].tick_params(axis='y', which='major', labelsize=14)
    axis[0] = results[['overall_acc', 'bal_acc', 'fitness', 'method']] \
        .plot.bar(x='method',rot=0, ax=axis[0], legend=False, color=colors[:3])
    axis[1] = results[['avg_odds_diff', 'stat_par_diff', 'eq_opp_diff', 'method']] \
        .plot.bar(x='method', rot=0, ax=axis[1], legend=False, color=colors[3:])
    axis[0].set_ylim(0, 1.0)
    axis[1].set_ylim(0, 0.35)
    axis[1].set(xlabel=None)
    axis[0].set(xlabel=None)

    axis[1].grid(visible=True, which='major', axis='y', color='black', linewidth=1)
    axis[0].grid(visible=True, which='major', axis='y', color='black', linewidth=1)

    bars_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    bars, labels = [sum(lol, []) for lol in zip(*bars_labels)]
    metric_explain = get_metric_explain()
    labels = [metric_explain[label] for label in labels]
    wrap_labels(axis[0], 12)
    wrap_labels(axis[1], 12)
    fig.legend(bars, labels,
               ##loc='upper left',
               fancybox=True, shadow=False, ncol=2,
               fontsize='large')
    fig.tight_layout()#rect=[0, 0.03, 1, 0.95])
    file_name = '%s_%s_plot.png' % (dataset_name, fairness_metric)
    file_name = file_name.lower().replace(' ', '_')

    fig.savefig(file_name, dpi=300)

def get_metric_explain():
    return {
        'overall_acc': 'Accuracy',
        'bal_acc': 'Balanced Accuracy',
        'avg_odds_diff': 'Equalized Odds',
        'disp_imp': 'Disparate Impact',
        'stat_par_diff': 'Statistical Parity',
        'eq_opp_diff': 'Equal Opportunity',
        'fitness': 'Fitness Function'
    }

def get_table_with_relative_difference(data, numerical_columns):
    numerical_data = data[numerical_columns]
    difference = (numerical_data - numerical_data.loc[0, :]) / numerical_data.loc[0, :]
    data_with_difference = data.copy()
    for column in difference:
        for i, x in enumerate(data[column]):
            data_with_difference.loc[i, column] = "{0:.3f} ({1:+.2%})".format(x, difference[column][i])

    return data_with_difference

metric_names = get_metric_explain()
numerical_columns = ['overall_acc', 'bal_acc', 'stat_par_diff', 'avg_odds_diff', 'eq_opp_diff', 'fitness']
latex_columns = ['method', 'fitness', 'overall_acc', 'bal_acc', 'stat_par_diff', 'avg_odds_diff', 'eq_opp_diff']

results = pd.read_csv('results.csv')

parity_results = pd.DataFrame(results[results['fitness_rule'] == 'linear_parity']).reset_index()
plot_comparison(parity_results, 'Income', 'Statistical Parity')
get_table_with_relative_difference(parity_results, numerical_columns)[latex_columns]\
    .rename(columns=metric_names).set_index('method').T.to_latex('statistical_parity.tex')

odds_results = pd.DataFrame(results[results['fitness_rule'] == 'linear_odds']).reset_index()
plot_comparison(odds_results, 'Income', 'Equalized Odds')
get_table_with_relative_difference(odds_results, numerical_columns)[latex_columns]\
    .rename(columns=metric_names).set_index('method').T.to_latex('equalized_odds.tex')

opportunity_results = pd.DataFrame(results[results['fitness_rule'] == 'linear_opportunity']).reset_index()
plot_comparison(opportunity_results, 'Income', 'Equal Opportunity')
get_table_with_relative_difference(opportunity_results, numerical_columns)[latex_columns]\
    .rename(columns=metric_names).set_index('method').T.to_latex('equal_opportunity.tex')


