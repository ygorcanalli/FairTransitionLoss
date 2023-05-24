import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_comparison(results, dataset_name, fairness_metric):
    my_cmap = plt.cm.get_cmap('Set2')
    colors = my_cmap([x for x in range(6)])
    print(results)
    fig, axis = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Minimizing %s Difference on %s\n" % (fairness_metric,dataset_name), fontsize=18)
    axis[1].tick_params(axis='x', which='major', labelsize=14)
    axis[1].tick_params(axis='y', which='major', labelsize=14)
    axis[0].tick_params(axis='x', which='major', labelsize=14)
    axis[0].tick_params(axis='y', which='major', labelsize=14)
    axis[0] = results[['overall_acc', 'bal_acc', 'fitness', 'method']] \
        .plot.bar(x='method',rot=15, ax=axis[0], legend=False, color=colors[:3])
    axis[1] = results[['avg_odds_diff', 'stat_par_diff', 'eq_opp_diff', 'method']] \
        .plot.bar(x='method', rot=15, ax=axis[1], legend=False, color=colors[3:])
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
    fig.legend(bars, labels,
               ##loc='upper left',
               fancybox=True, shadow=False, ncol=2,
               fontsize='large')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
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

results = pd.read_csv('parity_results.csv')
results['fitness'] = results['overall_acc'] - np.abs(results['stat_par_diff'])
plot_comparison(results, 'Income', 'Statistical Parity')
results[['overall_acc', 'bal_acc', 'stat_par_diff', 'avg_odds_diff', 'eq_opp_diff', 'fitness']].to_latex('parity_tb.txt')


results = pd.read_csv('odds_results.csv')
results['fitness'] = results['overall_acc'] - np.abs(results['avg_odds_diff'])
plot_comparison(results, 'Income', 'Equalized Odds')
results[['overall_acc', 'bal_acc', 'stat_par_diff', 'avg_odds_diff', 'eq_opp_diff', 'fitness']].to_latex('odds_tb.txt')

results = pd.read_csv('opportunity_results.csv')
results['fitness'] = results['overall_acc'] - np.abs(results['eq_opp_diff'])
plot_comparison(results, 'Income', 'Equal Opportunity')
results[['overall_acc', 'bal_acc', 'stat_par_diff', 'avg_odds_diff', 'eq_opp_diff', 'fitness']].to_latex('opp_tb.txt')

results = pd.read_csv('all_results.csv')
results['agg'] = np.sqrt(results['stat_par_diff']**2 + results['eq_opp_diff']**2 + results['avg_odds_diff']**2)
results['fitness'] = results['overall_acc'] - np.sqrt(results['stat_par_diff']**2 + results['eq_opp_diff']**2 + results['avg_odds_diff']**2)
plot_comparison(results, 'Income', 'All Three')
results[['overall_acc', 'bal_acc', 'stat_par_diff', 'avg_odds_diff', 'eq_opp_diff', 'fitness']].to_latex('all_tb.txt')


