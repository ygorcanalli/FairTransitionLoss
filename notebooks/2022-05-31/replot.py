import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_comparison(results, dataset_name, fairness_metric):
    my_cmap = plt.cm.get_cmap('tab20')
    colors = my_cmap([x for x in range(20)])
    print(results)
    fig, axis = plt.subplots(2, 1, figsize=(16, 9), sharex=True,
                             gridspec_kw={'height_ratios': [3, 2]})
    fig.suptitle("Minimizing %s Difference on %s\n" % (fairness_metric,dataset_name), fontsize=18)
    axis[1].tick_params(axis='x', which='major', labelsize=20)
    axis[1].tick_params(axis='y', which='major', labelsize=20)
    axis[0].tick_params(axis='x', which='major', labelsize=20)
    axis[0].tick_params(axis='y', which='major', labelsize=20)
    #axis[2].axis('off')
    axis[1] = results[['overall_acc', 'bal_acc', 'method']] \
        .plot.bar(x='method',rot=0, ax=axis[1], legend=False, color=colors[:2])
    axis[0] = results[['avg_odds_diff', 'stat_par_diff', 'eq_opp_diff', 'agg', 'fitness','method']] \
        .plot.bar(x='method', rot=0, ax=axis[0], legend=False, color=colors[[4,6,8,10,12]])
    axis[1].set_ylim(0, 0.9)
    axis[0].set_ylim(0, 0.9)
    axis[1].set_xlabel('Method', fontsize=24)
    axis[1].xaxis.labelpad = 20

    axis[1].grid(visible=True, which='major', axis='y', color='black', linewidth=1)
    axis[0].grid(visible=True, which='major', axis='y', color='black', linewidth=1)

    bars_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    bars, labels = [sum(lol, []) for lol in zip(*bars_labels)]
    metric_explain = get_metric_explain()
    labels = [metric_explain[label] for label in labels]
    fig.legend(bars, labels,
               loc='upper center',
               fancybox=True, shadow=False, ncol=3,
               fontsize='xx-large')
    fig.tight_layout()
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
        'agg': 'sqrt(partity² + opportunity² + odds²)',
        'fitness': 'acc - sqrt(partity² + opportunity² + odds²)'
    }

results = pd.read_csv('parity_results.csv')
results['agg'] = np.sqrt(results['stat_par_diff']**2 + results['eq_opp_diff']**2 + results['avg_odds_diff']**2)
results['fitness'] = results['overall_acc'] - np.sqrt(results['stat_par_diff']**2 + results['eq_opp_diff']**2 + results['avg_odds_diff']**2)
plot_comparison(results, 'Income', 'Statistical Parity')
results['overall_all', 'bal_acc', 'stat_par_diff', 'avg_odds_diff', ].to_latex('parity_tb.txt')


results = pd.read_csv('odds_results.csv')
results['agg'] = np.sqrt(results['stat_par_diff']**2 + results['eq_opp_diff']**2 + results['avg_odds_diff']**2)
results['fitness'] = results['overall_acc'] - np.sqrt(results['stat_par_diff']**2 + results['eq_opp_diff']**2 + results['avg_odds_diff']**2)
plot_comparison(results, 'Income', 'Equalized Odds')
results.to_latex('odds_tb.txt')

results = pd.read_csv('opportunity_results.csv')
results['agg'] = np.sqrt(results['stat_par_diff']**2 + results['eq_opp_diff']**2 + results['avg_odds_diff']**2)
results['fitness'] = results['overall_acc'] - np.sqrt(results['stat_par_diff']**2 + results['eq_opp_diff']**2 + results['avg_odds_diff']**2)
plot_comparison(results, 'Income', 'Equal Opportunity')
results.to_latex('opp_tb.txt')

results = pd.read_csv('all_results.csv')
results['agg'] = np.sqrt(results['stat_par_diff']**2 + results['eq_opp_diff']**2 + results['avg_odds_diff']**2)
results['fitness'] = results['overall_acc'] - np.sqrt(results['stat_par_diff']**2 + results['eq_opp_diff']**2 + results['avg_odds_diff']**2)
plot_comparison(results, 'Income', 'All Three')
results.to_latex('all_tb.txt')


