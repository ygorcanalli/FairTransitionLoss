import matplotlib.pyplot as plt
import pandas as pd

def plot_comparison(results, name):
    my_cmap = plt.cm.get_cmap('Dark2')
    n_bars = len(results.keys()) - 1
    colors = my_cmap([x for x in range(n_bars)])
    print(results)
    fig, axis = plt.subplots(3, 1, figsize=(14, 12), sharex=True,
                             gridspec_kw={'height_ratios': [1, 2, 2]})
    fig.suptitle("Comparação de Regras de Otimização em Fairness\n%s" % name, fontsize=16)
    axis[0].axis('off')
    axis[1] = results[['overall_acc', 'bal_acc', 'fitness_rule']] \
        .plot.bar(x='fitness_rule', rot=0, ax=axis[1], grid=True, legend=False, color=colors[:3])
    axis[2] = results[['avg_odds_diff', 'stat_par_diff', 'eq_opp_diff', 'fitness_rule']] \
        .plot.bar(x='fitness_rule', rot=0, ax=axis[2], grid=True, legend=False, color=colors[3:])
    axis[1].set_ylim(0, 0.9)
    axis[2].set_ylim(0, 0.7)
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

def get_metric_explain():
    return {
        'overall_acc': 'Accuracy - acurácia padrão',
        'bal_acc': 'Balanced Accuracy - média entre taxa de verdadeiros positivos e verdadeiros negativos',
        'avg_odds_diff': 'Equalized Odds - diferença de verdadeiros e falsos positivos dentre os grupos',
        'disp_imp': 'Disparate Impact - razão de previsões positivas dentre os grupos',
        'stat_par_diff': 'Statistical Parity - diferença de previsões positivas dentre os grupos',
        'eq_opp_diff': 'Equal Opportunity - diferença de falsos negativos dentre os grupos'
    }

fair_mlp = pd.read_csv('fair_mlp_results.csv')
plot_comparison(fair_mlp, 'fair_mlp')

meta_fair_classifier_sr = pd.read_csv('meta_fair_classifier_sr_results.csv')
plot_comparison(meta_fair_classifier_sr, 'meta_fair_classifier_sr')

prejudice_remover = pd.read_csv('prejudice_remover_results.csv')
plot_comparison(prejudice_remover, 'prejudice_remover')