
from multiprocessing import Pool

import pandas as pd

from datetime import datetime
from aif360.metrics import ClassificationMetric
from aif360.algorithms.inprocessing import PrejudiceRemover, AdversarialDebiasing, MetaFairClassifier, GerryFairClassifier
from models import FairTransitionLossMLP, SimpleMLP, describe_metrics
from fitness_rules import *
from dataset_readers import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow.compat.v1 as tf_old
tf_old.disable_eager_execution()
import numpy as np
import pygad
import json

start_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')

# Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
num_generations = 10 # Number of generations.
num_parents_mating = 5  # Number of solutions to be selected as parents in the mating pool.
parent_selection_type = "sss"  # Type of parent selection.
crossover_type = "single_point"  # Type of the crossover operator.
crossover_probability = 0.1
mutation_type = "random"  # Type of the mutation operator.
mutation_probability = 0.1  # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.
keep_parents = 2  # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.
sol_per_pop = 10

def eval(model, dataset, unprivileged_groups, privileged_groups, fitness_rule, solution):
    try:
        # sklearn classifier
        y_pred_prob = model.predict_proba(dataset.features)
        pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
    except AttributeError:
        # aif360 inprocessing algorithm
        y_pred_prob = model.predict(dataset).scores
        pos_ind = 0

    y_pred = (y_pred_prob[:, pos_ind] > 0.5).astype(np.float64)

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
    metrics['solution'] = solution.tolist()

    return metrics


def evolve_model(dataset_reader, model_initializer, fitness_rule):
    ga_val_metrics_history = []
    dataset_expanded_train, dataset_train, dataset_val, dataset_test, unprivileged_groups, privileged_groups, sens_attr = dataset_reader()

    def callback_generation(ga_instance):
        print("Generation = {generation}".format(generation=ga_instance.generations_completed))
        print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

    def fitness_function(solution, solution_idx):
        # training
        model = model_initializer(sens_attr, unprivileged_groups, privileged_groups, solution)

        scaler = StandardScaler()

        scaled_train = dataset_train.copy()
        scaled_train.features = scaler.fit_transform(scaled_train.features)

        scaled_val = dataset_val.copy()
        scaled_val.features = scaler.transform(scaled_val.features)

        model = model.fit(scaled_train)

        result = eval(model, scaled_val, unprivileged_groups, privileged_groups, fitness_rule, solution)
        ga_val_metrics_history.append(result)

        print('-----------------------------------')
        print('Solution %d' % solution_idx)
        print('Solution:', str(solution))
        describe_metrics(result)
        print('-----------------------------------')
        return result['fitness']

    if fitness_rule is not None:
        # best solution
        genes_space = get_genes_space(model_initializer)
        ga_instance = pygad.GA(num_generations=num_generations,
                               num_parents_mating=num_parents_mating,
                               fitness_func=fitness_function,
                               num_genes=len(genes_space['type']),
                               gene_space=genes_space['space'],
                               gene_type=genes_space['type'],
                               # initial_population=initial_population,
                               sol_per_pop=sol_per_pop,
                               parent_selection_type=parent_selection_type,
                               crossover_type=crossover_type,
                               crossover_probability=crossover_probability,
                               mutation_type=mutation_type,
                               mutation_probability=mutation_probability,
                               keep_parents=keep_parents,
                               on_generation=callback_generation)
        ga_instance.run()
        best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution()
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=best_solution_fitness))
        print("Index of the best solution : {solution_idx}".format(solution_idx=best_solution_idx))

        # eval on test set
        model = model_initializer(sens_attr, unprivileged_groups, privileged_groups, best_solution)
    else:
        model = model_initializer(sens_attr, unprivileged_groups, privileged_groups)

    scaler = StandardScaler()
    scaled_expanded_train = dataset_expanded_train.copy()

    scaled_expanded_train.features = scaler.fit_transform(scaled_expanded_train.features)

    model = model.fit(scaled_expanded_train)

    scaled_test = dataset_test.copy()
    scaled_test.features = scaler.transform(scaled_test.features)

    best_result = eval(model, scaled_test, unprivileged_groups, privileged_groups, fitness_rule, best_solution)

    if fitness_rule is not None:
        best_result['fitness_rule'] = fitness_rule.__name__
    else:
        best_result['fitness_rule'] = 'No optimization'

    print('-----------------------------------')
    describe_metrics(best_result)
    best_result['method'] = model_initializer.__name__
    print('-----------------------------------')
    best_result['ga_history'] = ga_val_metrics_history

    try:
        # tk classifier
        best_result['best_solution_tf_history'] = model.history.history
    except AttributeError:
        # aif360 inprocessing algorithm
        best_result['best_solution_tf_history'] = None


    return best_result

def gerry_fair_classifier_initializer(sens_attr, unprivileged_groups, privileged_groups, solution=None):
    if solution is not None:
        model = GerryFairClassifier(C=solution[0], gamma=solution[1], fairness_def='FP')
    else:
        model = GerryFairClassifier(fairness_def='FP')
    return model

def ftl_mlp_initializer(sens_attr, unprivileged_groups, privileged_groups, solution=None):
    if solution is not None:
        model = FairTransitionLossMLP(sensitive_attr=sens_attr,
                                      hidden_sizes=[solution[0]],
                                      dropout=solution[1],
                                      batch_size=32,
                                      privileged_demotion=solution[2], privileged_promotion=solution[3],
                                      protected_demotion=solution[4], protected_promotion=solution[5])
    else:
        model = FairTransitionLossMLP(sensitive_attr=sens_attr,
                                      hidden_sizes=[32],
                                      dropout=0.1,
                                      batch_size=32)
    return model

def simple_mlp_initializer(sens_attr, unprivileged_groups, privileged_groups, solution=None):
    if solution is not None:
        model = SimpleMLP(sensitive_attr=sens_attr,
                        hidden_sizes=[solution[0]],
                        dropout=solution[1],
                        batch_size=32)
    else:
        model = SimpleMLP(sensitive_attr=sens_attr,
                        hidden_sizes=[32],
                        dropout=0.1,
                        batch_size=32)
    return model

def prejudice_remover_initializer(sens_attr, unprivileged_groups, privileged_groups, solution=None):
    if solution is not None:
        model = PrejudiceRemover(eta=solution[0], sensitive_attr=sens_attr)
    else:
        model = PrejudiceRemover(sensitive_attr=sens_attr)
    return model

def meta_fair_classifier_sr_initializer(sens_attr, unprivileged_groups, privileged_groups, solution=None):
    if solution is not None:
        model = MetaFairClassifier(tau=solution[0], sensitive_attr=sens_attr, type='sr')
    else:
        model = MetaFairClassifier(sensitive_attr=sens_attr, type='sr')
    return model

def meta_fair_classifier_fdr_initializer(sens_attr, unprivileged_groups, privileged_groups, solution=None):
    if solution is not None:
        model = MetaFairClassifier(tau=solution[0], sensitive_attr=sens_attr, type='fdr')
    else:
        model = MetaFairClassifier(sensitive_attr=sens_attr, type='fdr')
    return model



def adversarial_debiasing_initializer(sens_attr, unprivileged_groups, privileged_groups, solution=None):
    if solution is not None:
        model = AdversarialDebiasing(unprivileged_groups, privileged_groups,
                                     'adv_debias_' + str(datetime.now().timestamp()), tf_old.Session(),
                                     classifier_num_hidden_units=solution[0], adversary_loss_weight=solution[1])
    else:
        model = AdversarialDebiasing(unprivileged_groups, privileged_groups,
                                     'adv_debias_' + str(datetime.now().timestamp()), tf_old.Session())
    return model

def get_genes_space(model_initializer):
    genes_space = {
        'simple_mlp_initializer':  {
                    'type': [int, float],
                    'space': [
                        [32],
                        {'low': 0.0, 'high': 0.2}
                    ]
                },
        'ftl_mlp_initializer':  {
                    'type': [int, float, float, float, float, float],
                    'space': [
                        [32],
                        {'low': 0.0, 'high': 0.2},
                        {'low': 0.0, 'high': 1.0},
                        {'low': 0.0, 'high': 1.0},
                        {'low': 0.0, 'high': 1.0},
                        {'low': 0.0, 'high': 1.0}
                    ]
                },
        'prejudice_remover_initializer':  {
                    'type': [float],
                    'space': [
                        {'low': 0.0, 'high': 50.0}
                    ]
                },
        'meta_fair_classifier_sr_initializer': {
                    'type': [float],
                    'space': [
                        {'low': 0.0, 'high': 2.0}
                    ]
                },
        'meta_fair_classifier_fdr_initializer': {
                    'type': [float],
                    'space': [
                        {'low': 0.0, 'high': 2.0}
                    ]
                },
        'adversarial_debiasing_initializer': {
                    'type': [int, float],
                    'space': [
                        [32],
                        {'low': 0.0, 'high': 1.0}
                    ]
                },
        'gerry_fair_classifier_initializer': {
                    'type': [float, float],
                    'space': [
                        {'low': 0.0, 'high': 20.0},
                        [0.1, 0.01, 0.001]
                    ]
                },
    }

    return genes_space[model_initializer.__name__]

datasets = [
    adult_dataset_reader,
]

rules = [
    mcc_parity
    #mcc_odds
    #mcc_opportunity
    #acc_parity,
    #acc_odds,
    #acc_opportunity
]

methods = [
    simple_mlp_initializer,
    ftl_mlp_initializer,
    prejudice_remover_initializer,
    gerry_fair_classifier_initializer,
    adversarial_debiasing_initializer,
]

results = []

for dataset_reader in datasets:
    for fitness_rule in rules:
        for model_initializer in methods:
            result = evolve_model(dataset_reader, model_initializer, fitness_rule)
            print('Best metrics')
            print('Dataset:', dataset_reader.__name__)
            print('Method:', model_initializer.__name__)
            if fitness_rule is not None:
                print('Fitness rule:', fitness_rule.__name__)
            else:
                print('Fitness rule: None')
            describe_metrics(result)
            results.append(result)
            results_df = pd.DataFrame(results)
            results_df.to_csv('raw_results/results_%s.csv' % start_time)
            with open('raw_results/results_%s.json' % start_time, 'w') as file:
                json.dump(results, file)