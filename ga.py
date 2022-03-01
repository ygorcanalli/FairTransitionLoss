import numpy as np
from matplotlib import cm
from collections import defaultdict

import numpy as np
import pygad
import time
from multiprocessing import Pool

from aif360.datasets import AdultDataset
from util import describe, describe_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# Bias mitigation techniques
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import PrejudiceRemover, AdversarialDebiasing, MetaFairClassifier
# Fair loss
from models import FairMLP, SimpleMLP
from util import eval_model, plot_comparison
import os


device = 'cpu'
if device == 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


label_map = {1.0: '>50K', 0.0: '<=50K'}
protected_attribute_maps = [{1.0: 'Male', 0.0: 'Female'}]
ad = AdultDataset(protected_attribute_names=['sex'],
                  categorical_features=['workclass', 'education', 'marital-status',
                                        'occupation', 'relationship', 'native-country', 'race'],
                  privileged_classes=[['Male']], metadata={'label_map': label_map,
                                                           'protected_attribute_maps': protected_attribute_maps})
(train,
 val,
 test) = AdultDataset().split([0.5, 0.8], shuffle=True)

sens_ind = 1
sens_attr = train.protected_attribute_names[sens_ind]

unprivileged_groups = [{sens_attr: v} for v in
                       train.unprivileged_protected_attributes[sens_ind]]
privileged_groups = [{sens_attr: v} for v in
                     train.privileged_protected_attributes[sens_ind]]

describe(train, val, test)

def fitness_rule(metrics):
    acc = metrics['overall_acc']
    odds = metrics['avg_odds_diff']
    par = metrics['stat_par_diff']
    opp = metrics['eq_opp_diff']

    return par**2 + (acc-1)**2

def fitness_function(solution, solution_idx):
    # training

    model = FairMLP(sensitive_attr=sens_attr,
                    hidden_sizes=[16, 32],
                    batch_size=32,
                    privileged_demotion=solution[0], privileged_promotion=solution[1],
                    protected_demotion=solution[2], protected_promotion=solution[3])

    scaler = StandardScaler()

    scaled_train = train.copy()
    scaled_train.features = scaler.fit_transform(scaled_train.features)

    scaled_val = val.copy()
    scaled_val.features = scaler.fit_transform(scaled_val.features)
    model = model.fit(scaled_train)

    # hyperparameter tunning in validation set
    val_metrics = eval_model(model, scaled_val, [0.5], unprivileged_groups, privileged_groups)
    # best solution
    print('-----------------------------------')
    print('Solution %d' % solution_idx)
    best_metrics = describe_metrics(val_metrics, [0.5])
    print('-----------------------------------')
    return fitness_rule(best_metrics)



def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=1/ga_instance.best_solution()[1]))


def fitness_wrapper(solution):
    return fitness_function(solution, 0)


class PooledGA(pygad.GA):

    def cal_pop_fitness(self):
        global pool

        pop_fitness = pool.map(fitness_wrapper, self.population)
        print(pop_fitness)
        pop_fitness = np.array(pop_fitness)
        return pop_fitness


start_time = time.time()



ga_instance = PooledGA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       num_genes=num_genes,
                       #initial_population=initial_population,
                       sol_per_pop=sol_per_pop,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       crossover_probability=crossover_probability,
                       mutation_type=mutation_type,
                       mutation_probability=mutation_probability,
                       keep_parents=keep_parents,
                       on_generation=callback_generation,
                       gene_space=gene_space)


with Pool(processes=4) as pool:
    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=1/solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

    print("--- %s seconds ---" % (time.time() - start_time))
    # After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
    ga_instance.plot_result(title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)




