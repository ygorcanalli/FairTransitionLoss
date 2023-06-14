import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
sns.set_theme()

# Load the example flights dataset and convert to long-form
flights_long = sns.load_dataset("flights")
flights = flights_long.pivot("month", "year", "passengers")

# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(flights, annot=True, fmt="d", linewidths=.5, ax=ax)

d_0s = [0.0, 0.1, 0.2]
p_0s = [0.0, 0.1, 0.2]
d_1s = [0.0, 0.1, 0.2]
p_1s = [0.0, 0.1, 0.2]
results = []

for d_0 in d_0s:
    for p_0 in p_0s:
        for d_1 in d_1s:
            for p_1 in p_1s:
                result = {'demotion': 'd_0={0:.1f},d_1={1:.1f}'.format(d_0,d_1),
                          'promotion': 'p_0={0:.1f},p_1={1:.1f}'.format(p_0,p_1),
                          'value': random.random()}
                results.append(result)
results_df = pd.DataFrame(results)

pivoted = results_df.pivot(index='demotion', columns='promotion', values='value')