from aif360.datasets import AdultDataset, GermanDataset
from util import describe
from sklearn.preprocessing import StandardScaler

from models import SimpleMLP
from util import eval_model, plot_comparison, describe_metrics
import numpy as np
import os

device = 'cpu'
if device == 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def get_T(model, X):
    x_prob = model.predict_proba(X)
    x_hat_0 = np.argmax(x_prob[:, 0])
    x_hat_1 = np.argmax(x_prob[:, 1])
    T_11 = x_prob[x_hat_1, 1]
    T_10 = x_prob[x_hat_1, 0]
    T_00 = x_prob[x_hat_0, 0]
    T_01 = x_prob[x_hat_0, 1]
    T = np.array([[T_00, T_01], [T_10, T_11]])
    return T


label_map = {1.0: '>50K', 0.0: '<=50K'}
protected_attribute_maps = [{1.0: 'Male', 0.0: 'Female'}]
data = AdultDataset(protected_attribute_names=['sex'],
                  categorical_features=['workclass', 'education', 'marital-status',
                                        'occupation', 'relationship', 'native-country', 'race'],
                  privileged_classes=[['Male']], metadata={'label_map': label_map,
                                                           'protected_attribute_maps': protected_attribute_maps})

sens_ind = 0
sens_attr = data.protected_attribute_names[sens_ind]

unprivileged_groups = [{sens_attr: v} for v in
                       data.unprivileged_protected_attributes[sens_ind]]
privileged_groups = [{sens_attr: v} for v in
                     data.privileged_protected_attributes[sens_ind]]

model = SimpleMLP(sensitive_attr=sens_attr,
                                  hidden_sizes=[16, 32],
                                  batch_size=32)

scaler = StandardScaler()
scaled_data = data.copy()
scaled_data.features = scaler.fit_transform(scaled_data.features)
privileged_subset = scaled_data.subset(np.where(scaled_data.labels == 0)[0])
unprivileged_subset = scaled_data.subset(np.where(scaled_data.labels == 1)[0])
model = model.fit(scaled_data)
T = get_T(model, scaled_data.features)
T_privileged = get_T(model, privileged_subset.features)
T_unprivileged = get_T(model, unprivileged_subset.features)


