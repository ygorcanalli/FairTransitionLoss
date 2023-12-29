import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import InputLayer, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.legacy import Adam
from aif360.algorithms import Transformer
from tensorflow.keras import backend as K

import logging
import os
from sklearn.linear_model import LogisticRegression


# tentar otimizar com outras métricas de performance
# estudar balanceamento dos dados
# verificar mudança de erro de valicação e treino ao longo das épocas

def fair_forward(P_privileged, P_protected):
    """
    Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach
    CVPR17 https://arxiv.org/abs/1609.03683
    :param P: noise model, a noisy label transition probability matrix
    :return:
    """
    P_privileged = K.constant(P_privileged)
    P_protected = K.constant(P_protected)

    def loss(y_true, y_pred):
        y_true_sensitive = y_true[:, :2]
        y_true_labels = y_true[:, -2:]

        privileged_pred = binary_crossentropy(y_true_labels, K.dot(y_pred, P_privileged))
        protected_pred = binary_crossentropy(y_true_labels, K.dot(y_pred, P_protected))

        # using onehootencoding to choose the transition matrix to use without if
        combined_pred = privileged_pred * y_true_sensitive[:, 0] + \
                        protected_pred * y_true_sensitive[:, 1]

        return combined_pred

    return loss


def fair_forward_2(P_privileged, P_protected):
    """
    Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach
    CVPR17 https://arxiv.org/abs/1609.03683
    :param P: noise model, a noisy label transition probability matrix
    :return:
    """
    P_privileged = K.constant(P_privileged)
    P_protected = K.constant(P_protected)

    def loss(y_true, y_pred):
        y_true_sensitive = y_true[:, :2]
        y_true_labels = y_true[:, -2:]
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        privileged_pred = -K.sum(y_true_labels * K.log(K.dot(y_pred, P_privileged)), axis=1)
        protected_pred = -K.sum(y_true_labels * K.log(K.dot(y_pred, P_protected)), axis=1)

        combined_pred = privileged_pred * y_true_sensitive[:, 0] + \
                        protected_pred * y_true_sensitive[:, 1]

        return combined_pred

    return loss


class SimpleMLP(Transformer):

    def __init__(self, sensitive_attr='',
                 hidden_sizes=[32, 64, 32], dropout=0.1,
                 num_epochs=20, batch_size=64, patience=5):

        self.model = None
        self.hidden_sizes = hidden_sizes
        self.input_shape = None
        self.num_classes = 2
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.patience = patience
        self.sensitive_attr = sensitive_attr
        self.classes_ = None
        self.history = None

    def _compile_model(self):
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=self.input_shape))

        for hidden_size in self.hidden_sizes:
            self.model.add(Dense(hidden_size, activation='relu'))
            self.model.add(Dropout(self.dropout))

        self.model.add(Dense(self.num_classes, activation="softmax"))
        self.model.compile(optimizer=Adam(learning_rate=3e-4),
                           loss='categorical_crossentropy')

    def fit(self, dataset, verbose=False):
        if self.model is None:
            self.input_shape = dataset.features.shape[1]
            self._compile_model()
            self.classes_ = np.array([dataset.unfavorable_label, dataset.favorable_label])

        callback = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)
        dataset_cp = dataset.copy()
        X = dataset_cp.features
        y_expanded = np.zeros(shape=(X.shape[0], 2))

        y_expanded[:, 0] = (dataset_cp.labels == dataset_cp.unfavorable_label).reshape(X.shape[0]).astype(int)
        y_expanded[:, 1] = (dataset_cp.labels == dataset_cp.favorable_label).reshape(X.shape[0]).astype(int)

        self.history = self.model.fit(X, y_expanded, epochs=self.num_epochs,
                                      batch_size=self.batch_size, callbacks=[callback],
                                      verbose=verbose, validation_split=0.1)

        return self

    def predict_proba(self, X):
        return self.model.predict(X)

    def predict(self, X):
        logits = self.predict_proba(X)
        return np.argmax(logits, axis=1)


class FairTransitionLossMLP(Transformer):

    def __init__(self, sensitive_attr='',
                 privileged_demotion=0.1, privileged_promotion=0.01,
                 protected_demotion=0.01, protected_promotion=0.1,
                 hidden_sizes=[32, 64, 32], dropout=0.1, patience=5,
                 num_epochs=50, batch_size=64):
        self.p_privileged = np.array([[1 - privileged_demotion, privileged_demotion],
                                      [privileged_promotion, 1 - privileged_promotion]])
        self.p_protected = np.array([[1 - protected_demotion, protected_demotion],
                                     [protected_promotion, 1 - protected_promotion]])
        self.model = None
        self.hidden_sizes = hidden_sizes
        self.input_shape = None
        self.num_classes = 2
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.patience = patience
        self.sensitive_attr = sensitive_attr
        self.classes_ = None
        self.history = None

    def _compile_model(self):
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=self.input_shape))

        for hidden_size in self.hidden_sizes:
            self.model.add(Dense(hidden_size, activation='relu'))
            self.model.add(Dropout(self.dropout))

        self.model.add(Dense(self.num_classes, activation="softmax"))
        self.model.compile(optimizer=Adam(learning_rate=3e-4),
                           loss=fair_forward(self.p_privileged, self.p_protected))

    def fit(self, dataset, verbose=False):
        if self.model is None:
            self.input_shape = dataset.features.shape[1]
            self._compile_model()
            self.classes_ = np.array([dataset.unfavorable_label, dataset.favorable_label])

        callback = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)

        X = dataset.features
        y_expanded = np.zeros(shape=(X.shape[0], 4))
        sensitive_index = dataset.protected_attribute_names.index(self.sensitive_attr)

        y_expanded[:, 0] = (dataset.protected_attributes[:, sensitive_index] == dataset.privileged_protected_attributes[
            sensitive_index]).astype(int)
        y_expanded[:, 1] = (
                    dataset.protected_attributes[:, sensitive_index] == dataset.unprivileged_protected_attributes[
                sensitive_index]).astype(int)
        y_expanded[:, 2] = (dataset.labels == dataset.unfavorable_label).reshape(X.shape[0]).astype(int)
        y_expanded[:, 3] = (dataset.labels == dataset.favorable_label).reshape(X.shape[0]).astype(int)

        self.history = self.model.fit(X, y_expanded, epochs=self.num_epochs,
                                      batch_size=self.batch_size, callbacks=[callback],
                                      verbose=verbose, validation_split=0.1)

        return self

    def predict_proba(self, X):
        return self.model.predict(X)

    def predict(self, X):
        logits = self.predict_proba(X)
        return np.argmax(logits, axis=1)


# wrapper to https://github.com/che2198/APW
class AdaptativePriorityReweightingDP(Transformer):

    def __init__(self, sensitive_attr='', epochs=50, eta=0, alpha=0):
        self.sensitive_attr = sensitive_attr
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.input_shape = None
        self.model = None
        self.classes_ = None
        self.history = None

    def _compute_sample_weights(self, true_labels, predicted_probabilities, protected_attributes, multipliers, eta,
                                decision_boundary=0.5):
        num_samples = len(true_labels)
        exponential_term = np.exp(-eta * abs(predicted_probabilities - decision_boundary))
        weight_component = np.zeros(num_samples)

        for attr in protected_attributes:
            weight_component += exponential_term * np.sum(attr) / np.sum(exponential_term * attr) * attr

        combined_weights = np.zeros(num_samples)
        for i, multiplier in enumerate(multipliers):
            combined_weights += multiplier * protected_attributes[i]

        sample_weights = combined_weights * weight_component

        return sample_weights

    def _compute_group_weights(self, predictions, true_labels, protected_attributes, alpha):
        group_weights = []

        num_samples = len(true_labels)

        for attr in protected_attributes:
            protected_indices = np.where(attr > 0)

            positive_protected_prediction = np.sum(predictions[protected_indices])
            negative_protected_prediction = np.sum(1 - predictions[protected_indices])

            # Calculate weights for positive and negative protected predictions
            weight_positive = (len(protected_indices[0]) * np.sum(predictions) + alpha) / (
                    num_samples * positive_protected_prediction)
            weight_negative = (len(protected_indices[0]) * np.sum(1 - predictions) + alpha) / (
                    num_samples * negative_protected_prediction)

            group_weights.extend([weight_positive, weight_negative])

        return group_weights

    def fit(self, dataset):
        if self.model is None:
            self.input_shape = dataset.features.shape[1]
            self.classes_ = np.array([dataset.unfavorable_label, dataset.favorable_label])

        sensitive_index = dataset.protected_attribute_names.index(self.sensitive_attr)

        features = dataset.features
        labels = (dataset.labels == dataset.favorable_label).reshape(features.shape[0]).astype(int)

        # labels[:, 0] = (dataset.labels == dataset.unfavorable_label).reshape(features.shape[0]).astype(int)
        # labels[:, 1] = (dataset.labels == dataset.favorable_label).reshape(features.shape[0]).astype(int)

        protected_attributes = (dataset.protected_attributes[:, sensitive_index] ==
                                dataset.unprivileged_protected_attributes[sensitive_index]).astype(int)

        protected_attributes_list = [protected_attributes, 1 - protected_attributes]
        label_combinations = [protected_attributes * labels, protected_attributes * (1 - labels), \
                              (1 - protected_attributes) * labels,
                              (1 - protected_attributes) * (1 - labels)]
        fairness_multipliers = np.ones(len(label_combinations))
        sample_weights = np.array([1] * features.shape[0])

        for epoch in range(self.epochs):
            # Train logistic regression model
            self.model = LogisticRegression(max_iter=10000)
            self.model = self.model.fit(features, labels, sample_weights)

            predictions_train = self.model.predict(features)
            prediction_probabilities = self.model.predict_proba(features)[:, 0].astype(np.float32)

            # Compute weights and multipliers
            sample_weights = self._compute_sample_weights(labels, prediction_probabilities, label_combinations,
                                                          fairness_multipliers, self.eta)
            group_fairness_weights = self._compute_group_weights(predictions_train, labels, protected_attributes_list,
                                                                 self.alpha)
            fairness_multipliers *= np.array(group_fairness_weights)

        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)


class AdaptativePriorityReweightingEOD(Transformer):

    def __init__(self, sensitive_attr='', epochs=50, eta=0, alpha=0):
        self.sensitive_attr = sensitive_attr
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.input_shape = None
        self.model = None
        self.classes_ = None
        self.history = None

    def _compute_sample_weights(self, true_labels, predicted_probabilities, protected_attributes, multipliers, eta,
                                decision_boundary=0.5):
        num_samples = len(true_labels)
        exponential_term = np.exp(-eta * abs(predicted_probabilities - decision_boundary))
        weight_component = np.zeros(num_samples)

        for attr in protected_attributes:
            weight_component += exponential_term * np.sum(attr) / np.sum(exponential_term * attr) * attr

        combined_weights = np.zeros(num_samples)
        for i, multiplier in enumerate(multipliers):
            combined_weights += multiplier * protected_attributes[i]

        sample_weights = combined_weights * weight_component

        return sample_weights

    def _compute_group_weights(self, predictions, true_labels, protected_attributes, alpha):
        group_weights = []

        for p in protected_attributes:
            protected_positive_idxs = np.where(np.logical_and(p > 0, true_labels > 0))
            protected_negative_idxs = np.where(np.logical_and(p > 0, true_labels <= 0))
            all_positive_idxs = np.where(true_labels > 0)
            all_negative_idxs = np.where(true_labels <= 0)

            weight1 = (np.sum(true_labels[protected_positive_idxs]) * np.sum(
                predictions[all_positive_idxs]) + alpha) / (
                              np.sum(true_labels[all_positive_idxs]) * np.sum(
                          predictions[protected_positive_idxs]) + alpha)

            weight2 = (np.sum(1 - true_labels[protected_negative_idxs]) * np.sum(
                1 - predictions[all_negative_idxs]) + alpha) / (
                              np.sum(1 - true_labels[all_negative_idxs]) * np.sum(
                          1 - predictions[protected_negative_idxs]) + alpha)

            group_weights.extend([weight1, weight2])

        return group_weights

    def fit(self, dataset):
        if self.model is None:
            self.input_shape = dataset.features.shape[1]
            self.classes_ = np.array([dataset.unfavorable_label, dataset.favorable_label])

        sensitive_index = dataset.protected_attribute_names.index(self.sensitive_attr)

        features = dataset.features
        labels = (dataset.labels == dataset.favorable_label).reshape(features.shape[0]).astype(int)

        protected_attributes = (dataset.protected_attributes[:, sensitive_index] ==
                                dataset.unprivileged_protected_attributes[sensitive_index]).astype(int)

        protected_attributes_list = [protected_attributes, 1 - protected_attributes]
        label_combinations = [protected_attributes * labels, protected_attributes * (1 - labels), \
                              (1 - protected_attributes) * labels,
                              (1 - protected_attributes) * (1 - labels)]
        fairness_multipliers = np.ones(len(label_combinations))
        sample_weights = np.array([1] * features.shape[0])

        for epoch in range(self.epochs):
            # Train logistic regression model
            self.model = LogisticRegression(max_iter=10000)
            self.model = self.model.fit(features, labels, sample_weights)

            predictions_train = self.model.predict(features)
            prediction_probabilities = self.model.predict_proba(features)[:, 0].astype(np.float32)

            # Compute weights and multipliers
            sample_weights = self._compute_sample_weights(labels, prediction_probabilities, label_combinations,
                                                          fairness_multipliers, self.eta)
            group_fairness_weights = self._compute_group_weights(predictions_train, labels, protected_attributes_list,
                                                                 self.alpha)
            fairness_multipliers *= np.array(group_fairness_weights)

        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)

class AdaptativePriorityReweightingEOP(Transformer):

    def __init__(self, sensitive_attr='', epochs=50, eta=0, alpha=0):
        self.sensitive_attr = sensitive_attr
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.input_shape = None
        self.model = None
        self.classes_ = None
        self.history = None

    def _compute_sample_weights(self, true_labels, predicted_probabilities, protected_attrs, multipliers, eta,
                                decision_boundary=0.5):
        num_samples = len(true_labels)
        sample_weights = np.zeros(num_samples)

        exponential_term = np.exp(-eta * abs(predicted_probabilities - decision_boundary))

        for attr in protected_attrs:
            positive_protected_indices = np.where(np.logical_and(attr > 0, true_labels > 0))

            sample_weights += exponential_term * len(positive_protected_indices[0]) / np.sum(
                exponential_term[positive_protected_indices]) * attr

        attribute_weighted_sum = sum(m * p for m, p in zip(multipliers, protected_attrs))

        sample_weights *= attribute_weighted_sum

        # Set the weight to 1 for samples with non-positive true labels
        sample_weights = np.where(true_labels > 0, sample_weights, 1)

        return sample_weights

    def _compute_group_weights(self, predictions, true_labels, protected_attrs, alpha):
        weights = []

        for attr in protected_attrs:
            positive_protected_indices = np.where(np.logical_and(attr > 0, true_labels > 0))
            positive_indices = np.where(true_labels > 0)

            weight = (np.sum(true_labels[positive_protected_indices]) * np.sum(predictions[positive_indices]) + alpha) \
                     / (np.sum(true_labels[positive_indices]) * np.sum(predictions[positive_protected_indices]) + alpha)

            weights.append(weight)

        return weights

    def fit(self, dataset):
        if self.model is None:
            self.input_shape = dataset.features.shape[1]
            self.classes_ = np.array([dataset.unfavorable_label, dataset.favorable_label])

        sensitive_index = dataset.protected_attribute_names.index(self.sensitive_attr)

        features = dataset.features
        labels = (dataset.labels == dataset.favorable_label).reshape(features.shape[0]).astype(int)

        protected_attributes = (dataset.protected_attributes[:, sensitive_index] ==
                                dataset.unprivileged_protected_attributes[sensitive_index]).astype(int)

        protected_attributes_list = [protected_attributes, 1 - protected_attributes]
        fairness_multipliers = np.ones(len(protected_attributes_list))
        sample_weights = np.array([1] * features.shape[0])

        for epoch in range(self.epochs):
            # Train logistic regression model
            self.model = LogisticRegression(max_iter=10000)
            self.model = self.model.fit(features, labels, sample_weights)

            predictions_train = self.model.predict(features)
            prediction_probabilities = self.model.predict_proba(features)[:, 0].astype(np.float32)

            # Compute weights and multipliers
            sample_weights = self._compute_sample_weights(labels, prediction_probabilities, protected_attributes_list,
                                                          fairness_multipliers, self.eta)
            group_fairness_weights = self._compute_group_weights(predictions_train, labels, protected_attributes_list,
                                                                 self.alpha)
            fairness_multipliers *= np.array(group_fairness_weights)

        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)


def describe_metrics(metrics):
    print("Fitness: {:6.4f}".format(metrics['fitness']))
    print("Overall accuracy: {:6.4f}".format(metrics['overall_acc']))
    print("F1 Score: {:6.4f}".format(metrics['f1_score']))
    print("Mathew Correlation Coefficient: {:6.4f}".format(metrics['MCC']))
    print("Balanced accuracy: {:6.4f}".format(metrics['bal_acc']))
    print("Average odds difference value: {:6.4f}".format(metrics['avg_odds_diff']))
    print("Statistical parity difference value: {:6.4f}".format(metrics['stat_par_diff']))
    print("Equal opportunity difference value: {:6.4f}".format(metrics['eq_opp_diff']))
