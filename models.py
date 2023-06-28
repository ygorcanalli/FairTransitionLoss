import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import InputLayer, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from aif360.algorithms import Transformer
from tensorflow.keras import backend as K
from dataset_readers import adult_dataset_reader

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
                 num_epochs=50, batch_size=16, patience=3):

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

    def _compile_model(self):
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=self.input_shape))

        for hidden_size in self.hidden_sizes:
            self.model.add(Dense(hidden_size, activation='relu'))
            self.model.add(Dropout(self.dropout))

        self.model.add(Dense(self.num_classes, activation="softmax"))
        self.model.compile(optimizer='adam', metrics=['accuracy'],
                           loss='categorical_crossentropy')
    def fit(self, dataset, verbose=False):
        if self.model is None:
            self.input_shape = dataset.features.shape[1]
            self._compile_model()
            self.classes_ = np.array([dataset.unfavorable_label, dataset.favorable_label])

        callback = EarlyStopping(monitor='loss', patience=self.patience)
        dataset_cp = dataset.copy()
        X = dataset_cp.features
        y_expanded = np.zeros( shape=(X.shape[0], 2) )

        y_expanded[:,0] = (dataset_cp.labels == dataset_cp.unfavorable_label).reshape(X.shape[0]).astype(int)
        y_expanded[:,1] = (dataset_cp.labels == dataset_cp.favorable_label).reshape(X.shape[0]).astype(int)

        self.model.fit(X, y_expanded, epochs=self.num_epochs,
                       batch_size=self.batch_size,# callbacks=[callback],
                       verbose=verbose)

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
                 hidden_sizes=[32, 64, 32], dropout=0.1, patience=3,
                 num_epochs=50, batch_size=16):
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

    def _compile_model(self):
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=self.input_shape))

        for hidden_size in self.hidden_sizes:
            self.model.add(Dense(hidden_size, activation='relu'))
            self.model.add(Dropout(self.dropout))

        self.model.add(Dense(self.num_classes, activation="softmax"))
        self.model.compile(optimizer='adam',
                           loss=fair_forward(self.p_privileged, self.p_protected))
    def fit(self, dataset, verbose=False):
        if self.model is None:
            self.input_shape = dataset.features.shape[1]
            self._compile_model()
            self.classes_ = np.array([dataset.unfavorable_label, dataset.favorable_label])

        callback = EarlyStopping(monitor='loss', patience=self.patience)

        X = dataset.features
        y_expanded = np.zeros( shape=(X.shape[0], 4) )
        sensitive_index = dataset.protected_attribute_names.index(self.sensitive_attr)

        y_expanded[:,0] = (dataset.protected_attributes[:,sensitive_index] == dataset.privileged_protected_attributes[sensitive_index]).astype(int)
        y_expanded[:,1] = (dataset.protected_attributes[:,sensitive_index] == dataset.unprivileged_protected_attributes[sensitive_index]).astype(int)
        y_expanded[:,2] = (dataset.labels == dataset.unfavorable_label).reshape(X.shape[0]).astype(int)
        y_expanded[:,3] = (dataset.labels == dataset.favorable_label).reshape(X.shape[0]).astype(int)

        self.model.fit(X, y_expanded, epochs=self.num_epochs,
                       batch_size=self.batch_size,# callbacks=[callback],
                       verbose=verbose)

        return self

    def predict_proba(self, X):
        return self.model.predict(X)

    def predict(self, X):
        logits = self.predict_proba(X)
        return np.argmax(logits, axis=1)

def describe_metrics(metrics):
    print("Fitness: {:6.4f}".format(metrics['fitness']))
    print("Overall accuracy: {:6.4f}".format(metrics['overall_acc']))
    print("Balanced accuracy: {:6.4f}".format(metrics['bal_acc']))
    print("Average odds difference value: {:6.4f}".format(metrics['avg_odds_diff']))
    print("Statistical parity difference value: {:6.4f}".format(metrics['stat_par_diff']))
    print("Equal opportunity difference value: {:6.4f}".format(metrics['eq_opp_diff']))