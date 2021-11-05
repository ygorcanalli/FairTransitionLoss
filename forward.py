import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import InputLayer, Dropout, Dense
from aif360.algorithms import Transformer
from tensorflow.keras import backend as K

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

class FairMLP(Transformer):

    def __init__(self, sensitive_attr='',
                 privileged_demotion=0.1, privileged_promotion=0.01,
                 protected_demotion=0.01, protected_promotion=0.1,
                 hidden_sizes=[32, 64, 32],
                 num_epochs=10, batch_size=64):
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
        self.sensitive_attr = sensitive_attr
        self.classes_ =  np.array([0, 1])

    def _compile_model(self):
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=self.input_shape))
        self.model.add(Dropout(0.1))

        for hidden_size in self.hidden_sizes:
            self.model.add(Dense(hidden_size, activation='relu'))

        self.model.add(Dense(self.num_classes, activation="softmax"))
        self.model.compile(optimizer='adam',
                           #loss=fair_forward_2(self.p_privileged, self.p_protected))
                           loss=binary_crossentropy, metrics=['accuracy'])
    def fit(self, dataset):
        if self.model is None:
            self.input_shape = dataset.features.shape[1]
            self._compile_model()

        X = dataset.features
        y_expanded = np.zeros( shape=(X.shape[0], 4) )
        sensitive_index = dataset.protected_attribute_names.index(self.sensitive_attr)

        y_expanded[:,0] = (dataset.protected_attributes[:,sensitive_index] == dataset.privileged_protected_attributes[sensitive_index]).astype(int)
        y_expanded[:,1] = (dataset.protected_attributes[:,sensitive_index] == dataset.unprivileged_protected_attributes[sensitive_index]).astype(int)
        y_expanded[:,2] = (dataset.labels == dataset.unfavorable_label).reshape(X.shape[0]).astype(int)
        y_expanded[:,3] = (dataset.labels == dataset.favorable_label).reshape(X.shape[0]).astype(int)

        #self.model.summary()
        self.model.fit(X, y_expanded[:,-2:], epochs=self.num_epochs, batch_size=self.batch_size, verbose=False)

    def predict_proba(self, X):
        return self.model.predict(X)

    def predict(self, X):
        logits = self.predict_proba(X)
        return np.argmax(logits, axis=1)