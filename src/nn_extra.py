"""Neural-network estimators used only in the post hoc revision analyses
(PROTOCOL.md, section 12). The prespecified estimators are in models.py and
are not modified.

ConfigurableMLP: a multilayer perceptron with the same training procedure as
the prespecified networks (Adam, binary cross-entropy, batch size 64, up to 200
epochs, early stopping on a stratified 20% validation split, patience 20) but
with architecture and regularisation given as arguments, for nested tuning.

CoxMLP: a multilayer perceptron with a linear output trained on the negative
Cox partial log-likelihood (Breslow handling of ties), full-batch, with early
stopping on a 20% validation split stratified on the event indicator.
Survival probabilities use the Breslow estimate of the baseline cumulative
hazard in the training data.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

from . import config
from .preprocessing import make_preprocessor


def _dense_stack(input_dim, hidden, dropout, l2, seed, out_activation, out_bias=True):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    reg = tf.keras.regularizers.l2(l2) if l2 > 0 else None
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(input_dim,))])
    for units in hidden:
        model.add(tf.keras.layers.Dense(units, activation="relu", kernel_regularizer=reg))
        if dropout:
            model.add(tf.keras.layers.Dropout(dropout, seed=seed))
    model.add(tf.keras.layers.Dense(1, activation=out_activation, use_bias=out_bias))
    return model


class ConfigurableMLP(BaseEstimator, ClassifierMixin):
    def __init__(self, features, hidden=(32, 16), dropout=0.2, l2=0.0, class_weight=False,
                 seed=config.SEED):
        self.features = features
        self.hidden = hidden
        self.dropout = dropout
        self.l2 = l2
        self.class_weight = class_weight
        self.seed = seed

    def fit(self, X, y):
        import tensorflow as tf
        hp = config.NN_TRAINING
        y = np.asarray(y).astype("float32")
        self.pre_ = make_preprocessor(X, self.features)
        Xt = self.pre_.fit_transform(X[self.features]).astype("float32")
        Xa, Xv, ya, yv = train_test_split(Xt, y, test_size=hp["validation_fraction"],
                                          random_state=self.seed, stratify=y)
        cw = None
        if self.class_weight:
            pos = float(ya.sum())
            cw = {0: 1.0, 1: (len(ya) - pos) / pos}
        self.model_ = _dense_stack(Xt.shape[1], self.hidden, self.dropout, self.l2, self.seed, "sigmoid")
        self.model_.compile(optimizer=tf.keras.optimizers.Adam(hp["learning_rate"]),
                            loss="binary_crossentropy")
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=hp["patience"],
                                              restore_best_weights=True)
        self.history_ = self.model_.fit(Xa, ya, validation_data=(Xv, yv), epochs=hp["max_epochs"],
                                        batch_size=hp["batch_size"], class_weight=cw, callbacks=[es],
                                        verbose=0, shuffle=True).history
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        Xt = self.pre_.transform(X[self.features]).astype("float32")
        p = np.asarray(self.model_(Xt, training=False)).ravel().astype(float)
        return np.column_stack([1 - p, p])


def _tie_end(t_desc):
    """For times sorted in descending order, index of the last element of each
    element's tie group (so that the risk set includes all tied subjects)."""
    n = len(t_desc)
    end = np.empty(n, dtype=np.int64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and t_desc[j + 1] == t_desc[i]:
            j += 1
        end[i:j + 1] = j
        i = j + 1
    return end


class _PartialLik:
    def __init__(self, t, e):
        order = np.argsort(-t, kind="mergesort")
        self.order = order
        self.e = e[order].astype("float32")
        self.end = _tie_end(t[order])
        self.n_events = max(float(self.e.sum()), 1.0)

    def __call__(self, risk):
        import tensorflow as tf
        r = tf.gather(tf.reshape(risk, [-1]), self.order)
        m = tf.reduce_max(r)
        lcs = tf.math.log(tf.cumsum(tf.exp(r - m))) + m
        lcs = tf.gather(lcs, self.end)
        return -tf.reduce_sum((r - lcs) * self.e) / self.n_events


class CoxMLP(BaseEstimator):
    def __init__(self, features, hidden=(32, 16), dropout=0.2, l2=0.0, max_epochs=1000, patience=50,
                 learning_rate=1e-3, seed=config.SEED):
        self.features = features
        self.hidden = hidden
        self.dropout = dropout
        self.l2 = l2
        self.max_epochs = max_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.seed = seed

    def fit(self, X, time, event):
        import tensorflow as tf
        time = np.asarray(time, float)
        event = np.asarray(event, int)
        self.pre_ = make_preprocessor(X, self.features)
        Xt = self.pre_.fit_transform(X[self.features]).astype("float32")
        ia, iv = train_test_split(np.arange(len(Xt)), test_size=config.NN_TRAINING["validation_fraction"],
                                  random_state=self.seed, stratify=event)
        la, lv = _PartialLik(time[ia], event[ia]), _PartialLik(time[iv], event[iv])
        Xa, Xv = tf.constant(Xt[ia]), tf.constant(Xt[iv])
        self.model_ = _dense_stack(Xt.shape[1], self.hidden, self.dropout, self.l2, self.seed, None,
                                   out_bias=False)
        opt = tf.keras.optimizers.Adam(self.learning_rate)
        best, best_w, wait, hist = np.inf, None, 0, {"loss": [], "val_loss": []}
        for _ in range(self.max_epochs):
            with tf.GradientTape() as tape:
                loss = la(self.model_(Xa, training=True))
                if self.model_.losses:
                    loss = loss + tf.add_n(self.model_.losses)
            grads = tape.gradient(loss, self.model_.trainable_variables)
            opt.apply_gradients(zip(grads, self.model_.trainable_variables))
            vl = float(lv(self.model_(Xv, training=False)))
            hist["loss"].append(float(loss))
            hist["val_loss"].append(vl)
            if vl < best - 1e-6:
                best, best_w, wait = vl, [w.numpy() for w in self.model_.weights], 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_w is not None:
            self.model_.set_weights(best_w)
        self.history_ = hist
        # Breslow baseline cumulative hazard on all training patients
        r = np.exp(self._risk_t(Xt))
        ut = np.unique(time[event == 1])
        h = np.array([event[(time == s)].sum() / r[time >= s].sum() for s in ut])
        self.base_times_, self.base_cumhaz_ = ut, np.cumsum(h)
        return self

    def _risk_t(self, Xt):
        return np.asarray(self.model_(Xt, training=False)).ravel().astype(float)

    def predict_risk(self, X):
        return self._risk_t(self.pre_.transform(X[self.features]).astype("float32"))

    def predict_survival(self, X, times):
        r = np.exp(self.predict_risk(X))
        H = np.array([self.base_cumhaz_[self.base_times_ <= t][-1] if (self.base_times_ <= t).any() else 0.0
                      for t in times])
        return np.exp(-np.outer(r, H))
