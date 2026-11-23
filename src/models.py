"""Model catalogue: transparent baselines, standard machine learning, and
TensorFlow neural networks (three multilayer perceptrons and an FT-Transformer).

Every model exposes fit/predict_proba and carries its own preprocessing, which
is fitted on the training data passed to fit() only. Neural networks are built
afresh at every fit (new weights and a new optimiser state).
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from . import config
from .preprocessing import make_preprocessor

TRANSPARENT = ["logistic_age_grade", "penalised_logistic"]
STANDARD_ML = ["random_forest", "hist_gboost"]
MLPS = ["tf_compact", "tf_medium", "tf_regularised"]
LABELS = {
    "base_rate": "Base rate",
    "logistic_age_grade": "Age + grade LR",
    "penalised_logistic": "Penalised LR",
    "random_forest": "Random forest",
    "hist_gboost": "Gradient boosting",
    "tf_compact": "Compact MLP",
    "tf_medium": "Medium MLP",
    "tf_regularised": "Regularised MLP",
    "ft_transformer": "FT-Transformer",
}

# Each layer: (units, dropout, batch normalisation)
ARCHITECTURES = {
    "compact": {"layers": [(32, 0.2, False), (16, 0.2, False)], "l2": 0.0},
    "medium": {"layers": [(128, 0.3, True), (64, 0.3, False), (32, 0.3, False)], "l2": 1e-4},
    "regularised": {"layers": [(64, 0.4, False), (32, 0.4, False)], "l2": 1e-3},
}


class AgeGradeLogistic(BaseEstimator, ClassifierMixin):
    """Unpenalised logistic regression on age and grade (those available)."""

    def __init__(self, features=None, seed: int = config.SEED):
        self.features = features
        self.seed = seed

    def fit(self, X, y):
        avail = self.features if self.features is not None else list(X.columns)
        self.cols_ = [c for c in ("age", "grade") if c in avail]
        self.pre_ = make_preprocessor(X, self.cols_)
        self.clf_ = LogisticRegression(penalty=None, max_iter=2000)
        self.clf_.fit(self.pre_.fit_transform(X[self.cols_]), y)
        self.classes_ = self.clf_.classes_
        return self

    def predict_proba(self, X):
        return self.clf_.predict_proba(self.pre_.transform(X[self.cols_]))


def _pipe(est, df, feats, scale=True):
    return Pipeline([("pre", make_preprocessor(df, feats, scale=scale)), ("clf", est)])


def build_mlp(input_dim, arch, seed):
    import tensorflow as tf
    spec = ARCHITECTURES[arch]
    tf.keras.utils.set_random_seed(seed)
    reg = tf.keras.regularizers.l2(spec["l2"]) if spec["l2"] > 0 else None
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(input_dim,))])
    for units, dropout, bn in spec["layers"]:
        model.add(tf.keras.layers.Dense(units, activation="relu", kernel_regularizer=reg))
        if bn:
            model.add(tf.keras.layers.BatchNormalization())
        if dropout:
            model.add(tf.keras.layers.Dropout(dropout, seed=seed))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer=tf.keras.optimizers.Adam(config.NN_TRAINING["learning_rate"]),
                  loss="binary_crossentropy")
    return model


def build_ft_transformer(n_features, seed, d_token=32, n_heads=4, n_blocks=2, dropout=0.1):
    """Compact feature-tokeniser transformer (after Gorishniy et al., 2021)."""
    import tensorflow as tf
    from tensorflow.keras import layers

    class CLSToken(layers.Layer):
        def build(self, input_shape):
            self.cls = self.add_weight(shape=(1, 1, input_shape[-1]), name="cls",
                                       initializer=tf.keras.initializers.RandomNormal(stddev=0.02))

        def call(self, x):
            return tf.concat([tf.tile(self.cls, [tf.shape(x)[0], 1, 1]), x], axis=1)

    tf.keras.utils.set_random_seed(seed)
    inp = layers.Input(shape=(n_features,))
    seq = CLSToken()(layers.Dense(d_token)(layers.Reshape((n_features, 1))(inp)))
    for _ in range(n_blocks):
        att = layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_token, dropout=dropout)(seq, seq)
        seq = layers.LayerNormalization()(layers.Add()([seq, att]))
        ff = layers.Dense(d_token, activation=None)(layers.Dense(2 * d_token, activation="gelu")(seq))
        seq = layers.LayerNormalization()(layers.Add()([seq, ff]))
    out = layers.Dense(1, activation="sigmoid")(layers.Dropout(dropout, seed=seed)(seq[:, 0, :]))
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(config.NN_TRAINING["learning_rate"]),
                  loss="binary_crossentropy")
    return model


class KerasClassifier(BaseEstimator, ClassifierMixin):
    """Neural-network wrapper. Training: Adam, binary cross-entropy, early
    stopping on a stratified 20% validation split of the training data, with
    optional inverse-frequency class weighting."""

    def __init__(self, features, arch="compact", class_weight=True, seed=config.SEED):
        self.features = features
        self.arch = arch
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
        if self.arch == "ft_transformer":
            self.model_ = build_ft_transformer(Xt.shape[1], self.seed)
        else:
            self.model_ = build_mlp(Xt.shape[1], self.arch, self.seed)
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=hp["patience"],
                                              restore_best_weights=True)
        self.history_ = self.model_.fit(Xa, ya, validation_data=(Xv, yv), epochs=hp["max_epochs"],
                                        batch_size=hp["batch_size"], class_weight=cw,
                                        callbacks=[es], verbose=0, shuffle=True).history
        self.n_input_ = Xt.shape[1]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        Xt = self.pre_.transform(X[self.features]).astype("float32")
        p = np.asarray(self.model_(Xt, training=False)).ravel().astype(float)
        return np.column_stack([1 - p, p])


def build_models(df, features, seed=config.SEED, include_nn=True, class_weight=True,
                 include_ftt=False, reference_vars=None):
    """reference_vars: variables for the age-and-grade reference model; by default
    age and grade wherever recorded in df. The reference model is the same
    clinical model for every feature set."""
    if reference_vars is None:
        reference_vars = [c for c in ("age", "grade") if c in df.columns and df[c].notna().any()]
    models = {
        "base_rate": DummyClassifier(strategy="prior"),
        "logistic_age_grade": AgeGradeLogistic(features=reference_vars, seed=seed),
        "penalised_logistic": _pipe(LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                                                       C=1.0, max_iter=10000, random_state=seed),
                                    df, features),
        "random_forest": _pipe(RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                                      random_state=seed, n_jobs=1), df, features, scale=False),
        "hist_gboost": _pipe(HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=400,
                                                            l2_regularization=1.0, random_state=seed),
                             df, features, scale=False),
    }
    if include_nn:
        for name in MLPS:
            models[name] = KerasClassifier(features, arch=name[3:], class_weight=class_weight, seed=seed)
    if include_ftt:
        models["ft_transformer"] = KerasClassifier(features, arch="ft_transformer",
                                                   class_weight=class_weight, seed=seed)
    return models
