"""
Supervised models over pangenome bin abundance.

Gradient-boosted classifiers are used here as a descriptive tool rather than a
predictor: how well the abundance of one organism's bins separates cases from
controls, and which bins carry that signal. Bin importance is measured with
SHAP values, which also give the pairwise interactions between bins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

#: Hyperparameters shared by the replicate models and the final SHAP model.
MODEL_PARAMS = dict(
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    n_jobs=-1,
)

#: Smallest number of participants in the minority class worth fitting a
#: model to. Counted in participants rather than samples, because a split
#: moves a participant's samples together.
MIN_CLASS_SIZE = 5

#: The validation fold is one of this many, about a quarter of the samples
VALIDATION_FOLDS = 4

#: The early-stopping fold is one of this many, from what is left to fit on
STOPPING_FOLDS = 5


@dataclass
class ClassifierResult:
    """
    What one organism-by-study classifier produced.

    Attributes
    ----------
    replicate_auc:
        Validation ROC-AUC from each split, no participant on both sides.
    shap:
        Mean absolute SHAP value per bin, from a model fit to all samples.
    interactions:
        Symmetric matrix of mean absolute SHAP interaction values among the
        most important bins. The total interaction between two distinct bins
        is twice the tabulated value, since it is split across both cells.
    n_samples:
        Samples the model was fit to.
    """

    replicate_auc: List[float]
    shap: pd.Series
    interactions: pd.DataFrame | None
    n_samples: int


def fit_classifier(
    features: pd.DataFrame,
    labels: pd.Series,
    participants: pd.Series,
    n_replicates: int = 10,
    n_estimators: int = 400,
    n_interaction_features: int = 10,
    max_interaction_samples: int = 200,
) -> ClassifierResult | None:
    """
    Fit replicate classifiers to measure how separable the two classes are,
    then one model on all samples to attribute that separation to bins.

    ``participants`` names the person each sample came from. Every split keeps
    a participant's samples on one side, so no model is validated on a person
    it was fit to; cohorts that sampled each person many times would
    otherwise be scored on how well a model recognizes individuals.

    Returns ``None`` when one class has fewer than ``MIN_CLASS_SIZE``
    participants.
    """
    import xgboost as xgb

    labeled = labels.dropna()
    X = np.log1p(features.reindex(index=labeled.index))
    y = labeled.values.astype(int)
    groups = participants.reindex(labeled.index)
    if groups.isna().any():
        raise ValueError(f"{int(groups.isna().sum())} labeled samples have no participant")
    groups = groups.values

    if min(len(set(groups[y == label])) for label in (0, 1)) < MIN_CLASS_SIZE:
        return None

    replicates = [_fit_one(X, y, groups, seed, n_estimators) for seed in range(n_replicates)]

    # Refit on every sample, with early stopping replaced by the typical number
    # of rounds the replicate models settled on
    rounds = max(int(np.median([model.best_iteration or 100 for model, _ in replicates])), 10)
    model = xgb.XGBClassifier(n_estimators=rounds, random_state=0, **MODEL_PARAMS)
    model.fit(X, y, verbose=False)

    shap_values = mean_abs_shap(model, X)
    return ClassifierResult(
        replicate_auc=[auc for _, auc in replicates],
        shap=shap_values,
        interactions=shap_interactions(
            model, X, shap_values, n_interaction_features, max_interaction_samples
        ),
        n_samples=len(y),
    )


def _grouped_split(y: np.ndarray, groups: np.ndarray, n_folds: int, seed: int) -> tuple:
    """
    Row positions of one fold (about ``1 / n_folds`` of the samples) and of
    the rest, stratified by class, with every group wholly on one side.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    folds = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rest, fold = next(folds.split(np.zeros(len(y)), y, groups))
    if len(np.unique(y[fold])) < 2 or len(np.unique(y[rest])) < 2:
        raise ValueError(
            f"A split by participant with seed {seed} left one side with a single class; "
            f"{len(np.unique(groups))} participants are too few for {n_folds} folds"
        )
    return rest, fold


def _fit_one(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, seed: int, n_estimators: int):
    """Fit one classifier on a split by participant, returning it with its validation AUC."""
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    fit, val = _grouped_split(y, groups, VALIDATION_FOLDS, seed)
    train, stop = _grouped_split(y[fit], groups[fit], STOPPING_FOLDS, seed)
    X_val, y_val = X.iloc[val], y[val]
    X_fit, y_fit = X.iloc[fit].iloc[train], y[fit][train]
    X_stop, y_stop = X.iloc[fit].iloc[stop], y[fit][stop]

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=seed,
        **MODEL_PARAMS,
    )
    model.fit(X_fit, y_fit, eval_set=[(X_stop, y_stop)], verbose=False)

    return model, float(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))


def mean_abs_shap(model, X: pd.DataFrame) -> pd.Series:
    """Mean absolute SHAP value for each feature."""
    import shap

    values = np.asarray(shap.TreeExplainer(model).shap_values(X))
    if values.ndim == 3:
        values = values.mean(axis=2)
    return pd.Series(np.abs(values).mean(axis=0), index=X.columns)


def shap_interactions(
    model,
    X: pd.DataFrame,
    shap_values: pd.Series,
    n_features: int,
    max_samples: int,
    seed: int = 42,
) -> pd.DataFrame | None:
    """
    Mean absolute SHAP interaction values among the most important features.

    XGBoost needs the full feature matrix to compute interactions, so the
    matrix is subsampled by row (the calculation is quadratic in features) and
    sliced down to the top features afterwards.
    """
    import shap

    features = list(shap_values.sort_values(ascending=False).head(n_features).index)
    if len(features) < 2:
        return None

    sample = X if len(X) <= max_samples else X.sample(max_samples, random_state=seed)
    values = np.asarray(shap.TreeExplainer(model).shap_interaction_values(sample))
    axes = (0, 3) if values.ndim == 4 else 0
    mean_abs = np.abs(values).mean(axis=axes)

    positions = [X.columns.get_loc(feature) for feature in features]
    return pd.DataFrame(
        mean_abs[np.ix_(positions, positions)], index=features, columns=features
    )
