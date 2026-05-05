import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss, roc_auc_score
from tabpfn import TabPFNClassifier

TABPFN_MAX_TRAIN = 10_000

_DEFAULT_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def _lgbm_objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'objective':         'binary',
        'metric':            'binary_logloss',
        'verbosity':         -1,
        'n_estimators':      trial.suggest_int('n_estimators', 100, 1000),
        'max_depth':         trial.suggest_int('max_depth', 3, 10),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves':        trial.suggest_int('num_leaves', 8, 128),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state':      42,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_val)[:, 1]
    return brier_score_loss(y_val, preds)


def train_lgbm(train_df, feature_cols, target, n_trials=50, cv=None):
    """
    LightGBM with per-fold Optuna tuning (minimises Brier score).

    Median imputation is fit on the train fold only to prevent leakage.

    Returns
    -------
    oof_preds : ndarray, shape (len(train_df),)
    best_params : dict — hyperparameters from the last fold (used for final retraining)
    """
    if cv is None:
        cv = _DEFAULT_CV

    X = train_df[feature_cols]
    y = train_df[target]
    oof_preds = np.zeros(len(train_df))
    best_params = {}

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        medians = X_tr.median()
        X_tr  = X_tr.fillna(medians)
        X_val = X_val.fillna(medians)

        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: _lgbm_objective(trial, X_tr, y_tr, X_val, y_val),
            n_trials=n_trials,
        )
        best_params = study.best_params | {'objective': 'binary', 'verbosity': -1, 'random_state': 42}

        model = lgb.LGBMClassifier(**best_params)
        model.fit(X_tr, y_tr)
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

        print(f"  Fold {fold+1}: Brier={brier_score_loss(y_val, oof_preds[val_idx]):.4f}  "
              f"AUC={roc_auc_score(y_val, oof_preds[val_idx]):.4f}")

    return oof_preds, best_params


def eval_lgbm_on_test(train_df, test_df, feature_cols, target, best_params):
    """
    Retrain LightGBM on the full training set and predict on the held-out test set.

    Imputation medians are computed from training data only.
    """
    medians = train_df[feature_cols].median()
    X_tr = train_df[feature_cols].fillna(medians)
    X_te = test_df[feature_cols].fillna(medians)

    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_tr, train_df[target])
    return model.predict_proba(X_te)[:, 1]


def train_tabpfn(train_df, feature_cols, target, cv=None):
    """
    TabPFN across CV folds with stratified subsampling if the training fold exceeds
    TABPFN_MAX_TRAIN (10,000) samples.

    TabPFN requires no hyperparameter tuning — this function only handles
    the CV loop, imputation, and subsampling.

    Returns
    -------
    oof_preds : ndarray, shape (len(train_df),)
    """
    if cv is None:
        cv = _DEFAULT_CV

    X = train_df[feature_cols]
    y = train_df[target]
    oof_preds = np.zeros(len(train_df))

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        medians = X_tr.median()
        X_tr  = X_tr.fillna(medians).values
        X_val = X_val.fillna(medians).values
        y_tr  = y_tr.values

        if len(X_tr) > TABPFN_MAX_TRAIN:
            rng = np.random.default_rng(42 + fold)
            pos_idx = np.where(y_tr == 1)[0]
            neg_idx = np.where(y_tr == 0)[0]
            n_pos = int(TABPFN_MAX_TRAIN * y_tr.mean())
            n_neg = TABPFN_MAX_TRAIN - n_pos
            sel = np.concatenate([
                rng.choice(pos_idx, min(n_pos, len(pos_idx)), replace=False),
                rng.choice(neg_idx, min(n_neg, len(neg_idx)), replace=False),
            ])
            X_tr, y_tr = X_tr[sel], y_tr[sel]
            print(f"  Fold {fold+1}: subsampled to {len(X_tr):,} training examples")

        model = TabPFNClassifier(device='cpu', ignore_pretraining_limits=True)
        model.fit(X_tr, y_tr)
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

        y_val_s = y.iloc[val_idx]
        print(f"  Fold {fold+1}: Brier={brier_score_loss(y_val_s, oof_preds[val_idx]):.4f}  "
              f"AUC={roc_auc_score(y_val_s, oof_preds[val_idx]):.4f}")

    return oof_preds


def eval_tabpfn_on_test(train_df, test_df, feature_cols, target):
    """
    Retrain TabPFN on the full training set (with subsampling if needed) and
    predict on the held-out test set.
    """
    X_tr = train_df[feature_cols]
    medians = X_tr.median()
    X_tr = X_tr.fillna(medians).values
    X_te = test_df[feature_cols].fillna(medians).values
    y_tr = train_df[target].values

    if len(X_tr) > TABPFN_MAX_TRAIN:
        rng = np.random.default_rng(42)
        pos_idx = np.where(y_tr == 1)[0]
        neg_idx = np.where(y_tr == 0)[0]
        n_pos = int(TABPFN_MAX_TRAIN * y_tr.mean())
        n_neg = TABPFN_MAX_TRAIN - n_pos
        sel = np.concatenate([
            rng.choice(pos_idx, min(n_pos, len(pos_idx)), replace=False),
            rng.choice(neg_idx, min(n_neg, len(neg_idx)), replace=False),
        ])
        X_tr, y_tr = X_tr[sel], y_tr[sel]
        print(f"  Final model: subsampled to {len(X_tr):,} training examples")

    model = TabPFNClassifier(device='cpu', ignore_pretraining_limits=True)
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_te)[:, 1]
