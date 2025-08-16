import io
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, RobustScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,mean_absolute_error, mean_squared_error, r2_score, confusion_matrix


from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor, BaggingClassifier, BaggingRegressor
from sklearn.ensemble import StackingClassifier, StackingRegressor

from sklearn.base import BaseEstimator, TransformerMixin


class ArrayToDataFrame(BaseEstimator, TransformerMixin):
    """
    Ensures numpy arrays are converted back into DataFrames 
    with correct column names before passing to base estimators.
    """

    def __init__(self, feature_names=None):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        # Save feature names if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        return self

    def transform(self, X):
        # Convert numpy arrays back to DataFrame
        if isinstance(X, pd.DataFrame):
            return X
        else:
            return pd.DataFrame(X, columns=self.feature_names)

# -----------------------------
# Utility helpers
# -----------------------------
def detect_task_type(y: pd.Series) -> str:
    """Infer task type from target column; allow override in UI."""
    if pd.api.types.is_numeric_dtype(y):
        # small unique numeric values can still be classification (e.g., 0/1)
        nunique = y.nunique(dropna=True)
        if nunique <= 20 and set(y.dropna().unique()).issubset(set(range(-10, 100000))):
            return "classification"
        else:
            return "regression"
    else:
        return "classification"


def split_feature_types(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    num_cols = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c != target and not pd.api.types.is_numeric_dtype(df[c])]
    return num_cols, cat_cols


def build_preprocessor(
    X: pd.DataFrame,
    target: str,
    drop_cols: List[str],
    per_col_imp: Dict[str, str],
    per_col_enc: Dict[str, str],
    per_col_scale: Dict[str, str],
    use_global_knn_for_numeric: bool,
) -> Tuple[ColumnTransformer, List[str], List[str], Dict[str, str]]:
    """
    Create a ColumnTransformer with per-column pipelines.
    Returns: (preprocessor, numeric_cols_used, categorical_cols_used, final_impute_plan)
    final_impute_plan maps each col to the imputation action actually applied (drop handled outside).
    """
    X = X.drop(columns=drop_cols, errors="ignore")
    feature_cols = [c for c in X.columns if c != target]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    transformers = []
    impute_plan = {}

    # Numeric columns: build per-column pipelines
    for col in numeric_cols:
        imp_choice = per_col_imp.get(col, "mean")
        impute_plan[col] = imp_choice

        # KNN is handled globally for numerics (pre-step), here we still add a no-op or simple imputer to handle any residual NaNs
        if imp_choice in ("mean", "median", "most_frequent"):
            num_imputer = SimpleImputer(strategy=imp_choice)
        elif imp_choice == "drop":
            # We'll drop in the outer flow; skip adding transformer for this column
            continue
        else:
            # 'knn' chosen per-column -> we will rely on global KNN step; use mean as fallback inside CT
            num_imputer = SimpleImputer(strategy="mean")

        scaler_name = per_col_scale.get(col, "none")
        if scaler_name == "standard":
            scaler = StandardScaler()
        elif scaler_name == "minmax":
            scaler = MinMaxScaler()
        elif scaler_name == "robust":
            scaler = RobustScaler()
        else:
            scaler = "passthrough"

        transformers.append((f"num_{col}", Pipeline([("imputer", num_imputer), ("scaler", scaler)]), [col]))

    # Categorical columns: per-column pipelines
    for col in categorical_cols:
        imp_choice = per_col_imp.get(col, "most_frequent")
        if imp_choice == "drop":
            continue
        if imp_choice in ("most_frequent", "mean", "median"):
            # mean/median are not sensible for cats; coerce to most_frequent
            cat_imputer = SimpleImputer(strategy="most_frequent")
            impute_plan[col] = "most_frequent"
        elif imp_choice == "knn":
            # Not supported for categorical directly; fall back gracefully
            cat_imputer = SimpleImputer(strategy="most_frequent")
            impute_plan[col] = "most_frequent"
        else:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            impute_plan[col] = "most_frequent"

        enc_choice = per_col_enc.get(col, "onehot")
        if enc_choice == "onehot":
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        elif enc_choice == "ordinal":
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        else:
            encoder = "passthrough"

        # Scaling for categorical is usually unnecessary; ignore per_col_scale for cats
        transformers.append((f"cat_{col}", Pipeline([("imputer", cat_imputer), ("encoder", encoder)]), [col]))

    preprocessor = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)

    # If global KNN is requested, it will be applied BEFORE this preprocessor on the numeric subset.
    # We'll return the lists so caller can run KNN on X[numeric_cols] if needed.
    return preprocessor, numeric_cols, categorical_cols, impute_plan


def apply_global_knn_numeric(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Apply KNNImputer to numeric columns only; returns a copy with imputed numerics."""
    if not numeric_cols:
        return df
    work = df.copy()
    subset = work[numeric_cols]
    if subset.isnull().any().any():
        imputer = KNNImputer(n_neighbors=5, weights="uniform")
        work[numeric_cols] = imputer.fit_transform(subset)
    return work


def get_models_for_task(task: str, selected_model_keys: List[str]):
    models = {}
    # Use hyperparameters from sidebar
    rf_n_estimators = st.session_state.get("rf_n_estimators", 200)
    dt_max_depth = st.session_state.get("dt_max_depth", None)
    alpha_val = st.session_state.get("alpha_val", 1.0)
    enet_l1_ratio = st.session_state.get("enet_l1_ratio", 0.5)

    if task == "classification":
        all_models = {
            "logreg_l2": LogisticRegression(penalty="l2", solver="liblinear", max_iter=2000),
            "logreg_l1": LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000),
            "logreg_en": LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=enet_l1_ratio, max_iter=4000),
            "dtree": DecisionTreeClassifier(
                random_state=42,
                max_depth=dt_max_depth if dt_max_depth is not None and dt_max_depth > 0 else None
            ),
            "rf": RandomForestClassifier(
                n_estimators=rf_n_estimators,
                random_state=42,
                n_jobs=-1,
                max_depth=dt_max_depth if dt_max_depth is not None and dt_max_depth > 0 else None
            )
        }
    else:
        all_models = {
            "ridge": Ridge(random_state=42, alpha=alpha_val),
            "lasso": Lasso(random_state=42, alpha=alpha_val),
            "enet": ElasticNet(random_state=42, alpha=alpha_val, l1_ratio=enet_l1_ratio),
            "dtree": DecisionTreeRegressor(
                random_state=42,
                max_depth=dt_max_depth if dt_max_depth is not None and dt_max_depth > 0 else None
            ),
            "rf": RandomForestRegressor(
                n_estimators=rf_n_estimators,
                random_state=42,
                n_jobs=-1,
                max_depth=dt_max_depth if dt_max_depth is not None and dt_max_depth > 0 else None
            ),
        }
    for k in selected_model_keys:
        if k in all_models:
            models[k] = all_models[k]
    return models


def train_and_eval_model(task: str, name: str, preprocessor, model, X_train, X_test, y_train, y_test):
    pipe = Pipeline([
        ("to_df", ArrayToDataFrame()),
        ("preprocess", preprocessor),
        ("model", model)
    ])

    # 🔥 ADD THIS BLOCK
    if isinstance(model, (BaggingClassifier, BaggingRegressor,
                          VotingClassifier, VotingRegressor,
                          StackingClassifier, StackingRegressor)):

        if hasattr(model, "estimators"):
            wrapped_estimators = []
            for name, est in model.estimators:
                if not isinstance(est, Pipeline):
                    est = Pipeline([
                        ("to_df", ArrayToDataFrame()),
                        ("est", est)
                    ])
                wrapped_estimators.append((name, est))
            model.estimators = wrapped_estimators

        if hasattr(model, "estimator") and model.estimator is not None:
            if not isinstance(model.estimator, Pipeline):
                model.estimator = Pipeline([
                    ("to_df", ArrayToDataFrame()),
                    ("est", model.estimator)
                ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    results = {"model": name}

    if task == "classification":
        results["accuracy"] = accuracy_score(y_test, y_pred)
        results["r2"] = r2_score(y_test, y_pred)
        # Confusion matrix as a string for table, full matrix for display
        cm = confusion_matrix(y_test, y_pred)
        results["confusion_matrix"] = str(cm.tolist())

        # ROC-AUC (binary or multiclass)
        try:
            if hasattr(pipe, "predict_proba"):
                proba = pipe.predict_proba(X_test)
                if proba.ndim == 1 or proba.shape[1] == 1:
                    # Unexpected shape; skip AUC
                    results["roc_auc"] = np.nan
                else:
                    if proba.shape[1] == 2:  # binary
                        results["roc_auc"] = roc_auc_score(y_test, proba[:, 1])
                    else:
                        # multiclass
                        results["roc_auc"] = roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")
            else:
                results["roc_auc"] = np.nan
        except Exception:
            results["roc_auc"] = np.nan

    else:
        mae = mean_absolute_error(y_test, y_pred)
        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        # Accuracy for regression: percent within 10% of true value
        tolerance = 0.1
        within_tol = np.abs(y_pred - y_test) <= (tolerance * np.abs(y_test))
        reg_accuracy = np.mean(within_tol)
        results["mae"] = mae
        results["rmse"] = rmse
        results["r2"] = r2
        results["accuracy"] = reg_accuracy

    return results, pipe


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AutoML (Classic Models)", layout="wide")
st.title("🔧 AutoML (Classic Models) — Streamlit")

st.markdown(
    "Upload a CSV, choose the target, configure per-column preprocessing, "
    "select models/ensembles, and get metrics."
)

with st.sidebar:
    st.header("1) Upload Dataset")
    file = st.file_uploader("CSV file", type=["csv"])
    test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random state", value=42, step=1)

    # --- Hyperparameter tuning options ---
    st.header("Hyperparameters")
    rf_n_estimators = st.number_input("Random Forest/Bagg. n_estimators", min_value=10, max_value=1000, value=200, step=10)
    dt_max_depth = st.number_input("Decision Tree max_depth (0 = None)", min_value=0, max_value=100, value=0, step=1)
    alpha_val = st.number_input("Ridge/Lasso/ElasticNet alpha", min_value=0.0001, max_value=10.0, value=1.0, step=0.1, format="%.4f")
    enet_l1_ratio = st.slider("ElasticNet l1_ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

if file is None:
    st.info("⬆️ Upload a CSV to begin.")
    st.stop()

# Load and preview
df = pd.read_csv(file)
st.subheader("Preview")
st.dataframe(df.head(20), use_container_width=True)

# Target selection
st.header("2) Target & Task")
target = st.selectbox("Select target column", df.columns)
if target is None:
    st.stop()

# Show basic info
st.write("**Shape:**", df.shape)
st.write("**Missing values per column:**")
st.write(df.isnull().sum()[df.isnull().sum() > 0])

# Task detection with override
auto_task = detect_task_type(df[target])
task = st.radio(
    "Task type",
    options=["auto-detect", "classification", "regression"],
    index=0,
    horizontal=True
)
if task == "auto-detect":
    task = auto_task
st.success(f"Task detected/selected: **{task}**")

# Feature deletion
st.header("3) Feature Selection & Deletion")
feature_cols = [c for c in df.columns if c != target]
drop_cols = st.multiselect("Drop columns (IDs/leaky/etc.)", feature_cols, default=[])

# Split numeric/categorical after drops
num_cols_all, cat_cols_all = split_feature_types(df.drop(columns=drop_cols, errors="ignore"), target)

st.write("**Numeric columns:**", num_cols_all)
st.write("**Categorical columns:**", cat_cols_all)

# Per-column imputation
st.header("4) Imputation (Per Column)")
st.caption("For categorical columns, 'most_frequent' will be used even if mean/median/knn is chosen.")
per_col_imp = {}
with st.expander("Configure imputation per column", expanded=False):
    for col in feature_cols:
        if col in drop_cols:
            continue
        if col in num_cols_all:
            options = ["mean", "median", "most_frequent", "drop", "knn"]
            default = "median" if df[col].isnull().any() else "mean"
        else:
            options = ["most_frequent", "drop"]  # keep simple & safe for cats
            default = "most_frequent" if df[col].isnull().any() else "most_frequent"
        per_col_imp[col] = st.selectbox(f"{col}", options, index=options.index(default), key=f"imp_{col}")

use_global_knn_for_numeric = st.checkbox(
    "Apply **global KNN imputation** to numeric columns with missing values (recommended only if many numeric NaNs).",
    value=False
)

# Per-column transforms
st.header("5) Per-Column Encoding & Scaling")
st.caption("Encoding applies to categorical columns; scaling applies to numeric columns.")
per_col_enc = {}
per_col_scale = {}
with st.expander("Configure encoding/scaling per column", expanded=False):
    for col in feature_cols:
        if col in drop_cols:
            continue
        if col in cat_cols_all:
            enc = st.selectbox(
                f"Encoding for {col}",
                ["onehot", "ordinal", "none"],
                index=0,
                key=f"enc_{col}"
            )
            per_col_enc[col] = enc
        else:
            per_col_enc[col] = "none"  # not used for numeric

        if col in num_cols_all:
            scale = st.selectbox(
                f"Scaling for {col}",
                ["none", "standard", "minmax", "robust"],
                index=0,
                key=f"scale_{col}"
            )
            per_col_scale[col] = scale
        else:
            per_col_scale[col] = "none"

# Train/test split
X = df.drop(columns=[target])
y = df[target]

# Optionally apply global numeric KNN imputation before CT
if use_global_knn_for_numeric:
    X = apply_global_knn_numeric(X, [c for c in X.columns if c in num_cols_all])

# Build preprocessor
preprocessor, num_cols_used, cat_cols_used, impute_plan = build_preprocessor(
    X=X.join(y),
    target=target,
    drop_cols=drop_cols,
    per_col_imp=per_col_imp,
    per_col_enc=per_col_enc,
    per_col_scale=per_col_scale,
    use_global_knn_for_numeric=use_global_knn_for_numeric,
)

# Drop columns marked for deletion or per-column "drop"
cols_to_drop_runtime = set(drop_cols)
for c, choice in per_col_imp.items():
    if choice == "drop":
        cols_to_drop_runtime.add(c)

X_work = X.drop(columns=list(cols_to_drop_runtime), errors="ignore")
feature_cols_final = [c for c in X_work.columns]  # after drops

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_work, y, test_size=test_size, random_state=random_state, stratify=y if task == "classification" else None
)

# Model selection
st.header("6) Models & Ensembling")

if task == "classification":
    model_options = {
        "logreg_l2": "Logistic (L2)",
        "logreg_l1": "Logistic (L1)",
        "logreg_en": "Logistic (ElasticNet)",
        "dtree": "Decision Tree",
        "rf": "Random Forest",
    }
else:
    model_options = {
        "ridge": "Ridge (L2)",
        "lasso": "Lasso (L1)",
        "enet": "ElasticNet",
        "dtree": "Decision Tree",
        "rf": "Random Forest",
    }

selected_keys = st.multiselect("Select models to train", list(model_options.keys()),
                               default=list(model_options.keys())[:2],
                               format_func=lambda k: model_options[k])

# Ensembling options
ens_choice = st.radio("Ensemble", ["None", "Voting", "Bagging"], horizontal=True)
voting_type = "hard"
bag_base_key = None
if ens_choice == "Voting":
    if task == "classification":
        voting_type = st.radio("Voting type", ["hard", "soft"], horizontal=True, index=0)
elif ens_choice == "Bagging":
    bag_base_key = st.selectbox("Base model for Bagging", selected_keys if selected_keys else list(model_options.keys()))

go = st.button("🚀 Run")

if go:
    # Build base models
    base_models = get_models_for_task(task, selected_keys)

    results_rows = []
    trained_pipes = {}

    # Train/eval each selected model
    for key, base_est in base_models.items():
        res, fitted = train_and_eval_model(task, model_options[key], preprocessor, base_est, X_train, X_test, y_train, y_test)
        results_rows.append(res)
        trained_pipes[model_options[key]] = fitted

    # Ensembling
    if ens_choice != "None" and selected_keys:
        if ens_choice == "Voting":
            estimators = []
            # Recreate fresh pipelines for estimators to avoid fitted-state reuse
            for key in selected_keys:
                base_est = get_models_for_task(task, [key])[key]
                # Pass correct feature_names to ArrayToDataFrame
                estimators.append((key, Pipeline([
                    ("to_df", ArrayToDataFrame(feature_names=feature_cols_final)),
                    ("pre", preprocessor),
                    ("est", base_est)
                ])))
            if task == "classification":
                ensemble = VotingClassifier(estimators=estimators, voting=voting_type, n_jobs=-1 if voting_type == "soft" else None)
            else:
                ensemble = VotingRegressor(estimators=estimators)
        elif ens_choice == "Bagging":
            if bag_base_key is None:
                st.warning("Select a base model for bagging.")
                ensemble = None
            else:
                base_est = get_models_for_task(task, [bag_base_key])[bag_base_key]
                base_pipe = Pipeline([
                    ("to_df", ArrayToDataFrame(feature_names=feature_cols_final)),
                    ("pre", preprocessor),
                    ("est", base_est)
                ])
                if task == "classification":
                    ensemble = BaggingClassifier(estimator=base_pipe, n_estimators=10, random_state=42, n_jobs=-1)
                else:
                    ensemble = BaggingRegressor(estimator=base_pipe, n_estimators=10, random_state=42, n_jobs=-1)
        else:
            ensemble = None

        if ensemble is not None:
            res, fitted = train_and_eval_model(task, f"{ens_choice}", preprocessor, ensemble, X_train, X_test, y_train, y_test)
            results_rows.append(res)
            trained_pipes[f"{ens_choice}"] = fitted

    # Results table
    st.subheader("Results")
    if results_rows:
        results_df = pd.DataFrame(results_rows).set_index("model")
        # Only show relevant columns
        if task == "classification":
            show_cols = ["accuracy", "r2", "confusion_matrix"]
        else:
            show_cols = ["mae", "rmse", "r2", "accuracy"]
        st.dataframe(results_df[show_cols].style.format(precision=4), use_container_width=True)

        # Download results
        csv_buf = io.StringIO()
        results_df[show_cols].to_csv(csv_buf)
        st.download_button("Download metrics CSV", data=csv_buf.getvalue(), file_name="metrics.csv", mime="text/csv")

        # Show confusion matrix for classification
        if task == "classification":
            st.subheader("Confusion Matrix")
            for model_name in results_df.index:
                cm_str = results_df.loc[model_name, "confusion_matrix"]
                cm = np.array(eval(cm_str))
                st.write(f"**{model_name}**")
                st.write(pd.DataFrame(cm))

        # Download transformed dataset (features only)
        st.subheader("Download Transformed Dataset")
        # Use the first trained pipeline for transformation
        first_pipe = next(iter(trained_pipes.values()))
        X_transformed = first_pipe.named_steps["preprocess"].transform(X_work)
        # If output is numpy array, convert to DataFrame with feature names
        if hasattr(first_pipe.named_steps["preprocess"], "get_feature_names_out"):
            feature_names = first_pipe.named_steps["preprocess"].get_feature_names_out()
        else:
            feature_names = [f"f{i}" for i in range(X_transformed.shape[1])]
        X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
        transformed_buf = io.StringIO()
        X_transformed_df.to_csv(transformed_buf, index=False)
        st.download_button("Download transformed dataset CSV", data=transformed_buf.getvalue(), file_name="transformed_dataset.csv", mime="text/csv")
    else:
        st.info("No models were selected.")

    # Show some details
    with st.expander("Preprocessing summary"):
        st.write("Dropped columns:", list(cols_to_drop_runtime))
        st.write("Numeric columns used:", num_cols_used)
        st.write("Categorical columns used:", cat_cols_used)
        st.write("Imputation plan (per column):", impute_plan)

