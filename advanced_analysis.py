"""
Advanced Analysis - PLID Surgical Outcomes
Implements all 6 recommendations from future work:
1. Feature Engineering (interaction terms & polynomial features)
2. Hyperparameter Tuning (GridSearchCV)
3. Cross-Validation (k-fold CV)
4. Ensemble Methods
5. External Validation
6. Statistical Significance Testing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    StratifiedKFold,
    KFold,
)
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    VotingRegressor,
    VotingClassifier,
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from scipy import stats
from scipy.stats import ttest_rel
import warnings

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)

# Visualization settings
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

print("=" * 80)
print("ADVANCED ANALYSIS - PLID SURGICAL OUTCOMES")
print("=" * 80)
print("\n")

# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading imputed dataset...")
df = pd.read_csv("imputed_dataset_full_349.csv")
print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
print(f"Missing values: {df.isnull().sum().sum()}")
print("\n")

# ============================================================================
# DATA PREPROCESSING
# ============================================================================
print("=" * 80)
print("DATA PREPROCESSING")
print("=" * 80)

# Drop non-predictive columns
df_clean = df.drop(["Timestamp", "Id"], axis=1)

# Encode categorical variables
categorical_columns = df_clean.select_dtypes(include=["object"]).columns
print(f"Categorical columns: {len(categorical_columns)}")

df_encoded = df_clean.copy()
for col in categorical_columns:
    df_encoded[col] = pd.Categorical(df_encoded[col]).codes

# Define target variables
target_columns = [
    "Post operative ODI",
    "Post operative NRS back pain",
    "Post operative NRS leg pain",
    "Surgery outcome according to Macnab criteria",
]

# Define feature columns
feature_columns = [col for col in df_encoded.columns if col not in target_columns]
print(f"Feature columns: {len(feature_columns)}")
print(f"Target columns: {len(target_columns)}")
print("\n")

# ============================================================================
# TASK 1: FEATURE ENGINEERING
# ============================================================================
print("=" * 80)
print("TASK 1: FEATURE ENGINEERING")
print("=" * 80)

# Original features
X_original = df_encoded[feature_columns]

# Identify numeric features for polynomial expansion
numeric_features = [
    "Age",
    "Pre operative ODI",
    "Pre operative NRS back pain",
    "Pre operative NRS leg pain",
]
numeric_indices = [
    feature_columns.index(col) for col in numeric_features if col in feature_columns
]

# Create polynomial features (degree 2)
print("Creating polynomial features (degree 2)...")
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_array = poly.fit_transform(X_original[numeric_features])
poly_feature_names = poly.get_feature_names_out(numeric_features)
X_poly = pd.DataFrame(X_poly_array, columns=poly_feature_names, index=X_original.index)

# Combine original categorical features with polynomial numeric features
categorical_features = [col for col in feature_columns if col not in numeric_features]
X_engineered = pd.concat([X_original[categorical_features], X_poly], axis=1)

print(f"Original features: {X_original.shape[1]}")
print(f"Engineered features: {X_engineered.shape[1]}")
print(f"New features added: {X_engineered.shape[1] - X_original.shape[1]}")
print("\n")

# ============================================================================
# TASK 5: EXTERNAL VALIDATION (Create holdout set first)
# ============================================================================
print("=" * 80)
print("TASK 5: EXTERNAL VALIDATION SETUP")
print("=" * 80)

# Create a completely independent external validation set (20%)
# This set will NEVER be used during training or hyperparameter tuning
X_dev, X_external, y_dev_dict, y_external_dict = {}, {}, {}, {}

for target in target_columns:
    y = df_encoded[target]

    # Stratify for classification target
    if target == "Surgery outcome according to Macnab criteria":
        X_dev_temp, X_external_temp, y_dev_temp, y_external_temp = train_test_split(
            X_engineered, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_dev_temp, X_external_temp, y_dev_temp, y_external_temp = train_test_split(
            X_engineered, y, test_size=0.2, random_state=42
        )

    X_dev[target] = X_dev_temp
    X_external[target] = X_external_temp
    y_dev_dict[target] = y_dev_temp
    y_external_dict[target] = y_external_temp

print(f"Development set: {X_dev[target_columns[0]].shape[0]} samples (80%)")
print(
    f"External validation set: {X_external[target_columns[0]].shape[0]} samples (20%)"
)
print("External set will be used ONLY for final validation\n")

# ============================================================================
# TASK 2: HYPERPARAMETER TUNING WITH GRIDSEARCHCV
# ============================================================================
print("=" * 80)
print("TASK 2: HYPERPARAMETER TUNING WITH GRIDSEARCHCV")
print("=" * 80)

# Define parameter grids
rf_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

gb_param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "min_samples_split": [2, 5],
}

best_params = {}
tuning_results = []

for target in target_columns:
    print(f"\nTuning models for: {target}")
    print("-" * 80)

    X_train, X_test, y_train, y_test = train_test_split(
        X_dev[target],
        y_dev_dict[target],
        test_size=0.2,
        random_state=42,
        stratify=y_dev_dict[target]
        if target == "Surgery outcome according to Macnab criteria"
        else None,
    )

    is_classification = target == "Surgery outcome according to Macnab criteria"

    # Random Forest tuning
    print("  Tuning Random Forest...")
    if is_classification:
        rf_model = RandomForestClassifier(random_state=42)
        scoring = "accuracy"
    else:
        rf_model = RandomForestRegressor(random_state=42)
        scoring = "r2"

    rf_grid = GridSearchCV(
        rf_model, rf_param_grid, cv=5, scoring=scoring, n_jobs=-1, verbose=0
    )
    rf_grid.fit(X_train, y_train)

    best_params[f"{target}_RF"] = rf_grid.best_params_
    print(f"    Best params: {rf_grid.best_params_}")
    print(f"    Best CV score: {rf_grid.best_score_:.4f}")

    tuning_results.append(
        {
            "Target": target,
            "Model": "Random Forest",
            "Best_Params": str(rf_grid.best_params_),
            "Best_CV_Score": rf_grid.best_score_,
        }
    )

    # Gradient Boosting tuning
    print("  Tuning Gradient Boosting...")
    if is_classification:
        gb_model = GradientBoostingClassifier(random_state=42)
    else:
        gb_model = GradientBoostingRegressor(random_state=42)

    gb_grid = GridSearchCV(
        gb_model, gb_param_grid, cv=5, scoring=scoring, n_jobs=-1, verbose=0
    )
    gb_grid.fit(X_train, y_train)

    best_params[f"{target}_GB"] = gb_grid.best_params_
    print(f"    Best params: {gb_grid.best_params_}")
    print(f"    Best CV score: {gb_grid.best_score_:.4f}")

    tuning_results.append(
        {
            "Target": target,
            "Model": "Gradient Boosting",
            "Best_Params": str(gb_grid.best_params_),
            "Best_CV_Score": gb_grid.best_score_,
        }
    )

# Save tuning results
tuning_df = pd.DataFrame(tuning_results)
tuning_df.to_csv("hyperparameter_tuning_results.csv", index=False)
print("\n✓ Hyperparameter tuning complete!")
print("✓ Results saved to hyperparameter_tuning_results.csv\n")

# ============================================================================
# TASK 3: K-FOLD CROSS-VALIDATION
# ============================================================================
print("=" * 80)
print("TASK 3: K-FOLD CROSS-VALIDATION")
print("=" * 80)

cv_results = []

for target in target_columns:
    print(f"\nCross-validating models for: {target}")
    print("-" * 80)

    X_data = X_dev[target]
    y_data = y_dev_dict[target]

    is_classification = target == "Surgery outcome according to Macnab criteria"

    # Use optimized parameters
    if is_classification:
        rf_model = RandomForestClassifier(
            **best_params[f"{target}_RF"], random_state=42
        )
        gb_model = GradientBoostingClassifier(
            **best_params[f"{target}_GB"], random_state=42
        )
        scoring = "accuracy"
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        rf_model = RandomForestRegressor(**best_params[f"{target}_RF"], random_state=42)
        gb_model = GradientBoostingRegressor(
            **best_params[f"{target}_GB"], random_state=42
        )
        scoring = "r2"
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # 5-fold CV
    print("  5-Fold Cross-Validation:")
    rf_scores_5 = cross_val_score(
        rf_model, X_data, y_data, cv=cv, scoring=scoring, n_jobs=-1
    )
    gb_scores_5 = cross_val_score(
        gb_model, X_data, y_data, cv=cv, scoring=scoring, n_jobs=-1
    )

    print(f"    RF: {rf_scores_5.mean():.4f} (+/- {rf_scores_5.std():.4f})")
    print(f"    GB: {gb_scores_5.mean():.4f} (+/- {gb_scores_5.std():.4f})")

    # 10-fold CV
    if is_classification:
        cv_10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    else:
        cv_10 = KFold(n_splits=10, shuffle=True, random_state=42)

    print("  10-Fold Cross-Validation:")
    rf_scores_10 = cross_val_score(
        rf_model, X_data, y_data, cv=cv_10, scoring=scoring, n_jobs=-1
    )
    gb_scores_10 = cross_val_score(
        gb_model, X_data, y_data, cv=cv_10, scoring=scoring, n_jobs=-1
    )

    print(f"    RF: {rf_scores_10.mean():.4f} (+/- {rf_scores_10.std():.4f})")
    print(f"    GB: {gb_scores_10.mean():.4f} (+/- {gb_scores_10.std():.4f})")

    cv_results.extend(
        [
            {
                "Target": target,
                "Model": "Random Forest",
                "CV_Folds": 5,
                "Mean_Score": rf_scores_5.mean(),
                "Std_Score": rf_scores_5.std(),
            },
            {
                "Target": target,
                "Model": "Gradient Boosting",
                "CV_Folds": 5,
                "Mean_Score": gb_scores_5.mean(),
                "Std_Score": gb_scores_5.std(),
            },
            {
                "Target": target,
                "Model": "Random Forest",
                "CV_Folds": 10,
                "Mean_Score": rf_scores_10.mean(),
                "Std_Score": rf_scores_10.std(),
            },
            {
                "Target": target,
                "Model": "Gradient Boosting",
                "CV_Folds": 10,
                "Mean_Score": gb_scores_10.mean(),
                "Std_Score": gb_scores_10.std(),
            },
        ]
    )

cv_df = pd.DataFrame(cv_results)
cv_df.to_csv("cross_validation_results.csv", index=False)
print("\n✓ Cross-validation complete!")
print("✓ Results saved to cross_validation_results.csv\n")

# ============================================================================
# TASK 4: ENSEMBLE METHODS
# ============================================================================
print("=" * 80)
print("TASK 4: ENSEMBLE METHODS")
print("=" * 80)

ensemble_results = []

for target in target_columns:
    print(f"\nBuilding ensemble for: {target}")
    print("-" * 80)

    X_train, X_test, y_train, y_test = train_test_split(
        X_dev[target],
        y_dev_dict[target],
        test_size=0.2,
        random_state=42,
        stratify=y_dev_dict[target]
        if target == "Surgery outcome according to Macnab criteria"
        else None,
    )

    is_classification = target == "Surgery outcome according to Macnab criteria"

    if is_classification:
        # Build classification ensemble
        rf = RandomForestClassifier(**best_params[f"{target}_RF"], random_state=42)
        gb = GradientBoostingClassifier(**best_params[f"{target}_GB"], random_state=42)

        ensemble = VotingClassifier(estimators=[("rf", rf), ("gb", gb)], voting="soft")

        # Train individual models
        rf.fit(X_train, y_train)
        gb.fit(X_train, y_train)
        ensemble.fit(X_train, y_train)

        # Predictions
        rf_pred = rf.predict(X_test)
        gb_pred = gb.predict(X_test)
        ensemble_pred = ensemble.predict(X_test)

        # Metrics
        rf_acc = accuracy_score(y_test, rf_pred)
        gb_acc = accuracy_score(y_test, gb_pred)
        ensemble_acc = accuracy_score(y_test, ensemble_pred)

        print(f"  Random Forest Accuracy: {rf_acc:.4f}")
        print(f"  Gradient Boosting Accuracy: {gb_acc:.4f}")
        print(f"  Ensemble Accuracy: {ensemble_acc:.4f}")
        print(
            f"  Improvement: {((ensemble_acc - max(rf_acc, gb_acc)) / max(rf_acc, gb_acc) * 100):.2f}%"
        )

        ensemble_results.append(
            {
                "Target": target,
                "RF_Accuracy": rf_acc,
                "GB_Accuracy": gb_acc,
                "Ensemble_Accuracy": ensemble_acc,
                "RF_F1": f1_score(y_test, rf_pred, average="weighted"),
                "GB_F1": f1_score(y_test, gb_pred, average="weighted"),
                "Ensemble_F1": f1_score(y_test, ensemble_pred, average="weighted"),
            }
        )

    else:
        # Build regression ensemble
        rf = RandomForestRegressor(**best_params[f"{target}_RF"], random_state=42)
        gb = GradientBoostingRegressor(**best_params[f"{target}_GB"], random_state=42)

        ensemble = VotingRegressor(estimators=[("rf", rf), ("gb", gb)])

        # Train individual models
        rf.fit(X_train, y_train)
        gb.fit(X_train, y_train)
        ensemble.fit(X_train, y_train)

        # Predictions
        rf_pred = rf.predict(X_test)
        gb_pred = gb.predict(X_test)
        ensemble_pred = ensemble.predict(X_test)

        # Metrics
        rf_r2 = r2_score(y_test, rf_pred)
        gb_r2 = r2_score(y_test, gb_pred)
        ensemble_r2 = r2_score(y_test, ensemble_pred)

        rf_mae = mean_absolute_error(y_test, rf_pred)
        gb_mae = mean_absolute_error(y_test, gb_pred)
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)

        print(f"  Random Forest R²: {rf_r2:.4f}, MAE: {rf_mae:.4f}")
        print(f"  Gradient Boosting R²: {gb_r2:.4f}, MAE: {gb_mae:.4f}")
        print(f"  Ensemble R²: {ensemble_r2:.4f}, MAE: {ensemble_mae:.4f}")

        ensemble_results.append(
            {
                "Target": target,
                "RF_R2": rf_r2,
                "GB_R2": gb_r2,
                "Ensemble_R2": ensemble_r2,
                "RF_MAE": rf_mae,
                "GB_MAE": gb_mae,
                "Ensemble_MAE": ensemble_mae,
            }
        )

ensemble_df = pd.DataFrame(ensemble_results)
ensemble_df.to_csv("ensemble_results.csv", index=False)
print("\n✓ Ensemble models complete!")
print("✓ Results saved to ensemble_results.csv\n")

# ============================================================================
# TASK 5: EXTERNAL VALIDATION (Final Testing)
# ============================================================================
print("=" * 80)
print("TASK 5: EXTERNAL VALIDATION (FINAL TEST)")
print("=" * 80)

external_results = []

for target in target_columns:
    print(f"\nExternal validation for: {target}")
    print("-" * 80)

    # Train on FULL development set with best ensemble
    X_train_full = X_dev[target]
    y_train_full = y_dev_dict[target]
    X_test_external = X_external[target]
    y_test_external = y_external_dict[target]

    is_classification = target == "Surgery outcome according to Macnab criteria"

    if is_classification:
        rf = RandomForestClassifier(**best_params[f"{target}_RF"], random_state=42)
        gb = GradientBoostingClassifier(**best_params[f"{target}_GB"], random_state=42)
        ensemble = VotingClassifier(estimators=[("rf", rf), ("gb", gb)], voting="soft")

        ensemble.fit(X_train_full, y_train_full)
        y_pred = ensemble.predict(X_test_external)

        acc = accuracy_score(y_test_external, y_pred)
        prec = precision_score(
            y_test_external, y_pred, average="weighted", zero_division=0
        )
        rec = recall_score(y_test_external, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test_external, y_pred, average="weighted", zero_division=0)

        print(f"  Accuracy: {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall: {rec:.4f}")
        print(f"  F1-Score: {f1:.4f}")

        external_results.append(
            {
                "Target": target,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1_Score": f1,
            }
        )

    else:
        rf = RandomForestRegressor(**best_params[f"{target}_RF"], random_state=42)
        gb = GradientBoostingRegressor(**best_params[f"{target}_GB"], random_state=42)
        ensemble = VotingRegressor(estimators=[("rf", rf), ("gb", gb)])

        ensemble.fit(X_train_full, y_train_full)
        y_pred = ensemble.predict(X_test_external)

        r2 = r2_score(y_test_external, y_pred)
        mae = mean_absolute_error(y_test_external, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_external, y_pred))

        print(f"  R²: {r2:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")

        external_results.append({"Target": target, "R2": r2, "MAE": mae, "RMSE": rmse})

external_df = pd.DataFrame(external_results)
external_df.to_csv("external_validation_results.csv", index=False)
print("\n✓ External validation complete!")
print("✓ Results saved to external_validation_results.csv\n")

# ============================================================================
# TASK 6: STATISTICAL SIGNIFICANCE TESTING
# ============================================================================
print("=" * 80)
print("TASK 6: STATISTICAL SIGNIFICANCE TESTING")
print("=" * 80)

stat_results = []

for target in target_columns:
    print(f"\nStatistical testing for: {target}")
    print("-" * 80)

    X_test_data = X_external[target]
    y_test_data = y_external_dict[target]

    is_classification = target == "Surgery outcome according to Macnab criteria"

    # Train baseline (RF only) and ensemble models
    X_train_full = X_dev[target]
    y_train_full = y_dev_dict[target]

    if is_classification:
        # Baseline: RF only
        baseline = RandomForestClassifier(
            **best_params[f"{target}_RF"], random_state=42
        )
        baseline.fit(X_train_full, y_train_full)
        baseline_pred = baseline.predict(X_test_data)

        # Advanced: Ensemble
        rf = RandomForestClassifier(**best_params[f"{target}_RF"], random_state=42)
        gb = GradientBoostingClassifier(**best_params[f"{target}_GB"], random_state=42)
        ensemble = VotingClassifier(estimators=[("rf", rf), ("gb", gb)], voting="soft")
        ensemble.fit(X_train_full, y_train_full)
        ensemble_pred = ensemble.predict(X_test_data)

        # McNemar's test for paired predictions
        # Create contingency table
        correct_both = np.sum(
            (baseline_pred == y_test_data) & (ensemble_pred == y_test_data)
        )
        baseline_only = np.sum(
            (baseline_pred == y_test_data) & (ensemble_pred != y_test_data)
        )
        ensemble_only = np.sum(
            (baseline_pred != y_test_data) & (ensemble_pred == y_test_data)
        )
        neither = np.sum(
            (baseline_pred != y_test_data) & (ensemble_pred != y_test_data)
        )

        contingency_table = [[correct_both, baseline_only], [ensemble_only, neither]]

        # McNemar's test
        if baseline_only + ensemble_only > 0:
            statistic = (abs(baseline_only - ensemble_only) - 1) ** 2 / (
                baseline_only + ensemble_only
            )
            p_value = 1 - stats.chi2.cdf(statistic, 1)
        else:
            statistic = 0
            p_value = 1.0

        baseline_acc = accuracy_score(y_test_data, baseline_pred)
        ensemble_acc = accuracy_score(y_test_data, ensemble_pred)

        print(f"  Baseline Accuracy: {baseline_acc:.4f}")
        print(f"  Ensemble Accuracy: {ensemble_acc:.4f}")
        print(f"  McNemar's statistic: {statistic:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")

        stat_results.append(
            {
                "Target": target,
                "Test": "McNemar",
                "Baseline_Metric": baseline_acc,
                "Ensemble_Metric": ensemble_acc,
                "Test_Statistic": statistic,
                "P_Value": p_value,
                "Significant": p_value < 0.05,
            }
        )

    else:
        # Baseline: RF only
        baseline = RandomForestRegressor(**best_params[f"{target}_RF"], random_state=42)
        baseline.fit(X_train_full, y_train_full)
        baseline_pred = baseline.predict(X_test_data)
        baseline_errors = np.abs(y_test_data - baseline_pred)

        # Advanced: Ensemble
        rf = RandomForestRegressor(**best_params[f"{target}_RF"], random_state=42)
        gb = GradientBoostingRegressor(**best_params[f"{target}_GB"], random_state=42)
        ensemble = VotingRegressor(estimators=[("rf", rf), ("gb", gb)])
        ensemble.fit(X_train_full, y_train_full)
        ensemble_pred = ensemble.predict(X_test_data)
        ensemble_errors = np.abs(y_test_data - ensemble_pred)

        # Paired t-test on absolute errors
        t_stat, p_value = ttest_rel(baseline_errors, ensemble_errors)

        baseline_mae = mean_absolute_error(y_test_data, baseline_pred)
        ensemble_mae = mean_absolute_error(y_test_data, ensemble_pred)

        print(f"  Baseline MAE: {baseline_mae:.4f}")
        print(f"  Ensemble MAE: {ensemble_mae:.4f}")
        print(f"  Paired t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")

        stat_results.append(
            {
                "Target": target,
                "Test": "Paired t-test",
                "Baseline_Metric": baseline_mae,
                "Ensemble_Metric": ensemble_mae,
                "Test_Statistic": t_stat,
                "P_Value": p_value,
                "Significant": p_value < 0.05,
            }
        )

stat_df = pd.DataFrame(stat_results)
stat_df.to_csv("statistical_significance_results.csv", index=False)
print("\n✓ Statistical significance testing complete!")
print("✓ Results saved to statistical_significance_results.csv\n")

# ============================================================================
# VISUALIZATION & SUMMARY REPORT
# ============================================================================
print("=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(
    "Advanced Analysis Results - PLID Surgical Outcomes", fontsize=16, fontweight="bold"
)

# 1. Hyperparameter Tuning Results
ax1 = axes[0, 0]
tuning_plot_data = tuning_df.pivot_table(
    values="Best_CV_Score", index="Target", columns="Model"
)
tuning_plot_data.plot(kind="bar", ax=ax1, rot=45)
ax1.set_title("Hyperparameter Tuning - Best CV Scores")
ax1.set_xlabel("")
ax1.set_ylabel("Score")
ax1.legend(title="Model", loc="lower right")
ax1.grid(axis="y", alpha=0.3)

# 2. Cross-Validation Comparison (5-fold vs 10-fold)
ax2 = axes[0, 1]
cv_5fold = cv_df[cv_df["CV_Folds"] == 5].groupby("Target")["Mean_Score"].mean()
cv_10fold = cv_df[cv_df["CV_Folds"] == 10].groupby("Target")["Mean_Score"].mean()
x_pos = np.arange(len(cv_5fold))
width = 0.35
ax2.bar(x_pos - width / 2, cv_5fold.values, width, label="5-Fold CV", alpha=0.8)
ax2.bar(x_pos + width / 2, cv_10fold.values, width, label="10-Fold CV", alpha=0.8)
ax2.set_title("Cross-Validation Comparison")
ax2.set_ylabel("Mean Score")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(
    [t.replace(" ", "\n") for t in cv_5fold.index], rotation=0, ha="center", fontsize=8
)
ax2.legend()
ax2.grid(axis="y", alpha=0.3)

# 3. Ensemble Performance (Classification)
ax3 = axes[0, 2]
class_ensemble = [r for r in ensemble_results if "Ensemble_Accuracy" in r][0]
models = ["RF", "GB", "Ensemble"]
accuracies = [
    class_ensemble["RF_Accuracy"],
    class_ensemble["GB_Accuracy"],
    class_ensemble["Ensemble_Accuracy"],
]
colors = ["#3498db", "#e74c3c", "#2ecc71"]
bars = ax3.bar(models, accuracies, color=colors, alpha=0.7)
ax3.set_title("Ensemble Performance\n(Surgery Outcome)")
ax3.set_ylabel("Accuracy")
ax3.set_ylim(0, 1)
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax3.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{acc:.3f}",
        ha="center",
        va="bottom",
        fontweight="bold",
    )
ax3.grid(axis="y", alpha=0.3)

# 4. Ensemble Performance (Regression)
ax4 = axes[1, 0]
reg_ensemble = [r for r in ensemble_results if "Ensemble_R2" in r]
targets_reg = [r["Target"].replace("Post operative ", "") for r in reg_ensemble]
rf_r2 = [r["RF_R2"] for r in reg_ensemble]
gb_r2 = [r["GB_R2"] for r in reg_ensemble]
ens_r2 = [r["Ensemble_R2"] for r in reg_ensemble]
x_pos = np.arange(len(targets_reg))
width = 0.25
ax4.bar(x_pos - width, rf_r2, width, label="RF", alpha=0.8)
ax4.bar(x_pos, gb_r2, width, label="GB", alpha=0.8)
ax4.bar(x_pos + width, ens_r2, width, label="Ensemble", alpha=0.8)
ax4.set_title("Ensemble Performance (R²)")
ax4.set_ylabel("R² Score")
ax4.set_xticks(x_pos)
ax4.set_xticklabels([t.replace(" ", "\n") for t in targets_reg], fontsize=8)
ax4.legend()
ax4.grid(axis="y", alpha=0.3)

# 5. External Validation Results
ax5 = axes[1, 1]
ext_class = [r for r in external_results if "Accuracy" in r][0]
ext_reg = [r for r in external_results if "R2" in r]
metrics_class = ["Accuracy", "Precision", "Recall", "F1"]
values_class = [
    ext_class["Accuracy"],
    ext_class["Precision"],
    ext_class["Recall"],
    ext_class["F1_Score"],
]
bars = ax5.bar(
    metrics_class,
    values_class,
    color=["#3498db", "#e74c3c", "#2ecc71", "#f39c12"],
    alpha=0.7,
)
ax5.set_title("External Validation\n(Surgery Outcome)")
ax5.set_ylabel("Score")
ax5.set_ylim(0, 1)
for bar, val in zip(bars, values_class):
    height = bar.get_height()
    ax5.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontweight="bold",
    )
ax5.grid(axis="y", alpha=0.3)

# 6. Statistical Significance
ax6 = axes[1, 2]
sig_data = stat_df.copy()
sig_data["Color"] = sig_data["Significant"].map({True: "#2ecc71", False: "#e74c3c"})
targets_stat = [
    t.replace("Post operative ", "").replace(
        "Surgery outcome according to Macnab criteria", "Surgery\nOutcome"
    )
    for t in sig_data["Target"]
]
bars = ax6.bar(
    range(len(sig_data)), sig_data["P_Value"], color=sig_data["Color"], alpha=0.7
)
ax6.axhline(y=0.05, color="red", linestyle="--", linewidth=2, label="α = 0.05")
ax6.set_title("Statistical Significance Tests")
ax6.set_ylabel("p-value")
ax6.set_xticks(range(len(sig_data)))
ax6.set_xticklabels(targets_stat, fontsize=8)
ax6.legend()
ax6.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("advanced_analysis_results.png", dpi=300, bbox_inches="tight")
print("✓ Visualization saved: advanced_analysis_results.png\n")

# ============================================================================
# GENERATE COMPREHENSIVE TEXT REPORT
# ============================================================================
print("=" * 80)
print("GENERATING COMPREHENSIVE REPORT")
print("=" * 80)

with open("advanced_analysis_summary.txt", "w") as f:
    f.write("=" * 80 + "\n")
    f.write("ADVANCED ANALYSIS - PLID SURGICAL OUTCOMES\n")
    f.write("Complete Implementation of 6 Future Work Recommendations\n")
    f.write("=" * 80 + "\n\n")

    # Dataset Info
    f.write("DATASET INFORMATION\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total samples: {df.shape[0]}\n")
    f.write(f"Original features: {X_original.shape[1]}\n")
    f.write(f"Engineered features: {X_engineered.shape[1]}\n")
    f.write(f"Development set: {X_dev[target_columns[0]].shape[0]} samples (80%)\n")
    f.write(
        f"External validation set: {X_external[target_columns[0]].shape[0]} samples (20%)\n\n"
    )

    # Task 1: Feature Engineering
    f.write("=" * 80 + "\n")
    f.write("TASK 1: FEATURE ENGINEERING\n")
    f.write("=" * 80 + "\n")
    f.write(f"Polynomial features created (degree 2)\n")
    f.write(f"New features added: {X_engineered.shape[1] - X_original.shape[1]}\n")
    f.write(f"Total features: {X_engineered.shape[1]}\n\n")

    # Task 2: Hyperparameter Tuning
    f.write("=" * 80 + "\n")
    f.write("TASK 2: HYPERPARAMETER TUNING (GridSearchCV)\n")
    f.write("=" * 80 + "\n")
    for _, row in tuning_df.iterrows():
        f.write(f"\n{row['Target']} - {row['Model']}\n")
        f.write(f"  Best CV Score: {row['Best_CV_Score']:.4f}\n")
        f.write(f"  Best Parameters: {row['Best_Params']}\n")
    f.write("\n")

    # Task 3: Cross-Validation
    f.write("=" * 80 + "\n")
    f.write("TASK 3: K-FOLD CROSS-VALIDATION\n")
    f.write("=" * 80 + "\n")
    for target in target_columns:
        f.write(f"\n{target}:\n")
        target_cv = cv_df[cv_df["Target"] == target]
        for _, row in target_cv.iterrows():
            f.write(
                f"  {row['Model']} ({row['CV_Folds']}-Fold): {row['Mean_Score']:.4f} (+/- {row['Std_Score']:.4f})\n"
            )
    f.write("\n")

    # Task 4: Ensemble Methods
    f.write("=" * 80 + "\n")
    f.write("TASK 4: ENSEMBLE METHODS\n")
    f.write("=" * 80 + "\n")
    for result in ensemble_results:
        f.write(f"\n{result['Target']}:\n")
        if "Ensemble_Accuracy" in result:
            f.write(f"  Random Forest Accuracy: {result['RF_Accuracy']:.4f}\n")
            f.write(f"  Gradient Boosting Accuracy: {result['GB_Accuracy']:.4f}\n")
            f.write(f"  Ensemble Accuracy: {result['Ensemble_Accuracy']:.4f}\n")
            f.write(f"  Ensemble F1-Score: {result['Ensemble_F1']:.4f}\n")
        else:
            f.write(
                f"  Random Forest R²: {result['RF_R2']:.4f}, MAE: {result['RF_MAE']:.4f}\n"
            )
            f.write(
                f"  Gradient Boosting R²: {result['GB_R2']:.4f}, MAE: {result['GB_MAE']:.4f}\n"
            )
            f.write(
                f"  Ensemble R²: {result['Ensemble_R2']:.4f}, MAE: {result['Ensemble_MAE']:.4f}\n"
            )
    f.write("\n")

    # Task 5: External Validation
    f.write("=" * 80 + "\n")
    f.write("TASK 5: EXTERNAL VALIDATION\n")
    f.write("=" * 80 + "\n")
    for result in external_results:
        f.write(f"\n{result['Target']}:\n")
        if "Accuracy" in result:
            f.write(f"  Accuracy: {result['Accuracy']:.4f}\n")
            f.write(f"  Precision: {result['Precision']:.4f}\n")
            f.write(f"  Recall: {result['Recall']:.4f}\n")
            f.write(f"  F1-Score: {result['F1_Score']:.4f}\n")
        else:
            f.write(f"  R²: {result['R2']:.4f}\n")
            f.write(f"  MAE: {result['MAE']:.4f}\n")
            f.write(f"  RMSE: {result['RMSE']:.4f}\n")
    f.write("\n")

    # Task 6: Statistical Significance
    f.write("=" * 80 + "\n")
    f.write("TASK 6: STATISTICAL SIGNIFICANCE TESTING\n")
    f.write("=" * 80 + "\n")
    for _, row in stat_df.iterrows():
        f.write(f"\n{row['Target']}:\n")
        f.write(f"  Test: {row['Test']}\n")
        f.write(f"  Baseline: {row['Baseline_Metric']:.4f}\n")
        f.write(f"  Ensemble: {row['Ensemble_Metric']:.4f}\n")
        f.write(f"  Test Statistic: {row['Test_Statistic']:.4f}\n")
        f.write(f"  p-value: {row['P_Value']:.4f}\n")
        f.write(
            f"  Statistically Significant: {'YES' if row['Significant'] else 'NO'}\n"
        )
    f.write("\n")

    # Summary
    f.write("=" * 80 + "\n")
    f.write("SUMMARY\n")
    f.write("=" * 80 + "\n")
    f.write("All 6 recommendations successfully implemented:\n")
    f.write("✓ Feature Engineering: Polynomial features (degree 2)\n")
    f.write("✓ Hyperparameter Tuning: GridSearchCV optimization\n")
    f.write("✓ Cross-Validation: 5-fold and 10-fold CV\n")
    f.write("✓ Ensemble Methods: Voting ensembles (RF + GB)\n")
    f.write("✓ External Validation: Independent holdout set (20%)\n")
    f.write("✓ Statistical Significance: Paired t-tests and McNemar's test\n\n")

    sig_count = stat_df["Significant"].sum()
    f.write(
        f"Statistical Significance: {sig_count}/{len(stat_df)} improvements are statistically significant (p < 0.05)\n"
    )

print("✓ Text report saved: advanced_analysis_summary.txt\n")

print("=" * 80)
print("ADVANCED ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  1. hyperparameter_tuning_results.csv")
print("  2. cross_validation_results.csv")
print("  3. ensemble_results.csv")
print("  4. external_validation_results.csv")
print("  5. statistical_significance_results.csv")
print("  6. advanced_analysis_results.png")
print("  7. advanced_analysis_summary.txt")
print("\nAll 6 recommendations from future work have been successfully implemented!")
print("=" * 80)
