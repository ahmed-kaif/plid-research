import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: DATA LOADING AND IDENTIFICATION OF 78 COMPLETE SAMPLES
# ============================================================================
print("=" * 80)
print("STEP 1: DATA LOADING AND IDENTIFICATION OF COMPLETE SAMPLES")
print("=" * 80)

df = pd.read_csv('plid.csv')
print(f"\n1. Original dataset shape: {df.shape}")
print(f"   Total records: {len(df)}")

# Define target columns
TARGET_COLUMNS = [
    'Post operative ODI',
    'Post operative NRS back pain',
    'Surgery outcome according to Macnab criteria',
    'Post operative NRS leg pain'
]

# Find samples with all target columns filled (complete cases for targets)
df_complete_targets = df.dropna(subset=TARGET_COLUMNS).copy()
print(f"\n2. Samples with ALL target columns filled: {len(df_complete_targets)}")
print(f"   These {len(df_complete_targets)} samples will be used for step 1 imputation")

# Identify numeric and categorical features (excluding targets)
all_features = df_complete_targets.drop(columns=['Timestamp', 'Id']).columns.tolist()
for target in TARGET_COLUMNS:
    if target in all_features:
        all_features.remove(target)

df_complete_targets_features = df_complete_targets[all_features]

# Separate numeric and categorical features
numeric_features = df_complete_targets_features.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df_complete_targets_features.select_dtypes(include=['object']).columns.tolist()

print(f"\n3. Feature types in complete samples:")
print(f"   Numeric features: {len(numeric_features)}")
print(f"   Categorical features: {len(categorical_features)}")

# Analyze missing patterns in 78 complete samples
print(f"\n4. Missing value patterns in 78 complete samples:")
missing_in_complete = df_complete_targets_features.isnull().sum()
missing_in_complete = missing_in_complete[missing_in_complete > 0].sort_values(ascending=False)

if len(missing_in_complete) > 0:
    print(f"   Columns with missing values:")
    for col, count in missing_in_complete.items():
        pct = (count / len(df_complete_targets)) * 100
        print(f"      - {col}: {count} missing ({pct:.1f}%)")
else:
    print("   No missing values in features of complete samples!")

# ============================================================================
# STEP 2: IMPUTE MISSING NUMERIC COLUMNS IN 78 COMPLETE SAMPLES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: IMPUTING MISSING VALUES IN 78 COMPLETE SAMPLES")
print("=" * 80)

# Step 2a: Impute categorical features with Mode in complete samples
print(f"\n1. Imputing categorical features with Mode in 78 samples...")
for cat_col in categorical_features:
    if df_complete_targets_features[cat_col].isnull().sum() > 0:
        mode_val = df_complete_targets_features[cat_col].mode()[0]
        df_complete_targets_features[cat_col].fillna(mode_val, inplace=True)
        print(f"   - {cat_col}: filled with mode '{mode_val}'")

# Step 2b: Train regression models on numeric features for the 78 complete samples
print(f"\n2. Training regression models on {len(numeric_features)} numeric features...")

# Remove rows with missing numeric values for training
df_train_numeric = df_complete_targets_features[numeric_features].dropna()
n_samples_for_training = len(df_train_numeric)
print(f"   Samples available for training numeric models: {n_samples_for_training}")

if n_samples_for_training < len(df_complete_targets_features):
    print(f"   Note: {len(df_complete_targets_features) - n_samples_for_training} samples have missing numeric values")

# Train a Random Forest model to predict missing numeric values
# We'll use the features that don't have missing values to predict those that do
numeric_imputation_models = {}

for numeric_col in numeric_features:
    col_missing = df_complete_targets_features[numeric_col].isnull().sum()
    if col_missing > 0:
        print(f"\n   Training model for '{numeric_col}' ({col_missing} missing values):")
        
        # Get features available for training (no missing values in this column)
        train_data = df_complete_targets_features[df_complete_targets_features[numeric_col].notna()].copy()
        
        # Use other numeric features as predictors
        predictor_cols = [col for col in numeric_features if col != numeric_col]
        
        X_train = train_data[predictor_cols].copy()
        y_train = train_data[numeric_col].copy()
        
        # Fill any remaining NaN in X_train with median
        X_train = X_train.fillna(X_train.median())
        
        if len(X_train) > 0 and X_train.shape[1] > 0:
            model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=min(5, len(X_train)//2), scoring='r2')
            print(f"      Cross-validation R² scores: {cv_scores}")
            print(f"      Mean R² score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
            numeric_imputation_models[numeric_col] = {
                'model': model,
                'predictor_cols': predictor_cols
            }

# Apply imputation using trained models
print(f"\n3. Applying imputation to 78 complete samples...")
for numeric_col, model_info in numeric_imputation_models.items():
    missing_mask = df_complete_targets_features[numeric_col].isnull()
    if missing_mask.sum() > 0:
        X_missing = df_complete_targets_features[missing_mask][model_info['predictor_cols']].copy()
        X_missing = X_missing.fillna(X_missing.median())
        
        if len(X_missing) > 0:
            predictions = model_info['model'].predict(X_missing)
            df_complete_targets_features.loc[missing_mask, numeric_col] = predictions
            print(f"   - {numeric_col}: imputed {missing_mask.sum()} values")

# Verify no missing values in 78 complete samples
remaining_missing = df_complete_targets_features.isnull().sum().sum()
print(f"\n4. After imputation - Remaining missing values in 78 samples: {remaining_missing}")

# ============================================================================
# STEP 3: IMPUTE REMAINING DATASET'S NUMERIC AND CATEGORICAL COLUMNS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: IMPUTING ENTIRE DATASET (349 SAMPLES)")
print("=" * 80)

# Create a dataset with imputed 60 complete samples
df_complete_targets[all_features] = df_complete_targets_features

# Prepare the full dataset for imputation
df_full = df.copy()

# Step 3a: Impute categorical features with Mode in entire dataset
print(f"\n1. Imputing categorical features with Mode in full dataset...")
print(f"   Starting with {len(df_full)} total samples")

for cat_col in categorical_features:
    if df_full[cat_col].isnull().sum() > 0:
        # Use mode from complete samples if available, otherwise from full dataset
        if df_complete_targets[cat_col].isnull().sum() == 0:
            mode_val = df_complete_targets[cat_col].mode()[0]
        else:
            mode_val = df_full[cat_col].mode()[0]
        
        n_filled = df_full[cat_col].isnull().sum()
        df_full[cat_col].fillna(mode_val, inplace=True)
        print(f"   - {cat_col}: filled {n_filled} values with mode '{mode_val}'")

# Step 3b: Use trained numeric models to impute entire dataset
print(f"\n2. Using trained regression models to impute numeric features in full dataset...")

# First, extract features from full dataset
df_full_features = df_full.drop(columns=['Timestamp', 'Id'] + TARGET_COLUMNS, errors='ignore')

for numeric_col in numeric_features:
    missing_mask = df_full_features[numeric_col].isnull()
    
    if missing_mask.sum() > 0 and numeric_col in numeric_imputation_models:
        model_info = numeric_imputation_models[numeric_col]
        
        X_missing = df_full_features[missing_mask][model_info['predictor_cols']].copy()
        # Fill any remaining NaN in predictors with median from complete samples
        for pred_col in model_info['predictor_cols']:
            if X_missing[pred_col].isnull().sum() > 0:
                median_val = df_complete_targets[pred_col].median()
                X_missing[pred_col].fillna(median_val, inplace=True)
        
        predictions = model_info['model'].predict(X_missing)
        df_full_features.loc[missing_mask, numeric_col] = predictions
        print(f"   - {numeric_col}: imputed {missing_mask.sum()} values")
    elif missing_mask.sum() > 0:
        # If no model available, use median from complete samples
        median_val = df_complete_targets[numeric_col].median()
        n_filled = missing_mask.sum()
        df_full_features.loc[missing_mask, numeric_col] = median_val
        print(f"   - {numeric_col}: filled {n_filled} values with median {median_val:.2f}")

# Update the full dataframe
for col in df_full_features.columns:
    df_full[col] = df_full_features[col]

# Verify no missing values in features
remaining_missing = df_full[all_features].isnull().sum().sum()
print(f"\n3. After full imputation - Remaining missing values in features: {remaining_missing}")
print(f"   Total imputed dataset size: {len(df_full)} samples")

# ============================================================================
# STEP 4: PREPARE COMPLETE DATASET AND SPLIT INTO TRAIN/TEST
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: PREPARING COMPLETE DATASET FOR MODELING")
print("=" * 80)

# Filter dataset: keep only samples with all target columns filled
df_final = df_full.dropna(subset=TARGET_COLUMNS).copy()
print(f"\n1. Final dataset with complete targets: {len(df_final)} samples")

# Verify no missing values in features
missing_in_final = df_final[all_features].isnull().sum()
if missing_in_final.sum() > 0:
    print(f"   Warning: Still have missing values in features!")
    print(missing_in_final[missing_in_final > 0])
else:
    print(f"   ✓ No missing values in features")

# Check target columns
print(f"\n2. Target columns - Missing value check:")
for target in TARGET_COLUMNS:
    missing = df_final[target].isnull().sum()
    print(f"   - {target}: {missing} missing")

# Prepare X and y
X = df_final[all_features].copy()
y_odi = df_final['Post operative ODI'].copy()
y_back_pain = df_final['Post operative NRS back pain'].copy()
y_leg_pain = df_final['Post operative NRS leg pain'].copy()
y_surgery_outcome = df_final['Surgery outcome according to Macnab criteria'].copy()

print(f"\n3. Feature matrix X shape: {X.shape}")
print(f"   Target shapes: ODI={y_odi.shape}, Back Pain={y_back_pain.shape}, "
      f"Leg Pain={y_leg_pain.shape}, Surgery Outcome={y_surgery_outcome.shape}")

# Encode categorical columns before model training
print(f"\n4. Encoding categorical columns...")

# Encode Age column if it's categorical
if 'Age' in X.columns and X['Age'].dtype == 'object':
    age_mapping = {
        '21-25': 23,
        '26-30': 28,
        '31-35': 33,
        '36-40': 38,
        '41-45': 43,
        '46-50': 48,
        '51-55': 53,
        '56-60': 58,
        'more than 60': 65
    }
    X['Age'] = X['Age'].map(age_mapping).astype(float)
    print("   - Age encoded to numeric values")

# Encode Sex column if present
if 'Sex' in X.columns and X['Sex'].dtype == 'object':
    sex_mapping = {'Male': 1, 'Female': 0}
    X['Sex'] = X['Sex'].map(sex_mapping).astype(float)
    print("   - Sex encoded (Male=1, Female=0)")

# Encode all remaining categorical columns
le_dict = {}
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        le_dict[col] = le
        print(f"   - {col} encoded ({len(le.classes_)} classes)")

# ============================================================================
# STEP 5: BUILD REGRESSION MODELS FOR NUMERIC TARGETS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: BUILDING REGRESSION MODELS FOR NUMERIC TARGETS")
print("=" * 80)

# Now all features are numeric after encoding
print(f"\nFeatures breakdown:")
print(f"  Total features (all numeric after encoding): {X.shape[1]}")

# Build preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, X.columns.tolist())
    ],
    verbose_feature_names_out=False
).set_output(transform="pandas")

# Split data into train/test
X_train, X_test, y_train_odi, y_test_odi, y_train_bp, y_test_bp, \
    y_train_lp, y_test_lp, y_train_so, y_test_so = train_test_split(
        X, y_odi, y_back_pain, y_leg_pain, y_surgery_outcome,
        test_size=0.2, random_state=42
    )

print(f"\nTrain/Test Split:")
print(f"  Training samples: {len(X_train)}")
print(f"  Testing samples: {len(X_test)}")

# ============================================================================
# REGRESSION MODELS - POST OPERATIVE ODI
# ============================================================================
print("\n" + "-" * 80)
print("MODEL 1: Post operative ODI (Regression)")
print("-" * 80)

rf_odi_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, max_depth=15, 
                                       random_state=42, n_jobs=-1))
])

rf_odi_pipeline.fit(X_train, y_train_odi)
y_pred_odi = rf_odi_pipeline.predict(X_test)

r2_odi = r2_score(y_test_odi, y_pred_odi)
mae_odi = mean_absolute_error(y_test_odi, y_pred_odi)
rmse_odi = np.sqrt(mean_squared_error(y_test_odi, y_pred_odi))

print(f"\nPerformance Metrics:")
print(f"  R² Score: {r2_odi:.4f}")
print(f"  MAE: {mae_odi:.4f} ODI points")
print(f"  RMSE: {rmse_odi:.4f} ODI points")

# ============================================================================
# REGRESSION MODELS - POST OPERATIVE NRS BACK PAIN
# ============================================================================
print("\n" + "-" * 80)
print("MODEL 2: Post operative NRS Back Pain (Regression)")
print("-" * 80)

rf_bp_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, max_depth=15, 
                                       random_state=42, n_jobs=-1))
])

rf_bp_pipeline.fit(X_train, y_train_bp)
y_pred_bp = rf_bp_pipeline.predict(X_test)

r2_bp = r2_score(y_test_bp, y_pred_bp)
mae_bp = mean_absolute_error(y_test_bp, y_pred_bp)
rmse_bp = np.sqrt(mean_squared_error(y_test_bp, y_pred_bp))

print(f"\nPerformance Metrics:")
print(f"  R² Score: {r2_bp:.4f}")
print(f"  MAE: {mae_bp:.4f} NRS points")
print(f"  RMSE: {rmse_bp:.4f} NRS points")

# ============================================================================
# REGRESSION MODELS - POST OPERATIVE NRS LEG PAIN
# ============================================================================
print("\n" + "-" * 80)
print("MODEL 3: Post operative NRS Leg Pain (Regression)")
print("-" * 80)

rf_lp_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, max_depth=15, 
                                       random_state=42, n_jobs=-1))
])

rf_lp_pipeline.fit(X_train, y_train_lp)
y_pred_lp = rf_lp_pipeline.predict(X_test)

r2_lp = r2_score(y_test_lp, y_pred_lp)
mae_lp = mean_absolute_error(y_test_lp, y_pred_lp)
rmse_lp = np.sqrt(mean_squared_error(y_test_lp, y_pred_lp))

print(f"\nPerformance Metrics:")
print(f"  R² Score: {r2_lp:.4f}")
print(f"  MAE: {mae_lp:.4f} NRS points")
print(f"  RMSE: {rmse_lp:.4f} NRS points")

# ============================================================================
# STEP 6: BUILD CLASSIFICATION MODEL FOR SURGERY OUTCOME
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: BUILDING CLASSIFICATION MODEL FOR SURGERY OUTCOME")
print("=" * 80)

print("\nSurgery Outcome Classes Distribution (Training Set):")
print(y_train_so.value_counts())

print(f"\nTraining set class distribution:")
print(y_train_so.value_counts())

print(f"\nTesting set class distribution:")
print(y_test_so.value_counts())

# Build classification pipeline
rf_so_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=15, 
                                         random_state=42, n_jobs=-1))
])

rf_so_pipeline.fit(X_train, y_train_so)
y_pred_so = rf_so_pipeline.predict(X_test)

acc_so = accuracy_score(y_test_so, y_pred_so)
precision_so = precision_score(y_test_so, y_pred_so, average='weighted', zero_division=0)
recall_so = recall_score(y_test_so, y_pred_so, average='weighted', zero_division=0)
f1_so = f1_score(y_test_so, y_pred_so, average='weighted', zero_division=0)

print(f"\n" + "-" * 80)
print("MODEL 4: Surgery Outcome According to Macnab Criteria (Classification)")
print("-" * 80)

print(f"\nPerformance Metrics:")
print(f"  Accuracy: {acc_so:.4f}")
print(f"  Precision (weighted): {precision_so:.4f}")
print(f"  Recall (weighted): {recall_so:.4f}")
print(f"  F1-Score (weighted): {f1_so:.4f}")

print(f"\nClassification Report:")
print(classification_report(y_test_so, y_pred_so))

# ============================================================================
# STEP 7: SUMMARY AND RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY OF RESULTS")
print("=" * 80)

print("\n1. DATA IMPUTATION SUMMARY:")
print(f"   - 60 complete samples imputed with trained regression models")
print(f"   - Entire dataset imputed using trained models + mode for categorical")
print(f"   - Final modeling dataset: {len(df_final)} samples")
print(f"   - Features: {X.shape[1]} (all numeric after encoding)")

print("\n2. MODEL PERFORMANCE SUMMARY:")
print(f"\n   Regression Models:")
print(f"   ┌─ Post operative ODI")
print(f"   │  ├─ R² Score: {r2_odi:.4f}")
print(f"   │  ├─ MAE: {mae_odi:.4f} points")
print(f"   │  └─ RMSE: {rmse_odi:.4f} points")
print(f"   │")
print(f"   ├─ Post operative NRS Back Pain")
print(f"   │  ├─ R² Score: {r2_bp:.4f}")
print(f"   │  ├─ MAE: {mae_bp:.4f} points")
print(f"   │  └─ RMSE: {rmse_bp:.4f} points")
print(f"   │")
print(f"   └─ Post operative NRS Leg Pain")
print(f"      ├─ R² Score: {r2_lp:.4f}")
print(f"      ├─ MAE: {mae_lp:.4f} points")
print(f"      └─ RMSE: {rmse_lp:.4f} points")

print(f"\n   Classification Model:")
print(f"   └─ Surgery Outcome (Macnab)")
print(f"      ├─ Accuracy: {acc_so:.4f}")
print(f"      ├─ Precision: {precision_so:.4f}")
print(f"      ├─ Recall: {recall_so:.4f}")
print(f"      └─ F1-Score: {f1_so:.4f}")

print("\n3. FILES GENERATED:")
print(f"   - imputed_dataset.csv: Complete imputed dataset")
print(f"   - modeling_results.txt: Detailed results summary")

# Save the imputed dataset
df_final.to_csv('imputed_dataset.csv', index=False)
print(f"\n✓ Imputed dataset saved to 'imputed_dataset.csv'")

# Save results summary
with open('modeling_results.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("PLID Research - Data Imputation and Modeling Results\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("DATA SUMMARY:\n")
    f.write(f"Final modeling dataset: {len(df_final)} samples\n")
    f.write(f"Features: {X.shape[1]} (all numeric after encoding)\n")
    f.write(f"Train/Test Split: {len(X_train)} / {len(X_test)}\n\n")
    
    f.write("REGRESSION MODELS:\n")
    f.write(f"1. Post operative ODI\n")
    f.write(f"   R² Score: {r2_odi:.4f}\n")
    f.write(f"   MAE: {mae_odi:.4f}\n")
    f.write(f"   RMSE: {rmse_odi:.4f}\n\n")
    
    f.write(f"2. Post operative NRS Back Pain\n")
    f.write(f"   R² Score: {r2_bp:.4f}\n")
    f.write(f"   MAE: {mae_bp:.4f}\n")
    f.write(f"   RMSE: {rmse_bp:.4f}\n\n")
    
    f.write(f"3. Post operative NRS Leg Pain\n")
    f.write(f"   R² Score: {r2_lp:.4f}\n")
    f.write(f"   MAE: {mae_lp:.4f}\n")
    f.write(f"   RMSE: {rmse_lp:.4f}\n\n")
    
    f.write("CLASSIFICATION MODEL:\n")
    f.write(f"4. Surgery Outcome (Macnab Criteria)\n")
    f.write(f"   Accuracy: {acc_so:.4f}\n")
    f.write(f"   Precision (weighted): {precision_so:.4f}\n")
    f.write(f"   Recall (weighted): {recall_so:.4f}\n")
    f.write(f"   F1-Score (weighted): {f1_so:.4f}\n")

print(f"✓ Results summary saved to 'modeling_results.txt'")

print("\n" + "=" * 80)
print("Process completed successfully!")
print("=" * 80)
