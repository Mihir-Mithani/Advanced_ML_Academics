"""
Case Study: Algorithmic Fairness & Risk Assessment in High-Stakes Financial Lending
Complete Solution - Parts A, B, and C

Part A: Evaluation & Baseline Modeling (35 Marks)
Part B: Algorithmic Analysis & Fairness Audit (35 Marks)
Part C: Interpretation & Explainable AI (XAI) (30 Marks)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb

# SHAP for Explainability
import shap

# Fairness Metrics
from sklearn.metrics import confusion_matrix

import json
import pickle
import os

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def business_cost_matrix(y_true, y_pred, fn_cost=5.0, fp_cost=1.0):
    """
    Custom Business Cost Matrix:
    - False Negative (defaulted loan approved): Cost = 5x
    - False Positive (good loan rejected): Cost = 1x
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = fn * fn_cost + fp * fp_cost
    return total_cost, tn, fp, fn, tp

def evaluate_model(y_true, y_pred_proba, y_pred, model_name="Model"):
    """Comprehensive model evaluation"""
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    cost, tn, fp, fn, tp = business_cost_matrix(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"{model_name} Evaluation")
    print(f"{'='*50}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"Business Cost: {cost:.2f} (FN={fn}, FP={fp}, TN={tn}, TP={tp})")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred):.4f}")

    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'business_cost': cost,
        'confusion_matrix': (tn, fp, fn, tp),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }

def find_optimal_threshold(y_true, y_pred_proba, fn_cost=5.0, fp_cost=1.0):
    """Find optimal threshold based on business cost"""
    thresholds = np.arange(0.01, 0.99, 0.01)
    best_threshold = 0.5
    best_cost = float('inf')

    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        cost, _, _, _, _ = business_cost_matrix(y_true, y_pred, fn_cost, fp_cost)
        if cost < best_cost:
            best_cost = cost
            best_threshold = thresh

    return best_threshold, best_cost

# ============================================================
# PART A: DATA PREPROCESSING & BASELINE MODELING
# ============================================================

def load_and_preprocess_data():
    """Load and preprocess the Home Credit dataset"""
    print("Loading datasets...")
    train = pd.read_csv('dataset/home-credit-default-risk/application_train.csv')
    test = pd.read_csv('dataset/home-credit-default-risk/application_test.csv')

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Default rate: {train['TARGET'].mean():.4f} ({train['TARGET'].mean()*100:.2f}%)")

    # Identify categorical and numerical columns
    cat_cols = train.select_dtypes(include=['object']).columns.tolist()
    num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    num_cols.remove('TARGET')
    num_cols.remove('SK_ID_CURR')

    print(f"Categorical columns: {len(cat_cols)}")
    print(f"Numerical columns: {len(num_cols)}")

    # Handle missing values
    # For numerical: median imputation
    # For categorical: mode imputation

    for col in num_cols:
        if train[col].isnull().sum() > 0:
            median_val = train[col].median()
            if pd.isna(median_val):
                median_val = 0
            train[col].fillna(median_val, inplace=True)
            test[col].fillna(median_val, inplace=True)

    for col in cat_cols:
        if train[col].isnull().sum() > 0:
            mode_val = train[col].mode()[0]
            train[col].fillna(mode_val, inplace=True)
            test[col].fillna(mode_val, inplace=True)

    # Double-check for any remaining NaN (some columns might be all NaN)
    for col in train.columns:
        if train[col].isnull().sum() > 0:
            if col in num_cols:
                train[col].fillna(0, inplace=True)
                test[col].fillna(0, inplace=True)
            else:
                train[col].fillna('Unknown', inplace=True)
                test[col].fillna('Unknown', inplace=True)

    # Encode categorical variables
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        # Fit on combined train + test to handle unseen categories
        combined = pd.concat([train[col], test[col]], axis=0)
        le.fit(combined.astype(str))
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
        label_encoders[col] = le

    return train, test, cat_cols, num_cols, label_encoders

def prepare_features(train, test, cat_cols, num_cols):
    """Prepare features for modeling"""
    # Features and target
    X = train.drop(['TARGET', 'SK_ID_CURR'], axis=1)
    y = train['TARGET']
    X_test = test.drop(['SK_ID_CURR'], axis=1)

    # Ensure same columns
    common_cols = list(set(X.columns) & set(X_test.columns))
    X = X[common_cols]
    X_test = X_test[common_cols]

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    return X_train, X_val, y_train, y_val, X_test, X, y

def train_baseline_models(X_train, X_val, y_train, y_val):
    """Train multiple baseline models"""
    models = {}
    results = {}

    # 1. Logistic Regression (with scaling)
    print("\n" + "="*50)
    print("Training Logistic Regression...")
    print("="*50)
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)

    # Check for any remaining NaN or inf
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    lr = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    lr.fit(X_train_scaled, y_train)
    y_pred_proba_lr = lr.predict_proba(X_val_scaled)[:, 1]

    # Find optimal threshold
    opt_thresh_lr, _ = find_optimal_threshold(y_val, y_pred_proba_lr)
    y_pred_lr = (y_pred_proba_lr >= opt_thresh_lr).astype(int)

    results['LogisticRegression'] = evaluate_model(y_val, y_pred_proba_lr, y_pred_lr, "Logistic Regression")
    models['LogisticRegression'] = {'model': lr, 'scaler': scaler, 'threshold': opt_thresh_lr}

    # 2. Random Forest
    print("\n" + "="*50)
    print("Training Random Forest...")
    print("="*50)
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_proba_rf = rf.predict_proba(X_val)[:, 1]

    opt_thresh_rf, _ = find_optimal_threshold(y_val, y_pred_proba_rf)
    y_pred_rf = (y_pred_proba_rf >= opt_thresh_rf).astype(int)

    results['RandomForest'] = evaluate_model(y_val, y_pred_proba_rf, y_pred_rf, "Random Forest")
    models['RandomForest'] = {'model': rf, 'scaler': None, 'threshold': opt_thresh_rf}

    # 3. LightGBM (optimized for imbalanced data)
    print("\n" + "="*50)
    print("Training LightGBM...")
    print("="*50)

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'is_unbalance': True,
        'random_state': 42,
        'n_jobs': -1
    }

    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=500,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )

    y_pred_proba_lgb = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    opt_thresh_lgb, _ = find_optimal_threshold(y_val, y_pred_proba_lgb)
    y_pred_lgb = (y_pred_proba_lgb >= opt_thresh_lgb).astype(int)

    results['LightGBM'] = evaluate_model(y_val, y_pred_proba_lgb, y_pred_lgb, "LightGBM")
    models['LightGBM'] = {'model': lgb_model, 'scaler': None, 'threshold': opt_thresh_lgb}

    # 4. XGBoost
    print("\n" + "="*50)
    print("Training XGBoost...")
    print("="*50)

    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
        random_state=42,
        n_jobs=-1,
        eval_metric='auc',
        early_stopping_rounds=50
    )

    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    y_pred_proba_xgb = xgb_model.predict_proba(X_val)[:, 1]
    opt_thresh_xgb, _ = find_optimal_threshold(y_val, y_pred_proba_xgb)
    y_pred_xgb = (y_pred_proba_xgb >= opt_thresh_xgb).astype(int)

    results['XGBoost'] = evaluate_model(y_val, y_pred_proba_xgb, y_pred_xgb, "XGBoost")
    models['XGBoost'] = {'model': xgb_model, 'scaler': None, 'threshold': opt_thresh_xgb}

    return models, results

# ============================================================
# PART B: FAIRNESS AUDIT
# ============================================================

def define_protected_attributes(df):
    """
    Define sensitive/protected attributes using available proxies:
    - Gender (CODE_GENDER)
    - Age groups (derived from DAYS_BIRTH)
    - Region rating (REGION_RATING_CLIENT)
    - Income brackets (AMT_INCOME_TOTAL)
    - Housing type (NAME_HOUSING_TYPE)
    """
    df = df.copy()

    # Gender
    if 'CODE_GENDER' in df.columns:
        df['protected_gender'] = df['CODE_GENDER'].map({'M': 'Male', 'F': 'Female', 'XNA': 'Other'})

    # Age groups
    if 'DAYS_BIRTH' in df.columns:
        df['age_years'] = (-df['DAYS_BIRTH'] / 365.25).astype(int)
        df['protected_age_group'] = pd.cut(
            df['age_years'],
            bins=[0, 25, 35, 45, 55, 65, 100],
            labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
        )

    # Region rating (proxy for geographic discrimination/redlining)
    if 'REGION_RATING_CLIENT' in df.columns:
        df['protected_region_rating'] = df['REGION_RATING_CLIENT'].map({
            1: 'Low', 2: 'Medium', 3: 'High'
        })

    # Income brackets
    if 'AMT_INCOME_TOTAL' in df.columns:
        df['protected_income_bracket'] = pd.qcut(
            df['AMT_INCOME_TOTAL'],
            q=4,
            labels=['Low', 'Lower-Middle', 'Upper-Middle', 'High'],
            duplicates='drop'
        )

    # Housing type (potential proxy for socioeconomic status)
    if 'NAME_HOUSING_TYPE' in df.columns:
        df['protected_housing'] = df['NAME_HOUSING_TYPE']

    return df

def compute_disparate_impact_ratio(y_pred, protected_attr, privileged_group, unprivileged_group):
    """
    Disparate Impact Ratio (DIR) = Selection rate of unprivileged / Selection rate of privileged
    Four-Fifths Rule: DIR >= 0.8 (80%)
    """
    # Selection rate = proportion of positive predictions
    priv_mask = (protected_attr == privileged_group)
    unpriv_mask = (protected_attr == unprivileged_group)

    if priv_mask.sum() == 0 or unpriv_mask.sum() == 0:
        return None

    selection_rate_priv = y_pred[priv_mask].mean()
    selection_rate_unpriv = y_pred[unpriv_mask].mean()

    if selection_rate_priv == 0:
        return None

    dir_ratio = selection_rate_unpriv / selection_rate_priv

    return {
        'disparate_impact_ratio': dir_ratio,
        'selection_rate_privileged': selection_rate_priv,
        'selection_rate_unprivileged': selection_rate_unpriv,
        'four_fifths_rule_pass': dir_ratio >= 0.8,
        'privileged_group': privileged_group,
        'unprivileged_group': unprivileged_group
    }

def compute_equalized_odds(y_true, y_pred, protected_attr):
    """
    Equalized Odds: TPR and FPR should be equal across groups
    """
    groups = protected_attr.unique()
    results = {}

    for group in groups:
        mask = (protected_attr == group)
        if mask.sum() == 0:
            continue

        y_t = y_true[mask]
        y_p = y_pred[mask]

        tn, fp, fn, tp = confusion_matrix(y_t, y_p).ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate

        results[group] = {
            'TPR': tpr,
            'FPR': fpr,
            'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn
        }

    # Compute disparities
    tprs = [v['TPR'] for v in results.values()]
    fprs = [v['FPR'] for v in results.values()]

    tpr_disparity = max(tprs) - min(tprs) if tprs else 0
    fpr_disparity = max(fprs) - min(fprs) if fprs else 0

    return {
        'per_group': results,
        'tpr_disparity': tpr_disparity,
        'fpr_disparity': fpr_disparity,
        'equalized_odds_satisfied': tpr_disparity < 0.1 and fpr_disparity < 0.1
    }

def fairness_audit(model, X_val, y_val, X_val_raw, protected_attrs, model_name="Model"):
    """Comprehensive fairness audit for a model"""
    print(f"\n{'='*60}")
    print(f"FAIRNESS AUDIT: {model_name}")
    print(f"{'='*60}")

    # Get predictions
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_val)[:, 1]
    else:
        # LightGBM
        y_pred_proba = model.predict(X_val)

    # Use optimal threshold from training
    # For simplicity, use 0.5 or find optimal
    opt_thresh, _ = find_optimal_threshold(y_val, y_pred_proba)
    y_pred = (y_pred_proba >= opt_thresh).astype(int)

    audit_results = {}

    for attr_name in protected_attrs:
        if attr_name not in X_val_raw.columns:
            print(f"  Skipping {attr_name} - not in data")
            continue

        print(f"\n  Analyzing: {attr_name}")
        protected = X_val_raw[attr_name]

        # Get unique groups
        groups = protected.unique()
        if len(groups) < 2:
            continue

        # Define privileged/unprivileged based on group size or domain knowledge
        # For gender: Male as privileged (historically)
        # For age: 26-35 or 36-45 as privileged (prime working age)
        # For region: High rating as privileged
        # For income: High bracket as privileged

        if attr_name == 'protected_gender':
            privileged, unprivileged = 'Male', 'Female'
        elif attr_name == 'protected_age_group':
            privileged, unprivileged = '36-45', '18-25'
        elif attr_name == 'protected_region_rating':
            privileged, unprivileged = 'High', 'Low'
        elif attr_name == 'protected_income_bracket':
            privileged, unprivileged = 'High', 'Low'
        elif attr_name == 'protected_housing':
            # House/apt as privileged, others as unprivileged
            privileged = 'House / apartment'
            unprivileged = 'Rented apartment'
        else:
            # Default: largest group as privileged, smallest as unprivileged
            group_counts = protected.value_counts()
            privileged = group_counts.index[0]
            unprivileged = group_counts.index[-1]

        # Disparate Impact Ratio
        dir_result = compute_disparate_impact_ratio(y_pred, protected, privileged, unprivileged)
        if dir_result:
            print(f"    DIR: {dir_result['disparate_impact_ratio']:.4f} "
                  f"(Priv: {dir_result['selection_rate_privileged']:.4f}, "
                  f"Unpriv: {dir_result['selection_rate_unprivileged']:.4f}) "
                  f"{'✓ PASS' if dir_result['four_fifths_rule_pass'] else '✗ FAIL'}")

        # Equalized Odds
        eo_result = compute_equalized_odds(y_val, y_pred, protected)
        print(f"    TPR Disparity: {eo_result['tpr_disparity']:.4f}, "
              f"FPR Disparity: {eo_result['fpr_disparity']:.4f} "
              f"{'✓ PASS' if eo_result['equalized_odds_satisfied'] else '✗ FAIL'}")

        audit_results[attr_name] = {
            'disparate_impact': dir_result,
            'equalized_odds': eo_result
        }

    return audit_results

def proxy_discrimination_analysis(X_val_raw, y_val, y_pred):
    """
    Investigate whether features act as indirect discriminators (redlining)
    """
    print(f"\n{'='*60}")
    print("PROXY DISCRIMINATION / REDLINING ANALYSIS")
    print(f"{'='*60}")

    # Key features to investigate
    proxy_features = [
        'CODE_GENDER',
        'REGION_RATING_CLIENT',
        'REGION_RATING_CLIENT_W_CITY',
        'NAME_HOUSING_TYPE',
        'NAME_TYPE_SUITE',
        'OCCUPATION_TYPE',
        'NAME_INCOME_TYPE',
        'NAME_EDUCATION_TYPE',
        'NAME_FAMILY_STATUS'
    ]

    results = {}

    for feat in proxy_features:
        if feat not in X_val_raw.columns:
            continue

        # Compute approval rates by feature value
        # Need to align indices properly
        X_val_raw_reset = X_val_raw.reset_index(drop=True)
        y_val_reset = y_val.reset_index(drop=True)
        y_pred_reset = pd.Series(y_pred, index=X_val_raw.index).reset_index(drop=True)

        approval_rates = X_val_raw_reset.groupby(feat).apply(
            lambda x: pd.Series({
                'approval_rate': (1 - y_pred_reset.loc[x.index]).mean(),
                'default_rate': y_val_reset.loc[x.index].mean(),
                'count': len(x)
            })
        ).reset_index()

        approval_rates = approval_rates[approval_rates['count'] > 50]  # Filter small groups

        if len(approval_rates) > 1:
            max_approval = approval_rates['approval_rate'].max()
            min_approval = approval_rates['approval_rate'].min()
            disparity = max_approval - min_approval

            results[feat] = {
                'approval_disparity': disparity,
                'by_group': approval_rates.to_dict('records')
            }

            if disparity > 0.15:  # Significant disparity threshold
                print(f"  ⚠ {feat}: Approval disparity = {disparity:.4f}")
                for _, row in approval_rates.iterrows():
                    print(f"    {row[feat]}: Approval={row['approval_rate']:.4f}, "
                          f"Default={row['default_rate']:.4f}, N={row['count']}")

    return results

# ============================================================
# PART C: EXPLAINABLE AI (SHAP & COUNTERFACTUALS)
# ============================================================

def generate_shap_explanations(model, X_train, X_val, feature_names, model_name="Model"):
    """Generate SHAP explanations"""
    print(f"\n{'='*60}")
    print(f"SHAP EXPLANATIONS: {model_name}")
    print(f"{'='*60}")

    # Create SHAP explainer
    if hasattr(model, 'predict_proba'):
        # Tree-based models
        if 'LGBM' in str(type(model)) or 'XGB' in str(type(model)) or 'RandomForest' in str(type(model)):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_val)

            # For binary classification, shap_values is list of [class0, class1]
            if isinstance(shap_values, list):
                shap_values_class1 = shap_values[1]
            else:
                shap_values_class1 = shap_values
        else:
            # Linear models - use KernelExplainer (sample for speed)
            explainer = shap.KernelExplainer(model.predict_proba, X_train.sample(100, random_state=42))
            shap_values = explainer.shap_values(X_val.sample(200, random_state=42))
            shap_values_class1 = shap_values[1]
    else:
        # LightGBM native
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val)
        shap_values_class1 = shap_values

    # Global feature importance (mean |SHAP|)
    mean_abs_shap = np.abs(shap_values_class1).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)

    print("\nTop 20 Global Feature Importance (SHAP):")
    print(feature_importance.head(20).to_string(index=False))

    # Summary plot data
    shap_summary = {
        'feature_importance': feature_importance.to_dict('records'),
        'shap_values_sample': shap_values_class1[:100].tolist() if len(shap_values_class1) > 100 else shap_values_class1.tolist()
    }

    return explainer, shap_values_class1, feature_importance, shap_summary

def counterfactual_analysis(model, X_val, X_val_raw, y_val, feature_names, n_samples=10):
    """
    Counterfactual Analysis: For rejected high-risk profiles,
    find minimum modifications to flip denial to approval
    """
    print(f"\n{'='*60}")
    print("COUNTERFACTUAL ANALYSIS")
    print(f"{'='*60}")

    # Get predictions
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_val)[:, 1]
    else:
        y_pred_proba = model.predict(X_val)

    # Find rejected applicants (high predicted probability of default)
    # Using threshold 0.5 for simplicity
    rejected_mask = (y_pred_proba >= 0.5)
    rejected_indices = np.where(rejected_mask)[0]

    print(f"Total rejected applicants: {len(rejected_indices)}")

    # Select a few samples for counterfactual analysis
    np.random.seed(42)
    sample_indices = np.random.choice(rejected_indices, min(n_samples, len(rejected_indices)), replace=False)

    counterfactuals = []

    for idx in sample_indices:
        x_orig = X_val.iloc[idx:idx+1].copy()
        x_raw = X_val_raw.iloc[idx]
        prob_orig = y_pred_proba[idx]

        # Identify actionable features (non-sensitive, modifiable)
        actionable_features = [
            'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL',
            'CNT_CHILDREN', 'CNT_FAM_MEMBERS',
            'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
            'DAYS_EMPLOYED', 'REGION_RATING_CLIENT'
        ]
        actionable_features = [f for f in actionable_features if f in X_val.columns]

        # Try modifying each actionable feature to see impact
        modifications = []

        for feat in actionable_features:
            x_mod = x_orig.copy()

            # Try reducing credit amount by 10%, 20%, 30%
            if feat in ['AMT_CREDIT', 'AMT_ANNUITY']:
                for pct in [0.9, 0.8, 0.7]:
                    x_mod[feat] = x_orig[feat] * pct
                    if hasattr(model, 'predict_proba'):
                        prob_new = model.predict_proba(x_mod)[0, 1]
                    else:
                        prob_new = model.predict(x_mod)[0]

                    if prob_new < 0.5:  # Flipped to approval
                        modifications.append({
                            'feature': feat,
                            'original_value': x_orig[feat].values[0],
                            'modified_value': x_mod[feat].values[0],
                            'change_pct': (pct - 1) * 100,
                            'original_prob': prob_orig,
                            'new_prob': prob_new,
                            'flip': True
                        })
                        break

            # Try increasing income
            elif feat == 'AMT_INCOME_TOTAL':
                for pct in [1.1, 1.2, 1.5]:
                    x_mod[feat] = x_orig[feat] * pct
                    if hasattr(model, 'predict_proba'):
                        prob_new = model.predict_proba(x_mod)[0, 1]
                    else:
                        prob_new = model.predict(x_mod)[0]

                    if prob_new < 0.5:
                        modifications.append({
                            'feature': feat,
                            'original_value': x_orig[feat].values[0],
                            'modified_value': x_mod[feat].values[0],
                            'change_pct': (pct - 1) * 100,
                            'original_prob': prob_orig,
                            'new_prob': prob_new,
                            'flip': True
                        })
                        break

            # Try improving external scores
            elif feat in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
                for new_val in [0.6, 0.7, 0.8]:
                    if x_orig[feat].values[0] < new_val:
                        x_mod[feat] = new_val
                        if hasattr(model, 'predict_proba'):
                            prob_new = model.predict_proba(x_mod)[0, 1]
                        else:
                            prob_new = model.predict(x_mod)[0]

                        if prob_new < 0.5:
                            modifications.append({
                                'feature': feat,
                                'original_value': x_orig[feat].values[0],
                                'modified_value': new_val,
                                'change_pct': ((new_val / x_orig[feat].values[0]) - 1) * 100 if x_orig[feat].values[0] > 0 else 0,
                                'original_prob': prob_orig,
                                'new_prob': prob_new,
                                'flip': True
                            })
                            break

        if modifications:
            best_mod = min(modifications, key=lambda m: abs(m['change_pct']))
            counterfactuals.append({
                'applicant_id': x_raw.get('SK_ID_CURR', idx),
                'original_prob': prob_orig,
                'actual_default': y_val.iloc[idx],
                'best_modification': best_mod,
                'all_modifications': modifications
            })
            print(f"  Applicant {x_raw.get('SK_ID_CURR', idx)}: "
                  f"Orig prob={prob_orig:.4f} -> "
                  f"Modify {best_mod['feature']} by {best_mod['change_pct']:.1f}% -> "
                  f"New prob={best_mod['new_prob']:.4f} ✓")
        else:
            print(f"  Applicant {x_raw.get('SK_ID_CURR', idx)}: "
                  f"Orig prob={prob_orig:.4f} -> No simple flip found")

    return counterfactuals

def generate_adverse_action_notice(applicant_data, shap_values, feature_names, top_k=3):
    """
    Generate human-readable Adverse Action Notice
    """
    # Get SHAP values for this applicant (top factors pushing toward default)
    # SHAP values > 0 push toward default (class 1)
    applicant_shap = shap_values

    # Get top factors contributing to rejection
    top_indices = np.argsort(applicant_shap)[-top_k:][::-1]  # Top positive SHAP

    notice = {
        'applicant_id': applicant_data.get('SK_ID_CURR', 'Unknown'),
        'decision': 'DENIED',
        'top_factors': []
    }

    for i, idx in enumerate(top_indices):
        feature = feature_names[idx]
        shap_val = applicant_shap[idx]
        value = applicant_data.get(feature, 'N/A')

        # Make feature names human-readable
        factor_desc = {
            'AMT_CREDIT': 'Requested loan amount',
            'AMT_INCOME_TOTAL': 'Annual income',
            'AMT_ANNUITY': 'Annual payment amount',
            'EXT_SOURCE_1': 'External credit score 1',
            'EXT_SOURCE_2': 'External credit score 2',
            'EXT_SOURCE_3': 'External credit score 3',
            'DAYS_EMPLOYED': 'Employment duration',
            'CNT_CHILDREN': 'Number of children',
            'CNT_FAM_MEMBERS': 'Family size',
            'REGION_RATING_CLIENT': 'Region risk rating',
            'CODE_GENDER': 'Gender',
            'NAME_HOUSING_TYPE': 'Housing type',
            'OCCUPATION_TYPE': 'Occupation',
            'NAME_INCOME_TYPE': 'Income type',
            'NAME_EDUCATION_TYPE': 'Education level'
        }.get(feature, feature)

        notice['top_factors'].append({
            'rank': i + 1,
            'factor': factor_desc,
            'feature_name': feature,
            'value': value,
            'impact': 'Increases default risk' if shap_val > 0 else 'Decreases default risk',
            'shap_value': float(shap_val)
        })

    return notice

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("="*70)
    print("CASE STUDY 1: ALGORITHMIC FAIRNESS & RISK ASSESSMENT")
    print("="*70)

    # ============ PART A: BASELINE MODELING ============
    print("\n" + "#"*70)
    print("# PART A: EVALUATION & BASELINE MODELING")
    print("#"*70)

    # Load and preprocess
    train, test, cat_cols, num_cols, label_encoders = load_and_preprocess_data()

    # Prepare features (keep raw for fairness audit)
    X_train, X_val, y_train, y_val, X_test, X_full, y_full = prepare_features(train, test, cat_cols, num_cols)

    # Keep raw validation data for fairness audit
    X_val_raw = train.iloc[X_val.index].copy()

    # Train baseline models
    models, results = train_baseline_models(X_train, X_val, y_train, y_val)

    # Select best model based on business cost
    best_model_name = min(results.keys(), key=lambda k: results[k]['business_cost'])
    best_model_info = models[best_model_name]
    best_model = best_model_info['model']
    best_threshold = best_model_info['threshold']

    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_model_name} (Business Cost: {results[best_model_name]['business_cost']:.2f})")
    print(f"Optimal Threshold: {best_threshold:.4f}")
    print(f"{'='*60}")

    # Save results
    with open('part_a_results.json', 'w') as f:
        # Convert numpy types to native Python
        def convert_to_native(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(x) for x in obj]
            elif isinstance(obj, dict):
                return {convert_to_native(k): convert_to_native(v) for k, v in obj.items()}
            else:
                return obj

        serializable_results = {}
        for k, v in results.items():
            serializable_results[k] = {k2: convert_to_native(v2) for k2, v2 in v.items()}
        json.dump(serializable_results, f, indent=2)

    # ============ PART B: FAIRNESS AUDIT ============
    print("\n" + "#"*70)
    print("# PART B: ALGORITHMIC ANALYSIS & FAIRNESS AUDIT")
    print("#"*70)

    # Define protected attributes on validation set
    X_val_raw = define_protected_attributes(X_val_raw)

    protected_attrs = [
        'protected_gender',
        'protected_age_group',
        'protected_region_rating',
        'protected_income_bracket',
        'protected_housing'
    ]

    # Run fairness audit on best model
    audit_results = fairness_audit(
        best_model, X_val, y_val, X_val_raw, protected_attrs, best_model_name
    )

    # Proxy discrimination analysis
    # Get predictions on validation set
    if hasattr(best_model, 'predict_proba'):
        y_pred_proba_val = best_model.predict_proba(X_val)[:, 1]
    else:
        y_pred_proba_val = best_model.predict(X_val)
    y_pred_val = (y_pred_proba_val >= best_threshold).astype(int)

    proxy_results = proxy_discrimination_analysis(X_val_raw, y_val, y_pred_val)

    # Save fairness audit results
    with open('part_b_fairness_audit.json', 'w') as f:
        # Convert to serializable
        def convert_to_native(obj):
            if isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (bool, np.bool_)):
                return bool(obj)
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(x) for x in obj]
            elif isinstance(obj, dict):
                return {convert_to_native(k): convert_to_native(v) for k, v in obj.items()}
            else:
                return obj

        serializable_audit = {}
        for attr, res in audit_results.items():
            serializable_audit[attr] = {
                'disparate_impact': convert_to_native(res['disparate_impact']),
                'equalized_odds': convert_to_native(res['equalized_odds'])
            }
        json.dump(serializable_audit, f, indent=2)

    with open('part_b_proxy_discrimination.json', 'w') as f:
        serializable_proxy = {}
        for feat, res in proxy_results.items():
            serializable_proxy[feat] = {
                'approval_disparity': float(res['approval_disparity']),
                'by_group': res['by_group']
            }
        json.dump(serializable_proxy, f, indent=2)

    # ============ PART C: EXPLAINABLE AI ============
    print("\n" + "#"*70)
    print("# PART C: INTERPRETATION & EXPLAINABLE AI (XAI)")
    print("#"*70)

    # SHAP explanations for best model
    feature_names = X_val.columns.tolist()

    explainer, shap_vals, feat_importance, shap_summary = generate_shap_explanations(
        best_model, X_train, X_val, feature_names, best_model_name
    )

    # Counterfactual analysis
    counterfactuals = counterfactual_analysis(
        best_model, X_val, X_val_raw, y_val, feature_names, n_samples=10
    )

    # Generate Adverse Action Notices for counterfactual samples
    print(f"\n{'='*60}")
    print("ADVERSE ACTION NOTICES (Sample)")
    print(f"{'='*60}")

    adverse_notices = []
    for cf in counterfactuals[:3]:
        # Find the correct row index in X_val for this applicant
        applicant_id = cf['applicant_id']
        if applicant_id in X_val_raw['SK_ID_CURR'].values:
            idx = X_val_raw[X_val_raw['SK_ID_CURR'] == applicant_id].index[0]
            idx = X_val.index.get_loc(idx)
        else:
            idx = 0
        applicant_raw = X_val_raw.iloc[idx]
        applicant_shap = shap_vals[idx]

        notice = generate_adverse_action_notice(applicant_raw, applicant_shap, feature_names)
        adverse_notices.append(notice)

        print(f"\n--- ADVERSE ACTION NOTICE ---")
        print(f"Applicant ID: {notice['applicant_id']}")
        print(f"Decision: {notice['decision']}")
        print(f"Top factors contributing to denial:")
        for factor in notice['top_factors']:
            print(f"  {factor['rank']}. {factor['factor']} (Value: {factor['value']}) - {factor['impact']}")

    # Save Part C results
    with open('part_c_shap_importance.json', 'w') as f:
        json.dump(feat_importance.to_dict('records'), f, indent=2)

    def convert_to_native(obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(x) for x in obj]
        elif isinstance(obj, dict):
            return {convert_to_native(k): convert_to_native(v) for k, v in obj.items()}
        else:
            return obj

    with open('part_c_counterfactuals.json', 'w') as f:
        serializable_cf = convert_to_native(counterfactuals)
        json.dump(serializable_cf, f, indent=2)

    with open('part_c_adverse_notices.json', 'w') as f:
        json.dump(convert_to_native(adverse_notices), f, indent=2)

    # ============ SUMMARY REPORT ============
    print("\n" + "#"*70)
    print("# EXECUTIVE SUMMARY")
    print("#"*70)

    print(f"""
    CASE STUDY 1: ALGORITHMIC FAIRNESS & RISK ASSESSMENT
    ============================================================

    PART A: BASELINE MODELING RESULTS
    ----------------------------------
    Best Model: {best_model_name}
    ROC-AUC: {results[best_model_name]['roc_auc']:.4f}
    PR-AUC: {results[best_model_name]['pr_auc']:.4f}
    Business Cost: {results[best_model_name]['business_cost']:.2f}
    Optimal Threshold: {best_threshold:.4f}

    All Models Comparison:
    """)

    for name, res in results.items():
        print(f"  {name}: ROC-AUC={res['roc_auc']:.4f}, PR-AUC={res['pr_auc']:.4f}, Cost={res['business_cost']:.2f}")

    print(f"""

    PART B: FAIRNESS AUDIT SUMMARY
    --------------------------------
    """)

    for attr, res in audit_results.items():
        dir_res = res['disparate_impact']
        eo_res = res['equalized_odds']
        if dir_res:
            print(f"  {attr}:")
            print(f"    Disparate Impact Ratio: {dir_res['disparate_impact_ratio']:.4f} "
                  f"({'PASS' if dir_res['four_fifths_rule_pass'] else 'FAIL'} - 4/5ths Rule)")
            print(f"    TPR Disparity: {eo_res['tpr_disparity']:.4f}, "
                  f"FPR Disparity: {eo_res['fpr_disparity']:.4f} "
                  f"({'PASS' if eo_res['equalized_odds_satisfied'] else 'FAIL'} - Equalized Odds)")

    print(f"""

    PART C: XAI & COUNTERFACTUAL INSIGHTS
    --------------------------------------
    Top 5 Global Features (SHAP):
    """)

    for i, row in feat_importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['mean_abs_shap']:.6f}")

    print(f"""
    Counterfactual Analysis: {len(counterfactuals)} rejected applicants analyzed
    Adverse Action Notices generated with top 3 actionable factors

    RECOMMENDATIONS:
    1. Deploy {best_model_name} with threshold {best_threshold:.4f} for optimal business cost
    2. Monitor fairness metrics monthly, especially for {max(audit_results.keys(), key=lambda k: audit_results[k]['disparate_impact']['disparate_impact_ratio'] if audit_results[k]['disparate_impact'] else 0)}
    3. Implement counterfactual explanations for rejected applicants
    4. Regular retraining with fairness constraints
    5. Establish human review process for borderline cases
    """)

    # Save executive summary
    with open('executive_summary.txt', 'w') as f:
        f.write(f"""
CASE STUDY 1: ALGORITHMIC FAIRNESS & RISK ASSESSMENT
============================================================

PART A: BASELINE MODELING RESULTS
----------------------------------
Best Model: {best_model_name}
ROC-AUC: {results[best_model_name]['roc_auc']:.4f}
PR-AUC: {results[best_model_name]['pr_auc']:.4f}
Business Cost: {results[best_model_name]['business_cost']:.2f}
Optimal Threshold: {best_threshold:.4f}

All Models Comparison:
""")
        for name, res in results.items():
            f.write(f"  {name}: ROC-AUC={res['roc_auc']:.4f}, PR-AUC={res['pr_auc']:.4f}, Cost={res['business_cost']:.2f}\n")

        f.write(f"""

PART B: FAIRNESS AUDIT SUMMARY
-------------------------------
""")
        for attr, res in audit_results.items():
            dir_res = res['disparate_impact']
            eo_res = res['equalized_odds']
            if dir_res:
                f.write(f"  {attr}:\n")
                f.write(f"    Disparate Impact Ratio: {dir_res['disparate_impact_ratio']:.4f} "
                      f"({'PASS' if dir_res['four_fifths_rule_pass'] else 'FAIL'} - 4/5ths Rule)\n")
                f.write(f"    TPR Disparity: {eo_res['tpr_disparity']:.4f}, "
                      f"FPR Disparity: {eo_res['fpr_disparity']:.4f} "
                      f"({'PASS' if eo_res['equalized_odds_satisfied'] else 'FAIL'} - Equalized Odds)\n")

        f.write(f"""

PART C: XAI & COUNTERFACTUAL INSIGHTS
--------------------------------------
Top 5 Global Features (SHAP):
""")
        for i, row in feat_importance.head(5).iterrows():
            f.write(f"  {row['feature']}: {row['mean_abs_shap']:.6f}\n")

        f.write(f"""
Counterfactual Analysis: {len(counterfactuals)} rejected applicants analyzed
Adverse Action Notices generated with top 3 actionable factors

RECOMMENDATIONS:
1. Deploy {best_model_name} with threshold {best_threshold:.4f} for optimal business cost
2. Monitor fairness metrics monthly
3. Implement counterfactual explanations for rejected applicants
4. Regular retraining with fairness constraints
5. Establish human review process for borderline cases
""")

    print("\n" + "="*70)
    print("CASE STUDY 1 COMPLETE - All outputs saved")
    print("="*70)

    return models, results, audit_results, proxy_results, counterfactuals, adverse_notices

if __name__ == "__main__":
    main()