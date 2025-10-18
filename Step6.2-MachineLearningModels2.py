#!/usr/bin/env python
# coding: utf-8

import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# AutoML Libraries
try:
    import autokeras as ak
    # Check what's actually available in AutoKeras
    available_classifiers = []
    if hasattr(ak, 'StructuredDataClassifier'):
        available_classifiers.append('StructuredDataClassifier')
    if hasattr(ak, 'AutoModel'):
        available_classifiers.append('AutoModel')
    if hasattr(ak, 'ImageClassifier'):
        available_classifiers.append('ImageClassifier')
    
    if len(available_classifiers) > 0:
        AUTOKERAS_AVAILABLE = True
        print(f"ℹ️  AutoKeras available classes: {', '.join(available_classifiers)}")
    else:
        print("⚠️  AutoKeras installed but no compatible classifiers found")
        AUTOKERAS_AVAILABLE = False
except ImportError:
    print("⚠️  AutoKeras not available. Install with: pip install autokeras")
    AUTOKERAS_AVAILABLE = False
except Exception as e:
    print(f"⚠️  AutoKeras import error: {e}")
    AUTOKERAS_AVAILABLE = False

try:
    from autosklearn.classification import AutoSklearnClassifier
    AUTOSKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️  Auto-sklearn not available. Install with: pip install auto-sklearn")
    AUTOSKLEARN_AVAILABLE = False

try:
    from flaml import AutoML
    FLAML_AVAILABLE = True
except ImportError:
    print("⚠️  FLAML not available. Install with: pip install flaml")
    FLAML_AVAILABLE = False

try:
    from h2o.automl import H2OAutoML
    import h2o
    H2O_AVAILABLE = True
except ImportError:
    print("⚠️  H2O AutoML not available. Install with: pip install h2o")
    H2O_AVAILABLE = False

# Import data loading functions
from Step6data_loaderMachineLearningModelstesting import load_all_data

# Get phenotype from command line
if len(sys.argv) < 2:
    print("Usage: python train_ml.py <phenotype_name>")
    print("Example: python train_ml.py Phenotype_1")
    sys.exit(1)

phenotype = sys.argv[1]

# Configuration
top_snp_counts = [ 500, 2000, 5000,10000]
n_folds = 5

# Training parameters for AutoML
automl_params = {
    'max_time_per_model': 300,  # 5 minutes per model
    'max_trials': 50,  # Maximum trials for AutoKeras
    'time_budget': 600,  # 10 minutes for FLAML
}

print("="*80)
print(f"TRAINING AutoML ENSEMBLE FOR: {phenotype}")
print(f"Features: SNPs + PCA + PRS")
print(f"Datasets: Top {top_snp_counts} SNPs")
print("="*80)
print(f"\nAutoML Ensemble includes:")
available_models = []
if AUTOKERAS_AVAILABLE:
    print(f"  1. AutoKeras (Neural Architecture Search)")
    available_models.append('AutoKeras')
if AUTOSKLEARN_AVAILABLE:
    print(f"  2. Auto-sklearn (Automated ML Pipeline)")
    available_models.append('Auto-sklearn')
if FLAML_AVAILABLE:
    print(f"  3. FLAML (Fast Lightweight AutoML)")
    available_models.append('FLAML')
if H2O_AVAILABLE:
    print(f"  4. H2O AutoML (Automated Machine Learning)")
    available_models.append('H2O')
print(f"  {len(available_models)+1}. Final: Weighted Average Ensemble")
print("="*80)

if len(available_models) == 0:
    print("❌ No AutoML libraries available. Please install at least one:")
    print("   pip install autokeras")
    print("   pip install auto-sklearn")
    print("   pip install flaml")
    print("   pip install h2o")
    sys.exit(1)

# Initialize H2O if available
if H2O_AVAILABLE:
    h2o.init()

# Store best dataset info for plotting
best_dataset_info = {
    'snp_count': None,
    'fold': None,
    'test_auc': 0,
    'train_pred_proba': None,
    'test_pred_proba': None,
    'val_pred_proba': None,
    'y_train': None,
    'y_test': None,
    'val_ids_df': None  # ADD THIS
}

def estimate_validation_auc_range(val_prs_scores, n_cases=2500, n_total=50000,
                                   train_auc=None, test_auc=None, n_simulations=10000):
    """Estimate possible AUC range for validation set using Monte Carlo simulation."""
    
    print("\n" + "="*80)
    print("ESTIMATING VALIDATION AUC RANGE (Monte Carlo Simulation)")
    print("="*80)
    print(f"  Validation samples: {n_total}")
    print(f"  Expected cases: {n_cases}")
    print(f"  Running {n_simulations} Monte Carlo simulations...")
    
    n_controls = n_total - n_cases
    
    # Remove NaN values
    val_prs_scores = val_prs_scores[~np.isnan(val_prs_scores)]
    
    if len(val_prs_scores) != n_total:
        print(f"  ⚠️  Warning: Expected {n_total} samples, got {len(val_prs_scores)}")
        n_total = len(val_prs_scores)
        n_cases = min(n_cases, n_total)
        n_controls = n_total - n_cases
    
    # Rank PRS scores (high to low)
    sorted_indices = np.argsort(val_prs_scores)[::-1]
    
    # Best case: top scores are all cases
    best_case_labels = np.zeros(n_total)
    best_case_labels[sorted_indices[:n_cases]] = 1
    best_case_auc = roc_auc_score(best_case_labels, val_prs_scores)
    
    # Worst case: bottom scores are cases
    worst_case_labels = np.zeros(n_total)
    worst_case_labels[sorted_indices[-n_cases:]] = 1
    worst_case_auc = roc_auc_score(worst_case_labels, val_prs_scores)
    
    # Random case: Monte Carlo simulation
    print("  Running Monte Carlo simulations...")
    random_aucs = []
    for i in range(n_simulations):
        if i % 2000 == 0 and i > 0:
            print(f"    Progress: {i}/{n_simulations}")
        
        random_labels = np.zeros(n_total)
        random_case_indices = np.random.choice(n_total, n_cases, replace=False)
        random_labels[random_case_indices] = 1
        
        try:
            random_auc = roc_auc_score(random_labels, val_prs_scores)
            random_aucs.append(random_auc)
        except:
            continue
    
    random_auc_mean = np.mean(random_aucs)
    random_auc_std = np.std(random_aucs)
    
    # Expected AUC based on test performance
    if test_auc is not None:
        expected_auc = test_auc * 0.95
        expected_range = (test_auc * 0.85, min(test_auc * 1.0, 1.0))
    else:
        expected_auc = random_auc_mean
        expected_range = (random_auc_mean - 2*random_auc_std,
                         random_auc_mean + 2*random_auc_std)
    
    print("  ✓ Monte Carlo simulations complete")
    print(f"\n  Results:")
    print(f"    Random AUC Mean: {random_auc_mean:.6f} ± {random_auc_std:.6f}")
    print(f"    Best Case AUC:   {best_case_auc:.6f}")
    print(f"    Worst Case AUC:  {worst_case_auc:.6f}")
    print(f"    Expected AUC:    {expected_auc:.6f}")
    print("="*80)
    
    return {
        'best_case_auc': best_case_auc,
        'worst_case_auc': worst_case_auc,
        'random_auc_mean': random_auc_mean,
        'random_auc_std': random_auc_std,
        'random_aucs': random_aucs,
        'expected_auc': expected_auc,
        'expected_range': expected_range,
        'theoretical_range': (worst_case_auc, best_case_auc)
    }

def estimate_validation_performance_by_threshold(train_metrics, test_metrics, val_prs_scores, 
                                                  n_cases=2500, n_total=50000):
    """Estimate validation AUC based on decision thresholds from train/test."""
    
    print("\n" + "="*80)
    print("VALIDATION PERFORMANCE ESTIMATION BY THRESHOLD ANALYSIS")
    print("="*80)
    
    val_scores = val_prs_scores[~np.isnan(val_prs_scores)]
    n_controls = n_total - n_cases
    
    # Find optimal thresholds using Youden's index
    train_j_scores = train_metrics['tpr'] - train_metrics['fpr']
    train_optimal_idx = np.argmax(train_j_scores)
    train_threshold = train_metrics['thresholds'][train_optimal_idx]
    train_tpr = train_metrics['tpr'][train_optimal_idx]
    train_fpr = train_metrics['fpr'][train_optimal_idx]
    
    test_j_scores = test_metrics['tpr'] - test_metrics['fpr']
    test_optimal_idx = np.argmax(test_j_scores)
    test_threshold = test_metrics['thresholds'][test_optimal_idx]
    test_tpr = test_metrics['tpr'][test_optimal_idx]
    test_fpr = test_metrics['fpr'][test_optimal_idx]
    
    print(f"\nTrain Optimal Threshold: {train_threshold:.6f}")
    print(f"  Sensitivity (TPR): {train_tpr:.4f}")
    print(f"  1-Specificity (FPR): {train_fpr:.4f}")
    
    print(f"\nTest Optimal Threshold: {test_threshold:.6f}")
    print(f"  Sensitivity (TPR): {test_tpr:.4f}")
    print(f"  1-Specificity (FPR): {test_fpr:.4f}")
    
    # Use average threshold
    avg_threshold = (train_threshold + test_threshold) / 2
    avg_tpr = (train_tpr + test_tpr) / 2
    avg_fpr = (train_fpr + test_fpr) / 2
    
    print(f"\nAverage Threshold: {avg_threshold:.6f}")
    print(f"  Average TPR: {avg_tpr:.4f}")
    print(f"  Average FPR: {avg_fpr:.4f}")
    
    # Count validation samples above threshold
    n_above = np.sum(val_scores >= avg_threshold)
    n_below = np.sum(val_scores < avg_threshold)
    
    print(f"\nValidation samples above threshold: {n_above:,} ({n_above/n_total*100:.2f}%)")
    print(f"Validation samples below threshold: {n_below:,} ({n_below/n_total*100:.2f}%)")
    
    # Scenario B: Expected case - use train/test performance
    expected_cases_above = int(n_cases * avg_tpr)
    expected_controls_above = int(n_controls * avg_fpr)
    
    scenario_b_tp = min(expected_cases_above, n_above)
    scenario_b_fp = n_above - scenario_b_tp
    scenario_b_fn = n_cases - scenario_b_tp
    scenario_b_tn = n_controls - scenario_b_fp
    
    scenario_b_tpr = scenario_b_tp / n_cases
    scenario_b_fpr = scenario_b_fp / n_controls
    scenario_b_auc = 0.5 + (scenario_b_tpr - scenario_b_fpr) / 2
    
    print(f"\nExpected Validation Performance (threshold-based):")
    print(f"  Expected TP: {scenario_b_tp:,}, FP: {scenario_b_fp:,}")
    print(f"  Expected FN: {scenario_b_fn:,}, TN: {scenario_b_tn:,}")
    print(f"  Expected TPR: {scenario_b_tpr:.4f}, FPR: {scenario_b_fpr:.4f}")
    print(f"  Estimated AUC: ~{scenario_b_auc:.6f}")
    
    print("="*80)
    
    return {
        'threshold': avg_threshold,
        'n_above': n_above,
        'n_below': n_below,
        'expected_auc': scenario_b_auc,
        'expected_tpr': scenario_b_tpr,
        'expected_fpr': scenario_b_fpr
    }

def train_autokeras(X_train, y_train, X_test, y_test, data_dir):
    """Train AutoKeras model"""
    print(f"  Training AutoKeras...", end=' ')
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        
        # Try different AutoKeras APIs
        if hasattr(ak, 'StructuredDataClassifier'):
            # Try the documented API
            clf = ak.StructuredDataClassifier(
                max_trials=automl_params['max_trials'],
                overwrite=True,
                directory=os.path.join(data_dir, 'autokeras'),
                seed=42
            )
        elif hasattr(ak, 'AutoModel'):
            # Use AutoModel with Input/Output specification
            input_node = ak.StructuredDataInput()
            output_node = ak.ClassificationHead()
            output_node = output_node(input_node)
            
            clf = ak.AutoModel(
                inputs=input_node,
                outputs=output_node,
                max_trials=automl_params['max_trials'],
                overwrite=True,
                directory=os.path.join(data_dir, 'autokeras'),
                seed=42
            )
        else:
            # Last resort: build a simple Keras model manually
            print("\n  AutoKeras API not compatible, using simple neural network instead...")
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout
            
            clf = Sequential([
                Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
            clf.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.15, verbose=0)
            
            train_pred_proba = clf.predict(X_train, verbose=0).flatten()
            test_pred_proba = clf.predict(X_test, verbose=0).flatten()
            test_auc = roc_auc_score(y_test, test_pred_proba)
            
            print(f"AUC: {test_auc:.4f}")
            clf.save(os.path.join(data_dir, 'autokeras_model'))
            
            return train_pred_proba, test_pred_proba, test_auc
        
        # Train AutoKeras model
        clf.fit(X_train, y_train, validation_split=0.15, verbose=0)
        
        # Predict
        train_pred = clf.predict(X_train, verbose=0)
        test_pred = clf.predict(X_test, verbose=0)
        
        # Handle different output formats
        if len(train_pred.shape) > 1 and train_pred.shape[1] > 1:
            train_pred_proba = train_pred[:, 1]
            test_pred_proba = test_pred[:, 1]
        else:
            train_pred_proba = train_pred.flatten()
            test_pred_proba = test_pred.flatten()
        
        test_auc = roc_auc_score(y_test, test_pred_proba)
        
        print(f"AUC: {test_auc:.4f}")
        
        # Export model
        try:
            model = clf.export_model()
            model.save(os.path.join(data_dir, 'autokeras_model'))
        except:
            # If export fails, save the AutoKeras model directly
            import joblib
            joblib.dump(clf, os.path.join(data_dir, 'autokeras_model.pkl'))
        
        return train_pred_proba, test_pred_proba, test_auc
    
    except Exception as e:
        print(f"Error: {e}")
        raise

def train_autosklearn(X_train, y_train, X_test, y_test, data_dir):
    """Train Auto-sklearn model"""
    print(f"  Training Auto-sklearn...", end=' ')
    
    # Create Auto-sklearn classifier
    clf = AutoSklearnClassifier(
        time_left_for_this_task=automl_params['max_time_per_model'],
        per_run_time_limit=60,
        memory_limit=8192,
        seed=42,
        n_jobs=-1
    )
    
    # Train
    clf.fit(X_train, y_train)
    
    # Predict
    train_pred_proba = clf.predict_proba(X_train)[:, 1]
    test_pred_proba = clf.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_pred_proba)
    
    print(f"AUC: {test_auc:.4f}")
    
    # Save model
    with open(os.path.join(data_dir, 'autosklearn_model.pkl'), 'wb') as f:
        pickle.dump(clf, f)
    
    return train_pred_proba, test_pred_proba, test_auc

def train_flaml(X_train, y_train, X_test, y_test, data_dir):
    """Train FLAML model"""
    print(f"  Training FLAML...", end=' ')
    
    # Create FLAML classifier
    clf = AutoML()
    
    # Train
    clf.fit(
        X_train, y_train,
        task='classification',
        metric='roc_auc',
        time_budget=automl_params['time_budget'],
        seed=42,
        verbose=0
    )
    
    # Predict
    train_pred_proba = clf.predict_proba(X_train)[:, 1]
    test_pred_proba = clf.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_pred_proba)
    
    print(f"AUC: {test_auc:.4f}")
    
    # Save model
    with open(os.path.join(data_dir, 'flaml_model.pkl'), 'wb') as f:
        pickle.dump(clf, f)
    
    return train_pred_proba, test_pred_proba, test_auc

def train_h2o(X_train, y_train, X_test, y_test, data_dir):
    """Train H2O AutoML model"""
    print(f"  Training H2O AutoML...", end=' ')
    
    # Convert to H2O frames
    train_df = pd.DataFrame(X_train)
    train_df['target'] = y_train
    test_df = pd.DataFrame(X_test)
    
    train_h2o = h2o.H2OFrame(train_df)
    test_h2o = h2o.H2OFrame(test_df)
    
    # Convert target to factor
    train_h2o['target'] = train_h2o['target'].asfactor()
    
    # Get feature names
    x = train_h2o.columns[:-1]
    y = 'target'
    
    # Train AutoML
    aml = H2OAutoML(
        max_runtime_secs=automl_params['max_time_per_model'],
        seed=42,
        balance_classes=True,
        verbosity='warn'
    )
    aml.train(x=x, y=y, training_frame=train_h2o)
    
    # Predict
    train_pred = aml.leader.predict(train_h2o)
    test_pred = aml.leader.predict(test_h2o)
    
    train_pred_proba = train_pred['p1'].as_data_frame().values.flatten()
    test_pred_proba = test_pred['p1'].as_data_frame().values.flatten()
    test_auc = roc_auc_score(y_test, test_pred_proba)
    
    print(f"AUC: {test_auc:.4f}")
    
    # Save model
    model_path = h2o.save_model(model=aml.leader, path=data_dir, force=True)
    
    return train_pred_proba, test_pred_proba, test_auc

# Process each SNP count
all_dataset_results = {}

for snp_count in top_snp_counts:
    print(f"\n{'#'*80}")
    print(f"DATASET: TOP {snp_count} SNPs")
    print(f"{'#'*80}")
    
    fold_results = {
        'train_auc': [],
        'test_auc': [],
        'train_acc': [],
        'test_acc': [],
        'train_cm': [],
        'test_cm': []
    }
    
    # Process each fold
    for fold in range(n_folds):
        print(f"\nProcessing Fold {fold}...")
        
        # Load all data using the data_loader function
        data = load_all_data(phenotype, fold, snp_count)
        
        # Check if data loaded successfully
        if data['X_train_combined'] is None or data['y_train'] is None:
            print(f"❌ Failed to load data")
            continue
        
        # Extract data
        X_train_combined = data['X_train_combined']
        X_test_combined = data['X_test_combined']
        X_val_combined = data['X_val_combined']
        y_train = data['y_train']
        y_test = data['y_test']
        feature_info = data['feature_info']
        val_compatible = data['val_compatible']
        
        # Handle missing values
        if pd.isna(X_train_combined).any() or pd.isna(X_test_combined).any():
            train_mean = pd.Series(X_train_combined.flatten()).mean()
            X_train_combined = pd.DataFrame(X_train_combined).fillna(train_mean).values
            X_test_combined = pd.DataFrame(X_test_combined).fillna(train_mean).values
            if X_val_combined is not None:
                X_val_combined = pd.DataFrame(X_val_combined).fillna(train_mean).values
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_combined)
        X_test_scaled = scaler.transform(X_test_combined)
        if X_val_combined is not None and val_compatible:
            X_val_scaled = scaler.transform(X_val_combined)
        else:
            X_val_scaled = None
        
        data_dir = os.path.join(phenotype, f"Fold_{fold}", f"Top_{snp_count}_SNPs")
        os.makedirs(data_dir, exist_ok=True)
        
        # Calculate class distribution
        n_cases = np.sum(y_train == 1)
        n_controls = np.sum(y_train == 0)
        
        print(f"  Class distribution - Cases: {n_cases}, Controls: {n_controls}")
        
        # Store predictions from each model
        train_preds = []
        test_preds = []
        val_preds = []
        model_aucs = []
        model_names = []
        
        # Train AutoKeras
        if AUTOKERAS_AVAILABLE:
            try:
                train_p, test_p, auc = train_autokeras(X_train_scaled, y_train, X_test_scaled, y_test, data_dir)
                train_preds.append(train_p)
                test_preds.append(test_p)
                model_aucs.append(auc)
                model_names.append('AutoKeras')
                
                # Validation prediction
                if X_val_scaled is not None:
                    try:
                        # Try to load the saved model
                        import tensorflow as tf
                        from tensorflow import keras
                        import joblib
                        
                        model_path = os.path.join(data_dir, 'autokeras_model')
                        pkl_path = os.path.join(data_dir, 'autokeras_model.pkl')
                        
                        if os.path.exists(pkl_path):
                            clf = joblib.load(pkl_path)
                            val_pred = clf.predict(X_val_scaled, verbose=0)
                        elif os.path.exists(model_path):
                            model = keras.models.load_model(model_path)
                            val_pred = model.predict(X_val_scaled, verbose=0)
                        else:
                            print("  Warning: Could not load AutoKeras model for validation")
                            val_preds.append(None)
                            continue
                        
                        if len(val_pred.shape) > 1 and val_pred.shape[1] > 1:
                            val_p = val_pred[:, 1]
                        else:
                            val_p = val_pred.flatten()
                        
                        val_preds.append(val_p)
                    except Exception as e:
                        print(f"  Warning: Could not generate validation predictions: {e}")
                        val_preds.append(None)
                else:
                    val_preds.append(None)
            except Exception as e:
                print(f"  Error training AutoKeras: {e}")
                print("  Continuing with other models...")

        # Train Auto-sklearn
        if AUTOSKLEARN_AVAILABLE:
            try:
                train_p, test_p, auc = train_autosklearn(X_train_scaled, y_train, X_test_scaled, y_test, data_dir)
                train_preds.append(train_p)
                test_preds.append(test_p)
                model_aucs.append(auc)
                model_names.append('Auto-sklearn')
                
                # Validation prediction
                if X_val_scaled is not None:
                    with open(os.path.join(data_dir, 'autosklearn_model.pkl'), 'rb') as f:
                        clf = pickle.load(f)
                    val_p = clf.predict_proba(X_val_scaled)[:, 1]
                    val_preds.append(val_p)
                else:
                    val_preds.append(None)
            except Exception as e:
                print(f"Error training Auto-sklearn: {e}")
        
        # Train FLAML
        if FLAML_AVAILABLE:
            try:
                train_p, test_p, auc = train_flaml(X_train_scaled, y_train, X_test_scaled, y_test, data_dir)
                train_preds.append(train_p)
                test_preds.append(test_p)
                model_aucs.append(auc)
                model_names.append('FLAML')
                
                # Validation prediction
                if X_val_scaled is not None:
                    with open(os.path.join(data_dir, 'flaml_model.pkl'), 'rb') as f:
                        clf = pickle.load(f)
                    val_p = clf.predict_proba(X_val_scaled)[:, 1]
                    val_preds.append(val_p)
                else:
                    val_preds.append(None)
            except Exception as e:
                print(f"Error training FLAML: {e}")
        
        # Train H2O AutoML
        if H2O_AVAILABLE:
            try:
                train_p, test_p, auc = train_h2o(X_train_scaled, y_train, X_test_scaled, y_test, data_dir)
                train_preds.append(train_p)
                test_preds.append(test_p)
                model_aucs.append(auc)
                model_names.append('H2O')
                
                # Validation prediction
                if X_val_scaled is not None:
                    val_df = pd.DataFrame(X_val_scaled)
                    val_h2o = h2o.H2OFrame(val_df)
                    # Load saved model
                    h2o_models = [f for f in os.listdir(data_dir) if any(x in f for x in ['GLM', 'GBM', 'DRF', 'XGBoost', 'DeepLearning', 'StackedEnsemble'])]
                    if h2o_models:
                        saved_model = h2o.load_model(os.path.join(data_dir, h2o_models[0]))
                        val_pred = saved_model.predict(val_h2o)
                        val_p = val_pred['p1'].as_data_frame().values.flatten()
                        val_preds.append(val_p)
                    else:
                        val_preds.append(None)
                else:
                    val_preds.append(None)
            except Exception as e:
                print(f"Error training H2O: {e}")
        
        if len(train_preds) == 0:
            print("❌ No models trained successfully")
            continue
        
        # ============================================================
        # ENSEMBLE: WEIGHTED AVERAGE BASED ON AUC
        # ============================================================
        print(f"\n  Creating AutoML Ensemble...", end=' ')
        
        # Normalize AUC scores to create weights
        model_aucs = np.array(model_aucs)
        ensemble_weights = model_aucs / model_aucs.sum()
        
        # Calculate ensemble predictions
        y_train_pred_proba = np.average(train_preds, axis=0, weights=ensemble_weights)
        y_test_pred_proba = np.average(test_preds, axis=0, weights=ensemble_weights)
        if X_val_scaled is not None:
            val_preds_filtered = [p for p in val_preds if p is not None]
            if len(val_preds_filtered) > 0:
                y_val_pred_proba = np.average(val_preds_filtered, axis=0, weights=ensemble_weights)
            else:
                y_val_pred_proba = None
        else:
            y_val_pred_proba = None
        
        y_train_pred = (y_train_pred_proba >= 0.5).astype(int)
        y_test_pred = (y_test_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        train_auc = roc_auc_score(y_train, y_train_pred_proba)
        test_auc = roc_auc_score(y_test, y_test_pred_proba)
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        train_cm = confusion_matrix(y_train, y_train_pred)
        test_cm = confusion_matrix(y_test, y_test_pred)
        
        # Calculate ROC curve data for threshold analysis
        from sklearn.metrics import roc_curve
        train_fpr, train_tpr, train_thresholds = roc_curve(y_train, y_train_pred_proba)
        test_fpr, test_tpr, test_thresholds = roc_curve(y_test, y_test_pred_proba)
        
        train_metrics = {
            'auc': train_auc,
            'y_true': y_train,
            'y_scores': y_train_pred_proba,
            'fpr': train_fpr,
            'tpr': train_tpr,
            'thresholds': train_thresholds
        }
        
        test_metrics = {
            'auc': test_auc,
            'y_true': y_test,
            'y_scores': y_test_pred_proba,
            'fpr': test_fpr,
            'tpr': test_tpr,
            'thresholds': test_thresholds
        }
        
        # ESTIMATE VALIDATION AUC USING MONTE CARLO AND THRESHOLD METHODS
        val_auc_estimates = None
        threshold_estimates = None
        
        if y_val_pred_proba is not None and len(y_val_pred_proba) > 0:
            # Monte Carlo estimation
            val_auc_estimates = estimate_validation_auc_range(
                y_val_pred_proba,
                n_cases=2500,
                n_total=50000,
                train_auc=train_auc,
                test_auc=test_auc,
                n_simulations=10000
            )
            
            # Threshold-based estimation
            threshold_estimates = estimate_validation_performance_by_threshold(
                train_metrics,
                test_metrics,
                y_val_pred_proba,
                n_cases=2500,
                n_total=50000
            )
            
            print(f"\n{'='*80}")
            print("📊 SUMMARY OF VALIDATION ESTIMATES")
            print(f"{'='*80}")
            print(f"  Monte Carlo Expected AUC: {val_auc_estimates['expected_auc']:.6f}")
            print(f"  Threshold-based Est AUC:  {threshold_estimates['expected_auc']:.6f}")
            print(f"  Random Assignment AUC:    {val_auc_estimates['random_auc_mean']:.6f} ± {val_auc_estimates['random_auc_std']:.6f}")
            print(f"  Theoretical Range:        [{val_auc_estimates['worst_case_auc']:.6f}, {val_auc_estimates['best_case_auc']:.6f}]")
            print(f"{'='*80}\n")
        
        print(f"Ensemble AUC: {test_auc:.4f}")
        print(f"  Model weights:")
        for i, model_name in enumerate(model_names):
            print(f"    {model_name}: {ensemble_weights[i]:.3f}")
        
        # Print validation set predictions distribution
        if y_val_pred_proba is not None and len(y_val_pred_proba) > 0:
            val_pred_above_threshold = np.sum(y_val_pred_proba >= 0.5)
            val_pred_below_threshold = np.sum(y_val_pred_proba < 0.5)
            print(f"\n  📊 Validation Set Predictions (n={len(y_val_pred_proba)}):")
            print(f"     • Predicted as Cases (≥0.5): {val_pred_above_threshold} ({val_pred_above_threshold/len(y_val_pred_proba)*100:.1f}%)")
            print(f"     • Predicted as Controls (<0.5): {val_pred_below_threshold} ({val_pred_below_threshold/len(y_val_pred_proba)*100:.1f}%)")
        else:
            val_pred_above_threshold = 0
            val_pred_below_threshold = 0
        
        # Calculate distribution similarities
        bins = np.linspace(0, 1, 50)
        
        train_hist, _ = np.histogram(y_train_pred_proba, bins=bins, density=True)
        train_hist = train_hist / train_hist.sum()
        
        test_hist, _ = np.histogram(y_test_pred_proba, bins=bins, density=True)
        test_hist = test_hist / test_hist.sum()
        
        # Train-Test similarity
        js_train_test = 1 - jensenshannon(train_hist, test_hist)
        ks_train_test = ks_2samp(y_train_pred_proba, y_test_pred_proba)
        
        # Train-Val and Test-Val similarity
        if y_val_pred_proba is not None and len(y_val_pred_proba) > 0:
            val_hist, _ = np.histogram(y_val_pred_proba, bins=bins, density=True)
            val_hist = val_hist / val_hist.sum()
            
            js_train_val = 1 - jensenshannon(train_hist, val_hist)
            ks_train_val = ks_2samp(y_train_pred_proba, y_val_pred_proba)
            
            js_test_val = 1 - jensenshannon(test_hist, val_hist)
            ks_test_val = ks_2samp(y_test_pred_proba, y_val_pred_proba)
        else:
            js_train_val = np.nan
            ks_train_val = (np.nan, np.nan)
            js_test_val = np.nan
            ks_test_val = (np.nan, np.nan)
        
        # Save comprehensive results to CSV
        results_data = {
            'Phenotype': phenotype,
            'SNP_Count': snp_count,
            'Fold': fold,
            'Feature_Info': feature_info,
            
            # Model Performance
            'Train_AUC': train_auc,
            'Test_AUC': test_auc,
            'Train_Accuracy': train_acc,
            'Test_Accuracy': test_acc,
            'AUC_Difference': train_auc - test_auc,
            
            # Confusion Matrix - Train
            'Train_TN': train_cm[0, 0],
            'Train_FP': train_cm[0, 1],
            'Train_FN': train_cm[1, 0],
            'Train_TP': train_cm[1, 1],
            
            # Confusion Matrix - Test
            'Test_TN': test_cm[0, 0],
            'Test_FP': test_cm[0, 1],
            'Test_FN': test_cm[1, 0],
            'Test_TP': test_cm[1, 1],
            
            # Derived Metrics - Train
            'Train_Sensitivity': train_cm[1, 1] / (train_cm[1, 1] + train_cm[1, 0]) if (train_cm[1, 1] + train_cm[1, 0]) > 0 else 0,
            'Train_Specificity': train_cm[0, 0] / (train_cm[0, 0] + train_cm[0, 1]) if (train_cm[0, 0] + train_cm[0, 1]) > 0 else 0,
            'Train_Precision': train_cm[1, 1] / (train_cm[1, 1] + train_cm[0, 1]) if (train_cm[1, 1] + train_cm[0, 1]) > 0 else 0,
            
            # Derived Metrics - Test
            'Test_Sensitivity': test_cm[1, 1] / (test_cm[1, 1] + test_cm[1, 0]) if (test_cm[1, 1] + test_cm[1, 0]) > 0 else 0,
            'Test_Specificity': test_cm[0, 0] / (test_cm[0, 0] + test_cm[0, 1]) if (test_cm[0, 0] + test_cm[0, 1]) > 0 else 0,
            'Test_Precision': test_cm[1, 1] / (test_cm[1, 1] + test_cm[0, 1]) if (test_cm[1, 1] + test_cm[0, 1]) > 0 else 0,
            
            # Validation AUC Estimates
            'Val_Expected_AUC_MonteCarlo': val_auc_estimates['expected_auc'] if val_auc_estimates else np.nan,
            'Val_Expected_AUC_Threshold': threshold_estimates['expected_auc'] if threshold_estimates else np.nan,
            'Val_Random_AUC_Mean': val_auc_estimates['random_auc_mean'] if val_auc_estimates else np.nan,
            'Val_Random_AUC_Std': val_auc_estimates['random_auc_std'] if val_auc_estimates else np.nan,
            'Val_Best_Case_AUC': val_auc_estimates['best_case_auc'] if val_auc_estimates else np.nan,
            'Val_Worst_Case_AUC': val_auc_estimates['worst_case_auc'] if val_auc_estimates else np.nan,
            'Val_Threshold_Based_TPR': threshold_estimates['expected_tpr'] if threshold_estimates else np.nan,
            'Val_Threshold_Based_FPR': threshold_estimates['expected_fpr'] if threshold_estimates else np.nan,
            
            # Validation Set Info
            'Val_Total': len(y_val_pred_proba) if y_val_pred_proba is not None else 0,
            'Val_Pred_Cases': val_pred_above_threshold,
            'Val_Pred_Controls': val_pred_below_threshold,
            'Val_Pred_Cases_Pct': (val_pred_above_threshold / len(y_val_pred_proba) * 100) if y_val_pred_proba is not None and len(y_val_pred_proba) > 0 else 0,
            'Val_Pred_Controls_Pct': (val_pred_below_threshold / len(y_val_pred_proba) * 100) if y_val_pred_proba is not None and len(y_val_pred_proba) > 0 else 0,
            
            # Distribution Similarities
            'JS_Similarity_Train_Test': js_train_test,
            'KS_Statistic_Train_Test': ks_train_test.statistic,
            'KS_Pvalue_Train_Test': ks_train_test.pvalue,
            
            'JS_Similarity_Train_Val': js_train_val,
            'KS_Statistic_Train_Val': ks_train_val[0] if not np.isnan(ks_train_val[0]) else np.nan,
            'KS_Pvalue_Train_Val': ks_train_val[1] if not np.isnan(ks_train_val[1]) else np.nan,
            
            'JS_Similarity_Test_Val': js_test_val,
            'KS_Statistic_Test_Val': ks_test_val[0] if not np.isnan(ks_test_val[0]) else np.nan,
            'KS_Pvalue_Test_Val': ks_test_val[1] if not np.isnan(ks_test_val[1]) else np.nan,
            
            # Models Used
            'Models_Used': ','.join(model_names),
            'Num_Models': len(model_names),
        }
        
        # Add individual model weights dynamically
        for i, model_name in enumerate(model_names):
            results_data[f'Weight_{model_name}'] = ensemble_weights[i]
            results_data[f'AUC_{model_name}'] = model_aucs[i]
        
        # Save to CSV
        results_df = pd.DataFrame([results_data])
        results_file = os.path.join(data_dir, "automl_ensemble_results.csv")
        results_df.to_csv(results_file, index=False)
        print(f"  💾 Results saved: {results_file}")
        
        # Also save predictions
        predictions_data = {
            'Train_Predictions': y_train_pred_proba,
            'Train_Labels': y_train,
            'Test_Predictions': y_test_pred_proba,
            'Test_Labels': y_test,
        }
        
        if y_val_pred_proba is not None and len(y_val_pred_proba) > 0:
            predictions_data['Val_Predictions'] = y_val_pred_proba
        predictions_file = os.path.join(data_dir, "automl_predictions.pkl")
        with open(predictions_file, 'wb') as f:
            pickle.dump(predictions_data, f)
        print(f"  💾 Predictions saved: {predictions_file}")
        
        # Generate submission file for this fold
        if y_val_pred_proba is not None and len(y_val_pred_proba) > 0:
            val_ids_file = os.path.join(phenotype, f"Fold_{fold}", "validation_ids.txt")
            
            if os.path.exists(val_ids_file):
                val_ids_df = pd.read_csv(val_ids_file, sep='\t')
                
                # Create submission dataframe
                submission_df = pd.DataFrame({
                    'FID': val_ids_df['FID'].values,
                    'IID': val_ids_df['IID'].values,
                    'PRS': y_val_pred_proba
                })
                
                # Save fold-specific submission file
                submission_file = os.path.join(data_dir, f"submission_fold_{fold}.txt")
                submission_df.to_csv(submission_file, sep='\t', index=False, float_format='%.10f')
                print(f"  📤 Submission file saved: {submission_file}")
            else:
                print(f"  ⚠️ Warning: Validation IDs file not found: {val_ids_file}")
        
        # Store results
        fold_results['train_auc'].append(train_auc)
        fold_results['test_auc'].append(test_auc)
        fold_results['train_acc'].append(train_acc)
        fold_results['test_acc'].append(test_acc)
        fold_results['train_cm'].append(train_cm)
        fold_results['test_cm'].append(test_cm)
        
        # Check if this is the best performing fold
        if test_auc > best_dataset_info['test_auc']:
            best_dataset_info['snp_count'] = snp_count
            best_dataset_info['fold'] = fold
            best_dataset_info['test_auc'] = test_auc
            best_dataset_info['train_pred_proba'] = y_train_pred_proba
            best_dataset_info['test_pred_proba'] = y_test_pred_proba
            best_dataset_info['val_pred_proba'] = y_val_pred_proba
            best_dataset_info['y_train'] = y_train
            best_dataset_info['y_test'] = y_test
            
            # SAVE THE VALIDATION IDs FOR BEST FOLD
            val_ids_file = os.path.join(phenotype, f"Fold_{fold}", "validation_ids.txt")
            if os.path.exists(val_ids_file):
                best_dataset_info['val_ids_df'] = pd.read_csv(val_ids_file, sep='\t')
            else:
                best_dataset_info['val_ids_df'] = None
        
        # Save scaler and ensemble weights
        with open(os.path.join(data_dir, "scaler.pkl"), 'wb') as f:
            pickle.dump(scaler, f)
        with open(os.path.join(data_dir, "ensemble_weights.pkl"), 'wb') as f:
            pickle.dump({'weights': ensemble_weights, 'model_names': model_names}, f)
        
        print(f"\n  ✓ {feature_info} Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f} (Diff: {train_auc-test_auc:.4f})")
        
        # Plot distributions for this fold using KDE curves
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot Train distribution
        train_cases = y_train_pred_proba[y_train == 1]
        train_controls = y_train_pred_proba[y_train == 0]
        
        if len(train_cases) > 0 and len(train_controls) > 0:
            pd.Series(train_cases).plot.kde(ax=axes[0], color='red', linewidth=2, 
                                           label=f'Cases (n={len(train_cases)})')
            pd.Series(train_controls).plot.kde(ax=axes[0], color='blue', linewidth=2, 
                                              label=f'Controls (n={len(train_controls)})')
        
        axes[0].set_xlabel('Predicted Probability', fontweight='bold')
        axes[0].set_ylabel('Density', fontweight='bold')
        axes[0].set_title(f'Training Set (AUC: {train_auc:.4f})', fontweight='bold', fontsize=12)
        axes[0].legend(frameon=True, shadow=True)
        axes[0].grid(alpha=0.3)
        
        # Plot Test distribution
        test_cases = y_test_pred_proba[y_test == 1]
        test_controls = y_test_pred_proba[y_test == 0]
        
        if len(test_cases) > 0 and len(test_controls) > 0:
            pd.Series(test_cases).plot.kde(ax=axes[1], color='red', linewidth=2, 
                                          label=f'Cases (n={len(test_cases)})')
            pd.Series(test_controls).plot.kde(ax=axes[1], color='blue', linewidth=2, 
                                             label=f'Controls (n={len(test_controls)})')
        
        axes[1].set_xlabel('Predicted Probability', fontweight='bold')
        axes[1].set_ylabel('Density', fontweight='bold')
        axes[1].set_title(f'Test Set (AUC: {test_auc:.4f})', fontweight='bold', fontsize=12)
        axes[1].legend(frameon=True, shadow=True)
        axes[1].grid(alpha=0.3)
        
        # Plot Validation distribution
        if y_val_pred_proba is not None and len(y_val_pred_proba) > 0:
            pd.Series(y_val_pred_proba).plot.kde(ax=axes[2], color='green', linewidth=2,
                                                 label=f'Validation (n={len(y_val_pred_proba)})')
            axes[2].set_xlabel('Predicted Probability', fontweight='bold')
            axes[2].set_ylabel('Density', fontweight='bold')
            axes[2].set_title('Validation Set', fontweight='bold', fontsize=12)
            axes[2].legend(frameon=True, shadow=True)
            axes[2].grid(alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, 'No Validation Data', ha='center', va='center', fontsize=14)
            axes[2].set_xlabel('Predicted Probability', fontweight='bold')
            axes[2].set_ylabel('Density', fontweight='bold')
            axes[2].set_title('Validation Set', fontweight='bold', fontsize=12)
        
        plt.suptitle(f'{phenotype} - AutoML Ensemble Predictions\n' + 
                     f'Top {snp_count} SNPs, Fold {fold}, Features: {feature_info}',
                     fontweight='bold', fontsize=14)
        plt.tight_layout()
        
        output_file = os.path.join(data_dir, f"AutoML_Ensemble_Distributions_Fold_{fold}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  📊 Distribution plot saved: {output_file}")
        plt.close()
    
    # Store dataset results
    all_dataset_results[snp_count] = fold_results
    
    # Save aggregated results across all folds
    if len(fold_results['train_auc']) > 0:
        aggregated_results = []
        
        for fold_idx in range(len(fold_results['train_auc'])):
            # Read individual fold results
            fold_dir = os.path.join(phenotype, f"Fold_{fold_idx}", f"Top_{snp_count}_SNPs")
            fold_results_file = os.path.join(fold_dir, "automl_ensemble_results.csv")
            
            if os.path.exists(fold_results_file):
                fold_df = pd.read_csv(fold_results_file)
                aggregated_results.append(fold_df)
        
        if aggregated_results:
            # Combine all folds
            combined_df = pd.concat(aggregated_results, ignore_index=True)
            
            # Save combined results
            combined_file = os.path.join(phenotype, f"AutoML_All_Folds_Top_{snp_count}_SNPs_Results.csv")
            combined_df.to_csv(combined_file, index=False)
            print(f"\n  💾 Combined AutoML results saved: {combined_file}")
            
            # Calculate and save summary statistics
            summary_stats = combined_df.describe()
            summary_file = os.path.join(phenotype, f"AutoML_Summary_Stats_Top_{snp_count}_SNPs.csv")
            summary_stats.to_csv(summary_file)
            print(f"  💾 AutoML summary statistics saved: {summary_file}")
    
    # Calculate averages
    if len(fold_results['train_auc']) > 0:
        avg_train_auc = pd.Series(fold_results['train_auc']).mean()
        avg_test_auc = pd.Series(fold_results['test_auc']).mean()
        std_train_auc = pd.Series(fold_results['train_auc']).std()
        std_test_auc = pd.Series(fold_results['test_auc']).std()
        
        avg_train_acc = pd.Series(fold_results['train_acc']).mean()
        avg_test_acc = pd.Series(fold_results['test_acc']).mean()
        std_train_acc = pd.Series(fold_results['train_acc']).std()
        std_test_acc = pd.Series(fold_results['test_acc']).std()
        
        # Fix confusion matrix averaging
        avg_train_cm = np.mean(np.array(fold_results['train_cm']), axis=0)
        avg_test_cm = np.mean(np.array(fold_results['test_cm']), axis=0)
        
        train_tn, train_fp, train_fn, train_tp = avg_train_cm.ravel()
        test_tn, test_fp, test_fn, test_tp = avg_test_cm.ravel()
        
        train_sensitivity = train_tp / (train_tp + train_fn) if (train_tp + train_fn) > 0 else 0
        train_specificity = train_tn / (train_tn + train_fp) if (train_tn + train_fp) > 0 else 0
        train_precision = train_tp / (train_tp + train_fp) if (train_tp + train_fp) > 0 else 0
        
        test_sensitivity = test_tp / (test_tp + test_fn) if (test_tp + test_fn) > 0 else 0
        test_specificity = test_tn / (test_tn + test_fp) if (test_tn + test_fp) > 0 else 0
        test_precision = test_tp / (test_tp + test_fp) if (test_tp + test_fp) > 0 else 0
        
        overfit_gap = avg_train_auc - avg_test_auc;
        
        # Print averaged results
        print(f"\n{'='*80}")
        print(f"AVERAGED RESULTS ACROSS {len(fold_results['train_auc'])} FOLDS - TOP {snp_count} SNPs")
        print(f"{'='*80}")
        
        print(f"\n📊 AVERAGE PERFORMANCE METRICS:")
        print(f"  {'Metric':<20} {'Train':<20} {'Test':<20}")
        print(f"  {'-'*60}")
        print(f"  {'AUC':<20} {avg_train_auc:.4f} ± {std_train_auc:.4f}     {avg_test_auc:.4f} ± {std_test_auc:.4f}")
        print(f"  {'Accuracy':<20} {avg_train_acc:.4f} ± {std_train_acc:.4f}     {avg_test_acc:.4f} ± {std_test_acc:.4f}")
        print(f"  {'Sensitivity':<20} {train_sensitivity:.4f}               {test_sensitivity:.4f}")
        print(f"  {'Specificity':<20} {train_specificity:.4f}               {test_specificity:.4f}")
        print(f"  {'Precision':<20} {train_precision:.4f}               {test_precision:.4f}")
        print(f"\n  {'Overfitting Gap':<20} {overfit_gap:.4f} (Train AUC - Test AUC)")
        
        print(f"\n📋 AVERAGE CONFUSION MATRICES:")
        print(f"\n  TRAIN SET (Averaged):")
        print(f"                  Predicted")
        print(f"                  Control    Case")
        print(f"  Actual Control  {train_tn:7.1f}    {train_fp:7.1f}")
        print(f"         Case     {train_fn:7.1f}    {train_tp:7.1f}")
        
        print(f"\n  TEST SET (Averaged):")
        print(f"                  Predicted")
        print(f"                  Control    Case")
        print(f"  Actual Control  {test_tn:7.1f}    {test_fp:7.1f}")
        print(f"         Case     {test_fn:7.1f}    {test_tp:7.1f}")

# Plot distributions for best performing dataset
print(f"\n{'='*80}")
print("PLOTTING DISTRIBUTIONS FOR BEST PERFORMING DATASET")
print(f"{'='*80}")

if best_dataset_info['snp_count'] is not None:
    print(f"\nBest Dataset: Top {best_dataset_info['snp_count']} SNPs, Fold {best_dataset_info['fold']}")
    print(f"Test AUC: {best_dataset_info['test_auc']:.4f}")
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot Train distribution
    train_cases = best_dataset_info['train_pred_proba'][best_dataset_info['y_train'] == 1]
    train_controls = best_dataset_info['train_pred_proba'][best_dataset_info['y_train'] == 0]
    
    if len(train_cases) > 0 and len(train_controls) > 0:
        pd.Series(train_cases).plot.kde(ax=axes[0], color='red', linewidth=2, 
                                       label=f'Cases (n={len(train_cases)})')
        pd.Series(train_controls).plot.kde(ax=axes[0], color='blue', linewidth=2, 
                                          label=f'Controls (n={len(train_controls)})')
    
    axes[0].set_xlabel('Predicted Probability', fontweight='bold')
    axes[0].set_ylabel('Density', fontweight='bold')
    axes[0].set_title('Training Set', fontweight='bold', fontsize=12)
    axes[0].legend(frameon=True, shadow=True)
    axes[0].grid(alpha=0.3)
    
    # Plot Test distribution
    test_cases = best_dataset_info['test_pred_proba'][best_dataset_info['y_test'] == 1]
    test_controls = best_dataset_info['test_pred_proba'][best_dataset_info['y_test'] == 0]
    
    if len(test_cases) > 0 and len(test_controls) > 0:
        pd.Series(test_cases).plot.kde(ax=axes[1], color='red', linewidth=2, 
                                      label=f'Cases (n={len(test_cases)})')
        pd.Series(test_controls).plot.kde(ax=axes[1], color='blue', linewidth=2, 
                                         label=f'Controls (n={len(test_controls)})')
    
    axes[1].set_xlabel('Predicted Probability', fontweight='bold')
    axes[1].set_ylabel('Density', fontweight='bold')
    axes[1].set_title('Test Set', fontweight='bold', fontsize=12)
    axes[1].legend(frameon=True, shadow=True)
    axes[1].grid(alpha=0.3)
    
    # Plot Validation distribution
    if best_dataset_info['val_pred_proba'] is not None and len(best_dataset_info['val_pred_proba']) > 0:
        pd.Series(best_dataset_info['val_pred_proba']).plot.kde(ax=axes[2], color='green', linewidth=2,
                                                                 label=f'Validation (n={len(best_dataset_info["val_pred_proba"])})')
        axes[2].set_xlabel('Predicted Probability', fontweight='bold')
        axes[2].set_ylabel('Density', fontweight='bold')
        axes[2].set_title('Validation Set', fontweight='bold', fontsize=12)
        axes[2].legend(frameon=True, shadow=True)
        axes[2].grid(alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'No Validation Data', ha='center', va='center', fontsize=14)
        axes[2].set_xlabel('Predicted Probability', fontweight='bold')
        axes[2].set_ylabel('Density', fontweight='bold')
        axes[2].set_title('Validation Set', fontweight='bold', fontsize=12)
    
    plt.suptitle(f'{phenotype} - AutoML Ensemble Predictions (BEST)\n' + 
                 f'Top {best_dataset_info["snp_count"]} SNPs, Fold {best_dataset_info["fold"]}, ' + 
                 f'Test AUC: {best_dataset_info["test_auc"]:.4f}',
                 fontweight='bold', fontsize=14)
    plt.tight_layout()
    
    output_file = f"AutoML_Ensemble_Distributions_Best_{phenotype}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Distribution plot saved: {output_file}")
    plt.close()

# Generate final consolidated submission files (BEST FOLD ONLY)
print(f"\n{'='*80}")
print("GENERATING FINAL SUBMISSION FILES (BEST FOLD ONLY)")
print(f"{'='*80}")

for snp_count in top_snp_counts:
    print(f"\nProcessing submission for Top {snp_count} SNPs...")
    
    # Use BEST fold submission instead of averaging
    if best_dataset_info['snp_count'] == snp_count and best_dataset_info['val_pred_proba'] is not None:
        best_fold = best_dataset_info['fold']
        best_test_auc = best_dataset_info['test_auc']
        
        print(f"  Using BEST fold: Fold {best_fold} (Test AUC: {best_test_auc:.6f})")
        
        # Create submission from best fold
        if best_dataset_info['val_ids_df'] is not None:
            submission_df = pd.DataFrame({
                'FID': best_dataset_info['val_ids_df']['FID'].values,
                'IID': best_dataset_info['val_ids_df']['IID'].values,
                'PRS': best_dataset_info['val_pred_proba']
            })
            
            # Save best fold submission
            submission_dir = os.path.join(phenotype, "Submission4")
            os.makedirs(submission_dir, exist_ok=True)
            
            final_submission_file = os.path.join(submission_dir, f"{phenotype}_submission4.txt")
            submission_df.to_csv(final_submission_file, sep='\t', index=False, float_format='%.10f')
            print(f"  ✅ Final submission saved: {final_submission_file}")
            print(f"     (Best Fold {best_fold}, Test AUC: {best_test_auc:.6f})")
            
            # Print prediction statistics
            if 'PRS' in submission_df.columns:
                n_pred_cases = int((submission_df['PRS'] >= 0.5).sum())
                n_pred_controls = int((submission_df['PRS'] < 0.5).sum())
                print(f"  🔢 Predicted (prob>=0.5) -> Cases: {n_pred_cases}, Controls: {n_pred_controls}")
            
            print(f"     Total individuals: {len(submission_df)}")
            print(f"     PRS range: [{submission_df['PRS'].min():.4f}, {submission_df['PRS'].max():.4f}]")
            
            # Generate submission statistics
            stats_data = {
                'Phenotype': phenotype,
                'SNP_Count': snp_count,
                'Best_Fold': best_fold,
                'Best_Test_AUC': best_test_auc,
                'Total_Individuals': len(submission_df),
                'Mean_PRS': submission_df['PRS'].mean(),
                'Std_PRS': submission_df['PRS'].std(),
                'Min_PRS': submission_df['PRS'].min(),
                'Max_PRS': submission_df['PRS'].max(),
                'Median_PRS': submission_df['PRS'].median(),
                'Q25_PRS': submission_df['PRS'].quantile(0.25),
                'Q75_PRS': submission_df['PRS'].quantile(0.75),
                'N_Predicted_Cases': n_pred_cases,
                'N_Predicted_Controls': n_pred_controls
            }
            
            stats_df = pd.DataFrame([stats_data])
            stats_file = os.path.join(submission_dir, f"{phenotype}_submission_stats.csv")
            stats_df.to_csv(stats_file, index=False)
            print(f"  📊 Submission statistics saved: {stats_file}")
            
            # Create a README for the submission
            readme_file = os.path.join(submission_dir, "README.txt")
            with open(readme_file, 'w') as f:
                f.write(f"CAGI7 PRS Challenge - AutoML Submission Files\n")
                f.write(f"{'='*60}\n\n")
                f.write(f"Phenotype: {phenotype}\n")
                f.write(f"Method: AutoML Ensemble\n")
                if len(available_models) > 0:
                    f.write(f"Models Used:\n")
                    for model in available_models:
                        f.write(f"  - {model}\n")
                f.write(f"\nFeatures: SNPs + PCA + PRS\n")
                f.write(f"Number of SNPs: {snp_count}\n")
                f.write(f"Cross-validation: {n_folds}-fold\n\n")
                f.write(f"BEST FOLD SELECTION:\n")
                f.write(f"  Selected Fold: {best_fold}\n")
                f.write(f"  Test AUC: {best_test_auc:.6f}\n")
                f.write(f"  Reason: Highest test AUC across all folds\n\n")
                f.write(f"Submission Format:\n")
                f.write(f"  - Tab-delimited text file\n")
                f.write(f"  - Columns: FID, IID, PRS\n")
                f.write(f"  - PRS: Predicted risk probability (higher = higher risk)\n")
                f.write(f"  - Total samples: {len(submission_df):,}\n\n")
                f.write(f"PRS Statistics:\n")
                f.write(f"  - Mean:   {submission_df['PRS'].mean():.6f}\n")
                f.write(f"  - Median: {submission_df['PRS'].median():.6f}\n")
                f.write(f"  - Std:    {submission_df['PRS'].std():.6f}\n")
                f.write(f"  - Range:  [{submission_df['PRS'].min():.6f}, {submission_df['PRS'].max():.6f}]\n\n")
                f.write(f"Prediction Summary:\n")
                f.write(f"  - Predicted Cases (≥0.5):     {n_pred_cases:,} ({n_pred_cases/len(submission_df)*100:.2f}%)\n")
                f.write(f"  - Predicted Controls (<0.5):  {n_pred_controls:,} ({n_pred_controls/len(submission_df)*100:.2f}%)\n\n")
                f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            print(f"  📝 README saved: {readme_file}")
            
            # ALSO save reference file with all folds for comparison
            all_fold_submissions = []
            for fold in range(n_folds):
                fold_dir = os.path.join(phenotype, f"Fold_{fold}", f"Top_{snp_count}_SNPs")
                submission_file = os.path.join(fold_dir, f"submission_fold_{fold}.txt")
                
                if os.path.exists(submission_file):
                    fold_sub = pd.read_csv(submission_file, sep='\t')
                    fold_sub['Fold'] = fold
                    
                    # Read fold results to get AUC
                    results_file = os.path.join(fold_dir, "automl_ensemble_results.csv")
                    if os.path.exists(results_file):
                        results_df = pd.read_csv(results_file)
                        if len(results_df) > 0:
                            fold_sub['Test_AUC'] = results_df['Test_AUC'].values[0]
                    
                    all_fold_submissions.append(fold_sub)
            
            if all_fold_submissions:
                combined_submissions = pd.concat(all_fold_submissions, ignore_index=True)
                all_folds_file = os.path.join(submission_dir, f"{phenotype}_all_folds_reference.txt")
                combined_submissions.to_csv(all_folds_file, sep='\t', index=False, float_format='%.10f')
                print(f"  📋 All folds reference file saved: {all_folds_file}")
        else:
            print(f"  ⚠️ Warning: Validation IDs not found for best fold")
    else:
        print(f"  ⚠️ No best fold data available for Top {snp_count} SNPs")

print(f"\n{'='*80}")
print("✅ SUBMISSION FILES GENERATION COMPLETED")
print(f"{'='*80}")
print(f"\n📁 Submission files are in: {phenotype}/Submission4/")
print(f"\nSubmission Strategy:")
print(f"  ✓ Used BEST performing fold (highest test AUC)")
print(f"  ✓ Best Fold: {best_dataset_info['fold']} (Test AUC: {best_dataset_info['test_auc']:.6f})")
print(f"  ✓ Format: Tab-delimited (FID, IID, PRS)")
print(f"  ✓ PRS = predicted probability (higher = higher risk)")

# Shutdown H2O if it was used
if H2O_AVAILABLE:
    h2o.cluster().shutdown(prompt=False)

print(f"\n{'='*80}")
print("✅ AutoML ENSEMBLE TRAINING COMPLETED")
print(f"{'='*80}")