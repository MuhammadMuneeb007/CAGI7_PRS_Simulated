#!/usr/bin/env python
# coding: utf-8

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
import umap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import data loading functions
from Step6data_loaderMachineLearningModelstesting import load_all_data

# Get phenotype from command line
if len(sys.argv) < 2:
    print("Usage: python train_ml.py <phenotype_name>")
    print("Example: python train_ml.py Phenotype_1")
    sys.exit(1)

phenotype = sys.argv[1]

# Configuration
top_snp_counts = [500, 2000,5000,10000]


n_folds = 5

# ============================================================
# DEEP NEURAL NETWORK ARCHITECTURES
# ============================================================

# Architecture 1: Wide and Deep Network
class WideDeepNN(nn.Module):
    def __init__(self, input_dim):
        super(WideDeepNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.4)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(0.2)
        
        self.fc5 = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout4(x)
        
        x = self.fc5(x)
        return x

# Architecture 2: Residual Network
class ResidualNN(nn.Module):
    def __init__(self, input_dim):
        super(ResidualNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.fc3 = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        
        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        
        self.fc6 = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # First layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Residual block 1
        identity = x
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = x + identity  # Skip connection
        x = self.relu(x)
        x = self.dropout(x)
        
        # Reduction layers
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc6(x)
        return x

# Architecture 3: Deep Narrow Network
class DeepNarrowNN(nn.Module):
    def __init__(self, input_dim):
        super(DeepNarrowNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(0.2)
        
        self.fc5 = nn.Linear(64, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.dropout5 = nn.Dropout(0.2)
        
        self.fc6 = nn.Linear(64, 32)
        self.bn6 = nn.BatchNorm1d(32)
        self.dropout6 = nn.Dropout(0.1)
        
        self.fc7 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout4(x)
        
        x = self.fc5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.dropout5(x)
        
        x = self.fc6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.dropout6(x)
        
        x = self.fc7(x)
        return x

# Architecture 4: Attention-like Network
class AttentionNN(nn.Module):
    def __init__(self, input_dim):
        super(AttentionNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        
        # Attention mechanism
        self.attention = nn.Linear(256, 256)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.fc4 = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Attention weights
        attention_weights = self.sigmoid(self.attention(x))
        x = x * attention_weights
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        return x

# Training parameters
training_params = {
    'batch_size': 64,
    'epochs': 150,
    'learning_rate': 0.0005,
    'weight_decay': 0.0001,
    'early_stopping_patience': 20
}

print("="*80)
print(f"TRAINING DEEP LEARNING ENSEMBLE FOR: {phenotype}")
print(f"Features: SNPs + PCA + PRS")
print(f"Datasets: Top {top_snp_counts} SNPs")
print("="*80)
print(f"\nDeep Learning Ensemble includes:")
print(f"  1. Wide & Deep Network (512->256->128->64->1)")
print(f"  2. Residual Network with Skip Connections")
print(f"  3. Deep Narrow Network (7 layers)")
print(f"  4. Attention-based Network")
print(f"  5. Final: Weighted Average Ensemble")
print("="*80)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

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
    'val_ids_df': None  # ADD THIS LINE
}

def train_deep_model(model, train_loader, X_test_tensor, y_test, optimizer, scheduler, 
                     criterion, device, data_dir, model_name, epochs, patience):
    """Train a deep learning model with early stopping"""
    best_test_auc = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test_tensor)
            test_probs = torch.sigmoid(test_logits)
            test_auc = roc_auc_score(y_test, test_probs.cpu().numpy())
        
        scheduler.step(test_auc)
        
        # Early stopping
        if test_auc > best_test_auc:
            best_test_auc = test_auc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(data_dir, f"{model_name}.pt"))
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(data_dir, f"{model_name}.pt")))
    return model, best_test_auc

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

# ============================================================
# TRAINING LOOP WITH CROSS-VALIDATION
# ============================================================

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
        
        # Calculate class weights for handling imbalance
        n_cases = np.sum(y_train == 1)
        n_controls = np.sum(y_train == 0)
        total = len(y_train)
        weight_case = total / (2 * n_cases)
        weight_control = total / (2 * n_controls)
        
        print(f"  Class distribution - Cases: {n_cases}, Controls: {n_controls}")
        print(f"  Class weights - Case: {weight_case:.2f}, Control: {weight_control:.2f}")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        if X_val_scaled is not None:
            X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
        
        # Create weighted sampler for balanced batches
        sample_weights = np.where(y_train == 1, weight_case, weight_control)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=training_params['batch_size'], 
                                 sampler=sampler)
        
        # Weighted BCE Loss
        pos_weight = torch.tensor([weight_case / weight_control]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        input_dim = X_train_combined.shape[1]
        
        # Store predictions from each model
        train_preds = []
        test_preds = []
        val_preds = []
        model_aucs = []
        
        # ============================================================
        # MODEL 1: WIDE & DEEP NETWORK
        # ============================================================
        print(f"\n  Training Wide & Deep Network...", end=' ')
        model1 = WideDeepNN(input_dim).to(device)
        optimizer1 = optim.AdamW(model1.parameters(), lr=training_params['learning_rate'],
                                weight_decay=training_params['weight_decay'])
        scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='max', 
                                                          factor=0.5, patience=7)
        
        model1, auc1 = train_deep_model(model1, train_loader, X_test_tensor, y_test,
                                        optimizer1, scheduler1, criterion, device, data_dir,
                                        "wide_deep_model", training_params['epochs'],
                                        training_params['early_stopping_patience'])
        
        model1.eval()
        with torch.no_grad():
            train_pred1 = torch.sigmoid(model1(X_train_tensor)).cpu().numpy().flatten()
            test_pred1 = torch.sigmoid(model1(X_test_tensor)).cpu().numpy().flatten()
            if X_val_scaled is not None:
                val_pred1 = torch.sigmoid(model1(X_val_tensor)).cpu().numpy().flatten()
            else:
                val_pred1 = None
        
        train_preds.append(train_pred1)
        test_preds.append(test_pred1)
        val_preds.append(val_pred1)
        model_aucs.append(auc1)
        print(f"AUC: {auc1:.4f}")
        
        # ============================================================
        # MODEL 2: RESIDUAL NETWORK
        # ============================================================
        print(f"  Training Residual Network...", end=' ')
        model2 = ResidualNN(input_dim).to(device)
        optimizer2 = optim.AdamW(model2.parameters(), lr=training_params['learning_rate'],
                                weight_decay=training_params['weight_decay'])
        scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='max', 
                                                          factor=0.5, patience=7)
        
        model2, auc2 = train_deep_model(model2, train_loader, X_test_tensor, y_test,
                                        optimizer2, scheduler2, criterion, device, data_dir,
                                        "residual_model", training_params['epochs'],
                                        training_params['early_stopping_patience'])
        
        model2.eval()
        with torch.no_grad():
            train_pred2 = torch.sigmoid(model2(X_train_tensor)).cpu().numpy().flatten()
            test_pred2 = torch.sigmoid(model2(X_test_tensor)).cpu().numpy().flatten()
            if X_val_scaled is not None:
                val_pred2 = torch.sigmoid(model2(X_val_tensor)).cpu().numpy().flatten()
            else:
                val_pred2 = None
        
        train_preds.append(train_pred2)
        test_preds.append(test_pred2)
        val_preds.append(val_pred2)
        model_aucs.append(auc2)
        print(f"AUC: {auc2:.4f}")
        
        # ============================================================
        # MODEL 3: DEEP NARROW NETWORK
        # ============================================================
        print(f"  Training Deep Narrow Network...", end=' ')
        model3 = DeepNarrowNN(input_dim).to(device)
        optimizer3 = optim.AdamW(model3.parameters(), lr=training_params['learning_rate'],
                                weight_decay=training_params['weight_decay'])
        scheduler3 = optim.lr_scheduler.ReduceLROnPlateau(optimizer3, mode='max', 
                                                          factor=0.5, patience=7)
        
        model3, auc3 = train_deep_model(model3, train_loader, X_test_tensor, y_test,
                                        optimizer3, scheduler3, criterion, device, data_dir,
                                        "deep_narrow_model", training_params['epochs'],
                                        training_params['early_stopping_patience'])
        
        model3.eval()
        with torch.no_grad():
            train_pred3 = torch.sigmoid(model3(X_train_tensor)).cpu().numpy().flatten()
            test_pred3 = torch.sigmoid(model3(X_test_tensor)).cpu().numpy().flatten()
            if X_val_scaled is not None:
                val_pred3 = torch.sigmoid(model3(X_val_tensor)).cpu().numpy().flatten()
            else:
                val_pred3 = None
        
        train_preds.append(train_pred3)
        test_preds.append(test_pred3)
        val_preds.append(val_pred3)
        model_aucs.append(auc3)
        print(f"AUC: {auc3:.4f}")
        
        # ============================================================
        # MODEL 4: ATTENTION NETWORK
        # ============================================================
        print(f"  Training Attention Network...", end=' ')
        model4 = AttentionNN(input_dim).to(device)
        optimizer4 = optim.AdamW(model4.parameters(), lr=training_params['learning_rate'],
                                weight_decay=training_params['weight_decay'])
        scheduler4 = optim.lr_scheduler.ReduceLROnPlateau(optimizer4, mode='max', 
                                                          factor=0.5, patience=7)
        
        model4, auc4 = train_deep_model(model4, train_loader, X_test_tensor, y_test,
                                        optimizer4, scheduler4, criterion, device, data_dir,
                                        "attention_model", training_params['epochs'],
                                        training_params['early_stopping_patience'])
        
        model4.eval()
        with torch.no_grad():
            train_pred4 = torch.sigmoid(model4(X_train_tensor)).cpu().numpy().flatten()
            test_pred4 = torch.sigmoid(model4(X_test_tensor)).cpu().numpy().flatten()
            if X_val_scaled is not None:
                val_pred4 = torch.sigmoid(model4(X_val_tensor)).cpu().numpy().flatten()
            else:
                val_pred4 = None
        
        train_preds.append(train_pred4)
        test_preds.append(test_pred4)
        val_preds.append(val_pred4)
        model_aucs.append(auc4)
        print(f"AUC: {auc4:.4f}")
        
        # ============================================================
        # ENSEMBLE: WEIGHTED AVERAGE BASED ON AUC
        # ============================================================
        print(f"\n  Creating Deep Learning Ensemble...", end=' ')
        
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
        print(f"  Model weights: WideDeep={ensemble_weights[0]:.3f}, Residual={ensemble_weights[1]:.3f}, " +
              f"DeepNarrow={ensemble_weights[2]:.3f}, Attention={ensemble_weights[3]:.3f}")
        
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
        # Create histograms for JS divergence and KS test
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
            
            # Validation Set Info
            'Val_Total': len(y_val_pred_proba) if y_val_pred_proba is not None else 0,
            'Val_Pred_Cases': val_pred_above_threshold,
            'Val_Pred_Controls': val_pred_below_threshold,
            'Val_Pred_Cases_Pct': (val_pred_above_threshold / len(y_val_pred_proba) * 100) if y_val_pred_proba is not None and len(y_val_pred_proba) > 0 else 0,
            'Val_Pred_Controls_Pct': (val_pred_below_threshold / len(y_val_pred_proba) * 100) if y_val_pred_proba is not None and len(y_val_pred_proba) > 0 else 0,
            
            # Validation AUC Estimates
            'Val_Expected_AUC_MonteCarlo': val_auc_estimates['expected_auc'] if val_auc_estimates else np.nan,
            'Val_Expected_AUC_Threshold': threshold_estimates['expected_auc'] if threshold_estimates else np.nan,
            'Val_Random_AUC_Mean': val_auc_estimates['random_auc_mean'] if val_auc_estimates else np.nan,
            'Val_Random_AUC_Std': val_auc_estimates['random_auc_std'] if val_auc_estimates else np.nan,
            'Val_Best_Case_AUC': val_auc_estimates['best_case_auc'] if val_auc_estimates else np.nan,
            'Val_Worst_Case_AUC': val_auc_estimates['worst_case_auc'] if val_auc_estimates else np.nan,
            'Val_Threshold_Based_TPR': threshold_estimates['expected_tpr'] if threshold_estimates else np.nan,
            'Val_Threshold_Based_FPR': threshold_estimates['expected_fpr'] if threshold_estimates else np.nan,
            
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
            
            # Ensemble Weights
            'Weight_WideDeep': ensemble_weights[0],
            'Weight_Residual': ensemble_weights[1],
            'Weight_DeepNarrow': ensemble_weights[2],
            'Weight_Attention': ensemble_weights[3],
            
            # Individual Model AUCs
            'AUC_WideDeep': model_aucs[0],
            'AUC_Residual': model_aucs[1],
            'AUC_DeepNarrow': model_aucs[2],
            'AUC_Attention': model_aucs[3],
        }
        
        # Save to CSV
        results_df = pd.DataFrame([results_data])
        results_file = os.path.join(data_dir, "ensemble_results.csv")
        results_df.to_csv(results_file, index=False)
        print(f"  💾 Results saved: {results_file}")
        
        # Save predictions to CSV
        predictions_data = {
            'Train_Predictions': y_train_pred_proba,
            'Train_Labels': y_train,
            'Test_Predictions': y_test_pred_proba,
            'Test_Labels': y_test,
        }
        
        if y_val_pred_proba is not None and len(y_val_pred_proba) > 0:
            predictions_data['Val_Predictions'] = y_val_pred_proba
            
            # Generate submission file for this fold (validation predictions)
            # Load validation IDs from the data loader
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
                submission_df.to_csv(submission_file, sep='\t', index=False)
                print(f"  📤 Submission file saved: {submission_file}")
            else:
                print(f"  ⚠️ Warning: Validation IDs file not found: {val_ids_file}")
        
        predictions_file = os.path.join(data_dir, "predictions.pkl")
        with open(predictions_file, 'wb') as f:
            pickle.dump(predictions_data, f)
        print(f"  💾 Predictions saved: {predictions_file}")
        
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
            pickle.dump(ensemble_weights, f)
        
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
        
        plt.suptitle(f'{phenotype} - Deep Learning Ensemble Predictions\n' + 
                     f'Top {snp_count} SNPs, Fold {fold}, Features: {feature_info}',
                     fontweight='bold', fontsize=14)
        plt.tight_layout()
        
        output_file = os.path.join(data_dir, f"DL_Ensemble_Distributions_Fold_{fold}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  📊 Distribution plot saved: {output_file}")
        plt.close()
        
        # ============================================================
        # PCA VISUALIZATION WITH DECISION BOUNDARIES
        # ============================================================
        print(f"  📊 Creating PCA visualization with decision boundaries...")
        
        # Apply PCA to reduce to 2D
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        if X_val_scaled is not None:
            X_val_pca = pca.transform(X_val_scaled)
        else:
            X_val_pca = None
        
        # Create meshgrid for decision boundary
        x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
        y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        
        # Get predictions on meshgrid using ensemble
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        
        # Inverse transform PCA to get back to original feature space
        mesh_points_original = pca.inverse_transform(mesh_points)
        mesh_tensor = torch.FloatTensor(mesh_points_original).to(device)
        
        # Get predictions from all models
        mesh_preds = []
        model1.eval()
        model2.eval()
        model3.eval()
        model4.eval()
        
        with torch.no_grad():
            mesh_pred1 = torch.sigmoid(model1(mesh_tensor)).cpu().numpy().flatten()
            mesh_pred2 = torch.sigmoid(model2(mesh_tensor)).cpu().numpy().flatten()
            mesh_pred3 = torch.sigmoid(model3(mesh_tensor)).cpu().numpy().flatten()
            mesh_pred4 = torch.sigmoid(model4(mesh_tensor)).cpu().numpy().flatten()
            mesh_preds = [mesh_pred1, mesh_pred2, mesh_pred3, mesh_pred4]
        
        # Ensemble predictions
        Z = np.average(mesh_preds, axis=0, weights=ensemble_weights)
        Z = Z.reshape(xx.shape)
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        
        # Plot 1: Training Set
        contour_train = axes[0].contourf(xx, yy, Z, levels=20, cmap='RdYlBu_r', alpha=0.6)
        axes[0].contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')
        
        train_cases_idx = y_train == 1
        train_controls_idx = y_train == 0
        axes[0].scatter(X_train_pca[train_controls_idx, 0], X_train_pca[train_controls_idx, 1],
                       c='blue', marker='o', s=30, alpha=0.6, edgecolors='k', linewidths=0.5,
                       label=f'Controls (n={np.sum(train_controls_idx)})')
        axes[0].scatter(X_train_pca[train_cases_idx, 0], X_train_pca[train_cases_idx, 1],
                       c='red', marker='^', s=30, alpha=0.6, edgecolors='k', linewidths=0.5,
                       label=f'Cases (n={np.sum(train_cases_idx)})')
        
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontweight='bold')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontweight='bold')
        axes[0].set_title(f'Training Set (AUC: {train_auc:.4f})', fontweight='bold', fontsize=12)
        axes[0].legend(frameon=True, shadow=True, loc='best')
        axes[0].grid(alpha=0.3)
        plt.colorbar(contour_train, ax=axes[0], label='Predicted Probability')
        
        # Plot 2: Test Set
        contour_test = axes[1].contourf(xx, yy, Z, levels=20, cmap='RdYlBu_r', alpha=0.6)
        axes[1].contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')
        
        test_cases_idx = y_test == 1
        test_controls_idx = y_test == 0
        axes[1].scatter(X_test_pca[test_controls_idx, 0], X_test_pca[test_controls_idx, 1],
                       c='blue', marker='o', s=30, alpha=0.6, edgecolors='k', linewidths=0.5,
                       label=f'Controls (n={np.sum(test_controls_idx)})')
        axes[1].scatter(X_test_pca[test_cases_idx, 0], X_test_pca[test_cases_idx, 1],
                       c='red', marker='^', s=30, alpha=0.6, edgecolors='k', linewidths=0.5,
                       label=f'Cases (n={np.sum(test_cases_idx)})')
        
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontweight='bold')
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontweight='bold')
        axes[1].set_title(f'Test Set (AUC: {test_auc:.4f})', fontweight='bold', fontsize=12)
        axes[1].legend(frameon=True, shadow=True, loc='best')
        axes[1].grid(alpha=0.3)
        plt.colorbar(contour_test, ax=axes[1], label='Predicted Probability')
        
        # Plot 3: Validation set (no labels)
        if X_val_pca is not None and len(X_val_pca) > 0:
            contour_val = axes[2].contourf(xx, yy, Z, levels=20, cmap='RdYlBu_r', alpha=0.6)
            axes[2].contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')
            
            axes[2].scatter(X_val_pca[:, 0], X_val_pca[:, 1],
                           c='green', marker='s', s=30, alpha=0.6, edgecolors='k', linewidths=0.5,
                           label=f'Validation (n={len(X_val_pca)})')
            
            axes[2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontweight='bold')
            axes[2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontweight='bold')
            axes[2].set_title('Validation Set (No Labels)', fontweight='bold', fontsize=12)
            axes[2].legend(frameon=True, shadow=True, loc='best')
            axes[2].grid(alpha=0.3)
            plt.colorbar(contour_val, ax=axes[2], label='Predicted Probability')
        else:
            axes[2].text(0.5, 0.5, 'No Validation Data', ha='center', va='center', fontsize=14)
            axes[2].set_xlabel(f'PC1', fontweight='bold')
            axes[2].set_ylabel(f'PC2', fontweight='bold')
            axes[2].set_title('Validation Set', fontweight='bold', fontsize=12)
        
        plt.suptitle(f'{phenotype} - PCA Visualization with Decision Boundaries\n' + 
                     f'Top {snp_count} SNPs, Fold {fold}, Features: {feature_info}',
                     fontweight='bold', fontsize=14)
        plt.tight_layout()
        
        pca_output_file = os.path.join(data_dir, f"DL_Ensemble_PCA_Fold_{fold}.png")
        plt.savefig(pca_output_file, dpi=300, bbox_inches='tight')
        print(f"  📊 PCA visualization saved: {pca_output_file}")
        plt.close()
        
        # ============================================================
        # t-SNE VISUALIZATION WITH DECISION BOUNDARIES
        # ============================================================
        print(f"  📊 Creating t-SNE visualization with decision boundaries...")
        
        # Apply t-SNE to reduce to 2D
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        X_train_tsne = tsne.fit_transform(X_train_scaled)
        
        # For test and val, we need to use the same transformation approach
        # Since t-SNE doesn't have transform method, we'll fit on combined data
        if X_val_scaled is not None:
            X_combined = np.vstack([X_train_scaled, X_test_scaled, X_val_scaled])
        else:
            X_combined = np.vstack([X_train_scaled, X_test_scaled])
        
        tsne_combined = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        X_combined_tsne = tsne_combined.fit_transform(X_combined)
        
        # Split back
        X_train_tsne = X_combined_tsne[:len(X_train_scaled)]
        X_test_tsne = X_combined_tsne[len(X_train_scaled):len(X_train_scaled)+len(X_test_scaled)]
        if X_val_scaled is not None:
            X_val_tsne = X_combined_tsne[len(X_train_scaled)+len(X_test_scaled):]
        else:
            X_val_tsne = None
        
        # Create meshgrid for decision boundary
        x_min, x_max = X_train_tsne[:, 0].min() - 5, X_train_tsne[:, 0].max() + 5
        y_min, y_max = X_train_tsne[:, 1].min() - 5, X_train_tsne[:, 1].max() + 5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        
        # For t-SNE, we approximate decision boundary using KNN on t-SNE space
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=15)
        knn.fit(X_train_tsne, y_train)
        Z_tsne = knn.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z_tsne = Z_tsne.reshape(xx.shape)
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        
        # Plot 1: Training Set
        contour_train = axes[0].contourf(xx, yy, Z_tsne, levels=20, cmap='RdYlBu_r', alpha=0.6)
        axes[0].contour(xx, yy, Z_tsne, levels=[0.5], colors='black', linewidths=2, linestyles='--')
        
        train_cases_idx = y_train == 1
        train_controls_idx = y_train == 0
        axes[0].scatter(X_train_tsne[train_controls_idx, 0], X_train_tsne[train_controls_idx, 1],
                       c='blue', marker='o', s=30, alpha=0.6, edgecolors='k', linewidths=0.5,
                       label=f'Controls (n={np.sum(train_controls_idx)})')
        axes[0].scatter(X_train_tsne[train_cases_idx, 0], X_train_tsne[train_cases_idx, 1],
                       c='red', marker='^', s=30, alpha=0.6, edgecolors='k', linewidths=0.5,
                       label=f'Cases (n={np.sum(train_cases_idx)})')
        
        axes[0].set_xlabel('t-SNE 1', fontweight='bold')
        axes[0].set_ylabel('t-SNE 2', fontweight='bold')
        axes[0].set_title(f'Training Set (AUC: {train_auc:.4f})', fontweight='bold', fontsize=12)
        axes[0].legend(frameon=True, shadow=True, loc='best')
        axes[0].grid(alpha=0.3)
        plt.colorbar(contour_train, ax=axes[0], label='Predicted Probability')
        
        # Plot 2: Test Set
        contour_test = axes[1].contourf(xx, yy, Z_tsne, levels=20, cmap='RdYlBu_r', alpha=0.6)
        axes[1].contour(xx, yy, Z_tsne, levels=[0.5], colors='black', linewidths=2, linestyles='--')
        
        test_cases_idx = y_test == 1
        test_controls_idx = y_test == 0
        axes[1].scatter(X_test_tsne[test_controls_idx, 0], X_test_tsne[test_controls_idx, 1],
                       c='blue', marker='o', s=30, alpha=0.6, edgecolors='k', linewidths=0.5,
                       label=f'Controls (n={np.sum(test_controls_idx)})')
        axes[1].scatter(X_test_tsne[test_cases_idx, 0], X_test_tsne[test_cases_idx, 1],
                       c='red', marker='^', s=30, alpha=0.6, edgecolors='k', linewidths=0.5,
                       label=f'Cases (n={np.sum(test_cases_idx)})')
        
        axes[1].set_xlabel('t-SNE 1', fontweight='bold')
        axes[1].set_ylabel('t-SNE 2', fontweight='bold')
        axes[1].set_title(f'Test Set (AUC: {test_auc:.4f})', fontweight='bold', fontsize=12)
        axes[1].legend(frameon=True, shadow=True, loc='best')
        axes[1].grid(alpha=0.3)
        plt.colorbar(contour_test, ax=axes[1], label='Predicted Probability')
        
        # Plot 3: Validation Set (no labels)
        if X_val_tsne is not None and len(X_val_tsne) > 0:
            contour_val = axes[2].contourf(xx, yy, Z_tsne, levels=20, cmap='RdYlBu_r', alpha=0.6)
            axes[2].contour(xx, yy, Z_tsne, levels=[0.5], colors='black', linewidths=2, linestyles='--')
            
            axes[2].scatter(X_val_tsne[:, 0], X_val_tsne[:, 1],
                           c='green', marker='s', s=30, alpha=0.6, edgecolors='k', linewidths=0.5,
                           label=f'Validation (n={len(X_val_tsne)})')
            
            axes[2].set_xlabel('t-SNE 1', fontweight='bold')
            axes[2].set_ylabel('t-SNE 2', fontweight='bold')
            axes[2].set_title('Validation Set (No Labels)', fontweight='bold', fontsize=12)
            axes[2].legend(frameon=True, shadow=True, loc='best')
            axes[2].grid(alpha=0.3)
            plt.colorbar(contour_val, ax=axes[2], label='Predicted Probability')
        else:
            axes[2].text(0.5, 0.5, 'No Validation Data', ha='center', va='center', fontsize=14)
            axes[2].set_xlabel('t-SNE 1', fontweight='bold')
            axes[2].set_ylabel('t-SNE 2', fontweight='bold')
            axes[2].set_title('Validation Set', fontweight='bold', fontsize=12)
        
        plt.suptitle(f'{phenotype} - t-SNE Visualization with Decision Boundaries\n' + 
                     f'Top {snp_count} SNPs, Fold {fold}, Features: {feature_info}',
                     fontweight='bold', fontsize=14)
        plt.tight_layout()
        
        tsne_output_file = os.path.join(data_dir, f"DL_Ensemble_tSNE_Fold_{fold}.png")
        plt.savefig(tsne_output_file, dpi=300, bbox_inches='tight')
        print(f"  📊 t-SNE visualization saved: {tsne_output_file}")
        plt.close()
        
        # ============================================================
        # UMAP VISUALIZATION WITH DECISION BOUNDARIES
        # ============================================================
        print(f"  📊 Creating UMAP visualization with decision boundaries...")
        
        # Apply UMAP to reduce to 2D
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        X_train_umap = reducer.fit_transform(X_train_scaled)
        X_test_umap = reducer.transform(X_test_scaled)
        if X_val_scaled is not None:
            X_val_umap = reducer.transform(X_val_scaled)
        else:
            X_val_umap = None
        
        # Create meshgrid for decision boundary
        x_min, x_max = X_train_umap[:, 0].min() - 1, X_train_umap[:, 0].max() + 1
        y_min, y_max = X_train_umap[:, 1].min() - 1, X_train_umap[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        
        # For UMAP, we approximate decision boundary using KNN on UMAP space
        knn_umap = KNeighborsClassifier(n_neighbors=15)
        knn_umap.fit(X_train_umap, y_train)
        Z_umap = knn_umap.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z_umap = Z_umap.reshape(xx.shape)
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        
        # Plot 1: Training Set
        contour_train = axes[0].contourf(xx, yy, Z_umap, levels=20, cmap='RdYlBu_r', alpha=0.6)
        axes[0].contour(xx, yy, Z_umap, levels=[0.5], colors='black', linewidths=2, linestyles='--')
        
        train_cases_idx = y_train == 1
        train_controls_idx = y_train == 0
        axes[0].scatter(X_train_umap[train_controls_idx, 0], X_train_umap[train_controls_idx, 1],
                       c='blue', marker='o', s=30, alpha=0.6, edgecolors='k', linewidths=0.5,
                       label=f'Controls (n={np.sum(train_controls_idx)})')
        axes[0].scatter(X_train_umap[train_cases_idx, 0], X_train_umap[train_cases_idx, 1],
                       c='red', marker='^', s=30, alpha=0.6, edgecolors='k', linewidths=0.5,
                       label=f'Cases (n={np.sum(train_cases_idx)})')
        
        axes[0].set_xlabel('UMAP 1', fontweight='bold')
        axes[0].set_ylabel('UMAP 2', fontweight='bold')
        axes[0].set_title(f'Training Set (AUC: {train_auc:.4f})', fontweight='bold', fontsize=12)
        axes[0].legend(frameon=True, shadow=True, loc='best')
        axes[0].grid(alpha=0.3)
        plt.colorbar(contour_train, ax=axes[0], label='Predicted Probability')
        
        # Plot 2: Test Set
        contour_test = axes[1].contourf(xx, yy, Z_umap, levels=20, cmap='RdYlBu_r', alpha=0.6)
        axes[1].contour(xx, yy, Z_umap, levels=[0.5], colors='black', linewidths=2, linestyles='--')
        
        test_cases_idx = y_test == 1
        test_controls_idx = y_test == 0
        axes[1].scatter(X_test_umap[test_controls_idx, 0], X_test_umap[test_controls_idx, 1],
                       c='blue', marker='o', s=30, alpha=0.6, edgecolors='k', linewidths=0.5,
                       label=f'Controls (n={np.sum(test_controls_idx)})')
        axes[1].scatter(X_test_umap[test_cases_idx, 0], X_test_umap[test_cases_idx, 1],
                       c='red', marker='^', s=30, alpha=0.6, edgecolors='k', linewidths=0.5,
                       label=f'Cases (n={np.sum(test_cases_idx)})')
        
        axes[1].set_xlabel('UMAP 1', fontweight='bold')
        axes[1].set_ylabel('UMAP 2', fontweight='bold')
        axes[1].set_title(f'Test Set (AUC: {test_auc:.4f})', fontweight='bold', fontsize=12)
        axes[1].legend(frameon=True, shadow=True, loc='best')
        axes[1].grid(alpha=0.3)
        plt.colorbar(contour_test, ax=axes[1], label='Predicted Probability')
        
        # Plot 3: Validation Set (no labels)
        if X_val_umap is not None and len(X_val_umap) > 0:
            contour_val = axes[2].contourf(xx, yy, Z_umap, levels=20, cmap='RdYlBu_r', alpha=0.6)
            axes[2].contour(xx, yy, Z_umap, levels=[0.5], colors='black', linewidths=2, linestyles='--')
            
            axes[2].scatter(X_val_umap[:, 0], X_val_umap[:, 1],
                           c='green', marker='s', s=30, alpha=0.6, edgecolors='k', linewidths=0.5,
                           label=f'Validation (n={len(X_val_umap)})')
            
            axes[2].set_xlabel('UMAP 1', fontweight='bold')
            axes[2].set_ylabel('UMAP 2', fontweight='bold')
            axes[2].set_title('Validation Set (No Labels)', fontweight='bold', fontsize=12)
            axes[2].legend(frameon=True, shadow=True, loc='best')
            axes[2].grid(alpha=0.3)
            plt.colorbar(contour_val, ax=axes[2], label='Predicted Probability')
        else:
            axes[2].text(0.5, 0.5, 'No Validation Data', ha='center', va='center', fontsize=14)
            axes[2].set_xlabel('UMAP 1', fontweight='bold')
            axes[2].set_ylabel('UMAP 2', fontweight='bold')
            axes[2].set_title('Validation Set', fontweight='bold', fontsize=12)
        
        plt.suptitle(f'{phenotype} - UMAP Visualization with Decision Boundaries\n' + 
                     f'Top {snp_count} SNPs, Fold {fold}, Features: {feature_info}',
                     fontweight='bold', fontsize=14)
        plt.tight_layout()
        
        umap_output_file = os.path.join(data_dir, f"DL_Ensemble_UMAP_Fold_{fold}.png")
        plt.savefig(umap_output_file, dpi=300, bbox_inches='tight')
        print(f"  📊 UMAP visualization saved: {umap_output_file}")
        plt.close()
    
    # Store dataset results
    all_dataset_results[snp_count] = fold_results
    
    # Save aggregated results across all folds
    if len(fold_results['train_auc']) > 0:
        aggregated_results = []
        
        for fold_idx in range(len(fold_results['train_auc'])):
            # Read individual fold results
            fold_dir = os.path.join(phenotype, f"Fold_{fold_idx}", f"Top_{snp_count}_SNPs")
            fold_results_file = os.path.join(fold_dir, "ensemble_results.csv")
            
            if os.path.exists(fold_results_file):
                fold_df = pd.read_csv(fold_results_file)
                aggregated_results.append(fold_df)
        
        if aggregated_results:
            # Combine all folds
            combined_df = pd.concat(aggregated_results, ignore_index=True)
            
            # Save combined results
            combined_file = os.path.join(phenotype, f"All_Folds_Top_{snp_count}_SNPs_Results.csv")
            combined_df.to_csv(combined_file, index=False)
            print(f"\n  💾 Combined results saved: {combined_file}")
            
            # Calculate and save summary statistics
            summary_stats = combined_df.describe()
            summary_file = os.path.join(phenotype, f"Summary_Stats_Top_{snp_count}_SNPs.csv")
            summary_stats.to_csv(summary_file)
            print(f"  💾 Summary statistics saved: {summary_file}")
    
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
        
        overfit_gap = avg_train_auc - avg_test_auc
        
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
    
    plt.suptitle(f'{phenotype} - Deep Learning Ensemble Predictions (BEST)\n' + 
                 f'Top {best_dataset_info["snp_count"]} SNPs, Fold {best_dataset_info["fold"]}, ' + 
                 f'Test AUC: {best_dataset_info["test_auc"]:.4f}',
                 fontweight='bold', fontsize=14)
    plt.tight_layout()
    
    output_file = f"DL_Ensemble_Distributions_Best_{phenotype}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Distribution plot saved: {output_file}")
    plt.close()

print(f"\n{'='*80}")
print("✅ DEEP LEARNING ENSEMBLE TRAINING COMPLETED")
print(f"{'='*80}")

# Generate final consolidated submission files
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
            submission_dir = os.path.join(phenotype, "Submission3")
            os.makedirs(submission_dir, exist_ok=True)
            
            final_submission_file = os.path.join(submission_dir, f"{phenotype}_submission3.txt")
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
                f.write(f"CAGI7 PRS Challenge - Submission Files\n")
                f.write(f"{'='*60}\n\n")
                f.write(f"Phenotype: {phenotype}\n")
                f.write(f"Method: Deep Learning Ensemble (4 models)\n")
                f.write(f"  - Wide & Deep Network\n")
                f.write(f"  - Residual Network with Skip Connections\n")
                f.write(f"  - Deep Narrow Network\n")
                f.write(f"  - Attention-based Network\n\n")
                f.write(f"Features: SNPs + PCA + PRS\n")
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
                    results_file = os.path.join(fold_dir, "ensemble_results.csv")
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
print(f"\n📁 Submission files are in: {phenotype}/Submission3/")
print(f"\nSubmission Strategy:")
print(f"  ✓ Used BEST performing fold (highest test AUC)")
print(f"  ✓ Best Fold: {best_dataset_info['fold']} (Test AUC: {best_dataset_info['test_auc']:.6f})")
print(f"  ✓ Format: Tab-delimited (FID, IID, PRS)")
print(f"  ✓ PRS = predicted probability (higher = higher risk)")