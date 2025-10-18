#!/usr/bin/env python
# coding: utf-8
"""
Data Loading Functions for Machine Learning Models
Author: Your Name
Description: Functions to load SNP, PCA, and PRS data for genomic prediction models
"""

import os
import pandas as pd
import numpy as np

def load_snp_data(phenotype, fold, snp_count):
    """
    Load SNP data for a specific fold and SNP count.
    
    Parameters:
    -----------
    phenotype : str
        Phenotype directory name
    fold : int
        Fold number (0-4)
    snp_count : int
        Number of top SNPs (e.g., 100, 500, 2000)
    
    Returns:
    --------
    tuple : (X_train, y_train, X_test, y_test, X_val)
        Training, testing, and validation SNP data
    """
    data_dir = os.path.join(phenotype, f"Fold_{fold}", f"Top_{snp_count}_SNPs")
    
    X_train = None
    y_train = None
    X_test = None
    y_test = None
    X_val = None
    
    try:
        X_train_file = os.path.join(data_dir, "X_train.npy")
        y_train_file = os.path.join(data_dir, "y_train.npy")
        X_test_file = os.path.join(data_dir, "X_test.npy")
        y_test_file = os.path.join(data_dir, "y_test.npy")
        X_val_file = os.path.join(data_dir, "X_val.npy")
        
        if os.path.exists(X_train_file):
            X_train = np.load(X_train_file)
        if os.path.exists(y_train_file):
            y_train = np.load(y_train_file)
        if os.path.exists(X_test_file):
            X_test = np.load(X_test_file)
        if os.path.exists(y_test_file):
            y_test = np.load(y_test_file)
        if os.path.exists(X_val_file):
            X_val = np.load(X_val_file)
    except Exception as e:
        print(f"Error loading SNP data: {e}")
    
    return X_train, y_train, X_test, y_test, X_val


def load_pca_data(phenotype, fold):
    """
    Load PCA data for train, test, and validation sets.
    
    Parameters:
    -----------
    phenotype : str
        Phenotype directory name
    fold : int
        Fold number (0-4)
    
    Returns:
    --------
    tuple : (train_pca, test_pca, val_pca)
        PCA features for train, test, and validation sets
    """
    train_pca_file = os.path.join(phenotype, f"Fold_{fold}", "train_data.eigenvec")
    test_pca_file = os.path.join(phenotype, f"Fold_{fold}", "test_data.eigenvec")
    val_pca_file = os.path.join(phenotype, f"Fold_{fold}", "val.PCA")
    
    train_pca = None
    test_pca = None
    val_pca = None
    
    # Load train PCA
    if os.path.exists(train_pca_file) and os.path.getsize(train_pca_file) > 0:
        try:
            train_pca = pd.read_csv(train_pca_file, sep=r'\s+', header=None)
            if not train_pca.empty:
                train_pca = train_pca.iloc[:, 2:].values  # Remove FID and IID columns
        except Exception as e:
            print(f"Error loading train PCA: {e}")
            train_pca = None
    
    # Load test PCA
    if os.path.exists(test_pca_file) and os.path.getsize(test_pca_file) > 0:
        try:
            test_pca = pd.read_csv(test_pca_file, sep=r'\s+', header=None)
            if not test_pca.empty:
                test_pca = test_pca.iloc[:, 2:].values  # Remove FID and IID columns
        except Exception as e:
            print(f"Error loading test PCA: {e}")
            test_pca = None
    
    # Load validation PCA (has 2 header rows)
    if os.path.exists(val_pca_file) and os.path.getsize(val_pca_file) > 0:
        try:
            val_pca = pd.read_csv(val_pca_file, sep=r'\s+', skiprows=2, header=None)
            if not val_pca.empty:
                val_pca = val_pca.iloc[:, 2:].values  # Remove FID and IID columns
        except Exception as e:
            print(f"Error loading validation PCA: {e}")
            val_pca = None
    
    return train_pca, test_pca, val_pca


def load_prs_for_method(phenotype, fold, method_dir_name, max_prs=3):
    """
    Load PRS data for a specific method using PRS1, PRS2, PRS3 naming convention.
    
    Parameters:
    -----------
    phenotype : str
        Phenotype directory name
    fold : int
        Fold number (0-4)
    method_dir_name : str
        Directory name for the method (e.g., 'Plink3', 'GCTA3', 'PRSice-2-3')
    max_prs : int
        Maximum number of PRS files to load (default: 3)
    
    Returns:
    --------
    tuple : (train_prs_list, test_prs_list, val_prs_list)
        Lists containing PRS arrays for train, test, and validation
    """
    # Correct path: Phenotype/Results/Fold_X/Method
    method_dir = os.path.join(phenotype, "Results", f"Fold_{fold}", method_dir_name)
    
    train_prs_list = []
    test_prs_list = []
    val_prs_list = []
    
    if not os.path.exists(method_dir):
        print(f"⚠️  Method directory not found: {method_dir}")
        return train_prs_list, test_prs_list, val_prs_list
    
    print(f"📂 Loading PRS from: {method_dir}")
    
    # Try to load PRS1, PRS2, PRS3, etc.
    for prs_num in range(1, max_prs + 1):
        # Determine file naming and score column based on method
        if method_dir_name == "PRSice-2-3":
            train_file = os.path.join(method_dir, f"PRS{prs_num}.train")
            test_file = os.path.join(method_dir, f"PRS{prs_num}.test")
            val_file = os.path.join(method_dir, f"PRS{prs_num}.val")
            score_column = 'PRS'
        else:
            train_file = os.path.join(method_dir, f"PRS{prs_num}.train")
            test_file = os.path.join(method_dir, f"PRS{prs_num}.test")
            val_file = os.path.join(method_dir, f"PRS{prs_num}.val")
            score_column = 'SCORE'
        
        # Track if all three files loaded successfully
        files_loaded = []
        
        # Load train PRS
        if os.path.exists(train_file) and os.path.getsize(train_file) > 0:
            try:
                df = pd.read_csv(train_file, sep=r'\s+')
                if not df.empty and score_column in df.columns:
                    train_prs_list.append(df[score_column].values.reshape(-1, 1))
                    files_loaded.append('train')
            except Exception as e:
                print(f"  ❌ Error loading train PRS{prs_num}: {e}")
        
        # Load test PRS
        if os.path.exists(test_file) and os.path.getsize(test_file) > 0:
            try:
                df = pd.read_csv(test_file, sep=r'\s+')
                if not df.empty and score_column in df.columns:
                    test_prs_list.append(df[score_column].values.reshape(-1, 1))
                    files_loaded.append('test')
            except Exception as e:
                print(f"  ❌ Error loading test PRS{prs_num}: {e}")
        
        # Load validation PRS
        if os.path.exists(val_file) and os.path.getsize(val_file) > 0:
            try:
                df = pd.read_csv(val_file, sep=r'\s+')
                if not df.empty and score_column in df.columns:
                    val_prs_list.append(df[score_column].values.reshape(-1, 1))
                    files_loaded.append('val')
            except Exception as e:
                print(f"  ❌ Error loading val PRS{prs_num}: {e}")
        
        if files_loaded:
            print(f"  ✅ Loaded PRS{prs_num} for: {', '.join(files_loaded)}")
    
    return train_prs_list, test_prs_list, val_prs_list
    
    return train_prs_list, test_prs_list, val_prs_list


def load_prs_data(phenotype, fold, methods=None, max_prs=3):
    """
    Load PRS data from all methods.
    
    Parameters:
    -----------
    phenotype : str
        Phenotype directory name
    fold : int
        Fold number (0-4)
    methods : list, optional
        List of method directory names to load. 
        Default: ['Plink3', 'GCTA3', 'LDAK-GWAS3', 'LDpred-gibbs3', 'PRSice-2-3']
    max_prs : int
        Maximum number of PRS files to load per method (default: 3)
    
    Returns:
    --------
    tuple : (train_prs, test_prs, val_prs)
        Combined PRS arrays for train, test, and validation sets
    """
    if methods is None:
    #    methods = ['Plink3', 'GCTA3', 'LDAK-GWAS3', 'LDpred-gibbs3', 'PRSice-2-3']
        methods = [ 'GCTA3', 'LDAK-GWAS3',  'PRSice-2-3']
    

    all_train_prs = []
    all_test_prs = []
    all_val_prs = []
    
    for method in methods:
        train_list, test_list, val_list = load_prs_for_method(phenotype, fold, method, max_prs)
        all_train_prs.extend(train_list)
        all_test_prs.extend(test_list)
        all_val_prs.extend(val_list)
    
    # Combine all PRS
    train_prs = np.hstack(all_train_prs) if all_train_prs else None
    test_prs = np.hstack(all_test_prs) if all_test_prs else None
    val_prs = np.hstack(all_val_prs) if all_val_prs else None
    
    return train_prs, test_prs, val_prs


def combine_features(X_snp, pca=None, prs=None):
    """
    Combine SNP, PCA, and PRS features.
    
    Parameters:
    -----------
    X_snp : numpy.ndarray
        SNP data
    pca : numpy.ndarray, optional
        PCA features
    prs : numpy.ndarray, optional
        PRS features
    
    Returns:
    --------
    tuple : (X_combined, feature_info)
        Combined feature array and description string
    """
    features = [X_snp]
    feature_info = f"SNPs({X_snp.shape[1]})"
    
    if pca is not None:
        features.append(pca)
        feature_info += f"+PCA({pca.shape[1]})"
    
    if prs is not None:
        features.append(prs)
        feature_info += f"+PRS({prs.shape[1]})"
    
    X_combined = np.hstack(features)
    
    return X_combined, feature_info


def load_all_data(phenotype, fold, snp_count, methods=None, max_prs=3):
    """
    Load all data (SNPs, PCA, PRS) for a specific fold and SNP count.
    
    Parameters:
    -----------
    phenotype : str
        Phenotype directory name
    fold : int
        Fold number (0-4)
    snp_count : int
        Number of top SNPs (e.g., 100, 500, 2000)
    methods : list, optional
        List of PRS method directory names to load
    max_prs : int
        Maximum number of PRS files to load per method (default: 3)
    
    Returns:
    --------
    dict : Dictionary containing all loaded data with keys:
        - 'X_train', 'y_train', 'X_test', 'y_test', 'X_val'
        - 'train_pca', 'test_pca', 'val_pca'
        - 'train_prs', 'test_prs', 'val_prs'
        - 'X_train_combined', 'X_test_combined', 'X_val_combined'
        - 'feature_info'
        - 'val_compatible' (bool indicating if validation can be used)
    """
    # Load SNP data
    X_train, y_train, X_test, y_test, X_val = load_snp_data(phenotype, fold, snp_count)
    
    # Load PCA data
    train_pca, test_pca, val_pca = load_pca_data(phenotype, fold)
    
    # Load PRS data
    train_prs, test_prs, val_prs = load_prs_data(phenotype, fold, methods, max_prs)
    
    # Combine train features
    X_train_combined, train_feature_info = combine_features(X_train, train_pca, train_prs)
    
    # Combine test features
    X_test_combined, test_feature_info = combine_features(X_test, test_pca, test_prs)
    
    # Combine validation features (with compatibility check)
    val_compatible = True
    X_val_combined = None
    
    if X_val is not None:
        val_features = [X_val]
        
        # Check if validation PCA matches
        if train_pca is not None:
            if val_pca is not None and val_pca.shape[0] == X_val.shape[0]:
                val_features.append(val_pca)
            else:
                val_compatible = False
        
        # Check if validation PRS matches
        if train_prs is not None:
            if val_prs is not None and val_prs.shape[0] == X_val.shape[0]:
                val_features.append(val_prs)
            else:
                val_compatible = False
        
        # Combine if compatible
        if val_compatible and len(val_features) > 0:
            try:
                X_val_combined = np.hstack(val_features)
                # Final check: feature count must match train
                if X_val_combined.shape[1] != X_train_combined.shape[1]:
                    val_compatible = False
                    X_val_combined = None
            except:
                val_compatible = False
                X_val_combined = None
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'X_val': X_val,
        'train_pca': train_pca,
        'test_pca': test_pca,
        'val_pca': val_pca,
        'train_prs': train_prs,
        'test_prs': test_prs,
        'val_prs': val_prs,
        'X_train_combined': X_train_combined,
        'X_test_combined': X_test_combined,
        'X_val_combined': X_val_combined,
        'feature_info': train_feature_info,
        'val_compatible': val_compatible
    }


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_loader.py <phenotype_name>")
        print("Example: python data_loader.py Phenotype_1")
        sys.exit(1)
    
    phenotype = sys.argv[1]
    fold = 0
    snp_count = 100
    
    print("="*80)
    print(f"Testing Data Loading for {phenotype}")
    print("="*80)
    
    # Load all data
    data = load_all_data(phenotype, fold, snp_count)
    
    print(f"\nFold {fold}, Top {snp_count} SNPs")
    print(f"Feature Info: {data['feature_info']}")
    print(f"\nTrain: {data['X_train_combined'].shape}")
    print(f"Test:  {data['X_test_combined'].shape}")
    
    if data['val_compatible'] and data['X_val_combined'] is not None:
        print(f"Val:   {data['X_val_combined'].shape} ✓")
        print("\n✅ All features compatible for training!")
    else:
        print(f"Val:   Incompatible ✗")
        print("\n⚠️  Validation will be skipped")
    
    # Show PRS counts
    if data['train_prs'] is not None:
        print(f"\n📊 PRS Features Loaded: {data['train_prs'].shape[1]}")
    else:
        print("\n⚠️  No PRS features found")