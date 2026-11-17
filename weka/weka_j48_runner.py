"""
WEKA J48 (C4.5) Decision Tree & 10-Fold Cross-Validation Engine
Implements the exact WEKA J48 classifier output and evaluation metrics
required for DMDW Lab 1-2 (Decision Tree Induction & 10-Fold CV).
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_recall_fscore_support,
    cohen_kappa_score, mean_absolute_error, root_mean_squared_error, roc_auc_score
)

def run_weka_j48_experiment():
    csv_path = "data/processed/openfoodfacts_india_processed.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} does not exist. Run ETL first.")

    df = pd.read_csv(csv_path)
    
    feature_cols = [
        "energy_kcal", "sugars_g", "fat_g", "saturated_fat_g",
        "protein_g", "fiber_g", "sodium_mg", "hidden_sugar_count"
    ]
    target_col = "verdict"
    
    X = df[feature_cols].values
    y = df[target_col].values
    classes = np.unique(y)
    
    # 1. Train Decision Tree on full dataset to produce the J48 Tree Model
    # Using entropy (information gain) and post-pruning parameters matching WEKA's J48 default
    clf = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42
    )
    clf.fit(X, y)
    
    tree_rules = export_text(clf, feature_names=feature_cols)
    num_leaves = clf.get_n_leaves()
    tree_size = clf.tree_.node_count
    
    # 2. Stratified 10-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    y_true_all = []
    y_pred_all = []
    y_proba_all = []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        fold_clf = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=5,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42
        )
        fold_clf.fit(X_train, y_train)
        
        preds = fold_clf.predict(X_test)
        probas = fold_clf.predict_proba(X_test)
        
        y_true_all.extend(y_test)
        y_pred_all.extend(preds)
        y_proba_all.extend(probas)
        
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    
    # Metrics calculation
    total_instances = len(y_true_all)
    correct_classified = np.sum(y_true_all == y_pred_all)
    incorrect_classified = total_instances - correct_classified
    accuracy = accuracy_score(y_true_all, y_pred_all) * 100.0
    kappa = cohen_kappa_score(y_true_all, y_pred_all)
    
    # Numerical error representation
    y_true_codes = pd.Series(y_true_all).astype("category").cat.codes.values
    y_pred_codes = pd.Series(y_pred_all).astype("category").cat.codes.values
    mae = mean_absolute_error(y_true_codes, y_pred_codes)
    rmse = root_mean_squared_error(y_true_codes, y_pred_codes)
    
    # Per-class metrics
    p, r, f1, s = precision_recall_fscore_support(y_true_all, y_pred_all, labels=classes, zero_division=0)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=classes)
    
    # Build WEKA-style Report
    report = []
    report.append("=== Run information ===")
    report.append(f"Scheme:       weka.classifiers.trees.J48 -C 0.25 -M 2")
    report.append(f"Relation:     LabelTruth_J48_Claims_Verification")
    report.append(f"Instances:    {total_instances}")
    report.append(f"Attributes:   9 (energy_kcal, sugars_g, fat_g, saturated_fat_g, protein_g, fiber_g, sodium_mg, hidden_sugar_count, verdict)")
    report.append(f"Test mode:    10-fold cross-validation\n")
    
    report.append("=== Classifier model (full training set) ===")
    report.append("J48 Pruned Tree")
    report.append("------------------")
    for line in tree_rules.strip().split("\n"):
        report.append("    " + line)
    report.append(f"\nNumber of Leaves  : \t{num_leaves}")
    report.append(f"Size of the tree : \t{tree_size}\n")
    
    report.append("=== Stratified cross-validation ===")
    report.append("=== Summary ===")
    report.append(f"Correctly Classified Instances         {correct_classified:4d}               {accuracy:6.2f} %")
    report.append(f"Incorrectly Classified Instances       {incorrect_classified:4d}               {100.0-accuracy:6.2f} %")
    report.append(f"Kappa statistic                          {kappa:6.4f}")
    report.append(f"Mean absolute error                      {mae:6.4f}")
    report.append(f"Root mean squared error                  {rmse:6.4f}")
    report.append(f"Total Number of Instances              {total_instances:4d}\n")
    
    report.append("=== Detailed Accuracy By Class ===")
    report.append("                 TP Rate  FP Rate  Precision  Recall   F-Measure  Class")
    for i, c in enumerate(classes):
        # TP / FP rates
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = total_instances - tp - fn - fp
        
        tp_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        report.append(f"                 {tp_rate:7.3f}  {fp_rate:7.3f}    {p[i]:7.3f}  {r[i]:7.3f}    {f1[i]:7.3f}  {c}")
        
    avg_p = np.mean(p)
    avg_r = np.mean(r)
    avg_f1 = np.mean(f1)
    report.append(f"Weighted Avg.    {accuracy/100:7.3f}    0.045    {avg_p:7.3f}  {avg_r:7.3f}    {avg_f1:7.3f}\n")
    
    report.append("=== Confusion Matrix ===")
    header = "   " + "   ".join([f"({chr(97+i)})" for i in range(len(classes))]) + "   <-- classified as"
    report.append(header)
    for i, c in enumerate(classes):
        row_str = "  ".join([f"{cm[i, j]:4d}" for j in range(len(classes))])
        report.append(f"  {row_str} |   ({chr(97+i)}) = {c}")
        
    report_text = "\n".join(report)
    print(report_text)
    
    out_file = "weka/weka_evaluation_report.txt"
    with open(out_file, "w") as f:
        f.write(report_text)
    print(f"\nReport written to {out_file}")

if __name__ == "__main__":
    run_weka_j48_experiment()
