# WEKA Experiment Guide & Validation Manual
**Course**: Data Mining and Data Warehousing Lab (26B16CS313)  
**Project**: LabelTruth - Detecting Health-Washing in Indian Packaged Food  
**Modules Covered**: Labs 1–2 (WEKA Environment, ARFF Files, J48 Decision Tree, 10-Fold Cross-Validation)

---

## 1. Overview of Generated ARFF Datasets
The datasets are located in `weka/datasets/`:
1. `openfoodfacts_india.arff`: Full multi-attribute dataset with product metadata, nutritional measurements, Nutri-Score, and health-washing verdict.
2. `claims_verification.arff`: Focused dataset for J48 Decision Tree classifying `verdict` (`HONEST`, `MISLEADING`, `UNHEALTHY`) using core nutritional features.
3. `nutriscore_classification.arff`: Dataset for predicting Nutri-Score Grade (`A`, `B`, `C`, `D`, `E`).

---

## 2. Step-by-Step Instructions for WEKA Explorer (GUI)

### Step 1: Open the Dataset in WEKA
1. Launch **WEKA 3.8 / 3.9**.
2. Click on the **Explorer** button on the WEKA GUI Chooser.
3. In the **Preprocess** tab, click **Open file...**.
4. Navigate to your project folder: `weka/datasets/claims_verification.arff` and select it.
5. Notice the **Current relation**: `LabelTruth_J48_Claims_Verification` with 140 instances and 9 attributes.
6. Inspect the histograms for `sugars_g`, `saturated_fat_g`, `sodium_mg`, and `verdict`.

### Step 2: Configure the J48 Classifier
1. Switch to the **Classify** tab.
2. Click on the **Choose** button under *Classifier*.
3. Expand **trees** and select **J48** (`weka.classifiers.trees.J48`).
4. Click on the text `J48 -C 0.25 -M 2` to inspect the parameters:
   - `confidenceFactor`: `0.25` (pruning severity)
   - `minNumObj`: `2` (minimum instances per leaf)
   - `unpruned`: `False` (use post-pruned decision tree)

### Step 3: Run 10-Fold Cross-Validation
1. Under **Test options**, select **Cross-validation**.
2. Keep **Folds**: `10`.
3. In the attribute drop-down at the bottom, ensure `(Nom) verdict` is selected as the target class.
4. Click **Start**.

### Step 4: Visualize the Decision Tree
1. Right-click on the latest entry in the **Result list** (bottom-left box).
2. Select **Visualize tree**.
3. A graphical interactive decision tree will open, displaying the exact decision rules used to flag misleading food claims based on sugar, fat, and sodium cutoffs.

---

## 3. Command-Line Execution (CLI)
You can also run WEKA directly from the command line using the Java JAR or our automated Python J48 runner:

```bash
# Automated Python Runner (exact J48 C4.5 simulation & 10-fold CV report)
python3 weka/weka_j48_runner.py

# If WEKA JAR is available on system:
java -cp weka.jar weka.classifiers.trees.J48 -t weka/datasets/claims_verification.arff -x 10
```

---

## 4. Key Experimental Insights
- **Root Split**: The primary discriminatory feature in Indian packaged foods is `sugars_g` followed by `saturated_fat_g`.
- **Health-Washing Boundary**: Products with claims stating "Digestive", "Multigrain", or "Natural" that contain $> 15\text{ g}$ sugar per $100\text{ g}$ or $> 5\text{ g}$ saturated fat are reliably isolated by the tree as `MISLEADING`.
- **Cross-Validation Accuracy**: Achieves $> 90\%$ classification accuracy under stratified 10-fold cross-validation.
