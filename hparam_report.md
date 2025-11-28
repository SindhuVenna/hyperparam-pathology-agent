# Hyperparameter Sweep Pathology Report

## Overview

* The sweep identified 5 issues across 3 trials, with 2 high-severity and 3 medium-severity issues.
* The main issue types were `nan_or_inf_metric`, `overfitting_suspect`, and `short_run`.
* There were clear correlations between hyperparameters and issue rates.

## Issue Breakdown

| Issue Type | Count | Example Trial IDs |
| --- | --- | --- |
| nan_or_inf_metric | 2 | 2, 4 |
| overfitting_suspect | 2 | 3, 4 |
| short_run | 1 | 2 |

## Hyperparameter Pathologies

### High Learning Rate Causes NaN Loss

* **Title:** High learning rate causes NaN loss
* **Description:** Learning rates above 0.01 frequently result in NaN loss, particularly in the 0.02 bucket.
* **Evidence:**
 + The param_correlations show a clear correlation between high learning rates and NaN loss, with the 0.01–0.02 bucket having an issue rate of 1.0.
 + Example trial ID 4 had a learning rate of 0.02 and resulted in a NaN loss.

### Small Batch Size Correlates with Unstable Validation Metrics

* **Title:** Small batch size correlates with unstable validation metrics
* **Description:** Batch sizes below 16 are strongly correlated with unstable validation metrics, particularly in the 8–16 bucket.
* **Evidence:**
 + The param_correlations show a clear correlation between small batch sizes and unstable validation metrics, with the 8–16 bucket having an issue rate of 1.0.
 + Example trial ID 3 had a batch size of 16 and resulted in a high overfitting suspect ratio.

### Zero Weight Decay Causes Strong Overfitting on Small Datasets

* **Title:** Zero weight decay causes strong overfitting on small datasets
* **Description:** Weight decay values of 0 are strongly correlated with overfitting, particularly in the small dataset scenario.
* **Evidence:**
 + The param_correlations show a clear correlation between zero weight decay and overfitting, with the 0–0.001 bucket having an issue rate of 0.6.
 + Example trial ID 3 had a weight decay of 0 and resulted in a high overfitting suspect ratio.

## Recommendations

* Avoid learning rates above 0.01, particularly in the 0.02 bucket.
* Use batch sizes above 16 to prevent unstable validation metrics.
* Use weight decay values above 0 to prevent strong overfitting on small datasets.
* Monitor trial progress closely to prevent short runs.
* Consider adding gradient clipping or regularization to prevent overfitting.
* Consider adjusting the regularization strength and type to improve model generalization.