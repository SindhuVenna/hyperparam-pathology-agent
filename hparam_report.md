# Hyperparameter Sweep Pathology Report

## Overview
- The sweep had a total of 5 issues across 3 trials.
- 2 issues were related to NaN or infinite metrics, 2 to overfitting, and 1 to a short run.
- Severity was high for 2 issues and medium for the remaining 3.

## Issue Breakdown
| Issue Type | Count |
| --- | --- |
| nan_or_inf_metric | 2 |
| short_run | 1 |
| overfitting_suspect | 2 |

- Example trial IDs:
  - nan_or_inf_metric: trial_id 2, 4
  - short_run: trial_id 2
  - overfitting_suspect: trial_id 3, 4

## Hyperparameter Pathologies

### High Learning Rate Causes Divergence
A high learning rate can lead to divergence in the model, resulting in NaN loss. This is evident in the param_correlations where learning rates in the range (0.01, 0.02] have an issue rate of 1.0, indicating a strong correlation with divergence. Additionally, trials with learning rates in this range, such as trial_id 4, also show NaN loss. Furthermore, trials with even higher learning rates, like trial_id 2, also experience NaN loss.

* Evidence:
  - lr learning rate in (0.01, 0.02] has issue rate of 1.0
  - trial_id 4 has learning rate of 0.02 and NaN loss
  - trial_id 2 has learning rate of 0.01 and NaN loss

### Small Batch Sizes Correlate with Unstable Validation Metrics
Small batch sizes can lead to unstable validation metrics, resulting in overfitting. This is evident in the param_correlations where batch sizes in the range (7.999, 16.0] have an issue rate of 0.5, indicating a moderate correlation with overfitting. Additionally, trials with batch sizes in this range, such as trial_id 3, also show signs of overfitting.

* Evidence:
  - batch_size in (7.999, 16.0] has issue rate of 0.5
  - trial_id 3 has batch size of 16 and shows signs of overfitting

### Zero Weight Decay Causes Strong Overfitting on Small Datasets
Zero weight decay can lead to strong overfitting on small datasets. This is evident in the issues_by_type where trials with weight decay of 0, such as trial_id 3, show signs of overfitting. Additionally, the param_correlations show a strong correlation between weight decay in the range (-0.001, 0.0001] and overfitting.

* Evidence:
  - weight_decay of 0 has issue rate of 1.0 in issues_by_type
  - trial_id 3 has weight decay of 0 and shows signs of overfitting

## Recommendations

- **Avoid learning rates in the range (0.01, 0.02]**: This range has a strong correlation with divergence, resulting in NaN loss.
- **Use batch sizes greater than 16**: Batch sizes in the range (7.999, 16.0] have a moderate correlation with overfitting.
- **Use weight decay greater than 0**: Zero weight decay has a strong correlation with overfitting on small datasets.
- **Increase regularization**: Regularization can help prevent overfitting, especially when using small datasets.
- **Monitor model performance closely**: Regular monitoring can help catch issues early and prevent longer runs from completing.
- **Adjust hyperparameter ranges**: Consider adjusting the ranges of the hyperparameters to avoid the problematic regions.