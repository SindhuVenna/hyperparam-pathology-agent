# Hyperparameter Sweep Pathology Report

## Overview
- The sweep contained 5 issues across 3 trials.
- The most severe issue types were "nan_or_inf_metric" with 2 high-severity issues and "overfitting_suspect" with 2 medium-severity issues.
- The average severity of issues was medium.

## Issue Breakdown
| Issue Type | Count | Example Trial IDs |
| --- | --- | --- |
| nan_or_inf_metric | 2 | 2, 4 |
| overfitting_suspect | 2 | 3, 4 |
| short_run | 1 | 2 |

## Hyperparameter Pathologies

### Learning Rate above 0.01 Frequently Leads to NaN Loss
The correlation between learning rate and NaN loss suggests that high learning rates are problematic. Specifically, buckets `(0.01, 0.02]` and `(0.005, 0.01]` have issue rates of 1.0, indicating that trials in these ranges are consistently experiencing NaN loss. For example, trial_id 2 had a learning rate of 0.01, while trial_id 4 had a learning rate of 0.02, both resulting in NaN loss.

* In the 'lr' correlation, buckets `(0.01, 0.02]` and `(0.005, 0.01]` have issue rates of 1.0.
* Trial_id 2 had a learning rate of 0.01 and trial_id 4 had a learning rate of 0.02, both experiencing NaN loss.

### Small Batch Sizes Correlate with Short Runs
The correlation between batch size and short runs suggests that small batch sizes lead to shorter training times. Specifically, buckets `(7.999, 16.0]` and `(16.0, 32.0]` have issue rates of 0.5 and 0.5, respectively. In contrast, bucket `(64.0, 128.0]` has an issue rate of 0.0. This indicates that batch sizes above 32 tend to result in longer training times.

* In the 'batch_size' correlation, buckets `(7.999, 16.0]` and `(16.0, 32.0]` have issue rates of 0.5 and 0.5, respectively.
* Bucket `(64.0, 128.0]` has an issue rate of 0.0, indicating that batch sizes above 32 tend to result in longer training times.

### Zero Weight Decay Causes Strong Overfitting on Small Datasets
The issue with zero weight decay on small datasets suggests that a lack of regularization leads to overfitting. Specifically, trial_id 3 had a weight decay of 0.0 and an extremely high val_loss/train_loss ratio of 5.33, indicating strong overfitting.

* Trial_id 3 had a weight decay of 0.0 and an extremely high val_loss/train_loss ratio of 5.33.
* The description of the overfitting_suspect issue type mentions a val_loss/train_loss ratio of inf for trial_id 4, which also had a weight decay of 0.0.

## Recommendations

- **Avoid learning rates above 0.01**: The correlation between learning rate and NaN loss suggests that high learning rates are problematic. Avoid setting learning rates above 0.01 to prevent NaN loss.
- **Increase batch size**: The correlation between batch size and short runs suggests that small batch sizes lead to shorter training times. Increase batch size above 32 to result in longer training times.
- **Use non-zero weight decay**: The issue with zero weight decay on small datasets suggests that a lack of regularization leads to overfitting. Use a non-zero weight decay, e.g., 0.0001, to prevent overfitting.
- **Monitor trials with high val_loss/train_loss ratios**: The description of the overfitting_suspect issue type mentions a val_loss/train_loss ratio of inf for trial_id 4, which also had a weight decay of 0.0. Monitor trials with high val_loss/train_loss ratios to prevent overfitting.
- **Use gradient clipping**: Gradient clipping can help prevent exploding gradients, which can cause NaN loss. Use gradient clipping to prevent NaN loss.
- **Constrain batch sizes**: Constrain batch sizes to a specific range, e.g., `(32, 128)`, to prevent short runs.
- **Adjust regularization**: Adjust regularization, e.g., weight decay, to prevent overfitting.