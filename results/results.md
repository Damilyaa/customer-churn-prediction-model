# results

## cv comparison (5-fold stratified)

| model               |   recall_mean |   precision_mean |   f1_mean |   roc_auc_mean |   recall_std |   precision_std |   f1_std |   roc_auc_std |
|:--------------------|--------------:|-----------------:|----------:|---------------:|-------------:|----------------:|---------:|--------------:|
| logistic_regression |         0.797 |            0.518 |     0.628 |          0.846 |        0.035 |           0.016 |    0.021 |         0.011 |
| random_forest       |         0.461 |            0.641 |     0.536 |          0.825 |        0.015 |           0.025 |    0.016 |         0.010 |
| gradient_boosting   |         0.522 |            0.665 |     0.584 |          0.848 |        0.020 |           0.034 |    0.021 |         0.011 |

## test set (frozen threshold)

| metric | value |
|---|---|
| threshold | 0.498 |
| recall | 0.794 |
| precision | 0.499 |
| f1 | 0.613 |
| roc-auc | 0.842 |

## confusion matrix

|                |   pred: no churn |   pred: churn |
|:---------------|-----------------:|--------------:|
| true: no churn |              737 |           298 |
| true: churn    |               77 |           297 |

## business interpretation

at threshold 0.50 the model flags ~422 customers per 1000, of whom ~210 are real churners → ~422 marketing calls per week to catch ~210 churners.
