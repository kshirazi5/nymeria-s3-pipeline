Rows: 841 | Participants: 197 | Positive rate: 0.600

Original V2 setup, LightGBM test ROC-AUC: 0.990

## Ablation (LightGBM)

| setup | accuracy | f1 | roc_auc | roc_auc_std |
|---|---|---|---|---|
| Original features, random CV | 0.941 | 0.95 | 0.984 | 0.009 |
| Original features, participant-grouped CV | 0.944 | 0.953 | 0.984 | 0.005 |
| Leakage-free features, random CV | 0.817 | 0.849 | 0.884 | 0.036 |
| Leakage-free features, participant-grouped CV | 0.81 | 0.842 | 0.883 | 0.011 |

## Corrected comparison

| model | accuracy | f1 | roc_auc | roc_auc_std |
|---|---|---|---|---|
| Majority baseline | 0.6 | 0.75 | 0.5 | 0.0 |
| Random Forest | 0.819 | 0.848 | 0.885 | 0.017 |
| XGBoost | 0.805 | 0.839 | 0.888 | 0.012 |
| LightGBM | 0.81 | 0.842 | 0.883 | 0.011 |
