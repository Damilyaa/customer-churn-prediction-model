import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    cross_val_predict, GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    recall_score, precision_score, f1_score, roc_auc_score,
    confusion_matrix, precision_recall_curve,
)

from preprocessing import FeatureEngineer, NUM_COLS, CAT_COLS

RANDOM_STATE = 42
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


# ── load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df['Churn'] = (df['Churn'] == 'Yes').astype(int)

X = df.drop(columns=['customerID', 'Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
)

print(f"train: {len(X_train)} rows  |  test: {len(X_test)} rows")
print(f"churn rate — train: {y_train.mean():.2%}  |  test: {y_test.mean():.2%}")


# ── build preprocessor ────────────────────────────────────────────────────────
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale',  StandardScaler()),
    ]), NUM_COLS),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_COLS),
])


def make_pipe(clf):
    return Pipeline([
        ('features', FeatureEngineer()),
        ('prep',     preprocessor),
        ('model',    clf),
    ])


# ── baseline ──────────────────────────────────────────────────────────────────
print("\n── dummy baseline ──")
for strategy in ('most_frequent', 'stratified'):
    dummy = make_pipe(DummyClassifier(strategy=strategy, random_state=RANDOM_STATE))
    dummy.fit(X_train, y_train)
    acc = (dummy.predict(X_test) == y_test).mean()
    print(f"  {strategy}: accuracy={acc:.3f}")

print("  → majority-class accuracy ~0.73 — accuracy is not useful here")


# ── cross-validation comparison ───────────────────────────────────────────────
classifiers = {
    'logistic_regression': LogisticRegression(
        class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE,
    ),
    'random_forest': RandomForestClassifier(
        class_weight='balanced', n_estimators=100, random_state=RANDOM_STATE,
    ),
    'gradient_boosting': GradientBoostingClassifier(
        n_estimators=100, random_state=RANDOM_STATE,
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
metrics = ['recall', 'precision', 'f1', 'roc_auc']

cv_results = {}
print("\n── 5-fold stratified cv ──")
for name, clf in classifiers.items():
    pipe = make_pipe(clf)
    row = {}
    for metric in metrics:
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=metric)
        row[metric] = scores
        print(f"  {name:25s}  {metric:12s}  mean={scores.mean():.3f}  std={scores.std():.3f}")
    cv_results[name] = row

# pick best by recall
best_name = max(cv_results, key=lambda n: cv_results[n]['recall'].mean())
print(f"\nbest model by recall: {best_name}")


# ── hyperparameter tuning on best model ───────────────────────────────────────
print("\n── grid search on best model ──")

if best_name == 'logistic_regression':
    param_grid = {
        'model__C': [0.01, 0.1, 1.0, 10.0],
        'model__solver': ['lbfgs', 'liblinear'],
    }
    best_clf = LogisticRegression(
        class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE,
    )
elif best_name == 'random_forest':
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 10, 20],
    }
    best_clf = RandomForestClassifier(
        class_weight='balanced', random_state=RANDOM_STATE,
    )
else:
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.05, 0.1],
    }
    best_clf = GradientBoostingClassifier(random_state=RANDOM_STATE)

best_pipe = make_pipe(best_clf)
gs = GridSearchCV(
    best_pipe,
    param_grid,
    cv=cv,
    scoring='recall',
    n_jobs=-1,
    verbose=1,
)
gs.fit(X_train, y_train)
print(f"best params: {gs.best_params_}")
print(f"best cv recall: {gs.best_score_:.3f}")

final_pipe = gs.best_estimator_


# ── threshold tuning on out-of-fold probabilities ────────────────────────────
print("\n── threshold tuning ──")
oof_proba = cross_val_predict(
    final_pipe, X_train, y_train, cv=cv, method='predict_proba',
)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_train, oof_proba)

# find threshold giving recall >= 0.80 with highest precision
mask = recalls[:-1] >= 0.80
if mask.any():
    best_idx = np.argmax(precisions[:-1][mask])
    candidates = np.where(mask)[0]
    frozen_threshold = float(thresholds[candidates[best_idx]])
    frozen_precision = float(precisions[:-1][mask][best_idx])
    frozen_recall    = float(recalls[:-1][mask][best_idx])
else:
    # fallback: best f1
    f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-9)
    idx = np.argmax(f1s)
    frozen_threshold = float(thresholds[idx])
    frozen_precision = float(precisions[idx])
    frozen_recall    = float(recalls[idx])

print(f"frozen threshold: {frozen_threshold:.3f}")
print(f"oof  — recall: {frozen_recall:.3f}  precision: {frozen_precision:.3f}")

# precision-recall curve plot
plt.figure(figsize=(7, 4))
plt.plot(recalls[:-1], precisions[:-1], color='steelblue', lw=1.5)
plt.axvline(x=frozen_recall, color='tomato', linestyle='--', label=f'threshold={frozen_threshold:.2f}')
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('precision-recall curve (out-of-fold)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'plots', 'precision_recall_curve.png'), dpi=120)
plt.close()


# ── refit final model on full train set and evaluate on test ──────────────────
final_pipe.fit(X_train, y_train)
test_proba = final_pipe.predict_proba(X_test)[:, 1]
y_pred = (test_proba >= frozen_threshold).astype(int)

test_recall    = recall_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_f1        = f1_score(y_test, y_pred)
test_roc_auc   = roc_auc_score(y_test, test_proba)

print("\n── test set results ──")
print(f"threshold:  {frozen_threshold:.3f}")
print(f"recall:     {test_recall:.3f}")
print(f"precision:  {test_precision:.3f}")
print(f"f1:         {test_f1:.3f}")
print(f"roc-auc:    {test_roc_auc:.3f}")

assert test_recall >= 0.75,    f"recall {test_recall:.3f} < 0.75 — threshold needs adjustment"
assert test_precision >= 0.45, f"precision {test_precision:.3f} < 0.45 — threshold needs adjustment"

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=['true: no churn', 'true: churn'],
    columns=['pred: no churn', 'pred: churn'],
)
print("\nconfusion matrix:")
print(cm_df)

# confusion matrix as png
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm_df, annot=True, fmt='d', cmap='Blues',
    linewidths=0.5, linecolor='gray', ax=ax,
)
ax.set_title(f'confusion matrix  (threshold={frozen_threshold:.2f})')
ax.set_ylabel('true label')
ax.set_xlabel('predicted label')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'plots', 'confusion_matrix.png'), dpi=120)
plt.close()
print("confusion matrix plot saved")


# ── business translation ──────────────────────────────────────────────────────
total_flagged = y_pred.sum()
true_positives = cm[1, 1]
per_1000 = int(total_flagged / len(y_pred) * 1000)
tp_per_1000 = int(true_positives / len(y_pred) * 1000)

business_note = (
    f"at threshold {frozen_threshold:.2f} the model flags ~{per_1000} customers per 1000, "
    f"of whom ~{tp_per_1000} are real churners "
    f"→ ~{per_1000} marketing calls per week to catch ~{tp_per_1000} churners."
)
print(f"\n{business_note}")


# ── save cv table ─────────────────────────────────────────────────────────────
rows = []
for name, row in cv_results.items():
    rows.append({
        'model': name,
        **{f'{m}_mean': row[m].mean() for m in metrics},
        **{f'{m}_std': row[m].std() for m in metrics},
    })
cv_df = pd.DataFrame(rows)

results_md = os.path.join(RESULTS_DIR, 'results.md')
with open(results_md, 'w') as f:
    f.write("# results\n\n")
    f.write("## cv comparison (5-fold stratified)\n\n")
    f.write(cv_df.to_markdown(index=False, floatfmt='.3f'))
    f.write("\n\n## test set (frozen threshold)\n\n")
    f.write(f"| metric | value |\n|---|---|\n")
    f.write(f"| threshold | {frozen_threshold:.3f} |\n")
    f.write(f"| recall | {test_recall:.3f} |\n")
    f.write(f"| precision | {test_precision:.3f} |\n")
    f.write(f"| f1 | {test_f1:.3f} |\n")
    f.write(f"| roc-auc | {test_roc_auc:.3f} |\n")
    f.write(f"\n## confusion matrix\n\n")
    f.write(cm_df.to_markdown())
    f.write(f"\n\n## business interpretation\n\n{business_note}\n")

print(f"\nresults written to {results_md}")


# ── save pipeline ─────────────────────────────────────────────────────────────
pipeline_path = os.path.join(RESULTS_DIR, 'churn_pipeline.pkl')
joblib.dump({'pipeline': final_pipe, 'threshold': frozen_threshold}, pipeline_path)
print(f"pipeline saved to {pipeline_path}")
