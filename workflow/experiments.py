"""Stress-test the model and the priority rule. Results go to experiments.json (shown in the console).

1. Can a different algorithm or more features beat the model?   (5-fold cross-validation)
2. How much does each signal matter?                            (ablation: remove it, measure the drop)

Run after train.py:  python workflow/experiments.py
"""
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from engine import HERE, design, from_crm

warnings.filterwarnings('ignore')
raw = pd.read_csv(HERE.parent / 'data' / 'Leads.csv')
L = from_crm(raw)
y = L.booked.values
cv = StratifiedKFold(5, shuffle=True, random_state=0)


def evaluate(X, model=None):
    model = HistGradientBoostingClassifier(random_state=0) if model is None else model
    p = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
    top = y[np.argsort(-p)][:int(.2 * len(y))]
    return {'auc': round(float(roc_auc_score(y, p)), 3), 'top20_precision': round(float(top.mean()), 3)}


X = design(L)
R = raw.replace('Select', np.nan)
form = X.copy()
for c in ['What is your current occupation', 'Specialization', 'City', 'Country']:
    form = form.join(pd.get_dummies(R[c].fillna('Unknown'), prefix=c[:12]).astype(float))

results = {'models': [
    {'name': 'Current model (gradient boosting)', **evaluate(X)},
    {'name': 'Logistic regression', **evaluate(X, LogisticRegression(max_iter=3000))},
    {'name': 'Random forest', **evaluate(X, RandomForestClassifier(400, min_samples_leaf=5, random_state=0, n_jobs=-1))},
    {'name': 'Tuned gradient boosting', **evaluate(X, HistGradientBoostingClassifier(
        learning_rate=.05, max_iter=300, l2_regularization=1, min_samples_leaf=30, random_state=0))},
], 'form_answers': {'without': evaluate(X), 'with': evaluate(form)},
   'ablations': [
    {'name': 'All signals', **evaluate(X)},
    {'name': 'Without form completeness', **evaluate(X.drop(columns=['profile_fields_filled']))},
    {'name': 'Without browsing behaviour', **evaluate(X.drop(columns=['visits', 'minutes_browsing', 'pages_per_visit']))},
    {'name': 'Channel only', **evaluate(X[[c for c in X if c.startswith('channel_')]])},
]}
for m in results['models']:
    print(f"{m['name']:36} AUC {m['auc']}  top-20% precision {m['top20_precision']}")
print('Form answers:', results['form_answers'])

# The priority rule's revenue tests (assumptions, extras, peak season) live in pipeline.py.

(HERE / 'experiments.json').write_text(json.dumps(results, indent=1))
print('Saved experiments.json')
