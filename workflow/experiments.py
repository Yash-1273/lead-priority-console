"""Stress-test the model and the priority rule. Results go to experiments.json (shown in the console).

1. Can a different algorithm or more features beat the model?   (5-fold cross-validation)
2. How much does each signal matter?                            (ablation: remove it, measure the drop)
3. Does the priority rule still win if the simulation's assumptions change?

Run after train.py:  python workflow/experiments.py
"""
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from engine import HERE, GBP_TO_INR, design, from_crm, daily_loss_rate, simulate_enquiry_context, LeadPriorityEngine

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

# Sensitivity of the 30-day simulation to the lead-loss assumption (25 calls a day)
leads = simulate_enquiry_context(L)
_, te = train_test_split(np.arange(len(leads)), test_size=0.3, random_state=42, stratify=leads.booked)
s = LeadPriorityEngine().score(leads.iloc[te].reset_index(drop=True))


def sim(by_value, mult, seed, cap=25, extras_inr=15_000):
    rng = np.random.default_rng(seed)
    pool = s.sample(frac=1, random_state=seed).reset_index(drop=True)
    pool['value'] = pool.commission + extras_inr / GBP_TO_INR
    pool['arrive'] = np.arange(len(pool)) * 30 // len(pool)
    q, earned = pool.iloc[:0], 0.0
    for day in range(30):
        q = pd.concat([q, pool[pool.arrive == day]])
        key = q.p_book * q.value if by_value else q.p_book
        called = q.loc[key.sort_values(ascending=False).index[:cap]]
        earned += float((called.booked * called.value).sum())
        q = q.drop(called.index).assign(days_to_move_in=lambda d: d.days_to_move_in - 1)
        q = q[rng.random(len(q)) >= np.minimum(1, mult * daily_loss_rate(q.days_to_move_in.values))]
    return earned


results['sensitivity'] = []
for mult in (0.25, 0.5, 1, 2, 4):
    std, prio = (np.mean([sim(v, mult, k) for k in range(5)]) for v in (False, True))
    results['sensitivity'].append({'loss_multiplier': mult, 'gain': round(prio / std - 1, 4)})
    print(f'Lead-loss rate x{mult}: priority vs standard {prio / std - 1:+.1%}')

# How much does counting per-student extras shrink the gain? (25 calls a day)
results['extras'] = []
for extras in (0, 7_500, 15_000, 30_000):
    std, prio = (np.mean([sim(v, 1, k, extras_inr=extras) for k in range(5)]) for v in (False, True))
    results['extras'].append({'extras_inr': extras, 'gain': round(prio / std - 1, 4)})
    print(f'Extras Rs {extras:,} per booking: priority vs standard {prio / std - 1:+.1%}')

(HERE / 'experiments.json').write_text(json.dumps(results, indent=1))
print('Saved experiments.json')
