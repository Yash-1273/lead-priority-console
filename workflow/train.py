"""Train the booking model, learn the call-fatigue and fade shapes, and export what the console needs.

Run:  python workflow/train.py   then   python workflow/pipeline.py   (60-day test of new + ongoing leads)
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split
from engine import (HERE, NUM, WEEKLY_RENT, COMMISSION_RATE, P_HIGH, GBP_TO_INR, ANCILLARY_INR, VISA_GRANT, design,
                    from_crm, simulate_enquiry_context, commission_value, LeadPriorityEngine)
from pipeline import learn_shapes

ROOT = HERE.parent
raw = pd.read_csv(ROOT / 'data' / 'Leads.csv')
leads = simulate_enquiry_context(from_crm(raw))

tr, te = train_test_split(np.arange(len(leads)), test_size=0.3, random_state=42, stratify=leads.booked)
train, test = leads.iloc[tr], leads.iloc[te].reset_index(drop=True)

# 1. Booking model: arrival-time signals only
medians = {c: float(train[c].median()) for c in NUM}
X_train = design(train, medians=medians)
model = HistGradientBoostingClassifier(random_state=0).fit(X_train, train.booked)
value_cut = float(np.median([commission_value(c, r, w) for c, r, w in zip(train.city, train.room, train.weeks)]))
joblib.dump(model, HERE / 'model.joblib')
shapes = learn_shapes()      # call fatigue (bank calls) and fade over time (Olist deals), from real data
(HERE / 'config.json').write_text(json.dumps({'columns': list(X_train.columns), 'medians': medians,
                                              'value_cut': value_cut, 'shapes': shapes}, indent=1))
print('Call fatigue (chance vs 1st call):', shapes['call_mult'][:8])
print('Fade (share of chance left) at days', shapes['fade_days'][:8], '->', shapes['fade'][:8])

s = LeadPriorityEngine().score(test)      # held-out leads, never seen in training
auc, brier = roc_auc_score(s.booked, s.p_book), brier_score_loss(s.booked, s.p_book)
print(f'Booking model: test AUC {auc:.3f}, Brier {brier:.3f}; high-value cut-off GBP {value_cut:.0f}')

# 2. Export the exact tree model so the browser scores new leads with the same model
def export_trees(m):
    trees = []
    for it in m._predictors:
        trees.append([[int(x['feature_idx']), round(float(x['num_threshold']), 6), int(x['left']), int(x['right']),
                       int(x['missing_go_to_left']), int(x['is_leaf']), round(float(x['value']), 6)] for x in it[0].nodes])
    return {'baseline': float(np.ravel(m._baseline_prediction)[0]), 'trees': trees}


# 3. Today's new leads for the console: 92 held-out leads with known outcomes (ongoing leads come from pipeline.py)
inbox = s.sample(92, random_state=5)
cols = ['lead_id', 'home_city', 'channel_origin', 'channel_source', 'no_email', 'no_call', 'visits',
        'minutes_browsing', 'pages_per_visit', 'profile_fields_filled', 'city', 'room', 'weeks', 'university',
        'days_to_move_in', 'p_book', 'booked']
inbox = inbox[cols].assign(minutes_browsing=lambda d: d.minutes_browsing.round(1), p_book=lambda d: d.p_book.round(4),
                           visits=lambda d: d.visits.fillna(0), pages_per_visit=lambda d: d.pages_per_visit.fillna(0).round(1))
eng = LeadPriorityEngine()
inbox['p_book'] = eng.model.predict_proba(design(inbox, eng.cfg['columns'], eng.cfg['medians']))[:, 1].round(4)
bundle = {
    'model': export_trees(model), 'columns': list(X_train.columns), 'medians': medians, 'value_cut': value_cut,
    'rent': [{'city': c, 'room': r, 'weekly': v} for (c, r), v in WEEKLY_RENT.items()],
    'commission_rate': COMMISSION_RATE, 'p_high': P_HIGH, 'fx': GBP_TO_INR, 'ancillary_inr': ANCILLARY_INR,
    'visa_grant': VISA_GRANT, 'shapes': {k: shapes[k] for k in ('call_mult', 'fade_days', 'fade', 'sources')},
    'metrics': {'auc': auc, 'brier': brier, 'test_leads': len(s), 'train_leads': len(train), 'per_day': len(s) / 30},
    'inbox': inbox.to_dict('records'),
}
(HERE / 'console_data.json').write_text(json.dumps(bundle, separators=(',', ':'), default=float))
print(f'Exported console_data.json ({(HERE / "console_data.json").stat().st_size / 1024:.0f} KB)')
