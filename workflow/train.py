"""Train the booking model, test ranking rules in a 30-day simulation, and export what the console needs.

Run:  python workflow/train.py
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split
from engine import (HERE, NUM, WEEKLY_RENT, COMMISSION_RATE, P_HIGH, GBP_TO_INR, ANCILLARY_INR, design, from_crm, daily_loss_rate,
                    simulate_enquiry_context, commission_value, LeadPriorityEngine)

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
(HERE / 'config.json').write_text(json.dumps({'columns': list(X_train.columns), 'medians': medians,
                                              'value_cut': value_cut}, indent=1))

s = LeadPriorityEngine().score(test)      # held-out leads, never seen in training
auc, brier = roc_auc_score(s.booked, s.p_book), brier_score_loss(s.booked, s.p_book)
print(f'Booking model: test AUC {auc:.3f}, Brier {brier:.3f}; high-value cut-off GBP {value_cut:.0f}')

# 2. 30-day simulation. Held-out leads arrive over 30 days (~92/day). Each day the team calls `cap` leads,
#    chosen by a ranking rule. A called lead books if it really booked (its true label). An uncalled lead
#    waits in the queue but may be lost each day (daily_loss_rate). Only revenue (commission + extras) from called leads counts.
RULES = {
    'arrival': lambda q: -q.arrive,                                             # first come, first served
    'probability': lambda q: q.p_book,                                          # standard lead scoring
    'priority': lambda q: q.p_book * q.value,                                   # ours: expected revenue
    'with_urgency': lambda q: q.p_book * q.value / (1 + q.days_to_move_in / 60),       # tested, rejected
}


def simulate(rule, cap, days=30, seed=0):
    rng = np.random.default_rng(seed)
    pool = s.sample(frac=1, random_state=seed).reset_index(drop=True)
    pool['arrive'] = np.arange(len(pool)) * days // len(pool)
    queue, commission, bookings = pool.iloc[:0], 0.0, 0
    for day in range(days):
        queue = pd.concat([queue, pool[pool.arrive == day]])
        called = queue.loc[RULES[rule](queue).sort_values(ascending=False).index[:cap]]
        commission += float((called.booked * called.value).sum())
        bookings += int(called.booked.sum())
        queue = queue.drop(called.index).assign(days_to_move_in=lambda q: q.days_to_move_in - 1)
        queue = queue[rng.random(len(queue)) >= daily_loss_rate(queue.days_to_move_in.values)]
    return commission, bookings


def run(rule, cap, seeds=5):
    r = np.mean([simulate(rule, cap, seed=k) for k in range(seeds)], axis=0)
    return {'commission': float(r[0]), 'bookings': float(r[1])}


CAPS = [10, 20, 25, 30, 40, 50, 60, 70, 80, 90]
sim = {rule: {cap: run(rule, cap) for cap in CAPS} for rule in RULES}
for cap in (25, 50):
    base = sim['probability'][cap]['commission']
    print(f'{cap} calls/day: ' + ' | '.join(f'{r} GBP {sim[r][cap]["commission"]/1e3:.0f}k ({sim[r][cap]["commission"]/base-1:+.1%}), '
                                          f'{sim[r][cap]["bookings"]:.0f} bookings' for r in RULES))


# 3. Export the exact tree model so the browser scores new leads with the same model
def export_trees(m):
    trees = []
    for it in m._predictors:
        trees.append([[int(x['feature_idx']), round(float(x['num_threshold']), 6), int(x['left']), int(x['right']),
                       int(x['missing_go_to_left']), int(x['is_leaf']), round(float(x['value']), 6)] for x in it[0].nodes])
    return {'baseline': float(np.ravel(m._baseline_prediction)[0]), 'trees': trees}


# 4. Today's inbox for the console: 92 held-out leads (one day's volume) with known outcomes for the replay
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
    'metrics': {'auc': auc, 'brier': brier, 'test_leads': len(s), 'train_leads': len(train), 'per_day': len(s) / 30},
    'sim': {r: [[c, v['commission'], v['bookings']] for c, v in d.items()] for r, d in sim.items()},
    'inbox': inbox.to_dict('records'),
}
(HERE / 'console_data.json').write_text(json.dumps(bundle, separators=(',', ':'), default=float))
print(f'Exported console_data.json ({(HERE / "console_data.json").stat().st_size / 1024:.0f} KB)')
