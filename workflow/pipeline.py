"""Ongoing leads: learn how chances fade over time, and test the one-queue workflow over 60 days.

A booking takes days to weeks, so every day the team chooses between new leads and ongoing ones. This module:
  1. learns from real data
       fade          Olist marketing funnel (8,000 leads): share of eventual deals still to come after d days open
       call limit    UCI Bank Marketing (41,188 customers called repeatedly): the k-th call's conversion rate, given
                     no "yes" before it. By the 5th call it is under half the 1st, so leads go to nurture after 5.
  2. simulates 60 days: held-out X Education leads arrive daily for 30 days; the team makes `cap` calls a day;
     a booker converts once they've had the calls they need, if they're still looking (otherwise they've moved on).
     EVERY way of working gets the same calls, the same follow-up spacing and the same 5-call limit.
  3. stress-tests the rule, including worlds where students behave differently from the curves it assumes.

Tested and rejected (see `rejected` in pipeline.json):
  move-in urgency       ranks students moving in soon first; pushes strong leads out of the queue
  precise call fatigue  lowers each lead's chance per unanswered call using the bank curve; wins when students behave
                        like the bank's customers, loses if they need one more call. The simple 5-call limit is robust.

Run after train.py:  python workflow/pipeline.py   (writes pipeline.json; takes several minutes)
"""
import json
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from engine import (HERE, GBP_TO_INR, VISA_GRANT, CALL_LIMIT, design, from_crm, simulate_enquiry_context,
                    LeadPriorityEngine, fade_factor)

DATA = HERE.parent / 'data'
MAX_CALLS = 12          # the bank data covers up to 12 calls per customer in one campaign
FADE_DAYS = [0, 3, 7, 14, 21, 30, 45, 60, 90, 120, 180, 365]


def learn_shapes():
    """Fade and call curves from the two real datasets, plus the raw distributions the simulation draws from."""
    b = pd.read_csv(DATA / 'bank_marketing_calls.csv', sep=';')
    yes = b.y == 'yes'                       # 'duration' is only known after a call, so it is never used
    k = np.arange(1, MAX_CALLS + 1)
    reached = np.array([(b.campaign >= i).sum() for i in k])
    converted = np.array([((b.campaign == i) & yes).sum() for i in k])
    hazard = IsotonicRegression(increasing=False).fit(k, converted / reached, sample_weight=reached).predict(k)

    leads = pd.read_csv(DATA / 'olist_leads.csv', parse_dates=['first_contact_date'])
    deals = pd.read_csv(DATA / 'olist_closed_deals.csv', parse_dates=['won_date'])
    won = leads.merge(deals, on='mql_id')
    won = won[won.first_contact_date <= deals.won_date.max() - pd.Timedelta(days=180)]   # leads with time to close
    days_to_close = (won.won_date - won.first_contact_date).dt.days.clip(lower=1).values

    return {
        'call_mult': [round(float(x), 4) for x in hazard / hazard[0]],        # k-th call vs the 1st (index 0 = 1st call)
        'call_hazard': [round(float(x), 5) for x in hazard],
        'attempt_dist': [round(float(x), 5) for x in converted / converted.sum()],
        'fade_days': FADE_DAYS,
        'fade': [round(float((days_to_close > d).mean()), 4) for d in FADE_DAYS],
        'days_to_close': days_to_close.tolist(),
        'sources': {'bank_customers': int(len(b)), 'olist_leads': int(len(leads)), 'olist_wins_used': int(len(won))},
    }


def held_out_pool(extras_inr=None):
    raw = pd.read_csv(DATA / 'Leads.csv')
    leads = simulate_enquiry_context(from_crm(raw))
    _, te = train_test_split(np.arange(len(leads)), test_size=0.3, random_state=42, stratify=leads.booked)
    kw = {} if extras_inr is None else {'extras_inr': extras_inr}
    return LeadPriorityEngine().score(leads.iloc[te].reset_index(drop=True), **kw)


def rank_key(rule, live, open_days, shapes):
    """Higher = called first. Every rule sees the same leads (same spacing, same 5-call limit)."""
    if rule == 'arrival':
        return -(live.arrive * 100 + live.attempts)                     # oldest first
    if rule == 'new_first':
        return (live.attempts == 0) * 10 + live.p_book                  # new leads first, then by chance
    if rule == 'standard':
        return live.p_book                                              # standard lead scoring
    if rule == 'value_only':
        return live.p_book * live.value                                 # ablation
    if rule == 'fade_only':
        return live.p_book * fade_factor(open_days, shapes)             # ablation
    key = live.p_book * fade_factor(open_days, shapes) * live.value     # PRIORITY: chance now x value
    if rule == 'with_urgency':                                          # rejected: move-in urgency on top
        key = key / (1 + (live.days_to_move_in - open_days).clip(lower=0) / 60)
    if rule == 'with_call_fatigue':                                     # rejected: precise per-call fatigue
        key = key * np.asarray(shapes['call_mult'])[np.minimum(live.attempts, MAX_CALLS - 1)]
    return key


def simulate(pool, shapes, rule, cap, seed=0, days=60, arrive_days=30, peak=False, followup_cost=1.0, gap=2,
             stop_day=None, need_shift=0, window_scale=1.0):
    """Returns (revenue in GBP after visa risk, bookings). With stop_day, returns the queue state that morning.

    need_shift / window_scale make the simulated students behave DIFFERENTLY from the curves the rule assumes
    (more or fewer calls needed, longer or shorter interest), so the rule can't win by construction."""
    rng = np.random.default_rng(seed)
    need_p = np.asarray(shapes['attempt_dist']) / np.sum(shapes['attempt_dist'])   # rounding-safe
    rate = np.full(arrive_days, len(pool) / arrive_days)
    if peak:
        rate[10:20] *= 2                                             # a 10-day peak: twice the enquiries
    parts = []
    for day, r in enumerate(rate):
        a = pool.sample(rng.poisson(r), replace=True, random_state=int(rng.integers(1e9))).copy()
        a['arrive'] = day
        a['need'] = np.clip(rng.choice(np.arange(1, MAX_CALLS + 1), len(a), p=need_p) + need_shift, 1, MAX_CALLS)
        a['window'] = rng.choice(shapes['days_to_close'], len(a)) * window_scale   # days the student keeps looking
        parts.append(a)
    L = pd.concat(parts, ignore_index=True)
    L['attempts'], L['done'], L['last_call'] = 0, False, -99
    revenue, bookings = 0.0, 0
    for day in range(days):
        live = L[(~L.done) & (L.arrive <= day) & (day - L.last_call >= gap) & (L.attempts < CALL_LIMIT)]
        open_days = day - live.arrive
        if stop_day is not None and day == stop_day:
            return live.assign(days_open=open_days, moved_on=open_days > live.window)
        budget = cap
        for i in rank_key(rule, live, open_days, shapes).sort_values(ascending=False).index:
            cost = 1.0 if L.at[i, 'attempts'] == 0 else followup_cost
            if budget < cost:
                break
            budget -= cost
            L.at[i, 'attempts'] += 1
            L.at[i, 'last_call'] = day
            moved_on = day - L.at[i, 'arrive'] > L.at[i, 'window']
            if L.at[i, 'booked'] == 1 and not moved_on and L.at[i, 'attempts'] >= L.at[i, 'need']:
                L.at[i, 'done'] = True
                bookings += 1
                revenue += L.at[i, 'value'] * VISA_GRANT
            elif moved_on:
                L.at[i, 'done'] = True
    return revenue, bookings


def mean_run(pool, shapes, rule, cap, seeds=3, **kw):
    r = np.mean([simulate(pool, shapes, rule, cap, seed=s, **kw) for s in range(seeds)], axis=0)
    return float(r[0]), float(r[1])


def gain(pool, shapes, rule='priority', cap=25, seeds=3, **kw):
    std = mean_run(pool, shapes, 'standard', cap, seeds, **kw)[0]
    return round(mean_run(pool, shapes, rule, cap, seeds, **kw)[0] / std - 1, 4)


if __name__ == '__main__':
    shapes = json.loads((HERE / 'config.json').read_text())['shapes']
    pool = held_out_pool()
    out = {'caps': [10, 20, 25, 30, 40, 50, 60, 90], 'curve': {}, 'call_limit': CALL_LIMIT}

    print('1/5 Capacity curve (60 days, new + ongoing leads, everyone stops after 5 calls)')
    for rule in ('arrival', 'new_first', 'standard', 'priority', 'with_urgency'):
        out['curve'][rule] = [mean_run(pool, shapes, rule, c) for c in out['caps']]
    i25 = out['caps'].index(25)
    for rule, v in out['curve'].items():
        print(f'   {rule:13} at 25 calls: Rs {v[i25][0] * GBP_TO_INR / 1e5:6.1f} lakh, {v[i25][1]:.0f} bookings, '
              f'{v[i25][0] / out["curve"]["standard"][i25][0] - 1:+.1%} vs standard')

    print('2/5 Where the gain comes from (25 calls)')
    out['parts'] = {r: mean_run(pool, shapes, r, 25) for r in ('standard', 'value_only', 'fade_only', 'priority')}
    for r, v in out['parts'].items():
        print(f'   {r:13} {v[0] / out["parts"]["standard"][0] - 1:+.1%}')

    print('3/5 Robustness (25 calls)')
    out['robust'] = [
        {'name': 'Normal season', 'gain': gain(pool, shapes, seeds=5)},
        {'name': 'Peak season (2x enquiries)', 'gain': gain(pool, shapes, peak=True)},
        {'name': 'Follow-ups take half the time', 'gain': gain(pool, shapes, followup_cost=0.5)},
        {'name': 'Follow-up every day', 'gain': gain(pool, shapes, gap=1)},
        {'name': 'Follow-up every 3 days', 'gain': gain(pool, shapes, gap=3)},
        {'name': 'No extras (commission only)', 'gain': gain(held_out_pool(extras_inr=0), shapes)},
        {'name': 'Extras doubled to Rs 30,000', 'gain': gain(held_out_pool(extras_inr=30_000), shapes)},
    ]
    for r in out['robust']:
        print(f"   {r['name']:32} {r['gain']:+.1%}")

    print('4/5 Students behave differently from what the rule assumes (25 calls)')
    worlds = [('Need 1 more call', dict(need_shift=1)), ('Need 1 fewer call', dict(need_shift=-1)),
              ('Stay interested 2x longer', dict(window_scale=2)), ('Stay interested half as long', dict(window_scale=0.5)),
              ('More calls and shorter interest', dict(need_shift=1, window_scale=0.5))]
    out['mismatch'] = []
    for name, kw in worlds:
        row = {'name': name, 'priority': gain(pool, shapes, **kw), 'with_call_fatigue': gain(pool, shapes, 'with_call_fatigue', **kw)}
        out['mismatch'].append(row)
        print(f"   {name:32} priority {row['priority']:+.1%} | precise call fatigue {row['with_call_fatigue']:+.1%}")
    out['rejected'] = {
        'with_urgency': gain(pool, shapes, 'with_urgency'),
        'with_call_fatigue_normal': gain(pool, shapes, 'with_call_fatigue'),
        'with_call_fatigue_worst': min(r['with_call_fatigue'] for r in out['mismatch']),
        'priority_worst': min(r['priority'] for r in out['mismatch']),
    }
    print('   rejected:', out['rejected'])

    print("5/5 Today's ongoing leads for the console (queue state on day 30, priority rule)")
    inbox_ids = {l['lead_id'] for l in json.loads((HERE / 'console_data.json').read_text())['inbox']}
    state = simulate(pool, shapes, 'priority', 25, seed=0, stop_day=30)
    ongoing = state[(state.attempts > 0) & ~state.lead_id.isin(inbox_ids)].drop_duplicates('lead_id')
    ongoing = ongoing.sample(min(40, len(ongoing)), random_state=3)
    cols = ['lead_id', 'home_city', 'channel_origin', 'channel_source', 'no_email', 'no_call', 'visits',
            'minutes_browsing', 'pages_per_visit', 'profile_fields_filled', 'city', 'room', 'weeks', 'university',
            'days_to_move_in', 'p_book', 'booked', 'attempts', 'days_open', 'moved_on']
    ongoing = ongoing[cols].assign(minutes_browsing=lambda d: d.minutes_browsing.round(1),
                                   visits=lambda d: d.visits.fillna(0), pages_per_visit=lambda d: d.pages_per_visit.fillna(0).round(1),
                                   days_to_move_in=lambda d: (d.days_to_move_in - d.days_open).clip(lower=1))
    eng = LeadPriorityEngine()          # score from the exported (rounded) values, exactly as the console and CLI see them
    ongoing['p_book'] = eng.model.predict_proba(design(ongoing, eng.cfg['columns'], eng.cfg['medians']))[:, 1].round(4)
    out['ongoing'] = ongoing.to_dict('records')
    print(f'   {len(ongoing)} ongoing leads ({int(ongoing.moved_on.sum())} have already moved on, unknown to the team)')

    (HERE / 'pipeline.json').write_text(json.dumps(out, separators=(',', ':'), default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    print('Saved pipeline.json')
