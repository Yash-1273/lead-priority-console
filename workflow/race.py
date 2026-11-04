"""Day-by-day results for the console's "Run the season" race, plus fresh enquiries for the live-events demo.

For each team size, the same 60 days are replayed under each way of working (same leads, same calls, same 5-call
limit); cumulative revenue and bookings are recorded every day and averaged over 3 runs.

Run after pipeline.py:  python workflow/race.py   (writes race.json; takes a few minutes)
"""
import json
import numpy as np
from engine import HERE, CALL_LIMIT, VISA_GRANT, design, LeadPriorityEngine
from pipeline import held_out_pool, rank_key
import pandas as pd

RULES = ['new_first', 'standard', 'priority', 'fade_only']
CAPS = [10, 25, 40, 60]
DAYS = 60


def daily(pool, shapes, rule, cap, seed, peak=False, gap=2):
    """Same mechanics as pipeline.simulate, recording cumulative revenue (GBP) and bookings each day."""
    rng = np.random.default_rng(seed)
    need_p = np.asarray(shapes['attempt_dist']) / np.sum(shapes['attempt_dist'])
    rate = np.full(30, len(pool) / 30)
    if peak:
        rate[10:20] *= 2
    parts = []
    for day, r in enumerate(rate):
        a = pool.sample(rng.poisson(r), replace=True, random_state=int(rng.integers(1e9))).copy()
        a['arrive'] = day
        a['need'] = rng.choice(np.arange(1, len(need_p) + 1), len(a), p=need_p)
        a['window'] = rng.choice(shapes['days_to_close'], len(a))
        parts.append(a)
    L = pd.concat(parts, ignore_index=True)
    L['attempts'], L['done'], L['last_call'] = 0, False, -99
    rev, bk, out_r, out_b = 0.0, 0, [], []
    for day in range(DAYS):
        live = L[(~L.done) & (L.arrive <= day) & (day - L.last_call >= gap) & (L.attempts < CALL_LIMIT)]
        open_days = day - live.arrive
        for i in rank_key(rule, live, open_days, shapes).sort_values(ascending=False).index[:cap]:
            L.at[i, 'attempts'] += 1
            L.at[i, 'last_call'] = day
            moved_on = day - L.at[i, 'arrive'] > L.at[i, 'window']
            if L.at[i, 'booked'] == 1 and not moved_on and L.at[i, 'attempts'] >= L.at[i, 'need']:
                L.at[i, 'done'] = True
                bk += 1
                rev += L.at[i, 'value'] * VISA_GRANT
            elif moved_on:
                L.at[i, 'done'] = True
        out_r.append(round(rev, 1))
        out_b.append(bk)
    return out_r, out_b


if __name__ == '__main__':
    shapes = json.loads((HERE / 'config.json').read_text())['shapes']
    pool = held_out_pool()
    out = {'caps': CAPS, 'days': DAYS, 'normal': {}, 'peak': {}}
    for season in ('normal', 'peak'):
        for cap in CAPS:
            out[season][str(cap)] = {}
            for rule in RULES:
                runs = [daily(pool, shapes, rule, cap, seed=s, peak=season == 'peak') for s in range(3)]
                out[season][str(cap)][rule] = {'revenue': np.mean([r[0] for r in runs], axis=0).round(1).tolist(),
                                               'bookings': np.mean([r[1] for r in runs], axis=0).round(1).tolist()}
            r = out[season][str(cap)]
            print(f"{season:6} {cap:2} calls: priority {r['priority']['revenue'][-1] / r['standard']['revenue'][-1] - 1:+.1%} revenue, "
                  f"fade-only {r['fade_only']['bookings'][-1] / r['standard']['bookings'][-1] - 1:+.1%} bookings")

    # Fresh enquiries for the console's "New enquiry arrives" button: held-out leads not already on screen
    seen = {l['lead_id'] for l in json.loads((HERE / 'console_data.json').read_text())['inbox']}
    seen |= {l['lead_id'] for l in json.loads((HERE / 'pipeline.json').read_text())['ongoing']}
    fresh = pool[~pool.lead_id.isin(seen)].drop_duplicates('lead_id').sample(12, random_state=21)
    cols = ['lead_id', 'home_city', 'channel_origin', 'channel_source', 'no_email', 'no_call', 'visits', 'minutes_browsing',
            'pages_per_visit', 'profile_fields_filled', 'city', 'room', 'weeks', 'university', 'days_to_move_in', 'booked']
    fresh = fresh[cols].assign(minutes_browsing=lambda d: d.minutes_browsing.round(1), visits=lambda d: d.visits.fillna(0),
                               pages_per_visit=lambda d: d.pages_per_visit.fillna(0).round(1))
    eng = LeadPriorityEngine()
    fresh['p_book'] = eng.model.predict_proba(design(fresh, eng.cfg['columns'], eng.cfg['medians']))[:, 1].round(4)
    out['fresh'] = fresh.to_dict('records')
    (HERE / 'race.json').write_text(json.dumps(out, separators=(',', ':'), default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    print('Saved race.json')
