"""Two CTO questions, answered with the same 60-day test.

1. Break-even: the property each lead enquired on is simulated independently of whether they booked. In reality,
   students enquiring about expensive rooms might book LESS often (price sensitivity), and the model wouldn't know.
   Here, a share of high-value bookers is turned into non-bookers, and the test re-run. How much less often would
   expensive-room enquirers have to book before Lead Desk stops beating standard scoring?

2. Pilot size: a real A/B test (half the leads routed by Lead Desk, half as today). How many weeks of leads would it
   take to detect the gain? Estimated from the 60-day test's day-to-day variation.

Run after pipeline.py:  python workflow/breakeven.py   (adds 'breakeven' and 'pilot' to pipeline.json)
"""
import json
import numpy as np
from engine import HERE, GBP_TO_INR
from pipeline import held_out_pool, simulate
from race import daily

shapes = json.loads((HERE / 'config.json').read_text())['shapes']
value_cut = json.loads((HERE / 'config.json').read_text())['value_cut']
pool = held_out_pool()
high = pool.commission >= value_cut          # the "higher value" half used for routing
print(f'High-value leads: {high.mean():.0%} of leads; real booking rate {pool.booked[high].mean():.1%} vs {pool.booked[~high].mean():.1%} for the rest')


def price_sensitive(pool, drop, seed=0):
    """Copy of the pool where high-value enquirers book `drop` (e.g. 0.2 = 20%) less often than in the data."""
    rng = np.random.default_rng(seed)
    p = pool.copy()
    flip = high & (p.booked == 1) & (rng.random(len(p)) < drop)
    p.loc[flip, 'booked'] = 0
    return p


def gain(p, cap=25, seeds=3):
    std = np.mean([simulate(p, shapes, 'standard', cap, seed=s)[0] for s in range(seeds)])
    pri = np.mean([simulate(p, shapes, 'priority', cap, seed=s)[0] for s in range(seeds)])
    return pri / std - 1


rows = []
for drop in (0, 0.1, 0.2, 0.3, 0.4, 0.5):
    p = price_sensitive(pool, drop)
    g = gain(p)
    rows.append({'drop': drop, 'gain': round(float(g), 4),
                 'high_rate': round(float(p.booked[high].mean()), 4), 'low_rate': round(float(p.booked[~high].mean()), 4)})
    print(f'High-value enquirers book {drop:.0%} less often ({rows[-1]["high_rate"]:.1%} vs {rows[-1]["low_rate"]:.1%}): '
          f'Lead Desk {g:+.1%} vs standard')
be = None
for a, b in zip(rows, rows[1:]):
    if a['gain'] > 0 >= b['gain']:
        be = a['drop'] + (b['drop'] - a['drop']) * a['gain'] / (a['gain'] - b['gain'])
print('Break-even:', f'{be:.0%} less often' if be is not None else 'not reached within 50%')

# Pilot size: daily revenue difference between the two ways of working, same days, same leads
days = 30
d = []
for s in range(6):
    pr, _ = daily(pool, shapes, 'priority', 25, seed=s)
    st, _ = daily(pool, shapes, 'standard', 25, seed=s)
    pr_day, st_day = np.diff([0] + pr[:days]), np.diff([0] + st[:days])
    d.append((pr_day, st_day))
pr_all = np.concatenate([x[0] for x in d]); st_all = np.concatenate([x[1] for x in d])
lift = pr_all.mean() / st_all.mean() - 1
sd = np.sqrt((pr_all.var(ddof=1) + st_all.var(ddof=1)) / 2)
# a 50/50 split halves each arm's daily volume: mean and variance scale with volume, so sd scales with sqrt(1/2)
mu_half, sd_half = st_all.mean() / 2, sd / np.sqrt(2)
delta = lift * mu_half
n_days = int(np.ceil(2 * (1.96 + 0.84) ** 2 * sd_half ** 2 / delta ** 2))
print(f'Daily lift {lift:+.1%}; a 50/50 pilot needs about {n_days} days of leads (95% confidence, 80% power)')

out = json.loads((HERE / 'pipeline.json').read_text())
out['breakeven'] = {'rows': rows, 'breakeven_drop': None if be is None else round(float(be), 3),
                    'high_share': round(float(high.mean()), 3)}
out['pilot'] = {'days': n_days, 'daily_lift': round(float(lift), 4), 'leads_per_day': round(len(pool) / 30)}
(HERE / 'pipeline.json').write_text(json.dumps(out, separators=(',', ':'), default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
print('Saved breakeven + pilot into pipeline.json')
