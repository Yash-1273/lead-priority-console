"""Route a batch of new leads: score -> priority -> play -> owner -> SLA.

    python workflow/run_workflow.py --calls 25

Uses one day of held-out leads (never seen in training). Writes workflow/output/queue.csv.
"""
import argparse
import sys
import json
import pandas as pd
from engine import HERE, LeadPriorityEngine, inr

ap = argparse.ArgumentParser()
ap.add_argument('--calls', type=int, default=25, help='expert calls the team can make today')
args = ap.parse_args()
sys.stdout.reconfigure(encoding='utf-8')   # so ₹ prints on Windows terminals

leads = pd.DataFrame(json.loads((HERE / 'console_data.json').read_text())['inbox']).drop(columns='p_book')
q = LeadPriorityEngine().score(leads).sort_values('priority', ascending=False).reset_index(drop=True)
q['gets_expert_today'] = q.index < args.calls

(HERE / 'output').mkdir(exist_ok=True)
cols = ['lead_id', 'university', 'city', 'room', 'days_to_move_in', 'p_book', 'commission', 'extras', 'value', 'priority',
        'play', 'owner', 'sla', 'action', 'gets_expert_today']
q[cols].round({'p_book': 2, 'commission': 0, 'extras': 0, 'value': 0, 'priority': 0}).to_csv(HERE / 'output' / 'queue.csv', index=False)

print(f'{len(q)} leads routed; top {args.calls} get an expert today\n')
print(q.groupby('play').agg(leads=('lead_id', 'size'), owner=('owner', 'first')).to_string(), '\n')
show = q.head(8).assign(**{c: q[c].head(8).map(inr) for c in ('value', 'priority')})
print(show[['lead_id', 'university', 'city', 'room', 'p_book', 'value', 'priority', 'play', 'owner', 'sla']].round({'p_book': 2}).to_string(index=False))
for rule, key in [('Standard score', 'p_book'), ('Priority', 'priority')]:
    top = q.sort_values(key, ascending=False).head(args.calls)
    print(f'\n{rule:15} real outcome: {int(top.booked.sum())} bookings, {inr((top.booked * top.value).sum())} revenue')
