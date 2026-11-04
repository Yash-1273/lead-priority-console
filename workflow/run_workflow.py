"""Route today's leads, new and ongoing, in one queue: chance now -> priority -> play -> owner -> SLA.

    python workflow/run_workflow.py --calls 25

Uses today's 92 new held-out leads plus the ongoing follow-ups from pipeline.py. Writes workflow/output/queue.csv.
"""
import argparse
import json
import sys
import pandas as pd
from engine import HERE, LeadPriorityEngine, inr, VISA_GRANT

ap = argparse.ArgumentParser()
ap.add_argument('--calls', type=int, default=25, help='expert calls the team can make today')
args = ap.parse_args()
sys.stdout.reconfigure(encoding='utf-8')   # so ₹ prints on Windows terminals

new = pd.DataFrame(json.loads((HERE / 'console_data.json').read_text())['inbox'])
ongoing = pd.DataFrame(json.loads((HERE / 'pipeline.json').read_text())['ongoing'])
new['arrival'] = range(len(new))                                   # today's enquiries, in arrival order
ongoing['arrival'] = -1000 - ongoing.days_open                     # earlier enquiries come first on ties
leads = pd.concat([new, ongoing], ignore_index=True).drop(columns='p_book')
leads['moved_on'] = leads.moved_on.eq(True)                         # ongoing leads may have booked elsewhere already

q = LeadPriorityEngine().score(leads).sort_values('priority', ascending=False).reset_index(drop=True)
q['gets_expert_today'] = q.index < args.calls
q['real_booker'] = (q.booked == 1) & ~q.moved_on

(HERE / 'output').mkdir(exist_ok=True)
cols = ['lead_id', 'stage', 'attempts', 'days_open', 'university', 'city', 'room', 'days_to_move_in', 'p_book', 'p_now',
        'commission', 'extras', 'value', 'priority', 'play', 'owner', 'sla', 'action', 'gets_expert_today']
q[cols].round({'p_book': 2, 'p_now': 2, 'commission': 0, 'extras': 0, 'value': 0, 'priority': 0}).to_csv(
    HERE / 'output' / 'queue.csv', index=False)

print(f'{len(q)} leads routed ({len(new)} new, {len(ongoing)} follow-ups); top {args.calls} get an expert today\n')
print(q.groupby('play').agg(leads=('lead_id', 'size'), owner=('owner', 'first')).to_string(), '\n')
show = q.head(8).assign(**{c: q[c].head(8).map(inr) for c in ('value', 'priority')})
print(show[['lead_id', 'stage', 'attempts', 'city', 'room', 'p_now', 'value', 'priority', 'play', 'sla']]
      .round({'p_now': 2}).to_string(index=False))
for rule, key in [('Standard score', 'p_book'), ('Priority', 'priority')]:
    top = q.assign(first=-q.arrival).sort_values([key, 'first'], ascending=False).head(args.calls)   # ties: earliest first
    print(f'\n{rule:15} real outcome: {int(top.real_booker.sum())} real bookers reached, '
          f'{inr((top.real_booker * top.value * VISA_GRANT).sum())} revenue after visa risk')
