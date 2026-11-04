"""University Living lead-priority engine.

The question is not "who is likely to book?" but "which lead is worth an expert's time right now?"

    Priority = chance to book NOW  x  revenue value  x  visa grant rate     -> decides WHO gets an expert next
    chance now = chance at enquiry (model)  x  fade (days the lead has been open)
    After 5 unanswered calls a lead moves to nurture                       -> stops wasting expert calls
    Days to move-in                                                        -> decides HOW FAST (response SLA)

New and ongoing leads are scored the same way and worked from one queue. Each part comes from real data:
  chance at enquiry   X Education leads (Kaggle): only information available when the lead arrives
  fade                Olist marketing funnel: days from first contact to a closed deal; half close within 14 days
  5-call limit        UCI Bank Marketing: 41,188 customers called repeatedly; by the 5th call the chance a call
                      converts is under half the 1st, and it keeps falling
  visa grant rate     Home Office: ~96% of Indian students' UK study visas granted (No Visa, No Pay)

Tested in a 60-day simulation and rejected: ranking by move-in urgency, and lowering each lead's chance per
unanswered call with the bank curve (it loses if students need even one more call than bank customers did).
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
NUM = ['visits', 'minutes_browsing', 'pages_per_visit']
FEATURES = ['channel_origin', 'channel_source', 'no_email', 'no_call', *NUM, 'profile_fields_filled']

# Filled in by the sales team after contacting a lead, so they don't exist at arrival. Never used.
POST_CONTACT_COLUMNS = ['Tags', 'Lead Quality', 'Lead Profile', 'Last Activity', 'Last Notable Activity',
                        'Asymmetrique Activity Index', 'Asymmetrique Profile Index',
                        'Asymmetrique Activity Score', 'Asymmetrique Profile Score']

ORIGIN_MAP = {'Landing Page Submission': 'Website enquiry form', 'API': 'App / platform lead',
              'Lead Add Form': 'Counsellor / partner submitted', 'Quick Add Form': 'Counsellor / partner submitted',
              'Lead Import': 'Bulk import (fairs, uni lists)'}

# ---- Value layer -------------------------------------------------------------------------------
# Weekly PBSA rent, GBP, 2025/26. London en-suite 259 / studio 341 and UK-average en-suite 153 / studio 228
# from published surveys (Unipol/HEPI, MyStudentHalls). Manchester and Birmingham en-suite are midpoints of
# published ranges (150-200, 140-190); their studio rents apply the UK studio/en-suite ratio (228/153 = 1.49).
WEEKLY_RENT = {
    ('London', 'En-suite'): 259, ('London', 'Studio'): 341,
    ('Manchester', 'En-suite'): 175, ('Manchester', 'Studio'): 260,
    ('Birmingham', 'En-suite'): 165, ('Birmingham', 'Studio'): 245,
    ('Other UK city', 'En-suite'): 153, ('Other UK city', 'Studio'): 228,
}
UNIVERSITIES = {
    'London': ['UCL', "King's College London", 'Queen Mary', 'University of Westminster'],
    'Manchester': ['University of Manchester', 'Manchester Metropolitan'],
    'Birmingham': ['University of Birmingham', 'Aston University'],
    'Other UK city': ['Coventry University', 'University of Leicester', 'University of Leeds', 'University of Glasgow'],
}
COMMISSION_RATE = 0.10   # ASSUMPTION: referral commission as a share of tenancy value; adjustable in the console
GBP_TO_INR = 127         # ASSUMPTION: exchange rate, Oct-Nov 2026. Rents are published in GBP; results are shown in INR


# ASSUMPTION: expected extra revenue per booked student (flights, money transfer, guarantor, room essentials),
# blended across take-up rates. It is earned per student, whatever the rent, so it narrows the gap between
# cheap and expensive rooms. Adjustable in the console.
ANCILLARY_INR = 15_000


def inr(gbp: float) -> str:
    """Format a GBP amount in rupees with Indian digit grouping, e.g. 2,20,866."""
    n = str(int(round(gbp * GBP_TO_INR)))
    head, tail = n[:-3], n[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    return '₹' + ','.join(([head] if head else []) + groups + [tail])


def commission_value(city, room, weeks, rate=COMMISSION_RATE):
    return WEEKLY_RENT[(city, room)] * weeks * rate


# ---- Ongoing leads: fading over time and the 5-call limit (curves learned in pipeline.learn_shapes) ----
VISA_GRANT = 0.96   # Home Office 2025: ~96% of Indian nationals' UK sponsored-study visas granted


CALL_LIMIT = 5      # bank data: by the 5th call the chance a call converts is under half the 1st; then nurture


def fade_factor(days_open, shapes):
    """Share of a lead's chance left after it has been open this many days (1.0 on the day it arrives)."""
    return np.interp(np.asarray(days_open, dtype=float), shapes['fade_days'], shapes['fade'])


# ---- Routing: match the cost of the channel to the value of the lead ---------------------------
P_HIGH = 0.5          # likely to book
ROUTES = {
    ('high', 'high'): ('Senior expert', 'Call now and lock the room before a competitor does', 'Protect'),
    ('low', 'high'):  ('Accommodation expert', 'Consultative call: shortlist, price match, answer visa worries', 'Persuade'),
    ('high', 'low'):  ('AI voice agent', 'Fast AI call with a booking link, expert on hand-off', 'Close cheaply'),
    ('low', 'low'):   ('Nurture automation', 'WhatsApp + email sequence; re-score on new activity', 'Nurture'),
}


def sla(days_to_move_in, p, high_value):
    if p >= P_HIGH and high_value:
        return '5 min'
    if days_to_move_in <= 30 and (p >= P_HIGH or high_value):
        return '15 min'
    if p >= P_HIGH or high_value:
        return '2 hours'
    return '24 hours'


# ---- Data mapping ------------------------------------------------------------------------------
def map_source(s):
    if pd.isna(s):
        return 'Unknown'
    s = str(s).lower()
    if s in ('reference', 'welingak website'):
        return 'Counsellor / referral'
    if s == 'olark chat':
        return 'Website chat (SABIE-like)'
    if s in ('google', 'organic search', 'bing'):
        return 'Search (SEO)'
    if s == 'direct traffic':
        return 'Direct traffic'
    if s in ('facebook', 'referral sites', 'social media'):
        return 'Social / affiliate sites'
    return 'Other'


def from_crm(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw CRM rows (X Education export format) -> University Living lead fields."""
    raw = raw.replace('Select', np.nan).drop(columns=[c for c in POST_CONTACT_COLUMNS if c in raw])
    ul = pd.DataFrame({
        'lead_id': raw['Lead Number'].values,
        'home_city': raw['City'].fillna('India').values,
        'channel_origin': raw['Lead Origin'].map(ORIGIN_MAP).fillna('Other').values,
        'channel_source': raw['Lead Source'].map(map_source).values,
        'no_email': (raw['Do Not Email'] == 'Yes').astype(int).values,
        'no_call': (raw['Do Not Call'] == 'Yes').astype(int).values,
        'visits': raw['TotalVisits'].values,
        'minutes_browsing': (raw['Total Time Spent on Website'] / 60).values,
        'pages_per_visit': raw['Page Views Per Visit'].values,
        'profile_fields_filled': raw[['What is your current occupation', 'Specialization', 'City']].notna().sum(axis=1).values,
    })
    if 'Converted' in raw:
        ul['booked'] = raw['Converted'].values
    return ul


def simulate_enquiry_context(leads: pd.DataFrame, seed=11) -> pd.DataFrame:
    """SIMULATED: the property each student enquired on (city, room, tenancy, move-in date).

    X Education has no such fields, so they are drawn from an assumed University Living mix. They are drawn
    independently of the booking outcome on purpose, so no pattern is invented between value and intent.
    """
    rng = np.random.default_rng(seed)
    n = len(leads)
    out = leads.copy()
    out['city'] = rng.choice(['London', 'Manchester', 'Birmingham', 'Other UK city'], n, p=[.35, .15, .15, .35])
    out['room'] = rng.choice(['En-suite', 'Studio'], n, p=[.65, .35])
    out['weeks'] = rng.choice([44, 51], n, p=[.5, .5])
    out['university'] = [rng.choice(UNIVERSITIES[c]) for c in out.city]
    bucket = rng.choice(3, n, p=[.4, .35, .25])     # intake this term / next term / next year
    out['days_to_move_in'] = np.select([bucket == 0, bucket == 1],
                                       [rng.integers(7, 61, n), rng.integers(61, 181, n)], rng.integers(181, 401, n))
    return out


def design(leads: pd.DataFrame, columns=None, medians=None) -> pd.DataFrame:
    X = leads[FEATURES].copy()
    for c in NUM:
        X[c] = pd.to_numeric(X[c]).fillna(medians[c] if medians else X[c].median())
    for c in ['channel_origin', 'channel_source']:
        X[c] = X[c].fillna('Unknown')
    X = pd.get_dummies(X).astype(float)
    for c in NUM:
        X[c] = np.log1p(X[c])
    return X.reindex(columns=columns, fill_value=0) if columns is not None else X


class LeadPriorityEngine:
    def __init__(self, model_dir: Path = HERE):
        self.model = joblib.load(model_dir / 'model.joblib')
        self.cfg = json.loads((model_dir / 'config.json').read_text())

    def score(self, leads: pd.DataFrame, rate=COMMISSION_RATE, extras_inr=ANCILLARY_INR) -> pd.DataFrame:
        out = leads.copy()
        for col in ('attempts', 'days_open'):               # new leads: no calls yet, open for 0 days
            out[col] = out[col].fillna(0) if col in out else 0
        out['p_book'] = self.model.predict_proba(design(out, self.cfg['columns'], self.cfg['medians']))[:, 1]
        shapes = self.cfg.get('shapes')
        out['p_now'] = out.p_book * (fade_factor(out.days_open, shapes) if shapes else 1)
        out['stage'] = np.where(out.attempts == 0, 'New', 'Follow-up')
        out['commission'] = [commission_value(c, r, w, rate) for c, r, w in zip(out.city, out.room, out.weeks)]
        out['extras'] = extras_inr / GBP_TO_INR
        out['value'] = out.commission + out.extras          # revenue if this student books (GBP)
        out['priority'] = out.p_now * out.value * VISA_GRANT  # expected revenue from working this lead now
        high_value = out.commission >= self.cfg['value_cut'] * rate / COMMISSION_RATE
        keys = zip(np.where(out.p_now >= P_HIGH, 'high', 'low'), np.where(high_value, 'high', 'low'))
        routes = [ROUTES[k] for k in keys]
        out['owner'] = [r[0] for r in routes]
        out['action'] = [r[1] for r in routes]
        out['play'] = [r[2] for r in routes]
        out['sla'] = [sla(d, p, h) for d, p, h in zip(out.days_to_move_in, out.p_now, high_value)]
        # Consent is a rule, not a feature: no phone calls to students who opted out
        # After 5 unanswered calls, stop spending expert calls: move to nurture until the student acts again
        tired = out.attempts >= CALL_LIMIT
        out.loc[tired, ['owner', 'play', 'sla']] = ['Nurture automation', 'Nurture', '24 hours']
        out.loc[tired, 'action'] = f'{CALL_LIMIT} unanswered calls: WhatsApp + email nurture; back to the queue if they act'
        opted_out = (out.no_call == 1) & (out.owner != 'Nurture automation')
        out.loc[opted_out, 'action'] = 'Personal WhatsApp from the owner (student opted out of calls)'
        return out

    def rescore(self, lead: dict, extra_minutes=0.0, extra_visits=0, profile_fields_filled=None, rate=COMMISSION_RATE,
                extras_inr=ANCILLARY_INR):
        """Leads keep changing after they arrive. Re-score on every new action (a return visit, a completed form)
        so a cold lead that heats up moves up the queue the moment it happens."""
        before = self.score(pd.DataFrame([lead]), rate, extras_inr).iloc[0]
        updated = dict(lead, minutes_browsing=lead['minutes_browsing'] + extra_minutes,
                       visits=(lead['visits'] or 0) + extra_visits)
        if profile_fields_filled is not None:
            updated['profile_fields_filled'] = profile_fields_filled
        after = self.score(pd.DataFrame([updated]), rate, extras_inr).iloc[0]
        return {'lead': updated, 'p_before': before.p_now, 'p_after': after.p_now,
                'play_before': before.play, 'play_after': after.play, 'sla_after': after.sla,
                'priority_before': before.priority, 'priority_after': after.priority}
