"""University Living lead-priority engine.

The question is not "who is likely to book?" but "which lead is worth an expert's time right now?"

    Priority = P(book)  x  Revenue value                       -> decides WHO gets an expert first
               (model)     (rent commission + per-student extras)
    Days to move-in                                 -> decides HOW FAST (response SLA)

Urgency was tested as a third ranking factor in a 30-day simulation and lost: lead volume exceeds team
capacity, so a lead not called today is rarely called later. Urgency belongs in the SLA, not the ranking.

Only information available when a lead arrives is used. At University Living a student enquires on a
specific property, so city, room type and move-in date are known at arrival, and so is the lead's value.
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


# ---- Lead decay (simulation assumption) -----------------------------------------------------------
def daily_loss_rate(days_to_move_in):
    """ASSUMPTION: chance per day that an uncontacted lead books elsewhere or drops out.
    Students close to move-in decide within days; next-year students can wait."""
    d = np.asarray(days_to_move_in, dtype=float)
    return np.select([d <= 30, d <= 90, d <= 180], [0.30, 0.12, 0.04], 0.015)


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
        out['p_book'] = self.model.predict_proba(design(out, self.cfg['columns'], self.cfg['medians']))[:, 1]
        out['commission'] = [commission_value(c, r, w, rate) for c, r, w in zip(out.city, out.room, out.weeks)]
        out['extras'] = extras_inr / GBP_TO_INR
        out['value'] = out.commission + out.extras          # revenue if this student books (GBP)
        out['priority'] = out.p_book * out.value            # expected revenue from working this lead
        high_value = out.commission >= self.cfg['value_cut'] * rate / COMMISSION_RATE
        keys = zip(np.where(out.p_book >= P_HIGH, 'high', 'low'), np.where(high_value, 'high', 'low'))
        routes = [ROUTES[k] for k in keys]
        out['owner'] = [r[0] for r in routes]
        out['action'] = [r[1] for r in routes]
        out['play'] = [r[2] for r in routes]
        out['sla'] = [sla(d, p, h) for d, p, h in zip(out.days_to_move_in, out.p_book, high_value)]
        # Consent is a rule, not a feature: no phone calls to students who opted out
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
        return {'lead': updated, 'p_before': before.p_book, 'p_after': after.p_book,
                'play_before': before.play, 'play_after': after.play, 'sla_after': after.sla,
                'priority_before': before.priority, 'priority_after': after.priority}
