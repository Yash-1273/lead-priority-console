# University Living — Business Context for Lead Scoring

*Why this exists: every modelling decision in this project should trace back to how University Living (UL) actually makes money.*

## 1. What UL is
A managed marketplace for international student housing. Students (mostly from India and South/South-East Asia) find and book beds near their university abroad: 65,000+ properties, 640+ cities, UK, Ireland, Australia, USA, Canada and more.

## 2. How a lead becomes revenue

```
Lead arrives ──► Expert contacts ──► Shortlist & ──► Booking ──► Visa ──► Move-in ──► Commission
(web form,        (call / WhatsApp /   compare          (deposit)    approved     (tenancy     paid by the
 chat/SABIE,       AI voice agent)                                                 starts)      property partner
 counsellor,
 university)
```

- **The student pays nothing.** UL's revenue is a **referral commission from the accommodation partner**, plus extras: flights, money transfer, guarantor, airport pickup, room essentials.
- **"No Visa, No Pay":** a booking whose visa is refused is cancelled and refunded, so revenue is only certain once the student moves in.
- **Two lead engines:**
  - **B2C:** website, app, SEO content, live chat, the SABIE AI assistant, WhatsApp, "schedule a call".
  - **B2B:** 1,500+ study-abroad counsellors, university partners, platforms like ApplyBoard.

## 3. What this means for lead scoring

| Business fact | So the lead score must… |
|---|---|
| Students pay nothing, so enquiring costs nothing | Separate the many casual browsers from serious bookers. Volume is high and intent varies a lot. |
| Sales experts are a limited, costly resource; the AI voice agent is cheap | Act as a **routing decision**: who gets a human, who gets the AI agent, who gets automated nurture. |
| Leads arrive continuously | Work **the moment a lead arrives**, using only information available then. |
| Demand peaks before the Sept and Jan intakes, and beds sell out | Put response speed first for the best leads. Hot leads go cold quickly. |
| Counsellor and university leads are pre-qualified by a third party | Treat **channel** as a key signal, and track which partners send leads that book. |
| Revenue = commission on booking value (city × room × weeks) | Eventually rank by **expected revenue**, not just probability (v2). |

**Success metric:** not model accuracy, but *"with the same sales team, how many more bookings do we capture?"*, measured as the share of bookings in the top-scored X% of leads.

## 4. Why X Education is the proxy dataset
There is no public UL lead data. X Education is the closest real funnel:
- Indian, education-driven buyers.
- Mixed digital and referral lead sources, including a live-chat channel (like SABIE).
- A sales team that calls leads.
- A real converted / not converted label for 9,240 leads.

| X Education field | UL equivalent |
|---|---|
| Lead Origin: Landing Page Submission | Website enquiry form |
| Lead Origin: API | App / platform-integrated lead |
| Lead Origin: Lead Add Form | Lead submitted by a counsellor/partner |
| Lead Origin: Lead Import | Bulk upload (events, fairs, university lists) |
| Lead Source: Olark Chat | Website chat / SABIE AI assistant |
| Lead Source: Reference / Welingak Website | Counsellor or friend referral / partner site |
| Lead Source: Google, Organic Search, Direct Traffic | SEO and direct traffic |
| Lead Source: Facebook / Referral Sites | Social / affiliate sites |
| TotalVisits, Time on Website, Page Views | Property browsing depth |
| Do Not Email / Do Not Call | Contact consent |
| Converted | Booked a bed |

**Honest limitation:** these patterns come from a real education funnel, not UL's own data. The method transfers; the exact numbers would be re-learned on UL's CRM data.

## 5. Columns we deliberately exclude: data leakage
These are filled in **by the sales team after contacting the lead**, so they are not available when the score is needed:
- **Tags** (e.g. "Ringing", "Closed by Horizzon", "Will revert after reading the email")
- **Lead Quality** (the dictionary says it's based on the employee's intuition)
- **Lead Profile** (a level assigned to the lead)
- **Last Activity / Last Notable Activity** (mix in sales actions like "SMS Sent" and "Had a Phone Conversation")
- **Asymmetrique Activity/Profile Index/Score** (opaque third-party scores, 46% missing, assignment time unknown)

## 6. A lead's whole life, and how each stage is handled
A booking takes days to weeks, so on any day the team juggles new leads **and** ongoing ones.

| Stage | What happens | How the engine handles it | Evidence |
|---|---|---|---|
| Enquiry arrives | Student enquires on a property | Chance at enquiry from the booking model; value known from the property | X Education leads |
| First contact | Expert or AI agent calls | High-value, likely leads: expert within 5 minutes | MIT/InsideSales: ~21x more likely to qualify within 5 min than at 30 |
| Follow-ups | More calls, shortlists, student thinks it over | Up to 5 calls; after 5 unanswered calls, nurture until the student acts again | UCI Bank Marketing: by the 5th call a call converts at under half the rate of the 1st |
| Lead goes cold | Stops replying, books elsewhere | Chance fades with days open, so fresh leads win the next call | Olist: half of deals close within 14 days, 66% within 30 |
| New activity | Student returns to the site, fills the form | Re-scored instantly and moves up | Model + live re-scoring |
| Booking → visa → move-in | No Visa, No Pay | Revenue counted after visa risk | Home Office: ~96% of Indian students' UK study visas granted |

**One queue, one rule:** every lead, new or ongoing, is ranked by *chance to book now × revenue value*. In a 60-day test where every way of working gets the same calls and the same 5-call limit, this earned 11% more revenue than standard scoring at 25 calls a day (29% when the team is most stretched), with about the same number of bookings. If the goal is more students rather than more revenue, ranking by chance × time alone gives about 7% more bookings and 7% more revenue. Move-in urgency and precise per-call fatigue were tested and dropped.
