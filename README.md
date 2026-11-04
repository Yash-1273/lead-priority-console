# Lead Priority Engine for University Living

**Rank leads by what they're likely to earn, not just whether they'll book.**
When experts can't call everyone, the top of the queue should hold the most expected revenue. Every other lead still gets the AI agent or nurture.

> **Live console:** https://yash-1273.github.io/lead-priority-console/ (the trained model runs inside the page, no install needed).
> Offline: download `Lead_Priority_Console.html` and open it in any browser.

Standard lead scoring ranks leads by how likely they are to convert. At University Living that misses half the picture.
Commission follows the rent, and students enquire on a specific property, so **the value of a lead is known the moment it arrives.**
And a booking takes days to weeks, so **new and ongoing leads compete for the same expert time.** They share one queue.

```
Priority      = chance to book NOW  ×  revenue value  ×  visa grant rate     → decides WHO gets an expert next
chance now    = chance at enquiry  ×  fade (days the lead has been open)
5-call limit  : after 5 unanswered calls → nurture, back in the queue when the student acts
revenue value = rent commission + per-student extras
Days to move-in                                                          → decides HOW FAST (response deadline)
```

## Four layers, each from real data
| Layer | Question | Real data |
|---|---|---|
| Chance at enquiry | How likely is this new lead to book? | **X Education** (Kaggle): 9,240 leads with real outcomes, mapped to University Living channels |
| When to stop calling | Is another call still worth it? | **UCI Bank Marketing**: 41,188 customers called repeatedly. By the 5th call a call converts at under half the rate of the 1st → 5-call limit |
| Fade over time | How fast do chances fade while a lead stays open? | **Olist marketing funnel**: 8,000 leads with first-contact and close dates. Half of deals close within 14 days, 66% within 30 |
| After booking | Will the booking turn into revenue? | **Home Office**: ~96% of Indian students' UK study visas granted (No Visa, No Pay) |

The datasets are **not merged**: they describe different people, and joining them would invent students. Each answers one question at one stage, and every lead keeps its own starting chance from the model.

## Routing: match the channel's cost to the lead's value
|  | Lower value | Higher value |
|---|---|---|
| **Likely to book (now)** | **Close cheaply:** AI voice agent with a booking link | **Protect:** senior expert calls within 5 min |
| **Less likely (now)** | **Nurture:** WhatsApp/email, re-scored on new activity | **Persuade:** expert consultative call |

- The 5-minute deadline is backed by the MIT/InsideSales study: leads contacted within 5 minutes are ~21x more likely to qualify than at 30 minutes.
- A lead that stays open slides towards Nurture as its chance fades; after 5 unanswered calls it moves to nurture.
- **Engaged students keep their expert:** a set number of expert calls is reserved for callbacks to students who have a shortlist or are at deposit or visa stage. The datasets can't tell engaged from unresponsive leads, so this is a setting, not a model.
- **On rollout:** merge duplicate enquiries from the same student.
- A student who opted out of calls is never phoned; the owner sends a WhatsApp message instead.
- **Continuous re-scoring:** every new action (a return visit, a completed form) re-scores the lead.

## Evidence: 60 days, new and ongoing leads together (`workflow/pipeline.py`)
2,772 held-out leads arrive at ~92 a day for 30 days. Bookers need several calls and stop looking after a while, as in the real bank and Olist data. **Every rule gets the same calls, the same follow-up spacing (2 days) and the same 5-call limit**, so standard scoring is a realistic baseline. Revenue is after visa risk, against standard scoring:

| Expert calls per day | 10 | 20 | 25 | 30 | 40 | 60 | 90 |
|---|---|---|---|---|---|---|---|
| **Priority (ours)** | **+29%** | **+16%** | **+11%** | **+9%** | **+6%** | **+3%** | **+2%** |

- **About the same bookings, more revenue:** at 25 calls a day, 380 bookings vs 387 (−2%) and +11% revenue. Across team sizes, bookings stay within −2% to +1%.
- **A dial, not a fixed answer:** ranking by chance × time alone gives **+7% bookings and +7% revenue**, if the goal is more students rather than more revenue.
- **Where the gain comes from:** value alone +8% (but −6% bookings), time fade alone +7% (+7% bookings), both together +11%.
- **The busier the team, the bigger the gain**, which matters most in peak season (July–September).
- New leads first earns 5% less than standard; arrival order 73% less.
- **One real day (92 new + 40 follow-ups, 25 calls):** standard and priority both reach 20 real bookers; priority's are worth ₹33.0 lakh vs ₹26.5 lakh.

## Ideas tested and dropped
- **Move-in urgency** as a ranking factor: −9% vs standard. Lead volume exceeds capacity, so a lead pushed to "later" is rarely called. Urgency sets the response deadline instead.
- **Precise call fatigue** (lowering each lead's chance per unanswered call with the bank curve): +13% when students behave like the bank's customers, but **−9%** if they need one more call. The simple 5-call limit never dropped below **+7%**.

## Stress tests
- **The gain holds in every scenario tried** (25 calls/day): normal +12%, peak season with 2x enquiries +13%, follow-ups taking half the time +7%, follow-ups every day or every 3 days +12%, no extras +12%, extras doubled +11%.
- **It holds when students behave differently from the data it learned from:** needing one more or one fewer call, staying interested twice as long or half as long, or both harder at once: always +7% to +16%.
- **No algorithm beats the booking model:** gradient boosting AUC 0.860; random forest 0.860; tuned GB 0.858; logistic regression 0.842. The limit is the data.
- **The biggest model gain is a product change:** adding enquiry-form answers lifts top-20% precision from 85% to 88%. Recommendation: three qualifying questions (offer letter / CAS? which intake? weekly budget?).
- **Risk to check on real data:** browsing time carries most of the model's weight (AUC 0.86 → 0.76 without it), and the dataset doesn't say when it was measured. In production, count browsing up to the enquiry, then re-score as more arrives.

## The biggest assumption, tested: price sensitivity (`workflow/breakeven.py`)
Each lead's property is simulated independently of whether they booked. If students enquiring about expensive rooms actually book less often, part of the value gain disappears. So high-value enquirers were made to book less often and the 60-day test re-run (25 calls a day):

| High-value enquirers book… | as in data | 10% less | 20% less | 30% less | 40% less | 50% less |
|---|---|---|---|---|---|---|
| Lead Desk vs standard | +11.2% | +9.6% | +5.1% | −0.6% | −3.5% | −10.8% |

**Break-even: Lead Desk keeps winning unless expensive-room enquirers book about 29% less often than others.** In the proxy data the two groups book at 38% and 39%. University Living can check its own figure in minutes from its CRM.

## From concept to production
1. **Event:** new enquiry, call outcome, return visit, SABIE chat → 2. **Score:** the same model, milliseconds per lead → 3. **Write back:** priority ₹, lane, deadline and reasons on the CRM record → 4. **Act:** expert call list, AI voice agent, WhatsApp → 5. **Learn:** bookings and move-ins feed retraining before each intake.
- **Run it safely:** retrain on University Living's own CRM history (including its own fade curve and call limit); check weekly that predicted chances still match real booking rates, and that the gain holds against a small holdout routed the old way.
- **Prove it with a pilot:** route half of new leads with Lead Desk and half as today, at random; compare revenue per expert-hour and bookings. At ~92 leads a day this takes roughly 67 days of leads (95% confidence, 80% power, a conservative estimate); with more leads a day it's proportionally faster.
- **Next, with AI:** the biggest model gain came from better information, not a better algorithm. An LLM can pull CAS status, intake and weekly budget out of SABIE chats and AI voice-agent calls and add them to the score, with no extra form for the student.
- **Promise:** every student gets a first response within 24 hours, whether from an expert, the AI voice agent or WhatsApp. Lead Desk decides who gets the expensive channel first; it never decides who gets ignored.

## Data: what's real, simulated, assumed
- **Real:** X Education leads and outcomes; bank call outcomes by call number; Olist days to close; Home Office visa grant rate; UK student rents from 2025/26 surveys (London en-suite ₹32,893 and studio ₹43,307 a week; UK average ₹19,431 and ₹28,956; £259, £341, £153, £228).
- **Simulated:** the property each lead enquired on (drawn independently of the outcome, so no link between value and intent is invented); today's ongoing leads' call history, and how many calls each booker needs and how long they keep looking (drawn from the real bank and Olist distributions, and stress-tested with different ones).
- **Conservative:** leads below the expert line count as not called in the test, though the AI agent and WhatsApp would reach some.
- **Assumed:** commission rate 10% and extras ₹15,000 per booking (both adjustable); exchange rate ₹127 per £; follow-ups at least 2 days apart, each as long as a first call (half the time also tested); Manchester and Birmingham rents (midpoints of published ranges).
- **From the proxy data:** the 38.5% conversion rate. University Living's raw enquiries likely convert lower, which stretches the team further and increases the gain.
- **Never used:** post-call fields, like sales tags, "lead quality", call logs, and the bank's call duration. They are only known after a call.

## Run it
```bash
pip install -r requirements.txt
python workflow/train.py           # booking model + fade and call curves from real data
python workflow/experiments.py     # model stress tests → workflow/experiments.json
python workflow/pipeline.py        # 60-day test of new + ongoing leads → workflow/pipeline.json (a few minutes)
python workflow/race.py            # day-by-day season race for the console → workflow/race.json
python workflow/breakeven.py       # price-sensitivity break-even + pilot size → added to pipeline.json
python workflow/run_workflow.py    # route today's new + ongoing leads → workflow/output/queue.csv
python workflow/build_console.py   # build Lead_Priority_Console.html (runs the same model in the browser)
```

## Files
| Path | What it is |
|---|---|
| `Lead_Priority_Console.html` | **Lead Desk**, the interactive console: today's routing (experts, AI agent, nurture) with reserved callbacks for engaged students, one call list for new and follow-up leads, live events, a 60-day season race, proof and a live scorer |
| `workflow/engine.py` | The engine: CRM row → chance at enquiry → chance now → value → priority → play, owner, SLA, 5-call limit |
| `workflow/breakeven.py` | Price-sensitivity break-even test and pilot-size estimate |
| `workflow/race.py` | Day-by-day results for the console's season race, plus fresh enquiries for the live demo |
| `workflow/pipeline.py` | Learns fade and the call limit from real data; runs the 60-day test, stress tests and rejected ideas |
| `workflow/train.py` | Trains the booking model, saves shapes, exports the model to the browser |
| `workflow/experiments.py` | Algorithm comparison, form-answer lift, signal ablations |
| `workflow/run_workflow.py` | Routes today's new and ongoing leads from the command line |
| `data/` | X Education leads, UCI bank calls, Olist leads and deals |
| `01_business_context.md` | How University Living makes money, the lead lifecycle, and why each decision follows from it |
| `Lead_Scoring_University_Living.ipynb` | Analysis: leakage audit, insights, booking model, priority |

**Data sources:** Kaggle – Lead Scoring X Education · [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) (CC BY 4.0) · [Olist Marketing Funnel](https://www.kaggle.com/datasets/olistbr/marketing-funnel-olist) (CC BY-NC-SA 4.0) · Home Office visa statistics.
