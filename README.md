# Lead Priority Engine for University Living

**Rank leads by what they're likely to earn, not just whether they'll book.**
When experts can't call everyone, the top of the queue should hold the most expected revenue. Every other lead still gets the AI agent or nurture.

> **Live console:** https://yash-1273.github.io/lead-priority-console/ (the trained model runs inside the page, no install needed).
> Offline: download `Lead_Priority_Console.html` and open it in any browser.

Standard lead scoring ranks leads by how likely they are to convert. At University Living that misses half the picture.
Commission follows the rent, and students enquire on a specific property, so **the value of a lead is known the moment it arrives.**

```
Priority = P(book)  ×  Revenue value                         → decides WHO gets an expert first
           (model)     (rent commission + per-student extras)
Days to move-in                               → decides HOW FAST (response deadline)
```

## Routing: match the channel's cost to the lead's value
|  | Lower value | Higher value |
|---|---|---|
| **Likely to book** | **Close cheaply:** AI voice agent with a booking link | **Protect:** senior expert calls within 5 min |
| **Less likely** | **Nurture:** WhatsApp/email sequence, re-scored on new activity | **Persuade:** expert consultative call |

A student who opted out of calls is never phoned; the owner sends a WhatsApp message instead.

**Continuous re-scoring:** leads change after they arrive. Every new action (a return visit, a completed form) re-scores the lead.
Example: lead #644288 browses 15 more minutes and moves from 17% to 86%, Persuade to Protect, rank #55 to #10.

## Evidence
30-day simulation on 2,772 held-out leads (~92 arriving per day). Revenue (commission + ₹15,000 extras per booking) is shown against standard scoring:

| Expert calls per day | 10 | 20 | 25 | 30 | 40 | 50+ |
|---|---|---|---|---|---|---|
| **Priority (P × revenue value)** | **+19%** | **+9%** | **+5.0%** | **+3%** | +1% | about 0 |
| Priority × urgency (tested) | +1% | −4% | −7% | −8% | −7% | −3% to 0 |
| Arrival order | −57% | −53% | −52% | −50% | −45% | −38% to −3% |

- **The tighter the team, the more the ranking rule matters.** The gain is largest in peak season (July–September), when leads spike.
- **Urgency was tested as a ranking factor and rejected.** Lead volume exceeds capacity, so a lead not called today is rarely called later. Urgency sets the response deadline instead.
- **Extras narrow the gap but don't close it.** Flights, money transfer and guarantor are earned per student, whatever the rent. Counting them lowers the gain at 25 calls/day from +6.5% (commission only) to +5.0% (₹15,000 extras) and +4.0% (₹30,000).
- **One real day (92 leads, 25 calls):** standard scoring captured ₹28.8 lakh of revenue; priority captured ₹34.0 lakh (+₹5.26 lakh), with 20 bookings vs 19.

## Stress tests (`workflow/experiments.py`)
- **No algorithm beats the model:** gradient boosting AUC 0.860; random forest 0.860; tuned GB 0.858; logistic regression 0.842. The limit is the data.
- **The biggest gain is a product change:** adding enquiry-form answers lifts top-20% precision from 85% to 88% (AUC 0.860 to 0.875).
  Recommendation: three qualifying questions on the enquiry form (offer letter / CAS? which intake? weekly budget?).
- **The priority result is robust:** +4.9% to +5.3% whether uncalled leads are lost 4x slower or 4x faster than assumed, and still +4.0% with ₹30,000 extras per booking.
- **Risk to check on real data:** browsing behaviour carries most of the weight (AUC 0.86 to 0.76 without it), and the dataset doesn't say when browsing time was measured. In production, count browsing up to the enquiry, then re-score as more arrives.

## Data: what's real, simulated, assumed
- **Real:** 9,240 leads with booked / not booked outcomes (Kaggle *Lead Scoring – X Education*), mapped to University Living channels. UK student rents from published 2025/26 surveys: London en-suite ₹32,893 and studio ₹43,307 a week; UK average ₹19,431 and ₹28,956 (£259, £341, £153, £228).
- **Simulated:** the property each lead enquired on (city, room, tenancy, move-in date). It's drawn independently of the outcome, so no link between value and intent is invented. Also the daily loss of uncalled leads in the simulation.
- **Assumed:** commission rate (10%, adjustable); extras per booking (₹15,000 from flights, money transfer, guarantor and essentials, adjustable); exchange rate ₹127 per £ (Oct–Nov 2026); Manchester and Birmingham rents (midpoints of published ranges).
- **From the proxy data:** the 38.5% conversion rate. University Living's raw enquiries likely convert lower, which stretches the team further and increases the gain from value ranking.
- **Never used:** post-call CRM fields (sales tags, "lead quality", call logs). They give 94% accuracy that doesn't exist when a lead arrives.

## Run it
```bash
pip install -r requirements.txt
python workflow/train.py           # train the model, run the 30-day simulation, export the data
python workflow/experiments.py     # stress tests → workflow/experiments.json
python workflow/run_workflow.py    # route today's leads → workflow/output/queue.csv
python workflow/build_console.py   # build Lead_Priority_Console.html (runs the same model in the browser)
```

## Files
| Path | What it is |
|---|---|
| `Lead_Priority_Console.html` | The interactive console: live queue, re-ranking, routing, evidence, new-lead scorer |
| `workflow/engine.py` | The engine: CRM row → UL fields → P(book) → value → priority → play, owner, SLA |
| `workflow/train.py` | Training, 30-day simulation, model export |
| `workflow/experiments.py` | Algorithm comparison, form-answer lift, ablations, assumption sensitivity |
| `workflow/run_workflow.py` | Routes a day's leads from the command line |
| `01_business_context.md` | How University Living makes money, and why each decision follows from it |
| `Lead_Scoring_University_Living.ipynb` | Full analysis: leakage audit, insights, model, priority, simulation, stress tests |
