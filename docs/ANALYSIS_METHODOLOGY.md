# The Analyst's Playbook — How to Do Science on This Dataset

> *"I feel like everything I analyze needs me to ask an LLM, and I don't have any
> actual creativity. I want my creativity back, which means I need to think."*

This document is written to answer exactly that. It is **not** another analysis
script. It is a thinking guide: a repeatable process for generating your **own**
questions, spotting **missing data**, and running experiments on the NO₂ / TEMPO
data in this repository — without needing anyone (human or model) to hand you the
next idea.

Read it once end-to-end. After that, use it as a checklist you return to whenever
you feel stuck or feel the urge to outsource your thinking.

---

## 0. The one mindset shift

Most people think a scientist *knows the answer* and then confirms it. That is
backwards, and believing it is what makes you feel un-creative.

A scientist starts from **a specific confusion** and treats it as fuel. Creativity
in data analysis is not inventing brilliant ideas from nothing — it is **noticing
that something doesn't fit, and refusing to look away from it.** Every question in
this playbook is a structured way to manufacture that "wait, that's weird" feeling
on purpose.

You do not need more intelligence for this. You need a habit: **look at data →
notice a gap or surprise → name it → design the smallest test that resolves it →
look again.** That loop *is* the creativity. It is a muscle, not a gift.

---

## 1. The scientific loop (and where LLMs actually belong)

This is the loop practicing scientists actually run. It answers your question
directly: *yes*, the process is "analyze → realize you need more data → get it →
see how it relates," and then repeat.

```
        ┌─────────────────────────────────────────────────────────┐
        │                                                         │
        ▼                                                         │
  (1) OBSERVE ──▶ (2) QUESTION ──▶ (3) HYPOTHESIZE ──▶ (4) PREDICT │
  Look at a           "Why is        "Maybe it's         "If I'm    │
  plot / number.      it like        because X."         right,     │
  Notice something    that?"                             then I'll  │
  you can't yet                                          see Y."    │
  explain.                                                  │       │
                                                            ▼       │
  (7) COMMUNICATE ◀── (6) INTERPRET ◀────────────── (5) TEST ───────┘
  Write down what     "Does the result      Run the smallest analysis
  you now believe     support, kill, or     that could prove you WRONG.
  and what's still    complicate the        (This is the step you keep
  open.               hypothesis?"          skipping.)
```

**Where should an LLM sit in this loop?** Only at the *edges*, never the center:

| Step | Do it yourself (this is the science) | OK to ask an LLM |
| --- | --- | --- |
| 1 Observe | ✅ You must look at your own plots | — |
| 2 Question | ✅ This is the creativity you want back | Ask it to *critique* your question, not invent one |
| 3 Hypothesize | ✅ Commit to a guess before you test | Ask "what mechanism could cause this?" as a menu, then pick |
| 4 Predict | ✅ You decide what "right" looks like | — |
| 5 Test | Mostly you | Ask for the *syntax* of a specific plot/stat, not the idea |
| 6 Interpret | ✅ You decide what it means | Ask it to argue the opposite of your conclusion |
| 7 Communicate | ✅ Your voice | Ask it to tighten wording |

The rule: **an LLM is a syntax lookup and a sparring partner, not an idea
generator.** If you catch yourself asking "what should I analyze next?", stop —
that question is *yours* to answer, and Section 3 shows you how.

---

## 2. What you already have (know your material)

You cannot brainstorm about data you don't know. A scientist has an almost
physical familiarity with their variables. Here is the raw material in this repo,
in plain language. Re-read this until each row feels concrete.

**The thing you are trying to explain (the "target"):**

- **Ground-level NO₂ (ppb)** — nitrogen dioxide measured by EPA surface monitors.
  Comes mostly from combustion (traffic, freight). This is what the models predict.

**The 20 engineered features** (from `run_temporal_holdout.py` and the master
`.npz`), grouped by what they physically represent:

| Group | Features | What it physically is |
| --- | --- | --- |
| Satellite | `TEMPO_NO2` | NO₂ seen from space (a whole vertical column) |
| Mixing / dilution | `blh` (boundary-layer height), `t2m` (2 m temp), `sp` (surface pressure) | How much air the pollution is diluted into |
| Wind | `u10`, `v10` (east/north wind at 10 m) | Transport — does it blow pollution in or out |
| Moisture | `d2m` (dew point), `tcc` (total cloud cover) | Proxies for humidity / overnight trapping |
| Sun | `solar_zenith_angle` | How high the sun is → how fast NO₂ is destroyed |
| Human timing | `traffic`, `road_density`, `pop`, `is_weekend` | Emission sources and schedules |
| Time encodings | `hour_sin/cos`, `day_of_week_sin/cos`, `month_sin/cos` | Cyclical clock/calendar position |
| Geography | `elev` (elevation) | Terrain (valley vs. coast) |

**The central story the repo already found** (your README's thesis): the *same
data* looks easy (R² ≈ 0.6) over 24 hours but hard (R² ≈ 0.2) in the TEMPO daylight
window, because daytime **photolysis** and **convective mixing** destroy the tidy
"humid = high NO₂" relationship. And coastal sites (Anaheim, Compton) behave
differently from shielded inland valleys (Pomona, Santa Clarita).

**Every good new question in this repo is a variation on: "the story above is
true *on average* — but where, when, and for whom does it break?"**

---

## 3. The idea-generation engine (your anti-LLM toolkit)

This is the heart of the document. When you don't know what to analyze, you do
**not** ask a model. You run one or more of these thinking moves. Each one is a
machine for manufacturing "that's weird."

### Move A — Interrogate a single number
Take any headline result (e.g. "Compton R² = 0.665"). Ask:
- What does the *worst* case behind this average look like? (Averages hide crises.)
- On which day/hour/station is it best? Worst? **Why those?**
- If I split this number in half (weekday vs weekend, summer vs winter, morning vs
  afternoon), do the halves agree? If not, you just found a new phenomenon.

> This repo already does this move in `diagnose_worst_spike.py` and
> `run_extreme_episode_test.py`. That *is* Move A in code. Copy the pattern.

### Move B — Look at what the model gets *wrong* (residuals are gold)
The prediction errors (`residual = observed − predicted`) are the single richest
source of new questions. A residual is literally *"the part reality does that my
current understanding cannot explain."* That is the definition of an open problem.
Ask:
- Are the errors random, or do they have a *pattern* in time, space, or weather?
- **A pattern in the residuals is a variable you forgot to include.** (See Move D.)

> `generate_spatial_residuals.py` and `plot_baseline_timeseries_residuals.py`
> already exist. Don't just generate them — *stare* at them and ask "why *there*?"

### Move C — Compare two things that "should" be the same
Creativity is often just a well-chosen contrast:
- Anaheim (coast) vs. Pomona (inland valley): same predictor, different skill — why?
- The same station in June vs. December.
- Weekday freight rush vs. Saturday.
- 24-hour data vs. TEMPO-window data (the repo's founding contrast).
Pick two cases you *expect* to match. When they don't, you have a finding.

### Move D — Hunt the missing variable ("unknown data")
This directly answers your question *"how do I realize the unknown data that would
help me understand what I'm looking at?"* You realize it by working **backwards
from a failure**:

1. Find where the model/relationship fails (Move B gives you this).
2. Ask: **"What is physically present in that situation that none of my 20
   features measures?"** Make a list of *mechanisms*, then ask which are missing.
3. Example reasoning: *"My errors spike on hot, stagnant afternoons near freeways.
   I have temperature and traffic… but I have no measure of **ozone** or **direct
   sunlight intensity**, and photochemistry depends on both. That's my unknown
   data."*

Concrete "unknown data" candidates for *this* project, and the confusion each
would resolve:

| If you're confused about… | The missing data might be… |
| --- | --- |
| Daytime NO₂ destruction | Ozone (O₃), NOₓ, actual UV/actinic flux |
| Sudden unexplained spikes | Wildfire smoke days, traffic incident logs, port ship schedules |
| Why inland ≠ coast | Terrain drainage flow, marine-layer depth, mixing-height soundings |
| Weekend/holiday dips | Freight/truck GPS counts, school calendars |
| Spatial error hot-spots | Local point sources (refineries, ports) not in `road_density` |

You don't need to *have* the data to name it. **Naming the missing variable is
itself a scientific result** — it tells you the next experiment to run or dataset
to download (`download_era5.py`, the EPA CSV pipeline, etc. show you already know
how to go get more data).

### Move E — Change the resolution / grain
Zoom in and out on purpose. The same data at a different grain tells a different
story:
- Time: yearly → monthly → daily → hourly. (The repo's whole thesis came from
  slicing to the *hourly* daylight window.)
- Space: California → LA Basin → single station.
- Population: all stations pooled → one freight-corridor station (Station 41).

### Move F — Ask "compared to what?" (baselines)
Never let a result stand alone. A number is only meaningful against a dumber
baseline. This repo is *built* on this move: persistence, cosine climatology,
wind-only, MLR, then XGBoost. When you get a result, always ask: **"Could a
one-line rule have done almost as well?"** If yes, your fancy result isn't a
finding yet.

### Move G — The "so what?" and "what would change my mind?" test
For any hypothesis, force two sentences:
1. *"If this is true, it matters because ______."* (If you can't finish it, drop it.)
2. *"I would abandon this belief if I saw ______."* (If nothing could change your
   mind, it's not science yet — it's a vibe.)

---

## 4. Turning a vague feeling into a testable hypothesis

A hypothesis is a *guess with a consequence*. Use this template every time:

> **"I think [target] changes because of [mechanism], so if I [do X to the data],
> I expect to see [specific, falsifiable pattern] — and NOT [what a rival
> explanation would predict]."**

Worked example, fully in this repo's world:

- Vague feeling: *"Inland valleys seem weird at night."*
- Sharpened: *"I think Pomona's overnight NO₂ is driven by cold-air drainage down
  the canyons, not by marine-layer humidity like at the coast. So if I plot NO₂
  against `d2m`/humidity separately for Pomona vs. Anaheim, I expect a strong
  slope at Anaheim and a **flat, weak** slope at Pomona — and if instead both are
  strong, my drainage idea is wrong and it's just humidity everywhere."*

Notice: it names the mechanism, the test, the expected result, **and** the result
that would kill it. That last clause is what separates a scientist from someone
looking for confirmation.

---

## 5. A 60-minute starter session (do this today, alone)

To rebuild the muscle, run this without asking anyone anything. Struggling here is
the point — that friction *is* the learning.

1. **(5 min) Pick one existing figure** in `graphs_for_paper/` or run one script
   in `scripts/`. Just *one*.
2. **(10 min) Write 5 observations** — literal statements of what you see. No
   interpretation yet. ("The scatter fans out above 20 ppb." "Weekends sit lower.")
3. **(10 min) Turn each observation into a "why?" question.** You now have 5
   questions you generated, not borrowed.
4. **(5 min) Pick the one question that bugs you most** and write a hypothesis
   using the Section 4 template.
5. **(20 min) Design and run the smallest possible test** — usually a subset +
   one plot. Try to prove yourself *wrong*.
6. **(10 min) Write 3 sentences:** what you found, whether it killed or supported
   your guess, and the **new** question it created.

Do this three times a week for a month. Keep every session in a running lab
notebook (see Section 7). Your "creativity" will visibly return — because you'll
have a stack of your own questions and dead ends to build on.

---

## 6. Making up for missing education (efficiently)

You said you feel undereducated for what you're looking at. Two truths:

1. **You don't need the whole textbook — you need the *concept behind the confusion
   in front of you*.** Learning is far faster when it's pulled by a real question
   ("why does photolysis destroy NO₂?") than pushed by a syllabus. Let your
   analysis *generate* the reading list.
2. Build a small "spine" of concepts this project actually rests on, and learn
   them just-in-time:

- **Domain physics:** the NOₓ–O₃ photochemical cycle; the atmospheric **boundary
  layer** and mixing height; temperature inversions; marine layer vs. drainage flow.
- **Statistics you'll reuse constantly:** distributions (look before you model),
  correlation vs. causation, linear regression + R²/RMSE/MAE (already in your
  baselines), residual analysis, train/test leakage and *why grouped/temporal
  splits matter* (`run_temporal_holdout.py`, `test_literature_split.py`).
- **Interpretability:** what SHAP values mean (`run_shap_analysis.py`) — read the
  plot as "which feature moved this prediction, and in which direction."

How to learn each: when a term blocks you, spend 20 minutes getting an *intuition*
(a diagram + a one-paragraph mental model), not a rigorous derivation. Then
immediately apply it to your own data — application is what makes it stick. Use an
LLM here the honest way: to **explain a concept you then verify**, not to make the
decision for you.

---

## 7. Keep a lab notebook (this is where creativity compounds)

Scientists don't hold ideas in their head — they externalize them so today's
dead-end becomes next month's insight. Keep a plain running log (a markdown file, a
notebook, anything). For each session, one entry:

```
DATE:
WHAT I LOOKED AT:
5 OBSERVATIONS:
QUESTIONS THEY RAISED:
HYPOTHESIS I TESTED (Section 4 template):
WHAT I EXPECTED / WHAT WOULD PROVE ME WRONG:
WHAT ACTUALLY HAPPENED:
WHAT I NOW BELIEVE / STILL DON'T UNDERSTAND:
NEXT QUESTION:
MISSING DATA I WISH I HAD:
```

The "NEXT QUESTION" and "MISSING DATA" lines are your idea backlog. You will
**never** run out of things to analyze again — every session spawns the next one.
That is the self-sustaining engine you're actually asking for.

---

## 8. Quick-reference checklist (pin this)

When you feel the urge to ask an LLM "what should I analyze?", run this instead:

- [ ] Did I actually **look** at a plot/number myself, and write down what I see?
- [ ] Have I named one thing that **surprises or confuses** me?
- [ ] Did I run a **Move** from Section 3 (interrogate a number, read the
      residuals, compare two "same" things, hunt a missing variable, change grain,
      compare to a dumb baseline)?
- [ ] Have I written a hypothesis with a clause that could **prove me wrong**?
- [ ] Have I designed the **smallest** test, not the fanciest?
- [ ] Did I log it, and write down the **next** question?

If you can tick these, you are doing science — and the ideas are coming from you.
