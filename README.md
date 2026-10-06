# Rebuilding NFL Expected Points Added (EPA) from Scratch

## The question

Expected Points Added is the foundational metric of modern NFL analytics — nearly every
team-side and public model builds on it. But how well do I actually understand it?
This project rebuilds EPA from raw play-by-play data and validates it against the
nflverse reference implementation.

## Data

- **nflverse play-by-play**, 2022–2024 seasons (~144k plays), via `nfl_data_py`
- Train: 2022–2023 | Test: 2024 (season-split — plays within a game are correlated,
  so random splits would leak)

## Method

**1. Expected Points model.** A multinomial classifier predicts the *next scoring
event* from game state, over 7 outcomes (TD +7, FG +3, Safety +2, no score 0,
opp. safety −2, opp. FG −3, opp. TD −7). EP = Σ P(event) × value.

Features: `yardline_100`, `down`, `ydstogo`, `half_seconds_remaining`,
`score_differential`, `posteam_timeouts_remaining`, `defteam_timeouts_remaining`.

Labels constructed by scanning forward to the next scoring play within the half,
from the possession team's perspective. Started with multinomial logistic
regression, then XGBoost.

**2. EPA computation.** `EPA = EP(after) − EP(before) + points scored on the play`,
with EP always measured from the perspective of the team in possession *after*
the play — so turnovers flip the sign (an interception isn't "their EP went up",
it's your EP going negative).

**3. Validation** against nflverse's `epa` column on the held-out 2024 season.

## Results

> **[FILL IN after Step 5]**

- Correlation (mine vs nflverse EPA): `[e.g. r = 0.97]`
- Mean absolute error: `[e.g. 0.08]`
- Mean EPA/play ≈ 0: `[confirm]`

![EPA validation scatter](notebooks/figures/epa_validation.png)
*My EPA vs nflverse EPA, 2024 season (held out).*

![Team EPA per play, 2024](notebooks/figures/team_epa_rankings.png)
*Offensive EPA per play by team — the payoff chart.*

**What the validation taught me:** `[1–2 sentences — e.g. where the model
disagrees with nflverse and why: end-of-half edge cases, turnover perspective,
etc.]`

## Repo structure

```
├── notebooks/
│   ├── 01-ep-model.ipynb          # label construction + EP model training
│   ├── 02-epa-validation.ipynb    # EPA computation + validation vs nflverse
│   └── 03-team-rankings.ipynb     # extension: team/QB leaderboards
├── src/
│   └── ep_model.py                # label builder + model, importable
└── requirements.txt
```

## What I'd do next

- [ ] Win probability model on top of the EP framework
- [ ] 4th-down aggressiveness index by coach
- [ ] CPOE vs EPA/dropback QB landscape

## References

- nflverse data dictionary — nflverse.com
- nflfastR EP/WP model paper (Baldwin)
- Brian Burke, Advanced Football Analytics — original EP primer
