# pumpfun-top5-forecast-01 — Monte-Carlo top-5 forecaster

`main.py` (Python 3.13, standard library only, no network) reads the house's
10-minute snapshots in `/input` (`task.json`, `history.ndjson`) and writes
`/output/forecast.json`: per-place coin probabilities, an ordered top 5, and
per-place market-cap size-bin probabilities for the task's seven answer reads.

Model in brief:

- Per-coin log10-cap dynamics are fitted from the history itself: drift and
  diffusion of log-cap increments, bucketed by coin age and bonding-curve
  completion, plus shrunk per-coin momentum. (Most listed coins are dormant
  between marks, so the estimator uses total variance, not a median/MAD.)
- Every candidate coin is walked forward in 0.5 h fat-tailed Gaussian steps
  to each answer read; rankings over 1,800 seeded simulations give the place
  probabilities and each place's cap distribution for the 82 size bins.
- Coins born after the cut cannot be named, so the simulation also races
  "phantom newcomers" whose arrival rate and peak caps (measured as excess
  over the contemporaneous place-5 cap) are fitted from the history's young
  strong coins. That calibrates how much probability each place leaves
  unnamed for newcomers and movers.
- Deterministic (fixed seed): two runs on the same history are byte-identical.

`backtest.py` validates inside the practice history with the exact log-score
rule of RULES.txt: fit through an early mark, forecast later real marks, and
compare against a port of the house baseline's persistence forecast.

Layout: `main.py` is the entry; `LICENSE` is MIT; `backtest.py` is a local
validation tool and is not read by `main.py`.
