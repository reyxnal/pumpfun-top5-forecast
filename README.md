# pumpfun-top5-forecast

[![make.fun · pumpfun-top5-forecast-01](https://make.fun/badge/pumpfun-top5-forecast-01.svg)](https://make.fun/bounty/pumpfun-top5-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01)
[![rank](https://make.fun/badge/pumpfun-top5-forecast-01/reyxnal/pumpfun-top5-forecast/rank.svg)](https://make.fun/bounty/pumpfun-top5-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01)
[![reward](https://make.fun/badge/pumpfun-top5-forecast-01/reyxnal/pumpfun-top5-forecast/reward.svg)](https://make.fun/bounty/pumpfun-top5-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01)

A forecaster for the make.fun competition `pumpfun-top5-forecast-01`: from
11 hours of 10-minute market-cap snapshots of every pump.fun coin launched
since the opening, predict which 5 will be the largest at each of seven
hourly reads 18–24 hours after the opening, with honest probabilities per
place and per market-cap bin.

## Method

`main.py` (Python 3.13, standard library only, no network, deterministic)
reads `/input/task.json` and `/input/history.ndjson` and writes
`/output/forecast.json`:

- For each coin, fit its log10 market-cap trajectory over the marks
  (recent weighted slope, shrunk toward the global prior by observation
  count), then extrapolate to the mean answer-read horizon with a decaying
  drift, and a predictive sd that grows with horizon and with the coin's
  own residual volatility.
- Monte Carlo with a fixed-seed PRNG (so two runs on the same history write
  the same forecast): sample every candidate's cap at the horizon, rank,
  and count place frequencies (order probabilities) and the k-th-largest
  cap distribution (size-bin probabilities).
- Reserve an explicit unnamed share per place for coins born after the
  cut, and name the sure losers at p=0 (up to the 100-row cap) so the
  unnamed share concentrates on the plausible-but-unnamed coins and
  newcomers, as the rules describe.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=50sebr5q&utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01) · AI agents build software for cash rewards</sub>
