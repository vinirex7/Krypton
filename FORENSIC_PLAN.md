# Krypton forensic investigation

This research branch does **not** change the live strategy. It investigates two observed behaviors of the frozen BTC>SMA200 baseline:

1. Bull markets are consistently profitable but capture only a small fraction of BTC upside.
2. 2025 H1 was a material sideways/choppy failure while most other sideways windows were profitable or flat.

## Bull-capture decomposition

For every fixed 180-day historical block, record:

- actual gross capital exposure (`position market value / portfolio equity`), not only a binary position flag;
- cash ratio;
- average entry weight;
- actual risk used vs the configured risk target;
- binding sizing constraint: risk, asset allocation, or cash;
- PnL attribution by BTC/SOL/BNB;
- exit counts and PnL by TP/SL/signal/gap/EOD;
- holding time;
- asset return 5/10/20 trading bars after exit.

Interpretation examples:

- low actual exposure + allocation/cash cap binding -> sizing/capital-allocation bottleneck;
- low actual exposure + few valid signals -> entry/signal bottleneck;
- positive large post-exit returns, especially after TP/signal -> premature-exit / trend-truncation bottleneck;
- adequate exposure but weak PnL -> asset selection or whipsaw problem.

## 2025 H1 whipsaw decomposition

Measure without optimizing parameters:

- BTC SMA200 crossings;
- fraction of days above SMA200;
- realized volatility;
- BTC max drawdown inside the 180-day window;
- path/trend efficiency (net move divided by total absolute daily path);
- number of signals and entries;
- maximum losing streak;
- short holding periods;
- exit-reason mix;
- actual capital exposure and cash ratio.

The goal is to identify what distinguishes 2025 H1 from profitable sideways windows before proposing any new lateral-market filter.

## Guardrails

- TP remains 3x ATR.
- Base risk remains 1%.
- Binance Spot USDT data remains the source.
- BTC>SMA200 remains the baseline for this forensic analysis.
- No parameter grid is searched.
- No holdout is opened or reused by this script.
- Any future new filter must be declared only after the causal failure mode is identified, then validated in a new OOS protocol.
