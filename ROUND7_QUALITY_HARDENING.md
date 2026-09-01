# Round 7 quality hardening

Round 7 keeps live authority disarmed and targets evidence quality before a canary.

## Forward evidence v2

Forward outcomes now start from the first 1h open at or after the actual observation timestamp. The prior candle-close outcome table is retained as legacy evidence but excluded from current metrics. Signal evidence is deduplicated into 15 minute decision clusters. MFE is non-negative and MAE is non-positive.

## Universe quality

The runtime universe remains 25 EUR spot markets, but public spread quality is tightened. Preferred markets are at or below 15 bps spread and the hard runtime ceiling is 35 bps. Missing spread data, positive 24h moves above the anti-hype threshold, stablecoins, leveraged tokens and non-allowed Shariah assets are rejected. Spread quality also affects the cheap screen used to choose deep scans.

## Agent qualification

Agent heads are qualified independently. Alpha, regime, return forecast and risk estimates cannot veto or influence decisions merely because another head passed. The alpha threshold search uses a finer grid, market breadth and minimum sample sizes. Return and risk heads must beat simple baselines. SHADOW remains the maximum authority after training.

## Cold-start research hygiene

Rolling feature warm-up is preserved across chronological partitions. Candidate ranking uses train and validation only. Only a bounded validation-selected shortlist touches the final test holdout. Research costs use a current per-market spread floor by default. Because current universe membership and current spreads are not historical point-in-time data, this bootstrap remains research-only.

No automatic paper or live promotion is introduced by this round.
