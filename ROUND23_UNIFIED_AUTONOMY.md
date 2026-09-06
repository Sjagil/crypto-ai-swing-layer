# Round 23 - Unified Autonomy Runtime

This round turns the project into one process-level autonomy service while keeping live authority fail-closed.

Design principles derived from the supplied finance references:
- optimize and compare on out-of-sample evidence, not training fit;
- keep Bayesian/online updating and model averaging in the research layer;
- treat RL as stochastic control and require independent evaluation before influence;
- evaluate results against benchmarks and risk-adjusted performance, not raw return alone;
- keep portfolio construction and allocation distinct from single-asset signal quality.

Runtime cadence:
- base trading/supervisor cycle: 60s
- causal forward outcome maturation: 15m
- meta-edge refresh: 15m
- promotion registry refresh: 1h
- RL tournament: 24h
- supervised training/economics/native research: existing supervisor schedules

Safety invariants:
- no automatic live authority
- no automatic model-to-live promotion
- RL remains research-only unless separately qualified
- champion configuration is immutable during an execution cycle
- challengers may retrain continuously in shadow/research
