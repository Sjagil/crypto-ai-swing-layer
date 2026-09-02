import numpy as np, pandas as pd
from datetime import datetime, timezone
from crypto_ai_swing.data.features import build_features
from crypto_ai_swing.quant.evidence import probability_diagnostics
from crypto_ai_swing.orchestration.timeframe_pipeline import evaluate_timeframe_pipeline

def frame(drift=.001,freq='1h',n=320):
    rng=np.random.default_rng(5); idx=pd.date_range('2025-01-01',periods=n,freq=freq,tz='UTC'); ret=rng.normal(drift,.003,n); close=100*np.exp(np.cumsum(ret)); op=np.r_[close[0],close[:-1]]; span=np.maximum(close*.001,.02)
    return pd.DataFrame({'open':op,'high':np.maximum(op,close)+span,'low':np.minimum(op,close)-span,'close':close,'volume':np.linspace(1000,1800,n)},index=idx)

def test_round11_quant_features():
    f=build_features(frame())
    for c in ('downside_rv_24','ewma_rv_24','skew_24','kurtosis_24','momentum_vol_adj_8','drawdown_48','atr_z_168','tail_q05_168'): assert c in f and len(f[c].dropna())>0

def test_round11_probability_quality():
    q=probability_diagnostics([0,0,0,1,1,1],[.05,.15,.25,.75,.85,.95]); assert q['brier_skill']>0 and q['expected_calibration_error']<.3

def test_round11_macro_cannot_be_overridden_by_15m():
    bull=frame(.001); bear=frame(-.001)
    d=evaluate_timeframe_pipeline({'1w':bear,'1d':bear,'4h':bull,'2h':bull,'1h':bull,'15m':bull},{'spread_bps':3,'book_imbalance':.2,'cvd_notional_ratio':.2},observed_at=datetime.now(timezone.utc))
    assert d.entry_blocked and 'MTF_MACRO_LONG_PERMISSION_DENIED' in d.blockers
