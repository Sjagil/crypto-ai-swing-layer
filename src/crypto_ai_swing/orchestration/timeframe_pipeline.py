from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Mapping
import numpy as np
import pandas as pd
from crypto_ai_swing.data.features import build_features

@dataclass(frozen=True)
class MTFDecision:
    observed_at: str; macro_score: float; trend_score: float; setup_score: float; trigger_score: float; execution_score: float; mtf_score: float
    entry_blocked: bool; blockers: tuple[str,...]; states: dict[str,dict[str,Any]]
    def to_dict(self): return asdict(self)

def _state(tf,frame):
    if frame is None or frame.empty: return None
    f=build_features(frame).replace([np.inf,-np.inf],np.nan).dropna()
    if f.empty:return None
    r=f.iloc[-1]; atr=max(1e-6,float(r.get('atr_pct',0))); rv=max(1e-6,float(r.get('rv_24',0)))
    trend=float(np.clip(.6*np.tanh(float(r.get('trend_20_50',0))/atr)+.4*np.tanh(float(r.get('trend_50_200',0))/atr),-1,1))
    momentum=float(np.tanh(float(r.get('ret_8',0))/max(1e-6,rv*np.sqrt(8))))
    breakout=float(np.tanh(float(r.get('breakout_20',0))/atr)); rsi=float(r.get('rsi_14',50)); rsi_score=float(np.clip((rsi-50)/20,-1,1))
    score=float(np.clip(.45*trend+.30*momentum+.15*breakout+.10*rsi_score,-1,1))
    return {'timeframe':tf,'rows':len(frame),'score':score,'trend':trend,'momentum':momentum,'breakout':breakout,'rsi':rsi,'latest_bar':pd.Timestamp(f.index[-1]).isoformat()}

def _weighted(states,weights):
    rows=[(states[k]['score'],v) for k,v in weights.items() if k in states]
    return float(np.clip(sum(a*b for a,b in rows)/sum(b for _,b in rows),-1,1)) if rows else 0.0

def evaluate_timeframe_pipeline(frames:Mapping[str,pd.DataFrame], micro:Mapping[str,Any], *, observed_at:datetime|pd.Timestamp, maximum_spread_bps:float=35.0)->MTFDecision:
    aliases={'1W':'1w'}; states={}
    for raw,frame in frames.items():
        tf=aliases.get(str(raw),str(raw)); row=_state(tf,frame)
        if row: states[tf]=row
    blockers=[]
    for tf in ('1d','4h','1h','15m'):
        if tf not in states:blockers.append(f'MTF_REQUIRED_FRAME_MISSING:{tf}')
    macro=_weighted(states,{'1d':.70,'1w':.30}); trend=_weighted(states,{'4h':.62,'2h':.38})
    setup=states.get('1h',{}).get('score',-1.0); trigger=states.get('15m',{}).get('score',-1.0)
    daily=states.get('1d'); weekly=states.get('1w'); h4=states.get('4h'); h2=states.get('2h')
    if daily and daily['score']<0 and (not weekly or weekly['score']<=0):blockers.append('MTF_MACRO_LONG_PERMISSION_DENIED')
    if h4 and h4['score']<0 and (not h2 or h2['score']<=0):blockers.append('MTF_TREND_LONG_PERMISSION_DENIED')
    if '1h' in states and setup<0:blockers.append('MTF_SETUP_NOT_BULLISH')
    if '15m' in states and trigger<-.35:blockers.append('MTF_TRIGGER_STRONGLY_ADVERSE')
    spread=float(micro.get('spread_bps',999) or 999); book=float(np.clip(float(micro.get('book_imbalance',0) or 0),-1,1)); cvd=float(np.clip(float(micro.get('cvd_notional_ratio',micro.get('cvd_ratio',0)) or 0),-1,1))
    execution=float(np.clip(.55*(1-spread/max(1,maximum_spread_bps))+.25*book+.20*cvd,-1,1))
    if not np.isfinite(spread) or spread>maximum_spread_bps:blockers.append('MTF_EXECUTION_SPREAD_TOO_WIDE')
    mtf=float(np.clip(.30*macro+.27*trend+.23*setup+.12*trigger+.08*execution,-1,1))
    ts=pd.Timestamp(observed_at); ts=ts.tz_localize('UTC') if ts.tzinfo is None else ts.tz_convert('UTC')
    return MTFDecision(ts.isoformat(),macro,trend,setup,trigger,execution,mtf,bool(blockers),tuple(blockers),states)
