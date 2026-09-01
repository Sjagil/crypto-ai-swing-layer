import numpy as np
import pandas as pd
from crypto_ai_swing.agents.dataset import build_agent_dataset,purged_chronological_split
from crypto_ai_swing.agents.council import build_council_decision

def _frame(rows=800):
    idx=pd.date_range("2025-01-01",periods=rows,freq="1h",tz="UTC")
    rng=np.random.default_rng(17);ret=rng.normal(.0002,.008,rows)
    close=50000*np.exp(np.cumsum(ret));open_=np.r_[close[0],close[:-1]];span=close*.003
    return pd.DataFrame({"open":open_,"high":np.maximum(open_,close)+span,
        "low":np.minimum(open_,close)-span,"close":close,
        "volume":rng.lognormal(7,.5,rows)},index=idx)

def test_future_labels_and_purged_split():
    ds=build_agent_dataset({"BTC-EUR":_frame(),"ETH-EUR":_frame()},horizon_bars=4)
    assert (pd.to_datetime(ds.frame["label_end_time"],utc=True)>pd.to_datetime(ds.frame["feature_time"],utc=True)).all()
    train,val,test=purged_chronological_split(ds)
    assert pd.Timestamp(train["label_end_time"].max())<pd.Timestamp(val["feature_time"].min())
    assert pd.Timestamp(val["label_end_time"].max())<pd.Timestamp(test["feature_time"].min())

def test_shadow_agent_cannot_silently_gain_live_authority():
    pred={"alpha_probability":.1,"regime_probability":.1,"predicted_return":-.02,"predicted_mae":.1}
    shadow=build_council_decision(pred,{"spread_bps":1},artifact_status="SHADOW",live_decision_influence=False,mode="shadow")
    live=build_council_decision(pred,{"spread_bps":1},artifact_status="SHADOW",live_decision_influence=False,mode="live")
    assert shadow.entry_blocked is True
    assert live.entry_blocked is False
    assert live.live_influence is False
