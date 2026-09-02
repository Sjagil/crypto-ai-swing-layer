from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge


def probability_diagnostics(y_true, probability, bins: int = 10) -> dict[str, Any]:
    y=np.asarray(y_true,dtype=int).reshape(-1); p=np.clip(np.asarray(probability,dtype=float).reshape(-1),1e-6,1-1e-6)
    if len(y)!=len(p) or len(y)==0: raise ValueError('unaligned probability arrays')
    base=float(y.mean()); baseline=np.full_like(p,base,dtype=float)
    brier=float(brier_score_loss(y,p)); base_brier=float(brier_score_loss(y,baseline))
    edges=np.linspace(0,1,max(2,int(bins))+1); rows=[]; ece=0.0
    for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        mask=(p>=lo)&((p<=hi) if i==len(edges)-2 else (p<hi)); count=int(mask.sum())
        if not count: continue
        pred=float(p[mask].mean()); obs=float(y[mask].mean()); ece += count/len(y)*abs(pred-obs)
        rows.append({'lower':float(lo),'upper':float(hi),'count':count,'mean_probability':pred,'observed_rate':obs,'absolute_gap':abs(pred-obs)})
    return {'observations':len(y),'positive_rate':base,'brier':brier,'baseline_brier':base_brier,
            'brier_skill':float(1-brier/max(base_brier,1e-12)),'log_loss':float(log_loss(y,p,labels=[0,1])),
            'average_precision':float(average_precision_score(y,p)) if len(np.unique(y))>1 else None,
            'expected_calibration_error':float(ece),'reliability_bins':rows}


def _trial_path(frame: pd.DataFrame, mask: np.ndarray, cost: float, horizon: int) -> pd.Series:
    realized=frame['target_forward_return'].to_numpy(float); mask=np.asarray(mask,bool)
    table=pd.DataFrame({'t':pd.to_datetime(frame['feature_time'],utc=True),'r':np.where(mask,realized-cost,0.0)})
    return table.groupby('t')['r'].mean().sort_index().iloc[::max(1,int(horizon))]


def native_model_selection_evidence(validation: pd.DataFrame, probabilities: Mapping[str,np.ndarray], threshold_rows: Mapping[str,list[dict]], *, cost_floor: float, horizon_bars: int, crypto_repo_root: Path) -> dict[str,Any]:
    paths={}
    for name,p in probabilities.items():
        for row in threshold_rows.get(name,[]):
            t=float(row['threshold']); key=f'{name}@{t:.3f}'
            paths[key]=_trial_path(validation,np.asarray(p)>=t,cost_floor,horizon_bars).rename(key)
    if not paths: return {'status':'NO_TRIAL_PATHS','known_trial_count':0}
    matrix=pd.concat(paths.values(),axis=1).fillna(0.0)
    if len(matrix)<12: return {'status':'INSUFFICIENT_NON_OVERLAPPING_OBSERVATIONS','known_trial_count':len(paths),'observation_count':len(matrix)}
    bridge=CryptoLibraryBridge(Path(crypto_repo_root)); fn=bridge.import_module('research.optimization').multiple_testing_bootstrap
    result=fn(matrix,bootstrap_samples=1000,block_size=max(1,min(5,int(horizon_bars))),seed=17,known_trial_count=len(paths))
    return {'status':'READY',**asdict(result),'native_backend':'Sjagil/crypto:research.optimization.multiple_testing_bootstrap'}


def native_hac_evidence(test: pd.DataFrame, selected: np.ndarray, *, cost_floor: float, horizon_bars: int, crypto_repo_root: Path) -> dict[str,Any]:
    path=_trial_path(test,selected,cost_floor,horizon_bars)
    if len(path)<3: return {'status':'INSUFFICIENT_PATH','observations':len(path)}
    bridge=CryptoLibraryBridge(Path(crypto_repo_root)); fn=bridge.import_module('research.statistical_evidence').hac_effective_sample_size
    return {'status':'READY','path_observations':len(path),'mean_net':float(path.mean()),'hac':fn(path),'native_backend':'Sjagil/crypto:research.statistical_evidence.hac_effective_sample_size'}
