"""Advisory enrichment of the canonical whole-market packet."""
from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping
import numpy as np

from crypto_ai_swing.agents.component_features import DEFAULT_PRIORS
from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge
from crypto_ai_swing.intelligence.cmc_context import CMCContextCollector
from crypto_ai_swing.intelligence.crypto_news import CryptoNewsCollector
from crypto_ai_swing.research.entry_selector import ProspectiveSwingEntrySelector


def _num(v: Any) -> float | None:
    try: x = float(v)
    except (TypeError, ValueError): return None
    return x if math.isfinite(x) else None


def _signed(v: Any) -> float | None:
    x = _num(v)
    return float(np.clip(x, -1, 1)) if x is not None else None


def _score01(v: Any) -> float | None:
    x = _signed(v)
    return (x + 1) / 2 if x is not None else None


def _cached_cmc(settings) -> dict[str, Any]:
    path = Path(settings.project_root) / "output/crypto_ai_swing/intelligence/cmc/latest.json"
    if not path.is_file(): return {"status": "MISSING", "authority": "CONTEXT_ONLY", "live_decision_influence": False, "reason": "NO_CACHED_CMC_CONTEXT"}
    try: result = json.loads(path.read_text(encoding="utf-8"))
    except Exception: return {"status": "INVALID", "authority": "CONTEXT_ONLY", "live_decision_influence": False}
    result["authority"], result["live_decision_influence"] = "CONTEXT_ONLY", False
    return result


def _cmc_score(cmc: Mapping[str, Any], market: str) -> float | None:
    asset = dict(dict(cmc.get("runtime_assets") or {}).get(market) or {})
    breadth = dict(cmc.get("breadth_top250") or {})
    rows = []
    for raw, scale, w in ((asset.get("percent_change_24h"), 5, .30), (asset.get("percent_change_7d"), 12, .30)):
        x = _num(raw)
        if x is not None: rows.append((float(np.tanh(x / scale)), w))
    for raw, w in ((breadth.get("positive_fraction_24h"), .20), (breadth.get("positive_fraction_7d"), .20)):
        x = _num(raw)
        if x is not None: rows.append(((x - .5) * 2, w))
    d = sum(w for _, w in rows)
    return sum(v * w for v, w in rows) / d if d else None


def _fresh_news(settings, markets: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    bridge = CryptoLibraryBridge(settings.crypto_repo_root)
    collector = CryptoNewsCollector(bridge, output_path=Path(settings.project_root) / "output/crypto_ai_swing/intelligence/news/latest.jsonl")
    try: snap = collector.collect(force=True, persist=True)
    except Exception as exc: return {}, {"status": "BLOCKED", "error": f"{type(exc).__name__}:{str(exc)[:240]}"}
    result = {}
    now = datetime.now(UTC)
    for market in markets:
        docs = [d for d in snap.documents if not (set(str(x).upper() for x in (d.metadata or {}).get("markets", ()) or ())) or market in set(str(x).upper() for x in (d.metadata or {}).get("markets", ()) or ())]
        weighted = total = 0.0
        top = []
        for doc in docs[:50]:
            m = dict(doc.metadata or {}); s = _num(m.get("sentiment_score")); r = max(.05, _num(m.get("relevance_score")) or 0); i = max(.05, _num(m.get("impact_score")) or 0)
            age = max(0, (now - doc.usable_at.astimezone(UTC)).total_seconds()/3600); w = r*i*math.exp(-age/48)
            if s is not None: weighted += s*w; total += w
            if len(top) < 12: top.append({"title": doc.title, "source": doc.source, "url": doc.url, "usable_at": doc.usable_at.isoformat(), "sentiment_score": s, "relevance_score": r, "impact_score": i})
        result[market] = {"status": "READY" if docs else "MISSING", "sentiment_score": weighted/total if total else None, "confidence": float(np.clip(1-math.exp(-len(docs)/8),0,1)), "record_count": len(docs), "top_events": top, "source": "Sjagil/crypto:scrapers.rss"}
    return result, {"status": snap.status, "observed_at": snap.observed_at, "source_statuses": list(snap.source_statuses)}


def _selector_context(market: str, canonical: Mapping[str, Any], cmc: float | None, news: Mapping[str, Any]) -> dict[str, Any]:
    mtf, states = dict(canonical.get("mtf") or {}), dict(canonical.get("timeframes") or {})
    h1, m15 = dict(states.get("1h") or {}), dict(states.get("15m") or {})
    state, over = "NEUTRAL", False
    for tf in ("1h","4h","1d"):
        row = dict(states.get(tf) or {}); over = over or bool(row.get("overextended", False))
        if row.get("breakout_state") and row.get("breakout_state") != "NEUTRAL": state = str(row["breakout_state"]); break
    return {"market": market, "technical": {"technical_score": _signed(mtf.get("swing_score")), "breakout_state": state}, "rally_capture": {"overextended": over}, "mtf_score": _signed(mtf.get("swing_score")), "timeframe_pipeline": {"states": {tf.lower(): {"score": _signed(row.get("technical_score"))} for tf,row in states.items()}}, "mtf_challenger": {"score": _score01(mtf.get("swing_score")), "alignment_score": _num(mtf.get("alignment_score_01")), "cmc_regime_score": _score01(cmc), "execution_score": None, "strategy_family": None, "diagnostics": {"macro": _score01(cmc), "trend": _score01(mtf.get("position_score")), "setup": _score01(mtf.get("swing_score")), "trigger": _score01(h1.get("technical_score")), "h1_rsi": h1.get("rsi_14"), "m15_rsi": m15.get("rsi_14"), "spread_bps": None}}, "nlp_score": _signed(news.get("sentiment_score")), "nlp_confidence": _num(news.get("confidence")), "orderflow_score": None, "authority": "ADVISORY_ONLY", "live_decision_influence": False}


def _descriptive(technical, mtf, cmc, news) -> dict[str, Any]:
    comps = {"technical": technical, "mtf": mtf, "cmc": cmc, "nlp": news}
    rows = [(k,float(v),float(DEFAULT_PRIORS[k])) for k,v in comps.items() if v is not None]
    d = sum(w for _,_,w in rows)
    return {"status": "READY" if d else "MISSING", "score": sum(v*w for _,v,w in rows)/d if d else None, "components": comps, "weights_used": {k:w for k,_,w in rows}, "weight_source": "crypto_ai_swing.agents.component_features.DEFAULT_PRIORS", "contract": "DESCRIPTIVE_ADVISORY_SCORE_NOT_EXECUTION_SCORE"}


def enrich_packet(packet: Mapping[str, Any], settings, *, refresh_external: bool = False) -> dict[str, Any]:
    markets = [str(x).upper() for x in packet.get("markets", [])]
    canonical = dict(packet.get("market_context") or {})
    collector = CMCContextCollector(settings)
    if refresh_external:
        try: cmc = collector.current(markets, force=True)
        except Exception as exc: cmc = {"status": "BLOCKED", "error": f"{type(exc).__name__}:{str(exc)[:240]}", "authority": "CONTEXT_ONLY", "live_decision_influence": False}
    else: cmc = _cached_cmc(settings)
    fresh, news_health = _fresh_news(settings, markets) if refresh_external else ({}, {"status": "CANONICAL_STORED_ONLY"})
    selector = ProspectiveSwingEntrySelector(settings, mode="paper")
    try: policy = selector.refresh(force=True); summary = selector.policy_summary(policy)
    except Exception as exc: policy = {"status": "ERROR", "qualified": False, "reason_codes": [f"{type(exc).__name__}:{str(exc)[:180]}"], "live_decision_influence": False}; summary = dict(policy)
    out = {}
    for market in markets:
        c = dict(canonical.get(market) or {}); n = dict(fresh.get(market) or c.get("news") or {}); cs = _cmc_score(cmc, market); ctx = _selector_context(market, c, cs, n)
        try: decision = selector.evaluate_context(ctx, policy=policy)
        except Exception as exc: decision = {"status": "ERROR", "action": "ABSTAIN", "passes": False, "error": f"{type(exc).__name__}:{str(exc)[:240]}", "authority": "ADVISORY_ONLY", "live_decision_influence": False}
        mtf = dict(c.get("mtf") or {})
        out[market] = {"fundamentals_cmc": {"status": cmc.get("status","MISSING"), "score": cs, "asset": dict(cmc.get("runtime_assets") or {}).get(market), "global_metrics": cmc.get("global_metrics"), "breadth_top250": cmc.get("breadth_top250"), "fear_and_greed": cmc.get("fear_and_greed"), "altcoin_season": cmc.get("altcoin_season"), "source_health": cmc.get("source_health")}, "news": n, "descriptive_advisory": _descriptive(_signed(mtf.get("swing_score")), _signed(mtf.get("all_timeframe_consensus")), cs, _signed(n.get("sentiment_score"))), "selector_context": ctx, "prospective_selector": decision}
    ready = all(out[m]["descriptive_advisory"]["status"] == "READY" for m in markets)
    return {"schema_version": "crypto_ai_swing_whole_context_advisory_v1", "generated_at": datetime.now(UTC).isoformat(), "status": "READY" if ready else "DEGRADED", "markets": markets, "cmc": {"status": cmc.get("status","MISSING"), "authority": "CONTEXT_ONLY", "live_decision_influence": False, "failures": cmc.get("failures") or []}, "news_health": news_health, "selector_policy": summary, "market_advisory": out, "authority": {"authority": "ADVISORY_ONLY", "canonical_execution_authority": "SJAGIL_CRYPTO_ONLY", "live_decision_influence": False, "automatic_live_promotion": False}, "safety": {"private_exchange_requests": 0, "orders_generated": 0, "orders_submitted": 0}}
