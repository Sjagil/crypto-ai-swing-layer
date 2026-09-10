from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pandas as pd

from crypto_ai_swing.agents.manager import AgentManager
from crypto_ai_swing.agents.rl_runtime import RLRuntime
from crypto_ai_swing.agents.runtime import AgentRuntime
from crypto_ai_swing.bridge.canonical_portfolio import (
    CanonicalPortfolioRiskBridge,
)
from crypto_ai_swing.bridge.crypto_library import CryptoLibraryBridge
from crypto_ai_swing.contracts import Authority, ModelVote, Side, Signal
from crypto_ai_swing.data.features import build_features
from crypto_ai_swing.decision_packet import DecisionPacket


class Round41DataFlowAudit:
    """Read-only end-to-end canonical dataflow audit. Never submits orders."""

    def __init__(self, settings) -> None:
        self.settings = settings
        self.crypto = CryptoLibraryBridge(settings.crypto_repo_root)
        self.agent_runtime = AgentRuntime(settings, mode="shadow")
        self.rl_runtime = RLRuntime(settings)
        self.agent_manager = AgentManager(settings, mode="shadow")
        self.portfolio = CanonicalPortfolioRiskBridge(
            settings.crypto_repo_root,
            project_root=settings.project_root,
            risk_config=settings.risk,
        )

    @staticmethod
    def _frame_check(
        market: str,
        timeframe: str,
        frame: pd.DataFrame,
    ) -> dict[str, Any]:
        required = {"open", "high", "low", "close", "volume"}
        columns = {str(column) for column in frame.columns}
        positive = False
        if not frame.empty and "close" in frame.columns:
            close = pd.to_numeric(frame["close"], errors="coerce").dropna()
            positive = bool(not close.empty and (close > 0).all())
        return {
            "market": market,
            "timeframe": timeframe,
            "rows": len(frame),
            "columns": sorted(columns),
            "has_required_columns": required <= columns,
            "positive_close_values": positive,
            "latest_timestamp": (
                frame.index[-1].isoformat() if not frame.empty else None
            ),
            "ready": bool(
                len(frame) >= 30
                and required <= columns
                and positive
            ),
        }

    def run(
        self,
        *,
        markets: list[str],
        timeframes: list[str],
        network: bool,
        require_trained_agents: bool = False,
        deep_context: bool = False,
    ) -> dict[str, Any]:
        selected_markets = [
            str(value).upper()
            for value in markets
            if str(value).strip()
        ]
        selected_timeframes = [
            str(value)
            for value in timeframes
            if str(value).strip()
        ]
        imports = self.crypto.integration_status()
        frame_rows: list[dict[str, Any]] = []
        frames_by_tf: dict[str, dict[str, pd.DataFrame]] = {}
        errors: list[str] = []

        if network:
            for timeframe in selected_timeframes:
                try:
                    frames = self.crypto.ohlcv_many(
                        selected_markets,
                        timeframe,
                        persist=False,
                        concurrency=min(4, max(1, len(selected_markets))),
                    )
                    frames_by_tf[timeframe] = frames
                    for market in selected_markets:
                        frame_rows.append(
                            self._frame_check(
                                market,
                                timeframe,
                                frames.get(market, pd.DataFrame()),
                            )
                        )
                except Exception as exc:
                    errors.append(
                        f"OHLCV_{timeframe}:"
                        f"{type(exc).__name__}:{str(exc)[:300]}"
                    )
        else:
            errors.append("NETWORK_PROBE_SKIPPED")

        primary_tf = (
            "1h"
            if "1h" in frames_by_tf
            else next(iter(frames_by_tf), None)
        )
        primary_market = (
            selected_markets[0] if selected_markets else "BTC-EUR"
        )
        primary_frame = (
            frames_by_tf.get(primary_tf, {}).get(primary_market)
            if primary_tf
            else None
        )

        bundle_result: dict[str, Any] = {
            "ready": False,
            "status": "NOT_RUN",
        }
        if network and primary_tf:
            try:
                bundle = self.crypto.market_bundle(
                    primary_market,
                    primary_tf,
                    mode="shadow",
                    persist=False,
                    depth=20,
                )
                bundle_audit = self.crypto.market_bundle_audit(bundle)
                best_bid = float(
                    bundle.microstructure.get("best_bid", 0.0) or 0.0
                )
                best_ask = float(
                    bundle.microstructure.get("best_ask", 0.0) or 0.0
                )
                bundle_result = {
                    "ready": bool(
                        not bundle.frame.empty
                        and bool(bundle.ticker)
                        and bool(bundle.orderbook)
                        and best_bid > 0.0
                        and best_ask > 0.0
                    ),
                    "status": "RECEIVED",
                    "market": bundle.market,
                    "timeframe": bundle.timeframe,
                    "ohlcv_rows": len(bundle.frame),
                    "ticker_keys": sorted(bundle.ticker)[:40],
                    "trade_count": len(bundle.trades),
                    "orderbook_keys": sorted(bundle.orderbook)[:40],
                    "microstructure": dict(bundle.microstructure),
                    "audit": bundle_audit,
                }
            except Exception as exc:
                bundle_result = {
                    "ready": False,
                    "status": "ERROR",
                    "error": f"{type(exc).__name__}:{str(exc)[:500]}",
                }

        deep_result: dict[str, Any] = {
            "requested": deep_context,
            "status": "NOT_REQUESTED",
            "ready": not deep_context,
        }
        if network and deep_context:
            try:
                context = self.crypto.prospective_context(
                    selected_markets,
                    cache_seconds=0.0,
                )
                status = str(context.get("status") or "UNKNOWN").upper()
                deep_result = {
                    "requested": True,
                    "status": status,
                    "ready": status not in {
                        "FAILED",
                        "BLOCKED",
                        "NOT_READY",
                    },
                    "keys": sorted(str(key) for key in context),
                    "payload": context,
                }
            except Exception as exc:
                deep_result = {
                    "requested": True,
                    "status": "ERROR",
                    "ready": False,
                    "error": f"{type(exc).__name__}:{str(exc)[:500]}",
                }

        feature_result: dict[str, Any] = {
            "ready": False,
            "status": "NO_FRAME",
        }
        agent_result: dict[str, Any] = {
            "ready": False,
            "status": "NO_FRAME",
        }
        risk_result: dict[str, Any] = {
            "ready": False,
            "status": "NO_FRAME",
        }

        if primary_frame is not None and not primary_frame.empty:
            features = build_features(primary_frame).dropna()
            feature_result = {
                "ready": not features.empty,
                "status": "BUILT" if not features.empty else "EMPTY",
                "rows": len(features),
                "columns": list(features.columns),
            }

            agent = self.agent_runtime.predict_frame(
                primary_market,
                primary_frame,
                {"data_source": "Sjagil/crypto"},
            )
            rl = self.rl_runtime.predict_frame(primary_frame)
            supervised_status = self.agent_runtime.status()
            rl_status = self.rl_runtime.status()
            supervised_state = str(
                supervised_status.get("status") or "NOT_TRAINED"
            ).upper()
            trained_ok = supervised_state not in {
                "NOT_TRAINED",
                "ERROR",
                "EXPIRED",
            }
            agent_result = {
                "ready": bool(trained_ok or not require_trained_agents),
                "supervised_status": supervised_status,
                "supervised_decision": {
                    key: getattr(agent, key)
                    for key in (
                        "alpha_probability",
                        "forecast_score",
                        "regime_score",
                        "predicted_return",
                        "predicted_mae",
                        "entry_blocked",
                    )
                    if hasattr(agent, key)
                },
                "rl_status": rl_status,
                "rl_decision": rl,
                "trained_required": require_trained_agents,
            }

            price = float(primary_frame["close"].iloc[-1])
            probe_signal = Signal(
                market=primary_market,
                timestamp=(
                    primary_frame.index[-1].to_pydatetime()
                    if hasattr(primary_frame.index[-1], "to_pydatetime")
                    else datetime.now(UTC)
                ),
                side=Side.BUY,
                score=0.74,
                confidence=0.70,
                expected_edge_bps=140.0,
                stop_pct=0.02,
                take_profit_pct=0.06,
                trailing_stop_pct=0.02,
                strategy="ROUND41_READ_ONLY_PLUMBING_PROBE",
                edge_source="READ_ONLY_DATAFLOW_PROBE",
                votes=(
                    ModelVote(
                        source="round41_probe",
                        score=0.74,
                        confidence=0.70,
                    ),
                ),
                features={"price": price, "atr_pct": 0.02},
            )
            packet = DecisionPacket.from_signal(
                probe_signal,
                context={
                    "ml_probability": 0.58,
                    "agents": {
                        "alpha_probability": 0.58,
                        "predicted_return": 0.01,
                        "predicted_mae": 0.015,
                        "diagnostics": {
                            "head_influence": {"alpha": True}
                        },
                    },
                    "data_source": "Sjagil/crypto",
                    "prospective_context_status": "READY",
                },
                authority=Authority.SHADOW,
            )
            decision = self.portfolio.assess(
                packet,
                equity_eur=Decimal(2000),
                cash_eur=Decimal(2000),
                exposure_eur=Decimal(0),
                open_risk_eur=Decimal(0),
                positions=[],
                frames=frames_by_tf.get(primary_tf, {}),
                reconciled=True,
                data_healthy=True,
                risk_manager_healthy=True,
                intelligence_timing_healthy=True,
            )
            risk_result = {
                "ready": True,
                "status": "CANONICAL_RISK_PATH_REACHED",
                "decision_packet_hash": packet.canonical_hash(),
                "canonical_decision": decision.to_dict(),
                "orders_generated": 0,
                "orders_submitted": 0,
            }

        manager_status = self.agent_manager.status()
        data_ready = bool(frame_rows) and all(
            bool(row.get("ready")) for row in frame_rows
        )
        overall_ready = bool(
            imports.get("ready")
            and data_ready
            and bundle_result.get("ready")
            and deep_result.get("ready")
            and feature_result.get("ready")
            and agent_result.get("ready")
            and risk_result.get("ready")
        )
        return {
            "schema_version": "round41_dataflow_audit_v1",
            "checked_at": datetime.now(UTC).isoformat(),
            "ready": overall_ready,
            "network_probe": network,
            "canonical_imports": imports,
            "data": {
                "ready": data_ready,
                "frames": frame_rows,
                "market_bundle": bundle_result,
                "deep_context": deep_result,
            },
            "features": feature_result,
            "agents": agent_result,
            "agent_manager": manager_status,
            "canonical_portfolio_risk": risk_result,
            "errors": errors,
            "live_authority_granted": False,
            "orders_generated": 0,
            "orders_submitted": 0,
        }
