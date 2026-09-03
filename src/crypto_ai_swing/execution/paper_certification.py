from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, getcontext
import json
from pathlib import Path
from typing import Any


getcontext().prec = 28
BPS = Decimal("10000")


@dataclass(frozen=True)
class PaperFill:
    side: str
    reference_price: Decimal
    fill_price: Decimal
    quantity: Decimal
    notional_eur: Decimal
    fee_eur: Decimal
    slippage_bps: Decimal


@dataclass(frozen=True)
class PaperRoundTrip:
    entry: PaperFill
    exit: PaperFill
    realized_pnl_eur: Decimal
    realized_return: Decimal


def _fill(
    *,
    side: str,
    reference_price: Decimal,
    quantity: Decimal | None,
    notional_eur: Decimal | None,
    fee_bps: Decimal,
    slippage_bps: Decimal,
) -> PaperFill:
    selected = side.upper()
    direction = Decimal("1") if selected == "BUY" else Decimal("-1")
    fill_price = reference_price * (
        Decimal("1") + direction * slippage_bps / BPS
    )

    if selected == "BUY":
        if notional_eur is None or notional_eur <= 0:
            raise ValueError("BUY requires positive notional")
        fee = notional_eur * fee_bps / BPS
        spendable = notional_eur - fee
        selected_quantity = spendable / fill_price
        return PaperFill(
            selected,
            reference_price,
            fill_price,
            selected_quantity,
            notional_eur,
            fee,
            slippage_bps,
        )

    if quantity is None or quantity <= 0:
        raise ValueError("SELL requires positive quantity")
    gross = quantity * fill_price
    fee = gross * fee_bps / BPS
    return PaperFill(
        selected,
        reference_price,
        fill_price,
        quantity,
        gross,
        fee,
        slippage_bps,
    )


def simulate_round_trip(
    *,
    entry_reference: Decimal,
    exit_reference: Decimal,
    notional_eur: Decimal,
    fee_bps_per_side: Decimal,
    slippage_bps_per_side: Decimal,
) -> PaperRoundTrip:
    entry = _fill(
        side="BUY",
        reference_price=entry_reference,
        quantity=None,
        notional_eur=notional_eur,
        fee_bps=fee_bps_per_side,
        slippage_bps=slippage_bps_per_side,
    )
    exit_fill = _fill(
        side="SELL",
        reference_price=exit_reference,
        quantity=entry.quantity,
        notional_eur=None,
        fee_bps=fee_bps_per_side,
        slippage_bps=slippage_bps_per_side,
    )
    proceeds = (
        exit_fill.notional_eur
        - exit_fill.fee_eur
    )
    pnl = proceeds - notional_eur
    return PaperRoundTrip(
        entry=entry,
        exit=exit_fill,
        realized_pnl_eur=pnl,
        realized_return=pnl / notional_eur,
    )


def _json_fill(fill: PaperFill) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Decimal) else value
        for key, value in asdict(fill).items()
    }


def certify_paper_lifecycle(
    project_root: Path,
    *,
    fee_bps_per_side: float,
    slippage_bps_per_side: float,
) -> dict[str, Any]:
    """Offline accounting certification. No network path is imported."""

    notional = Decimal("1000")
    entry = Decimal("100")
    target_exit = Decimal("103")
    stop_exit = Decimal("98")

    target = simulate_round_trip(
        entry_reference=entry,
        exit_reference=target_exit,
        notional_eur=notional,
        fee_bps_per_side=Decimal(
            str(fee_bps_per_side)
        ),
        slippage_bps_per_side=Decimal(
            str(slippage_bps_per_side)
        ),
    )
    stop = simulate_round_trip(
        entry_reference=entry,
        exit_reference=stop_exit,
        notional_eur=notional,
        fee_bps_per_side=Decimal(
            str(fee_bps_per_side)
        ),
        slippage_bps_per_side=Decimal(
            str(slippage_bps_per_side)
        ),
    )

    checks = {
        "target_roundtrip_positive_after_costs": (
            target.realized_pnl_eur > 0
        ),
        "stop_roundtrip_negative_after_costs": (
            stop.realized_pnl_eur < 0
        ),
        "entry_fee_positive": target.entry.fee_eur > 0,
        "exit_fee_positive": target.exit.fee_eur > 0,
        "buy_slippage_adverse": (
            target.entry.fill_price > entry
        ),
        "sell_slippage_adverse": (
            target.exit.fill_price < target_exit
        ),
        "network_calls": 0,
        "live_authority_used": False,
        "orders_submitted": 0,
    }
    boolean_checks = (
        "target_roundtrip_positive_after_costs",
        "stop_roundtrip_negative_after_costs",
        "entry_fee_positive",
        "exit_fee_positive",
        "buy_slippage_adverse",
        "sell_slippage_adverse",
    )
    passed = (
        all(bool(checks[key]) for key in boolean_checks)
        and checks["network_calls"] == 0
        and checks["orders_submitted"] == 0
        and checks["live_authority_used"] is False
    )

    payload = {
        "schema_version": (
            "crypto_ai_swing_paper_certification_v1"
        ),
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "costs": {
            "fee_bps_per_side": fee_bps_per_side,
            "slippage_bps_per_side": slippage_bps_per_side,
        },
        "target_scenario": {
            "entry": _json_fill(target.entry),
            "exit": _json_fill(target.exit),
            "realized_pnl_eur": str(
                target.realized_pnl_eur
            ),
            "realized_return": str(
                target.realized_return
            ),
        },
        "stop_scenario": {
            "entry": _json_fill(stop.entry),
            "exit": _json_fill(stop.exit),
            "realized_pnl_eur": str(
                stop.realized_pnl_eur
            ),
            "realized_return": str(
                stop.realized_return
            ),
        },
        "scope": (
            "DETERMINISTIC_OFFLINE_PAPER_ECONOMICS_"
            "NO_EXCHANGE_ROUTE"
        ),
        "live_authority_used": False,
        "orders_submitted": 0,
    }

    out = (
        Path(project_root)
        / "output/crypto_ai_swing/modes/paper"
        / "certification/latest.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    payload["output"] = str(out)
    return payload
