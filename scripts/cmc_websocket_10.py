from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv

CMC_WS_ENDPOINT = "wss://pro-stream.coinmarketcap.com/v1"
CMC_MAP_ENDPOINT = (
    "https://pro-api.coinmarketcap.com/public-api/v1/cryptocurrency/map"
)
CMC_WS_CHANNEL = "market@crypto_latest_price"
CMC_WS_CREDIT_PER_MESSAGE = 0.025


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _extract_markets(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("markets")
    if not isinstance(raw, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for value in raw:
        market = (
            str(value)
            .strip()
            .upper()
            .replace("/", "-")
            .replace("_", "-")
        )
        if market and market not in seen:
            seen.add(market)
            result.append(market)
    return result


def _base_symbol(market: str) -> str:
    return str(market).upper().split("-", 1)[0].strip()


def _split_shards(values: list[int], count: int) -> list[list[int]]:
    if count < 1:
        raise ValueError("connection count must be positive")
    unique = list(dict.fromkeys(int(value) for value in values))
    if not unique:
        return []
    shards = [[] for _ in range(min(count, len(unique)))]
    for index, value in enumerate(unique):
        shards[index % len(shards)].append(value)
    return shards


def _pick_asset_rows(
    rows: Iterable[dict[str, Any]],
    symbols: Iterable[str],
) -> dict[str, int]:
    wanted = {str(symbol).upper() for symbol in symbols}
    candidates: dict[str, list[dict[str, Any]]] = {
        symbol: [] for symbol in wanted
    }
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").upper()
        if symbol in candidates:
            candidates[symbol].append(row)

    result: dict[str, int] = {}
    for symbol, options in candidates.items():

        def key(row: dict[str, Any]) -> tuple[int, float, int]:
            active = 0 if int(row.get("is_active") or 0) == 1 else 1
            try:
                rank = float(row.get("rank"))
            except (TypeError, ValueError):
                rank = float("inf")
            try:
                cmc_id = int(row["id"])
            except (KeyError, TypeError, ValueError):
                cmc_id = 2**31 - 1
            return active, rank, cmc_id

        for row in sorted(options, key=key):
            try:
                result[symbol] = int(row["id"])
                break
            except (KeyError, TypeError, ValueError):
                continue
    return result


def _resolve_asset_ids(
    symbols: list[str],
    cache_path: Path,
    *,
    cache_ttl_seconds: int,
) -> dict[str, int]:
    symbols = sorted(
        {
            str(symbol).upper()
            for symbol in symbols
            if str(symbol).strip()
        }
    )
    cached = _load_json(cache_path)
    if (
        cached
        and cache_path.is_file()
        and time.time() - cache_path.stat().st_mtime <= cache_ttl_seconds
    ):
        mapping = cached.get("mapping")
        if isinstance(mapping, dict) and set(symbols).issubset(
            {str(key).upper() for key in mapping}
        ):
            return {
                symbol: int(mapping[symbol])
                for symbol in symbols
            }

    query = urllib.parse.urlencode(
        {
            "symbol": ",".join(symbols),
            "aux": "is_active,status",
        }
    )
    request = urllib.request.Request(
        f"{CMC_MAP_ENDPOINT}?{query}",
        headers={
            "Accept": "application/json",
            "User-Agent": "crypto-ai-swing-layer/cmc-ws",
        },
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))

    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError("CMC map returned no data list")

    mapping = _pick_asset_rows(rows, symbols)
    missing = sorted(set(symbols) - set(mapping))
    if missing:
        raise RuntimeError(
            f"CMC map missing symbols: {','.join(missing)}"
        )

    _atomic_json(
        cache_path,
        {
            "schema_version": "cmc_ws_asset_map_v1",
            "generated_at": _utc_now(),
            "mapping": mapping,
        },
    )
    return mapping


@dataclass
class ConnectionState:
    connection_id: int
    crypto_ids: tuple[int, ...]
    connected: bool = False
    reconnects: int = 0
    messages: int = 0
    data_messages: int = 0
    last_message_at: str | None = None
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "connection_id": self.connection_id,
            "crypto_ids": list(self.crypto_ids),
            "connected": self.connected,
            "reconnects": self.reconnects,
            "messages": self.messages,
            "data_messages": self.data_messages,
            "last_message_at": self.last_message_at,
            "last_error": self.last_error,
        }


class CMCShardedWebSocketCollector:
    def __init__(
        self,
        *,
        api_key: str,
        root: Path,
        market_to_id: dict[str, int],
        connections: int = 10,
        endpoint: str = CMC_WS_ENDPOINT,
        flush_seconds: float = 1.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("CoinMarketCap API key is required")
        if connections < 1 or connections > 10:
            raise ValueError("connections must be between 1 and 10")
        if not market_to_id:
            raise ValueError("market_to_id cannot be empty")

        self.api_key = api_key.strip()
        self.root = Path(root)
        self.state_root = self.root / "state"
        self.raw_root = self.root / "websocket" / "raw"
        self.latest_path = self.state_root / "websocket_latest.json"
        self.status_path = self.state_root / "websocket_status.json"
        self.endpoint = endpoint
        self.flush_seconds = max(0.25, float(flush_seconds))
        self.market_to_id = dict(
            sorted(
                (str(market).upper(), int(cmc_id))
                for market, cmc_id in market_to_id.items()
            )
        )
        self.id_to_market = {
            cmc_id: market
            for market, cmc_id in self.market_to_id.items()
        }

        shards = _split_shards(
            list(self.market_to_id.values()),
            connections,
        )
        self.states = {
            index: ConnectionState(index, tuple(shard))
            for index, shard in enumerate(shards, start=1)
        }
        self.requested_connections = int(connections)
        self.events: dict[str, dict[str, Any]] = {}
        self.message_count_total = 0
        self.data_message_count = 0
        self.error_count = 0
        self.started_at = _utc_now()
        self._dirty = True
        self._lock = asyncio.Lock()
        self._archive_queue: asyncio.Queue[dict[str, Any]] = (
            asyncio.Queue(maxsize=50_000)
        )
        self._stop = asyncio.Event()

    async def _snapshot(self) -> dict[str, Any]:
        async with self._lock:
            return {
                "schema_version": "cmc_websocket_sharded_v1",
                "generated_at": _utc_now(),
                "started_at": self.started_at,
                "endpoint": self.endpoint,
                "channel": CMC_WS_CHANNEL,
                "requested_connections": self.requested_connections,
                "configured_connections": len(self.states),
                "active_connections": sum(
                    1
                    for state in self.states.values()
                    if state.connected
                ),
                "asset_count": len(self.market_to_id),
                "market_to_cmc_id": self.market_to_id,
                "connections": {
                    str(index): state.to_dict()
                    for index, state in sorted(self.states.items())
                },
                "message_count_total": self.message_count_total,
                "data_message_count": self.data_message_count,
                "estimated_ws_credits": round(
                    self.message_count_total
                    * CMC_WS_CREDIT_PER_MESSAGE,
                    6,
                ),
                "error_count": self.error_count,
                "events": dict(self.events),
                "execution_authority": False,
                "capital_authority": False,
                "context_only": True,
            }

    async def _writer(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(self.flush_seconds)
            if not self._dirty:
                continue
            payload = await self._snapshot()
            _atomic_json(self.latest_path, payload)
            _atomic_json(
                self.status_path,
                {
                    key: value
                    for key, value in payload.items()
                    if key != "events"
                },
            )
            self._dirty = False

    async def _archiver(self) -> None:
        current_date = ""
        handle = None
        try:
            while (
                not self._stop.is_set()
                or not self._archive_queue.empty()
            ):
                try:
                    event = await asyncio.wait_for(
                        self._archive_queue.get(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue

                date = datetime.now(UTC).strftime("%Y-%m-%d")
                if date != current_date:
                    if handle is not None:
                        handle.close()
                    self.raw_root.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    handle = (
                        self.raw_root
                        / f"{date}.jsonl"
                    ).open(
                        "a",
                        encoding="utf-8",
                    )
                    current_date = date

                handle.write(
                    json.dumps(
                        event,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )
                handle.flush()
                self._archive_queue.task_done()
        finally:
            if handle is not None:
                handle.close()

    async def _record_message(
        self,
        connection_id: int,
        message: dict[str, Any],
    ) -> None:
        now = _utc_now()
        async with self._lock:
            state = self.states[connection_id]
            state.messages += 1
            state.last_message_at = now
            self.message_count_total += 1
            message_type = str(message.get("type") or "")

            if message_type == "data":
                state.data_messages += 1
                self.data_message_count += 1
                data = message.get("data")
                cmc_id = None
                if isinstance(data, dict):
                    try:
                        cmc_id = int(data.get("cid"))
                    except (TypeError, ValueError):
                        cmc_id = None

                event_key = (
                    f"{message.get('channel') or 'unknown'}:"
                    f"{cmc_id if cmc_id is not None else connection_id}"
                )
                event = dict(message)
                event["received_at"] = now
                event["connection_id"] = connection_id
                if (
                    cmc_id is not None
                    and cmc_id in self.id_to_market
                ):
                    event["market"] = self.id_to_market[cmc_id]
                self.events[event_key] = event

            elif message_type == "error":
                self.error_count += 1
                state.last_error = json.dumps(
                    message,
                    separators=(",", ":"),
                )[:500]

            self._dirty = True

        archive = dict(message)
        archive["received_at"] = now
        archive["connection_id"] = connection_id
        await self._archive_queue.put(archive)

    async def _run_connection(
        self,
        connection_id: int,
    ) -> None:
        import websockets

        state = self.states[connection_id]
        attempt = 0

        while not self._stop.is_set():
            try:
                async with websockets.connect(
                    self.endpoint,
                    additional_headers={
                        "X-CMC_PRO_API_KEY": self.api_key
                    },
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=10,
                    max_queue=4096,
                ) as websocket:
                    state.connected = True
                    state.last_error = None
                    self._dirty = True
                    attempt = 0

                    await websocket.send(
                        json.dumps(
                            {
                                "id": connection_id,
                                "method": "subscribe",
                                "channel": CMC_WS_CHANNEL,
                                "params": {
                                    "crypto_ids": list(
                                        state.crypto_ids
                                    )
                                },
                            }
                        )
                    )

                    async for raw in websocket:
                        if self._stop.is_set():
                            break
                        try:
                            message = json.loads(raw)
                        except (
                            TypeError,
                            ValueError,
                            json.JSONDecodeError,
                        ):
                            message = {
                                "type": "error",
                                "code": "LOCAL_JSON_DECODE",
                                "raw": str(raw)[:500],
                            }

                        if isinstance(message, dict):
                            await self._record_message(
                                connection_id,
                                message,
                            )

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                state.connected = False
                state.reconnects += 1
                state.last_error = (
                    f"{type(exc).__name__}:"
                    f"{str(exc)[:400]}"
                )
                self.error_count += 1
                self._dirty = True
                attempt += 1

                delay = (
                    min(
                        60.0,
                        1.5 ** min(attempt, 10),
                    )
                    + connection_id * 0.13
                )
                try:
                    await asyncio.wait_for(
                        self._stop.wait(),
                        timeout=delay,
                    )
                except asyncio.TimeoutError:
                    pass
            finally:
                state.connected = False
                self._dirty = True

    async def run(self) -> None:
        connection_tasks = [
            asyncio.create_task(
                self._run_connection(connection_id),
                name=f"cmc-ws-{connection_id}",
            )
            for connection_id in self.states
        ]
        writer_task = asyncio.create_task(
            self._writer(),
            name="cmc-ws-state-writer",
        )
        archiver_task = asyncio.create_task(
            self._archiver(),
            name="cmc-ws-archiver",
        )

        try:
            await self._stop.wait()
        finally:
            for task in connection_tasks:
                task.cancel()
            await asyncio.gather(
                *connection_tasks,
                return_exceptions=True,
            )

            writer_task.cancel()
            await asyncio.gather(
                writer_task,
                return_exceptions=True,
            )

            try:
                await asyncio.wait_for(
                    self._archive_queue.join(),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                pass

            archiver_task.cancel()
            await asyncio.gather(
                archiver_task,
                return_exceptions=True,
            )

            payload = await self._snapshot()
            _atomic_json(self.latest_path, payload)
            _atomic_json(
                self.status_path,
                {
                    key: value
                    for key, value in payload.items()
                    if key != "events"
                },
            )

    def stop(self) -> None:
        self._stop.set()


def _api_key(canonical_env: Path) -> str:
    load_dotenv(canonical_env, override=False)
    return str(
        os.getenv("COINMARKETCAP_API_KEY")
        or os.getenv("CMC_API_KEY")
        or ""
    ).strip()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run ten sharded CoinMarketCap WebSocket "
            "market-data connections."
        )
    )
    parser.add_argument(
        "--markets-file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            os.getenv(
                "CMC_STARTUP_ROOT",
                "/mnt/data/quantdata/cmc-startup",
            )
        ),
    )
    parser.add_argument(
        "--canonical-env",
        type=Path,
        default=Path("/home/pi/sjagil/crypto/.env"),
    )
    parser.add_argument(
        "--connections",
        type=int,
        default=int(
            os.getenv("CMC_WS_CONNECTIONS", "10")
        ),
    )
    parser.add_argument(
        "--max-markets",
        type=int,
        default=int(
            os.getenv("CMC_WS_MAX_MARKETS", "25")
        ),
    )
    parser.add_argument(
        "--map-cache-ttl-seconds",
        type=int,
        default=86400,
    )
    parser.add_argument(
        "--flush-seconds",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    args = _build_parser().parse_args(argv)

    if not 1 <= int(args.connections) <= 10:
        raise SystemExit(
            "--connections must be between 1 and 10"
        )

    payload = _load_json(args.markets_file)
    markets = _extract_markets(payload)
    if not markets:
        raise SystemExit(
            f"No markets found in {args.markets_file}"
        )

    markets = markets[
        : max(1, int(args.max_markets))
    ]
    symbols = [
        _base_symbol(market)
        for market in markets
    ]

    mapping = _resolve_asset_ids(
        symbols,
        (
            args.root
            / "state"
            / "websocket_asset_map.json"
        ),
        cache_ttl_seconds=max(
            60,
            int(args.map_cache_ttl_seconds),
        ),
    )

    market_to_id = {
        market: mapping[_base_symbol(market)]
        for market in markets
        if _base_symbol(market) in mapping
    }
    shards = _split_shards(
        list(market_to_id.values()),
        int(args.connections),
    )

    plan = {
        "schema_version": "cmc_websocket_plan_v1",
        "markets": len(market_to_id),
        "requested_connections": int(
            args.connections
        ),
        "configured_connections": len(shards),
        "shard_sizes": [
            len(shard)
            for shard in shards
        ],
        "root": str(args.root),
        "endpoint": CMC_WS_ENDPOINT,
        "channel": CMC_WS_CHANNEL,
        "duplicated_subscriptions": False,
        "execution_authority": False,
    }

    if args.dry_run:
        print(
            json.dumps(
                plan,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    try:
        import websockets  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "websockets>=14 is required"
        ) from exc

    key = _api_key(args.canonical_env)
    if not key:
        raise SystemExit(
            "Missing COINMARKETCAP_API_KEY / "
            "CMC_API_KEY in canonical environment"
        )

    collector = CMCShardedWebSocketCollector(
        api_key=key,
        root=args.root,
        market_to_id=market_to_id,
        connections=int(args.connections),
        flush_seconds=float(args.flush_seconds),
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    for sig in (
        signal.SIGTERM,
        signal.SIGINT,
    ):
        try:
            loop.add_signal_handler(
                sig,
                collector.stop,
            )
        except NotImplementedError:
            pass

    print(
        json.dumps(
            {
                "event": "CMC_WS_START",
                **plan,
            },
            sort_keys=True,
        )
    )

    if float(args.duration_seconds) > 0:
        loop.call_later(
            float(args.duration_seconds),
            collector.stop,
        )

    try:
        loop.run_until_complete(
            collector.run()
        )
    finally:
        loop.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
