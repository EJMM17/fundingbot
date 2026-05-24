# CLAUDE.md

Guidance for AI assistants working in this repository.

## What this is

A **KuCoin Futures funding-fee scalper** ("Funding Fee Scalper v4 / Quant Edition").
The bot opens **LONG-only** positions on USD-M perpetual swaps when the funding
rate is deeply negative, holds through the funding snapshot to **collect the
funding payment**, then exits. The core thesis: *the bot exists to collect the
funding fee and get out — any time held after the snapshot is exposure without a
thesis.*

It runs as a single long-lived async process (`main.py`) that loops every
`SCAN_INTERVAL_SECONDS` (30s), scanning the market, evaluating open positions,
and entering/DCAing/exiting per a layered rule set. State persists to a local
SQLite file (`bot.db`). Operational tracking is via Telegram.

Note: most code comments and log messages are in **Spanish**. Match that style
when editing existing files.

## Layout

| File | Role |
|------|------|
| `main.py` | Entry point. Logging setup, signal handling, the main async loop, graceful shutdown. |
| `config.py` | All tunables. Loads secrets from `.env`. Heavily commented with the economic rationale per constant. **No secrets live here.** |
| `trading_engine.py` | The core. `TradingEngine` class: scanning, scoring, entry/DCA/exit rules, risk gates, circuit breakers, state reconciliation. ~2000 lines — the heart of the system. |
| `db_local.py` | SQLite persistence (`bot.db`). Async writers run in an executor so they never block the event loop. Tables: `trade_logs`, `pnl_snapshots`, `math_snapshots`, `monte_carlo_runs`, `learning_stats`. |
| `auto_learning.py` | Conservative per-symbol multipliers (score + size) derived from closed trades in `trade_logs`. Bounded boosts/penalties. Does **not** predict prices. |
| `telegram_tracker.py` | Read-only Telegram command interface: `/status`, `/positions`, `/risk`, `/learn`, `/help`, `/ping`. Polls for updates and pushes periodic status. |
| `notifications.py` | Webhook + Telegram message senders (fire-and-forget, never raise to caller). |
| `math_engine.py` | Pure, side-effect-free quant library: Power-Law tails (Clauset-Shalizi-Newman), EVT/GPD, Hurst (R/S + DFA), entropy/transfer-entropy, log-volatility, multifractal spectrum, and the integrating `power_score`. |
| `kalman_engine.py` | Kalman filter for smoothing/predicting the funding rate series. |
| `microstructure_engine.py` | Order-flow/microstructure proxies from ticker data (bid-ask imbalance, VPIN proxy, toxicity). |
| `monte_carlo_engine.py` | Monte Carlo P&L sim with fat tails (Normal+GPD mixture) for sizing/stop recommendations and Kelly fraction. |
| `regime_engine.py` | 2-regime ("calm"/"chaos") model over funding-rate changes via simplified EM. |
| `schema.sql` | PostgreSQL/Supabase schema (legacy/reference — runtime uses SQLite via `db_local.py`). |
| `deploy/` | `install_ubuntu.sh` (systemd installer) + `fundingbot.service` unit file. |
| `README_DEPLOY.md` | DigitalOcean/Ubuntu deployment guide (Spanish). |
| `test_trading_engine.py`, `test_auto_learning.py` | Unit tests. |

## Architecture & data flow

```
main.py loop ──► TradingEngine.run_cycle() every 30s
                   │
                   ├─ circuit breaker + daily drawdown gate (skip cycle if open)
                   ├─ auto_learning.refresh() (throttled)
                   ├─ scan()  ──► qualified pairs inside the entry window
                   ├─ fetch_positions() ──► open LONGs
                   └─ for each symbol in (qualified ∪ open):
                        ├─ record OI snapshot
                        ├─ compute math metrics (math/kalman/regime/MC engines)
                        ├─ _evaluate_position() for open positions (exit logic)
                        └─ entry/DCA rules for qualified pairs
```

Exchange access is via `ccxt.async_support` with class id **`kucoinfutures`**
(a separate ccxt class — **not** `ccxt.kucoin` with `defaultType='swap'`).

### Entry & DCA rule layers (in `trading_engine.py`)
- **Initial entry**: pair qualifies in `scan()` (volume, FR threshold, predicted-FR
  check, spread/liquidity, math gates, opportunity score), and is within the
  scaled entry window before the funding snapshot.
- **DCA on dips** (price drop between `DCA_MIN_DROP_PCT` and `DCA_MAX_DROP_PCT`),
  classified by Open Interest trend:
  - **Rule 3 — Squeeze** (`SQUEEZE_MARGIN`): FR worsening + OI rising strong, *aborted if OI crowding detected*.
  - **Rule 1 — Defense** (`DEFENSE_MARGIN`): OI falling.
  - **Rule 2 — Standard DCA** (`STANDARD_MARGIN`): OI lateral.
- **Blindfold**: shortly after the snapshot, DCA is disabled (anti-dump); only exit evaluation runs.
- **Exit**: take-profit ROE, post-funding reduced TP, OI dump, max-hold timeout, absolute stop-loss ROE, or FR turning non-negative.

### Risk gates (all in config + enforced in engine)
Global margin cap, max open positions, per-coin margin cap, price-drift gate,
slippage gate, spread gate, daily-drawdown circuit breaker, consecutive-error
circuit breaker, OI-crowding abort, and the quant "math gates" (tail-α, entropy,
multifractal width, Hurst, volatility regime).

## Conventions

- **Python 3.11**, `from __future__ import annotations` at the top of every module.
- **Async-first**: never block the event loop. DB/network I/O is awaited; sync DB work runs through `db_local._run_sync` (executor).
- **Secrets only via `.env`** (loaded in `config.py` with `_require_env`). Required: `KUCOIN_API_KEY`, `KUCOIN_SECRET`, `KUCOIN_PASSPHRASE`. Never hardcode credentials or commit `.env`.
- **`DRY_RUN` defaults to `true`** — the bot runs the full cycle, logs orders it *would* send, and writes to the DB but sends no real orders. Keep this default; only real operators flip it.
- **LONG-only.** Do not add short logic without explicit instruction — it breaks the funding thesis and the reconciliation/close paths.
- Config constants carry their economic rationale in comments. When changing a threshold, update the rationale too.
- `math_engine.py` is **pure** — keep it free of side effects, I/O, and global state so it stays testable.
- Logging: module-scoped `logging.getLogger("<short-name>")`; INFO for cycle events, DEBUG for skip reasons.
- `numpy`/`scipy` are used throughout the quant engines; guard array operations against short histories (the engines already check `len(...)` minimums before computing).

## Development workflow

Use a virtualenv (the systemd deploy uses `/opt/fundingbot/.venv`):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies are **pinned** in `requirements.txt` (ccxt, python-dotenv, numpy,
scipy, aiohttp) so prod matches local. Update pins deliberately.

### Verify changes
```bash
python -m compileall .          # syntax/import sanity
python -m unittest -v           # run the test suite
```

Tests set dummy KuCoin env vars before importing `config`, so they run without a
real `.env`. When adding tests that import `config` or `trading_engine`, do the
same (`os.environ.setdefault("KUCOIN_API_KEY", "dummy")`, etc.) at the top.

### Run locally (dry-run)
```bash
cp .env.example .env            # fill in at least the KuCoin keys
python main.py                  # DRY_RUN=true by default; Ctrl+C to stop
```

## Deployment

Production target is an Ubuntu droplet under `systemd`:
- `deploy/install_ubuntu.sh <git_repo_url>` provisions `/opt/fundingbot`, a venv, the `.env`, and installs the service.
- `deploy/fundingbot.service` runs `main.py`; `KillSignal=SIGINT` triggers the graceful shutdown path.
- Update flow: stop service → `git pull --ff-only` → reinstall deps → run tests → start service. See `README_DEPLOY.md`.

Pre-go-live checklist (from README): hours of stable `DRY_RUN=true`, Telegram
commands responding, KuCoin key with minimal perms + IP whitelist, risk caps
reviewed, `bot.db` backed up. Only then flip `DRY_RUN=false`.

## Gotchas

- `db_local.py` initializes `bot.db` **at import time** (`_init_db()`); importing it has a filesystem side effect.
- `bot.db`, `bot.log`, `.env`, and `.venv/` are git-ignored — never commit them.
- `schema.sql` is PostgreSQL and is **not** what the runtime uses; the live store is SQLite defined inline in `db_local.py`. Keep them in sync if you change tables.
- The funding interval (8h for BTC/ETH, 4h/1h for many alts) scales the entry and blindfold windows dynamically — they are *fractions of the cycle*, not fixed minutes.
- Telegram commands are **read-only by design**; do not add trade-mutating commands without explicit instruction.
