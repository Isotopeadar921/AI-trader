# AI Trader — NSE F&O Algorithmic Trading System

> **Full-stack intraday options trading system** — ML regime detection, RL exit agents, tick-level backtesting, and a live retro terminal dashboard for NIFTY.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Next.js 15](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

Intraday NSE F&O options trading system for NIFTY. Combines LightGBM regime detection, RL exit agents, tick-level backtesting, and a live retro terminal dashboard.

**Stack:** Python 3.13 · Flask API · Next.js 15 · PostgreSQL · LightGBM · PyTorch DQN · TrueData · Recharts

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                  │
│  TrueData → tick_collector / ingest_historical → PostgreSQL  │
│  NIFTY-I 1m candles + options tick data                      │
└──────────────────────┬───────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────┐
│  INTELLIGENCE LAYER                                          │
│  RegimeDetector (EMA/ATR/range) → LightGBM Macro Model      │
│  StrategyPredictor (per-strategy LightGBM) → TradeScorer     │
│  VolSurface (IV-based strike selection)                      │
└──────────────────────┬───────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────┐
│  RL EXIT AGENTS                                              │
│  Tabular Q-Learning (247K episodes, 11,606 states)           │
│  DQN Agent (64→64→32 LayerNorm, Double DQN + Huber)          │
│  Actions: HOLD / EXIT / TIGHTEN_SL                           │
└──────────────────────┬───────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────┐
│  EXECUTION LAYER                                             │
│  KellySizer → RiskManager → OrderManager → Kite Connect      │
│  3 risk profiles: LOW / MEDIUM / HIGH                        │
└──────────────────────┬───────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────┐
│  DASHBOARD LAYER                                             │
│  Flask API (port 5050) ← → Next.js UI (port 3000)           │
│  Retro terminal theme · Live P&L · Charts · Backtest viewer  │
└──────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.13+ | |
| Node.js | 18+ | For dashboard |
| PostgreSQL | 14+ | With TimescaleDB extension |
| TrueData API | — | For market data |
| Zerodha Kite Connect | — | For live order execution |

---

## Setup

### 1. Clone & Python environment

```bash
git clone https://github.com/yourusername/ai-trader.git
cd ai-trader

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/trading_db

# TrueData
TRUEDATA_USERNAME=your_username
TRUEDATA_PASSWORD=your_password

# Kite Connect
KITE_API_KEY=your_api_key
KITE_ACCESS_TOKEN=your_access_token

# Trading
INITIAL_CAPITAL=50000
RISK_PER_TRADE=0.01
MAX_TRADES_PER_DAY=5
MAX_DAILY_LOSS=0.05
```

### 3. Database setup

```bash
# macOS: install TimescaleDB
brew install timescaledb

# Create DB and enable extension
psql -U postgres -c "CREATE DATABASE trading_db;"
psql -U postgres -d trading_db -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"

# Run schema
psql -U postgres -d trading_db -f database/schema.sql
```

### 4. Dashboard dependencies

```bash
cd dashboard
npm install
```

---

## Running the System

Three processes need to run simultaneously. Open three terminals:

### Terminal 1 — Flask API (port 5050)
```bash
# From project root, with .venv activated
python frontend/app.py
```

### Terminal 2 — Next.js Dashboard (port 3000)
```bash
cd dashboard
npm run dev
```

### Terminal 3 — (Optional) Paper trading engine
```bash
# Replay a specific day
python scripts/paper_trade.py --replay 2026-03-20

# Live paper trading via TrueData WebSocket
python scripts/paper_trade.py
```

Open **http://localhost:3000** in your browser.

---

## Scripts Reference

| Script | Purpose |
|---|---|
| `python scripts/tick_replay_backtest.py --risk high` | Full tick-level backtest, HIGH risk |
| `python scripts/tick_replay_backtest.py --risk medium` | Full tick-level backtest, MEDIUM risk |
| `python scripts/forward_test.py --risk medium` | Out-of-sample forward test |
| `python scripts/train_rl_exit.py --epochs 15` | Train tabular Q-learning exit agent |
| `python scripts/train_dqn_exit.py --epochs 10` | Train DQN neural net exit agent |
| `python scripts/paper_trade.py --replay 2026-03-20` | Paper trade replay for a day |
| `python scripts/paper_trade.py` | Live paper trading (TrueData WebSocket) |
| `python scripts/ingest_historical.py` | Ingest historical 1m candles from TrueData |
| `python scripts/collect_ticks.py` | Collect live tick data |
| `python scripts/retrain_models.py` | Retrain all ML models |
| `python frontend/app.py` | Start Flask API backend (port 5050) |
| `cd dashboard && npm run dev` | Start Next.js dashboard (port 3000) |

---

## Risk Profiles

Three configurable risk profiles, selectable from the dashboard:

| Profile | Lot Size | Stop Loss | Target | Max Trades/Day |
|---|---|---|---|---|
| LOW | 1× | 1.5% | 2.0% | 3 |
| MEDIUM | 2× | 2.0% | 3.0% | 4 |
| HIGH | 3× | 2.5% | 4.0% | 5 |

Position sizing uses **half-Kelly Criterion** based on rolling 20-trade win rate, capped by profile limits.

---

## ML Models

| Model | Type | Purpose |
|---|---|---|
| `models/saved/macro_model.pkl` | LightGBM | Bull/Bear regime probability (80+ features, AUC 0.98) |
| `models/saved/strategy_*.pkl` | LightGBM | Per-strategy success probability |
| `models/saved/rl_exit_agent.pkl` | Tabular Q-Table | Exit timing (HOLD/EXIT/TIGHTEN) |
| `models/saved/dqn_exit_agent.pt` | PyTorch DQN | Neural net exit agent |
| `strategy/vol_surface.py` | Scoring | IV-based strike selection |

---

## Project Structure

```
ai-trader/
├── backtest/               # Backtest engine + option resolver
├── config/                 # Settings, constants
├── dashboard/              # Next.js 15 retro terminal UI
│   ├── app/                # Pages: /, /live, /trades, /charts, /backtest, /ai, /settings
│   ├── components/         # Sidebar, StatCard, EquityChart, TradeTable, etc.
│   └── lib/                # API client (fetchJSON/postJSON)
├── data/                   # TrueData adapter, tick collector, aggregator
│   └── historical/         # Stored option tick CSVs
├── database/               # schema.sql + db.py (SQLAlchemy + psycopg2)
├── execution/              # OrderManager + broker adapter (Kite Connect)
├── features/               # indicators.py (80+ features), feature engine
├── frontend/               # Flask API (app.py) — serves all /api/* routes
├── models/                 # LightGBM + DQN training, prediction, strategy models
├── risk/                   # Kelly sizer, risk manager
├── scripts/                # All runnable scripts (backtest, train, paper trade)
├── strategy/               # RegimeDetector, SignalGenerator, VolSurface, TradeScorer
├── utils/                  # Logger, helpers
├── main.py                 # Legacy entry point (mock/ingest/train/backtest/live)
├── requirements.txt
└── .env.example
```

---

## Dashboard Pages

| Page | Route | Description |
|---|---|---|
| Dashboard | `/` | Live stats, equity curve, recent trades, ticker bar |
| Live | `/live` | System status, market regime, trade suggestions |
| Trades | `/trades` | Full trade history, P&L chart, strategy breakdown |
| Charts | `/charts` | NIFTY candles, option chain, tick charts, analytics |
| Backtest | `/backtest` | Run backtests, compare risk profiles, equity curves |
| AI Models | `/ai` | RL agent status, model info, Kelly sizer |
| Settings | `/settings` | Risk profile selection, system info, CLI reference |

---

## Risk Management

| Rule | Value |
|---|---|
| Risk per trade | 1% of capital |
| Max trades/day | 3–5 (by profile) |
| Max daily loss | 5% of capital |
| Stop loss | ATR-based × profile multiplier |
| Target | 2× stop loss |
| Position sizing | Half-Kelly, min 1 lot, max 5 lots |

---

## Data Requirements

| Type | Source | Stored |
|---|---|---|
| 1m NIFTY candles | TrueData | PostgreSQL `candles_1m` |
| Options tick data | TrueData | PostgreSQL + `data/historical/` CSVs |
| Option chain (live) | TrueData WebSocket | In-memory |
| Trade execution | Kite Connect | PostgreSQL `trades` |

---

## Important Notes

**Do NOT train models on mock data** — synthetic data has no real market patterns and will produce useless models.

**Paper trade before going live:**
1. Run `tick_replay_backtest.py` on historical data
2. Verify results in the Backtest dashboard page
3. Run `paper_trade.py` in replay mode for a few days
4. Only then enable live order execution via Kite Connect

**Market hours:** System operates 9:15 AM – 3:30 PM IST. The scanner runs every 30s.

---

## License

MIT — see [LICENSE](LICENSE)

---

## Disclaimer

For educational purposes only. Trading derivatives involves substantial risk of loss. Past backtest performance does not guarantee future results. Always paper trade first and never risk capital you cannot afford to lose.
