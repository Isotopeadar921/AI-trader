"""
Risk Profile Configuration
──────────────────────────
Defines LOW / MEDIUM / HIGH risk profiles that control every
aspect of the trading system:

  - Position sizing (lot multiplier)
  - SL / TGT ranges
  - Score thresholds (entry selectivity)
  - Max trades per day
  - Max premium cap
  - Trailing stop behaviour
  - Afternoon cut-off
  - Regime-aware lot scaling
  - News sensitivity

Usage:
  from config.risk_profiles import get_risk_profile, RiskLevel
  profile = get_risk_profile(RiskLevel.HIGH)
  sl_pct = profile.sl_pct
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class RiskProfile:
    """Immutable trading parameter set tied to a risk level."""

    name: str
    level: RiskLevel

    # ── Position sizing ─────────────────────────────────────────────────
    base_lot_size: int          # base NIFTY lot count (65 = 1 lot)
    lot_multiplier: float       # applied on top of regime multiplier
    max_capital_per_trade: float  # fraction of capital risked per trade

    # ── SL / TGT ────────────────────────────────────────────────────────
    sl_pct: float               # default SL as fraction of premium
    tgt_pct: float              # default TGT as fraction of premium
    sl_min_pct: float           # floor for dynamic SL
    sl_max_pct: float           # ceiling for dynamic SL
    tgt_min_pct: float          # floor for dynamic TGT
    tgt_max_pct: float          # ceiling for dynamic TGT

    # ── Trailing stop ───────────────────────────────────────────────────
    trailing_trigger: float     # activate after +X% move
    trailing_lock: float        # lock at Y% of entry (0 = breakeven)

    # ── Entry selectivity ───────────────────────────────────────────────
    score_threshold: float      # minimum composite score for CALL
    put_score_threshold: float  # minimum composite score for PUT
    max_trades_day: int         # maximum trades per day
    max_premium: float          # don't buy options above this ₹

    # ── Time filters ────────────────────────────────────────────────────
    skip_first_min: int         # skip N minutes after open
    skip_last_min: int          # skip last N minutes before close
    afternoon_cut: int          # no new trades after X minutes from open
    max_hold_bars: int          # timeout after N minutes

    # ── News sensitivity ────────────────────────────────────────────────
    news_block_threshold: float   # block trades below this sentiment
    news_boost_threshold: float   # boost score above this sentiment
    news_boost_amount: float      # how much to add to final_score

    # ── Regime lot multipliers ──────────────────────────────────────────
    regime_multipliers: Dict[str, float]

    # ── Strike selection ────────────────────────────────────────────────
    use_vol_surface: bool       # use volatility surface for strike selection
    max_strike_offset: int      # max strikes away from ATM (in strike gaps)


# ── LOW RISK — Conservative, capital preservation ────────────────────────────
LOW_RISK = RiskProfile(
    name="Conservative",
    level=RiskLevel.LOW,
    # Position sizing
    base_lot_size=65,
    lot_multiplier=0.5,
    max_capital_per_trade=0.005,
    # SL / TGT — tight stops, modest targets
    sl_pct=0.20,
    tgt_pct=0.35,
    sl_min_pct=0.15,
    sl_max_pct=0.25,
    tgt_min_pct=0.25,
    tgt_max_pct=0.45,
    # Trailing — activate at 12%, lock at 5%
    trailing_trigger=0.12,
    trailing_lock=0.05,
    # Entry selectivity — very selective
    score_threshold=0.70,
    put_score_threshold=0.80,
    max_trades_day=3,
    max_premium=150,
    # Time filters — conservative windows
    skip_first_min=10,
    skip_last_min=30,
    afternoon_cut=150,   # no trades after 11:45 IST
    max_hold_bars=20,
    # News — very sensitive to negative news
    news_block_threshold=-0.15,
    news_boost_threshold=0.30,
    news_boost_amount=0.03,
    # Regime
    regime_multipliers={
        "TRENDING_BULL": 1.0,
        "TRENDING_BEAR": 0.75,
        "SIDEWAYS": 0.50,
        "HIGH_VOLATILITY": 0.25,
        "LOW_VOLATILITY": 0.75,
        "UNKNOWN": 0.50,
    },
    # Strike selection
    use_vol_surface=True,
    max_strike_offset=1,
)


# ── MEDIUM RISK — Balanced (current system defaults) ─────────────────────────
MEDIUM_RISK = RiskProfile(
    name="Balanced",
    level=RiskLevel.MEDIUM,
    # Position sizing
    base_lot_size=65,
    lot_multiplier=1.0,
    max_capital_per_trade=0.01,
    # SL / TGT — tight 20% SL, 50% target
    sl_pct=0.20,
    tgt_pct=0.50,
    sl_min_pct=0.12,
    sl_max_pct=0.28,
    tgt_min_pct=0.35,
    tgt_max_pct=0.70,
    # Trailing — activate at 15%, lock at 10%
    trailing_trigger=0.15,
    trailing_lock=0.10,
    # Entry selectivity — balanced
    score_threshold=0.60,
    put_score_threshold=0.70,
    max_trades_day=5,
    max_premium=250,
    # Time filters
    skip_first_min=5,
    skip_last_min=15,
    afternoon_cut=195,   # no trades after 12:30 IST
    max_hold_bars=25,
    # News
    news_block_threshold=-0.30,
    news_boost_threshold=0.20,
    news_boost_amount=0.05,
    # Regime
    regime_multipliers={
        "TRENDING_BULL": 1.25,
        "TRENDING_BEAR": 1.25,
        "SIDEWAYS": 0.75,
        "HIGH_VOLATILITY": 0.50,
        "LOW_VOLATILITY": 1.00,
        "UNKNOWN": 0.75,
    },
    # Strike selection
    use_vol_surface=True,
    max_strike_offset=1,
)


# ── HIGH RISK — Aggressive, maximum profit potential ─────────────────────────
HIGH_RISK = RiskProfile(
    name="Aggressive",
    level=RiskLevel.HIGH,
    # Position sizing — larger positions
    base_lot_size=65,
    lot_multiplier=1.75,
    max_capital_per_trade=0.02,
    # SL / TGT — tight 20% SL, wide 80% target for high RR
    sl_pct=0.20,
    tgt_pct=0.80,
    sl_min_pct=0.12,
    sl_max_pct=0.25,
    tgt_min_pct=0.50,
    tgt_max_pct=1.00,
    # Trailing — early activation, lock at 10%
    trailing_trigger=0.15,
    trailing_lock=0.10,
    # Entry selectivity — slightly more selective
    score_threshold=0.58,
    put_score_threshold=0.65,
    max_trades_day=8,
    max_premium=400,
    # Time filters — wider windows
    skip_first_min=3,
    skip_last_min=10,
    afternoon_cut=240,   # trades until 1:15 IST
    max_hold_bars=30,
    # News — less sensitive
    news_block_threshold=-0.50,
    news_boost_threshold=0.15,
    news_boost_amount=0.08,
    # Regime — bigger in trending, still size down in chaos
    regime_multipliers={
        "TRENDING_BULL": 1.75,
        "TRENDING_BEAR": 1.75,
        "SIDEWAYS": 1.00,
        "HIGH_VOLATILITY": 0.75,
        "LOW_VOLATILITY": 1.25,
        "UNKNOWN": 1.00,
    },
    # Strike selection
    use_vol_surface=True,
    max_strike_offset=3,
)


_PROFILES = {
    RiskLevel.LOW: LOW_RISK,
    RiskLevel.MEDIUM: MEDIUM_RISK,
    RiskLevel.HIGH: HIGH_RISK,
}


def get_risk_profile(level: RiskLevel) -> RiskProfile:
    """Get risk profile by level."""
    return _PROFILES[level]


def list_profiles() -> list:
    """Return all available profiles with summary info."""
    summaries = []
    for profile in _PROFILES.values():
        summaries.append({
            "level": profile.level.value,
            "name": profile.name,
            "lot_multiplier": profile.lot_multiplier,
            "sl_range": f"{profile.sl_min_pct:.0%}–{profile.sl_max_pct:.0%}",
            "tgt_range": f"{profile.tgt_min_pct:.0%}–{profile.tgt_max_pct:.0%}",
            "score_threshold": profile.score_threshold,
            "max_trades": profile.max_trades_day,
            "max_premium": profile.max_premium,
        })
    return summaries
