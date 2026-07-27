"""Unit tests for social_intel schema, rubric, date window, extract."""
from __future__ import annotations

from datetime import date

from trad_research.social_intel.config import SocialIntelConfig
from trad_research.social_intel.extract import enrich_card_from_text, extract_claims, extract_rules
from trad_research.social_intel.schema import Claims, Rules, StrategyCard
from trad_research.social_intel.truth_rubric import score_card, verdict_from_gates


def test_within_window():
    cfg = SocialIntelConfig(cutoff_date=date(2026, 4, 27), as_of_date=date(2026, 7, 27))
    assert cfg.within_window(date(2026, 5, 1)) is True
    assert cfg.within_window(date(2026, 4, 26)) is False
    assert cfg.within_window(date(2026, 7, 27)) is True
    assert cfg.within_window(date(2026, 7, 28)) is False


def test_verdict_strong():
    gates = {"G1": True, "G2": True, "G3": True, "G4": False, "G5": False, "G6": True}
    assert verdict_from_gates(gates, within_3m=True, transcript_coverage="full_auto") == "EVIDENCE_STRONG"


def test_verdict_too_old():
    gates = {"G1": True, "G2": True, "G3": True, "G4": True, "G5": True, "G6": True}
    assert verdict_from_gates(gates, within_3m=False) == "TOO_OLD"


def test_score_card_marketing():
    card = StrategyCard(
        source="youtube",
        url="https://youtu.be/x",
        id="x",
        title="Best strategy ever guaranteed profits",
        within_3m=True,
        transcript_coverage="partial",
        rules=Rules(entry="just feel the market", exit="when ready"),
    )
    card = score_card(card, "guaranteed profits no risk")
    assert card.verdict in ("MARKETING_ONLY", "UNPARSEABLE", "EVIDENCE_WEAK", "NO_TRANSCRIPT")
    assert card.gates["G1"] is False


def test_score_card_rules_orb():
    text = (
        "First step is daily bias with SMA 200. Entry on 15m opening range breakout "
        "close above ORB high. Stop loss below ORB low. Take profit at 2R. "
        "Backtested 500 trades walk-forward 2018-2025 with commission and slippage vs SPY."
    )
    card = StrategyCard(
        source="youtube",
        url="https://youtu.be/y",
        id="y",
        title="ORB strategy walk forward",
        within_3m=True,
        transcript_coverage="full_auto",
    )
    card = enrich_card_from_text(card, text)
    card = score_card(card, text)
    assert card.gates["G1"] is True
    assert card.gates["G2"] is True
    assert card.gates["G3"] is True
    assert card.gates["G4"] is True
    assert card.gates["G5"] is True
    assert card.verdict in ("EVIDENCE_STRONG", "EVIDENCE_WEAK")


def test_extract_claims_winrate():
    c = extract_claims("I even back tested it 1000 times to a win rate of 79.45%.")
    assert c.sample_n == 1000
    assert c.win_rate is not None
    assert 0.7 < c.win_rate < 0.85


def test_extract_rules_entry():
    r = extract_rules(
        "The first step in this intraday model is having a daily bias. "
        "Entries are taken on the lower timeframe using continuation order blocks with 2R targets. "
        "Stop loss goes below the order block."
    )
    assert len(r.entry) > 10
