import pytest

from app.domain.scoring import ScoringConfig, combine, text_score_from_fuzz

CONFIG = ScoringConfig(text_weight=0.25, semantic_weight=0.75, override_threshold=0.75)


def test_weighted_sum_below_thresholds():
    assert combine(0.4, 0.6, CONFIG) == pytest.approx(0.25 * 0.4 + 0.75 * 0.6)


def test_text_override_wins():
    assert combine(0.8, 0.1, CONFIG) == pytest.approx(0.8)


def test_semantic_override_wins():
    assert combine(0.1, 0.9, CONFIG) == pytest.approx(0.9)


def test_both_zero_dropped():
    assert combine(0.0, 0.0, CONFIG) is None


def test_text_only_keeps_full_text_score():
    # El servicio viejo aplicaba w_text incluso sin embeddings, capando los matches a 0.25.
    assert combine(0.6, None, CONFIG) == pytest.approx(0.6)
    assert combine(0.0, None, CONFIG) is None


def test_text_score_blend():
    assert text_score_from_fuzz(100.0, 100.0) == pytest.approx(1.0)
    assert text_score_from_fuzz(50.0, 100.0) == pytest.approx(0.8)
