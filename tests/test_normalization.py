from app.domain.normalization import normalize_text


def test_lowercases_and_strips_accents():
    assert normalize_text("Cámara RÉFLEX") == "camara reflex"


def test_removes_punctuation_and_collapses_whitespace():
    assert normalize_text("  hola,   mundo!! ") == "hola mundo"


def test_drops_non_ascii_leftovers():
    assert normalize_text("café 電話 sofá") == "cafe sofa"


def test_empty_and_symbol_only():
    assert normalize_text("¡¿!?") == ""
    assert normalize_text("") == ""
