from app.domain.normalization import normalize_text


def test_lowercases_and_strips_accents():
    assert normalize_text("Cámara RÉFLEX") == "camara reflex"


def test_removes_punctuation_and_collapses_whitespace():
    assert normalize_text("  hola,   mundo!! ") == "hola mundo"


def test_keeps_non_latin_scripts():
    # Antes se eliminaba todo lo no ASCII: una lista en ruso o japonés quedaba en blanco.
    assert normalize_text("café 電話 sofá") == "cafe 電話 sofa"
    assert normalize_text("Фотоаппарат Canon!") == "фотоаппарат canon"
    assert normalize_text("Ελληνικά, ναι") == "ελληνικα ναι"


def test_folds_compatibility_forms():
    # Anchos completos y ligaduras (NFKC) se comparan como sus equivalentes simples.
    assert normalize_text("Ｃａｎｏｎ") == "canon"
    assert normalize_text("ﬁlm") == "film"


def test_empty_and_symbol_only():
    assert normalize_text("¡¿!?") == ""
    assert normalize_text("") == ""
    assert normalize_text("___") == ""
