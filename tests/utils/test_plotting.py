from matplotlib import font_manager

from utils.plotting import resolve_korean_font


def test_resolve_korean_font_returns_available_font(
    monkeypatch,
) -> None:
    resolve_korean_font.cache_clear()
    monkeypatch.setattr(
        font_manager,
        "findfont",
        lambda family, *, fallback_to_default: "/fonts/korean.ttf",
    )

    font_properties = resolve_korean_font()

    assert font_properties is not None
    assert font_properties.get_file() == "/fonts/korean.ttf"
    resolve_korean_font.cache_clear()
