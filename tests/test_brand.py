from tensorboard_book import brand, cli


def test_logo_files_match_the_drawing():
    assert brand.LOGO_SVG.read_text().strip() == brand.logo_svg()
    dark = brand.logo_svg(ink=brand.DARK_BLUE)
    assert brand.LOGO_DARK_SVG.read_text().strip() == dark
    assert brand.LOGO_PNG.is_file()


def test_logo_has_a_three_by_three_matrix():
    assert len(brand.cells()) == 9
    assert brand.logo_svg().count("<rect") == 9 + len(brand.SLICES)


def test_fonts_ship_with_the_package():
    fonts = brand.ASSETS.parent / "static" / "fonts"
    for name in (
        "IBMPlexSans.woff2",
        "IBMPlexMono-Regular.woff2",
        "IBMPlexMono-Medium.woff2",
        "Newsreader.woff2",
    ):
        assert (fonts / name).is_file()


def test_theme_flags_are_known_streamlit_options():
    flags = cli.theme_flags()
    assert "--theme.light.primaryColor" in flags
    assert "--theme.dark.backgroundColor" in flags
    assert len(flags) % 2 == 0
