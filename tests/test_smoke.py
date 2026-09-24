def test_package_importable():
    from net import cards
    from net.sim import meld  # noqa: F401
    assert cards.decode(77) == "K♦"
