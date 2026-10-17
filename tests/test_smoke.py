from underwrite.smoke import render_sample_receipt


def test_sample_receipt_is_a_rgb_image_of_requested_size():
    image = render_sample_receipt(width=200, height=300)
    assert image.size == (200, 300)
    assert image.mode == "RGB"


def test_sample_receipt_renders_some_ink():
    image = render_sample_receipt(width=200, height=300)
    assert image.getextrema() != ((255, 255), (255, 255), (255, 255))
