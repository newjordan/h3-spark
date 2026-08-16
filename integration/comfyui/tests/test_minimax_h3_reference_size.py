from comfy_extras.nodes_minimax_h3 import reference_image_size


def test_reference_half_uses_quarter_area_at_aligned_shape():
    match = reference_image_size(1536, 1024, 864, 480, "match")
    half = reference_image_size(1536, 1024, 864, 480, "half")

    assert match == (800, 512)
    assert half == (384, 256)
    assert half[0] * half[1] < match[0] * match[1] / 4 * 1.05


def test_reference_size_modes_preserve_existing_match_and_max_behavior():
    assert reference_image_size(1536, 1024, 864, 480, "match") == (800, 512)
    assert reference_image_size(1536, 1024, 864, 480, "max") == (1536, 1024)
    assert reference_image_size(320, 192, 864, 480, "match") == (320, 192)
