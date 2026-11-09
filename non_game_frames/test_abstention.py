import numpy as np
from abstention import N_FRAMES, HALF_BLOCKS, non_game_mask, apply_abstention


def test_counts():
    m = non_game_mask()
    assert len(m) == 199_319
    assert (~m).sum() == 158_857  # graded frames, from CHALLENGE.md


def test_boundaries():
    m = non_game_mask()
    assert m[11_184] and not m[11_185]
    assert not m[90_651] and m[90_652]
    assert m[110_807] and not m[110_808]
    assert not m[190_197] and m[190_198]


def test_apply_only_touches_non_game():
    labels = np.array(["H"] * N_FRAMES, dtype=object)
    out = apply_abstention(labels)
    m = non_game_mask()
    assert (out[m] == "X").all()
    assert (out[~m] == "H").all()


def test_input_not_modified():
    labels = np.array(["A"] * N_FRAMES, dtype=object)
    apply_abstention(labels)
    assert (labels == "A").all()