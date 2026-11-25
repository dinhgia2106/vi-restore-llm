from virestore.boundary import boundary_pair, boundary_scores, join


def test_boundary_pair_strips_marks_and_spaces():
    src, labels = boundary_pair("Hôm nay  trời đẹp")
    assert src == "Homnaytroidep"
    assert labels == [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    assert join(src, labels) == "Hom nay troi dep"


def test_join_ignores_a_trailing_boundary():
    assert join("ab", [1, 1]) == "a b"


def test_boundary_scores():
    gold = [[0, 1, 0, 0], [1, 0, 0]]
    pred = [[0, 1, 0, 1], [0, 0, 0]]  # the last label of each row is not scored
    s = boundary_scores(pred, gold)
    assert s["precision"] == 1.0 and s["recall"] == 0.5 and s["sentence_exact"] == 0.5


def test_error_breakdown_separates_letter_boundaries():
    from virestore.boundary import error_breakdown

    rows = [boundary_pair("ab cd , ef"), boundary_pair("gh ij")]
    preds = [list(rows[0][1]), list(rows[1][1])]
    preds[0][4] = 0  # miss the space after "cd" (next to the comma): not a letter-letter boundary
    preds[1][1] = 0  # miss the space between "gh" and "ij"
    b = error_breakdown(rows, preds)
    assert b["errors_other"] == 1 and b["errors_letters"] == 1 and b["letters_exact"] == 0.5
