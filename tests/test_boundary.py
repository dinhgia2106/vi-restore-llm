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


def test_lexicon_decode_prefers_known_tokens():
    from virestore.boundary import lexicon_decode

    src = "thuongmai"
    probs = [0.1] * len(src)  # the model sees no boundary
    probs[4] = 0.3            # weak evidence after "thuon"
    probs[5] = 0.2            # weaker after "thuong"
    best = lexicon_decode(src, probs, {"thuong", "mai"}, unknown_logp=-8.0)[0][1]
    assert best[5] == 1 and sum(best) == 1


def test_lexicon_decode_keeps_model_decisions_next_to_punctuation():
    from virestore.boundary import lexicon_decode

    src = "ab,cd"
    probs = [0.1, 0.9, 0.1, 0.1, 0.1]
    labels = lexicon_decode(src, probs, {"ab", "cd"}, unknown_logp=-8.0)[0][1]
    assert labels == [0, 1, 0, 0, 0]


def test_word_span_cuts_whole_words():
    import random

    from virestore.boundary import word_span

    src, labels = boundary_pair("mot hai ba bon nam")
    for seed in range(20):
        s, l = word_span(src, labels, random.Random(seed))
        assert join(s, l) in "mot hai ba bon nam" and s and not join(s, l).startswith(" ")


class _FakeBoundary:
    lexicon = {"tieng", "anh", "sang"}

    def plausible(self, segmentation):
        return "hsang" not in segmentation

    def segment_nbest(self, text, k):
        return {"anhsang": ["anh sang", "an hsang"], "homnay": ["hom nay"]}.get(text, [text])[:k]


class _FakeRestorer:
    def restore_nbest(self, inputs, k):
        return [[t.replace("anh sang", "ánh sáng").replace("hom nay", "hôm nay"), t] for t in inputs]

    def restore(self, inputs):
        return list(inputs)


def test_chunk_suggester_segments_the_raw_keys():
    from virestore.boundary import ChunkSuggester

    cs = ChunkSuggester(_FakeBoundary(), _FakeRestorer())
    out = cs.suggest("Hôm qua ", "ánhang", "anhsang", 5)  # Telex turned the "s" into a mark on screen
    assert out == {"candidates": ["ánh sáng", "anh sang"], "telex": False}  # "an hsang" is not plausible


def test_chunk_suggester_leaves_telex_words_alone():
    from virestore.boundary import ChunkSuggester

    cs = ChunkSuggester(_FakeBoundary(), _FakeRestorer())
    assert cs.suggest("", "tiếng", "tieengs")["telex"] is True
    assert cs.suggest("", "homnay", "homnay")["candidates"][0] == "hôm nay"
