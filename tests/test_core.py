import json
import random
from pathlib import Path

from virestore.data import build, iter_documents
from virestore.metrics import levenshtein, score
from virestore.models.baselines import Identity, Lexicon
from virestore.noise import NoiseConfig, make_input, telex_spelling
from virestore.text import has_diacritics, split_sentences, strip_diacritics

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "sample.txt"


def test_strip_diacritics_keeps_case_and_punctuation():
    assert strip_diacritics("Tiếng Việt đẹp quá!") == "Tieng Viet dep qua!"
    assert strip_diacritics("ĐƯỢC") == "DUOC"
    assert not has_diacritics("hom nay")
    assert has_diacritics("hôm nay")


def test_split_sentences():
    assert split_sentences("Một câu. Hai câu!  Ba câu?") == ["Một câu.", "Hai câu!", "Ba câu?"]


def test_telex_spelling():
    assert telex_spelling("tiếng") == "tieengs"
    assert telex_spelling("Việt") == "vieetj"
    assert telex_spelling("được") == "dduwowcj"


def test_make_input_clean_is_exact_stripping():
    s = "Hôm nay trời đẹp quá"
    assert make_input(s, random.Random(0)) == "Hom nay troi dep qua"


def test_noise_is_deterministic():
    s = "Hôm nay trời đẹp quá, chúng ta đi dạo công viên nhé"
    cfg = NoiseConfig(p_telex=0.3, p_drop=0.1, p_double=0.1, p_swap=0.1)
    assert make_input(s, random.Random(7), cfg) == make_input(s, random.Random(7), cfg)


def test_levenshtein_and_score():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein(["a", "b"], ["a", "c", "b"]) == 1
    perfect = score(["hôm nay"], ["hôm nay"])
    assert perfect.cer == 0 and perfect.sent_acc == 1
    wrong = score(["hom nay"], ["hôm nay"])
    assert wrong.cer > 0 and wrong.word_acc == 0.5


def test_build_and_baselines(tmp_path):
    stats = build(SAMPLE, tmp_path, noisy=False, seed=1, min_words=3, max_words=40, limit=None)
    total = sum(stats["counts"].values())
    assert total > 20
    rows = [json.loads(l) for name in ("train", "dev", "test") for l in (tmp_path / f"{name}.jsonl").open(encoding="utf-8")]
    assert len(rows) == total
    assert all(r["input"] == strip_diacritics(r["target"]) for r in rows)
    assert len({r["target"] for r in rows}) == total  # no duplicates across splits

    inputs = [r["input"] for r in rows]
    targets = [r["target"] for r in rows]
    floor = score(Identity().restore(inputs), targets)
    lex = Lexicon.from_jsonl(tmp_path / "train.jsonl").restore(inputs)
    assert score(lex, targets).cer < floor.cer


def test_ambiguous_syllables():
    from virestore.hardset import ambiguous_syllables, is_hard

    # 'ma': top form is 1/6 of uses -> ambiguous; 'hoa' always 'hoa'; 'cua': 'của' 90 % -> not ambiguous
    rows = [{"target": t} for t in ["ma", "mà", "má", "mả", "mã", "mạ", "hoa"] * 5 + ["của"] * 9 + ["cua"]]
    amb = ambiguous_syllables(rows, max_top_share=0.5, min_total=5)
    assert amb == {"ma"}
    assert is_hard("ma ma hoa", amb, min_ratio=0.5, min_hits=2)
    assert not is_hard("ma hoa hoa hoa", amb, min_ratio=0.3, min_hits=1)


def test_jsonl_gz_corpus_and_filters(tmp_path):
    import gzip

    from virestore.data import is_clean_sentence, iter_sentences

    docs = [
        {"content": "Hôm nay trời đẹp quá. Xem thêm tại https://example.com nhé bạn ơi."},
        {"content": "Mã đơn 123456789 0987654321 thanh toán."},
        {"id": 3, "content": "Chúng tôi học máy mỗi ngày."},
    ]
    path = tmp_path / "part-000.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    assert list(iter_sentences(tmp_path, 3, 40)) == ["Hôm nay trời đẹp quá.", "Chúng tôi học máy mỗi ngày."]
    assert not is_clean_sentence("Liên hệ admin@site.vn để biết thêm", 3, 40)


def test_iter_documents_interleaves_shards(tmp_path):
    (tmp_path / "a.txt").write_text("a1\na2\na3\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b1\n", encoding="utf-8")
    docs = [d.strip() for d in iter_documents(tmp_path)]
    assert docs == ["a1", "b1", "a2", "a3"]


def test_analysis_separates_diacritic_errors_from_rewrites():
    from virestore.analysis import analyse

    rows = [
        {"input": "hom nay troi dep", "pred": "hôm nay trời đẹp", "target": "hôm nay trời đẹp"},
        {"input": "ma oi", "pred": "mà ơi", "target": "má ơi"},
        {"input": "toi di hoc", "pred": "tôi đi làm", "target": "tôi đi học"},
        {"input": "chao ban", "pred": "xin chào bạn", "target": "chào bạn"},
    ]
    a = analyse(rows)
    assert a["n"] == 4
    assert a["base_letter_fidelity"] == 0.5
    assert a["more_words"] == 0.25
    assert a["top_diacritic_confusions"] == [["má", "mà", 1]]
    assert a["top_rewrites"] == [["học", "làm", 1]]


def test_summary_skips_analysis_files(tmp_path):
    from virestore.eval import write_summary

    run = {
        "tag": "t", "model": "identity", "test_file": "data/processed/test.jsonl",
        "quality": {"n": 1, "cer": 0.1, "wer": 0.2, "word_acc": 0.8, "sent_acc": 0.0},
        "latency": {}, "rss_mb": None, "machine": {"device": "cpu"},
    }
    (tmp_path / "t.json").write_text(json.dumps(run), encoding="utf-8")
    (tmp_path / "t.analysis.json").write_text(json.dumps({"n": 1}), encoding="utf-8")
    text = write_summary(tmp_path).read_text(encoding="utf-8")
    assert "| t | identity | processed/test | 1 |" in text
    assert "| – / – / – |" in text


def test_em_at_k_and_lexicon_nbest(tmp_path):
    from virestore.metrics import em_at_k

    assert em_at_k([["a", "b", "c"], ["x", "y"]], ["b", "z"], (1, 3)) == {"em@1": 0.0, "em@3": 0.5}

    data = tmp_path / "train.jsonl"
    rows = [{"input": "", "target": t} for t in ["ma ơi"] * 3 + ["má ơi"] * 2 + ["mà thôi"]]
    data.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    lex = Lexicon.from_jsonl(data)
    nb = lex.restore_nbest(["ma oi"], k=3)[0]
    assert nb[0] == lex.restore(["ma oi"])[0] == "ma ơi"
    assert set(nb) == {"ma ơi", "má ơi", "mà ơi"}
    assert Identity().restore_nbest(["ma oi"], 3) == [["ma oi"]]
