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


def test_tagger_labels_round_trip():
    from virestore.tagger import N_LABELS, apply_label, char_label, decode, labels_for

    for line in SAMPLE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            assert decode(strip_diacritics(line), labels_for(line)) == line
    for ch in "ăâêôơưđĂÂÊÔƠƯĐ" + "áàảãạ" + "ỳýỷỹỵ" + "ữựừứử":
        assert apply_label(strip_diacritics(ch), char_label(ch)) == ch
    assert N_LABELS == 30
    assert apply_label("b", 7) == "b"  # consonants are copied
    assert apply_label("i", char_label("ơ")) == "i"  # a modifier the letter cannot take is dropped


def test_serve_restore_and_candidates(tmp_path):
    import pytest

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from virestore.serve import create_app

    build(SAMPLE, tmp_path, noisy=False, seed=13, min_words=3, max_words=40, limit=None)
    client = TestClient(create_app(Lexicon.from_jsonl(tmp_path / "train.jsonl")))
    assert client.get("/health").json()["ok"]
    r = client.post("/restore", json={"text": "hom nay troi dep qua"}).json()
    assert r["text"] and r["ms"] >= 0
    c = client.post("/candidates", json={"text": "hom nay troi dep qua", "k": 3}).json()
    assert c["candidates"][0] == r["text"] and len(c["candidates"]) <= 3
    assert TestClient(create_app(Identity())).post("/candidates", json={"text": "a"}).status_code == 200


def test_tagger_restorer_untrained_smoke(tmp_path):
    import pytest

    pytest.importorskip("torch")
    from virestore.models.tagger import TaggerRestorer
    from virestore.tagger import Vocab, build_model, save, valid_labels

    assert len(valid_labels("a")) == 18 and valid_labels("d") == [0, 24] and valid_labels("b") == [0]
    cfg = {"model": {"d_model": 16, "layers": 1, "heads": 2, "ff": 32, "max_len": 64, "dropout": 0.0}}
    vocab = Vocab(sorted(set("hom nay troi dep qua")))
    save(build_model(len(vocab), 16, 1, 2, 32, 64, 0.0), vocab, cfg, tmp_path / "t")
    r = TaggerRestorer(str(tmp_path / "t"), threads=1)
    src = "hom nay troi dep qua"
    out = r.restore([src])[0]
    assert strip_diacritics(out) == src  # never changes letters
    nb = r.restore_nbest([src], 3)[0]
    assert nb[0] == out and len(nb) == 3 and all(strip_diacritics(x) == src for x in nb)


def test_ime_chunks_lock_and_tone_rules():
    from virestore.ime import ImeRestorer, has_stacked_tones, lock_typed_words, split_chunks

    text = "mot hai ba. bon nam sau bay tam chin muoi"
    chunks = split_chunks(text, 16)
    assert "".join(c for c, _ in chunks) == text
    assert chunks[0] == ("mot hai ba. ", True)
    assert all(len(c) <= 16 for c, _ in chunks)
    assert split_chunks("x" * 30 + " mot", 10)[0] == ("x" * 30 + " ", False)
    assert "".join(c for c, _ in split_chunks(" " * 30, 10)) == " " * 30

    assert lock_typed_words("hôm nay troi", "hom này trời") == "hôm này trời"
    assert has_stacked_tones("thỏả") and not has_stacked_tones("và/hoặc") and not has_stacked_tones("người")

    class Fake:
        name, max_len = "fake", 16
        seen: list[str] = []

        def restore_one_tone(self, xs):
            self.seen += xs
            return [x.replace("nay", "này") for x in xs]

        def restore_nbest(self, xs, k):
            return [[x, "thỏả"] for x in xs]

    fake = Fake()
    ime = ImeRestorer(fake)
    assert ime.restore(["", "  "]) == ["", "  "] and fake.seen == []
    assert ime.restore(["hôm nay"]) == ["hôm này"] and fake.seen == ["hom nay"]
    assert "thỏả" not in ime.restore_nbest(["hom nay"], 3)[0]


def test_resolve_stacked_tones_keeps_the_likelier_one():
    import torch

    from virestore.models.tagger import resolve_stacked_tones
    from virestore.tagger import char_label, decode, valid_labels

    labels = [char_label(c) for c in "thỏả"]
    lp = torch.full((4, 30), -9.0)
    lp[2, labels[2]], lp[3, labels[3]] = -0.1, -1.0
    lp[2, 0], lp[3, 0] = -3.0, -2.0
    assert decode("thoa", resolve_stacked_tones("thoa", labels, lp, valid_labels)) == "thỏa"
