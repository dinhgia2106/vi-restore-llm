"""Export the character tagger to ONNX, so a C++ input method can run it with onnxruntime.

    python -m virestore.export_onnx --tagger runs/tagger --out runs/tagger/tagger.onnx

The graph takes one sentence per call: `ids` (int64, [1, chars]; 0 = padding, 1 = unknown, others from
the `vocab` list in config.json, offset by 2) and returns `logits` ([1, chars, 30]). Decoding a label: modifier =
label // 6 (none, breve, circumflex, horn, stroke), tone = label % 6 (none, grave, acute, hook,
tilde, dot below); see `virestore.tagger.apply_label`. After export the script checks that
onnxruntime gives the same labels as PyTorch on a few sentences.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import utf8_stdio

CHECK = [
    "Hom nay troi dep qua, chung ta di dao cong vien nhe.",
    "Toi di hoc",
    "Truoc Viet Nam, ong Abe se tham lan luot Philippines, Australia va Indonesia.",
]


def _max_len(path: Path) -> int:
    import json

    return json.loads((path / "config.json").read_text(encoding="utf-8"))["model"]["max_len"]


def main() -> None:
    utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tagger", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--quantize", type=Path, help="also write an int8 (dynamic quantisation) copy here")
    a = ap.parse_args()

    import numpy as np
    import onnxruntime as ort
    import torch

    from .tagger import load

    model, vocab, _ = load(a.tagger, "cpu")

    class Wrapped(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, ids):
            return self.m(ids, ids != 0)

    w = Wrapped(model).eval()
    # the fused attention fast path bakes the example's length into reshapes; trace the plain path
    torch.backends.mha.set_fastpath_enabled(False)
    example = torch.tensor([vocab.encode(CHECK[0])])
    batch, chars = torch.export.Dim("batch", max=1024), torch.export.Dim("chars", min=2, max=_max_len(a.tagger))
    torch.onnx.export(
        w, (example,), str(a.out), input_names=["ids"], output_names=["logits"],
        dynamic_shapes={"ids": {0: batch, 1: chars}}, dynamo=True, external_data=False,
    )
    sess = ort.InferenceSession(str(a.out), providers=["CPUExecutionProvider"])
    for s in CHECK:
        ids = torch.tensor([vocab.encode(s)])
        with torch.inference_mode():
            ref = w(ids).argmax(-1).numpy()
        got = sess.run(["logits"], {"ids": ids.numpy()})[0].argmax(-1)
        assert np.array_equal(ref, got), f"onnxruntime labels differ from PyTorch on: {s}"
    print(f"exported {a.out} ({a.out.stat().st_size / 2**20:.1f} MB); onnxruntime labels match PyTorch on {len(CHECK)} sentences")
    if a.quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(str(a.out), str(a.quantize), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
        print(f"quantised {a.quantize} ({a.quantize.stat().st_size / 2**20:.1f} MB); score it with virestore.eval, labels can differ")


if __name__ == "__main__":
    main()
