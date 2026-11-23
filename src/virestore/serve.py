"""Minimal local inference server for VIMEK.

    python -m virestore.serve --model tagger --tagger runs/tagger --threads 4 --port 8765

POST /restore    {"text": "hom nay troi dep qua"}       -> {"text": <restored sentence>, "ms": <server time>}
POST /candidates {"text": "hom nay troi dep qua", "k": 5} -> {"candidates": [<best first>], "ms": <server time>}
GET  /health                                             -> {"ok": true, "model": <name>}

`/candidates` needs a model with n-best output (tagger, lexicon, hf).
"""

# No `from __future__ import annotations` here: FastAPI must see the request models defined inside
# create_app() as real classes, not as strings it cannot resolve.
import argparse
import time

from . import utf8_stdio


def create_app(restorer):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    app = FastAPI(title="vi-restore-llm")

    class Req(BaseModel):
        text: str

    class NbestReq(BaseModel):
        text: str
        k: int = 5

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "model": restorer.name}

    @app.post("/restore")
    def restore(req: Req) -> dict:
        t0 = time.perf_counter()
        out = restorer.restore([req.text])[0]
        return {"text": out, "ms": round((time.perf_counter() - t0) * 1000, 2)}

    @app.post("/candidates")
    def candidates(req: NbestReq) -> dict:
        if not hasattr(restorer, "restore_nbest"):
            raise HTTPException(400, f"{restorer.name} has no n-best output")
        t0 = time.perf_counter()
        try:
            cands = restorer.restore_nbest([req.text], max(1, min(req.k, 10)))[0]
        except NotImplementedError as e:
            raise HTTPException(400, str(e)) from e
        return {"candidates": cands, "ms": round((time.perf_counter() - t0) * 1000, 2)}

    return app


def main() -> None:
    utf8_stdio()
    import uvicorn

    from . import models

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["lexicon", "tcn", "hf", "gguf", "tagger", "tagger-onnx"])
    ap.add_argument("--gguf")
    ap.add_argument("--hf-model")
    ap.add_argument("--adapter")
    ap.add_argument("--tagger", help="directory of a trained virestore.tagger model")
    ap.add_argument("--onnx-file", help="ONNX file inside --tagger for tagger-onnx (default: tagger.onnx)")
    ap.add_argument("--train", help="training JSONL for the lexicon baseline")
    ap.add_argument("--tcn-repo")
    ap.add_argument("--tcn-weights")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (tagger, gguf)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    restorer = models.load(
        a.model, model=a.hf_model or a.gguf or a.tagger, adapter=a.adapter, train=a.train,
        tcn_repo=a.tcn_repo, tcn_weights=a.tcn_weights, threads=a.threads, onnx_file=a.onnx_file,
    )
    if a.model in ("tagger", "tagger-onnx"):
        from .ime import ImeRestorer

        restorer = ImeRestorer(restorer)
    restorer.restore(["khoi dong"])  # the first call is about 100x slower; do not spend it on the user
    uvicorn.run(create_app(restorer), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
