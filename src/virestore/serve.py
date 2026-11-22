"""Minimal local inference server for VIMEK.

    python -m virestore.serve --model gguf --gguf models/qwen05-lora-q4_k_m.gguf --port 8765

POST /restore {"text": "hom nay troi dep qua"} -> {"text": "hôm nay trời đẹp quá", "ms": 31.2}
"""

from __future__ import annotations

import argparse
import time

from . import utf8_stdio


def create_app(restorer):
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="vi-restore-llm")

    class Req(BaseModel):
        text: str

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "model": restorer.name}

    @app.post("/restore")
    def restore(req: Req) -> dict:
        t0 = time.perf_counter()
        out = restorer.restore([req.text])[0]
        return {"text": out, "ms": round((time.perf_counter() - t0) * 1000, 1)}

    return app


def main() -> None:
    utf8_stdio()
    import uvicorn

    from . import models

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["lexicon", "tcn", "hf", "gguf"])
    ap.add_argument("--gguf")
    ap.add_argument("--hf-model")
    ap.add_argument("--adapter")
    ap.add_argument("--train", help="training JSONL for the lexicon baseline")
    ap.add_argument("--tcn-repo")
    ap.add_argument("--tcn-weights")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    restorer = models.load(a.model, model=a.hf_model or a.gguf, adapter=a.adapter, train=a.train, tcn_repo=a.tcn_repo, tcn_weights=a.tcn_weights)
    uvicorn.run(create_app(restorer), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
