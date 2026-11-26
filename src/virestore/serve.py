"""Minimal local inference server for VIMEK.

    python -m virestore.serve --model tagger --tagger runs/tagger --threads 4 --port 8765

POST /restore    {"text": "hom nay troi dep qua"}       -> {"text": <restored sentence>, "ms": <server time>}
POST /candidates {"text": "hom nay troi dep qua", "k": 5} -> {"candidates": [<best first>], "ms": <server time>}
POST /chunk      {"context": "Hôm ", "screen": "naytroi", "raw": "naytroi", "k": 5}
                 -> {"candidates": ["nay trời", ...], "telex": false, "ms": <server time>}
GET  /health                                             -> {"ok": true, "model": <name>}

`/candidates` needs a model with n-best output (tagger, lexicon, hf). `/chunk` needs `--boundary`
(a virestore.boundary model; its lexicon.json is used) and a tagger: it segments the chunk being
typed, which may hold several words without spaces, then marks it (virestore.boundary.ChunkSuggester).
With `--model tagger-onnx` and `--boundary-onnx`, nothing at run time needs PyTorch.

FastAPI and uvicorn are used when installed; otherwise (and with `--stdlib`) the same endpoints are
served by Python's http.server, which is what the bundled VIMEK server does.
"""

# No `from __future__ import annotations` here: FastAPI must see the request models defined inside
# create_app() as real classes, not as strings it cannot resolve.
import argparse
import time

from . import utf8_stdio


def create_app(restorer, chunks=None):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    app = FastAPI(title="vi-restore-llm")

    class Req(BaseModel):
        text: str

    class NbestReq(BaseModel):
        text: str
        k: int = 5

    class ChunkReq(BaseModel):
        context: str = ""
        screen: str
        raw: str = ""
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

    @app.post("/chunk")
    def chunk(req: ChunkReq) -> dict:
        if chunks is None:
            raise HTTPException(400, "start the server with --boundary to segment unspaced input")
        t0 = time.perf_counter()
        out = chunks.suggest(req.context, req.screen, req.raw, max(1, min(req.k, 10)))
        return {**out, "ms": round((time.perf_counter() - t0) * 1000, 2)}

    return app


class _Handler:
    """Endpoints of create_app() without FastAPI, for a frozen build with as few modules as possible."""

    @staticmethod
    def make(restorer, chunks):
        import json
        from http.server import BaseHTTPRequestHandler

        def ms(t0: float) -> float:
            return round((time.perf_counter() - t0) * 1000, 2)

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # no console: requests are not logged
                pass

            def _send(self, code: int, obj: dict) -> None:
                body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/health":
                    self._send(200, {"ok": True, "model": restorer.name})
                else:
                    self._send(404, {"detail": "Not Found"})

            def do_POST(self):
                try:
                    req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                    if not isinstance(req, dict):
                        raise ValueError
                    k = max(1, min(int(req.get("k", 5)), 10))
                except (ValueError, TypeError):
                    return self._send(422, {"detail": "body must be a JSON object with an integer k"})
                t0 = time.perf_counter()
                if self.path == "/restore" and isinstance(req.get("text"), str):
                    return self._send(200, {"text": restorer.restore([req["text"]])[0], "ms": ms(t0)})
                if self.path == "/candidates" and isinstance(req.get("text"), str):
                    if not hasattr(restorer, "restore_nbest"):
                        return self._send(400, {"detail": f"{restorer.name} has no n-best output"})
                    return self._send(200, {"candidates": restorer.restore_nbest([req["text"]], k)[0], "ms": ms(t0)})
                if self.path == "/chunk" and isinstance(req.get("screen"), str):
                    if chunks is None:
                        return self._send(400, {"detail": "start the server with --boundary to segment unspaced input"})
                    out = chunks.suggest(str(req.get("context") or ""), req["screen"], str(req.get("raw") or ""), k)
                    return self._send(200, {**out, "ms": ms(t0)})
                self._send(404 if self.path not in ("/restore", "/candidates", "/chunk") else 422, {"detail": "bad request"})

        return H


def serve_stdlib(restorer, chunks, host: str, port: int) -> None:
    from http.server import ThreadingHTTPServer

    ThreadingHTTPServer.allow_reuse_address = False  # never share the port with a second server
    ThreadingHTTPServer((host, port), _Handler.make(restorer, chunks)).serve_forever()


def main() -> None:
    utf8_stdio()
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
    ap.add_argument("--boundary", help="virestore.boundary model directory: enables POST /chunk")
    ap.add_argument("--boundary-onnx", help="ONNX file inside --boundary (virestore.export_onnx --boundary): no PyTorch")
    ap.add_argument("--unknown-logp", type=float, default=-4.0, help="lexicon cost of an unknown token (boundary)")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (tagger, gguf)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--stdlib", action="store_true", help="serve with http.server even if FastAPI is installed")
    a = ap.parse_args()
    restorer = models.load(
        a.model, model=a.hf_model or a.gguf or a.tagger, adapter=a.adapter, train=a.train,
        tcn_repo=a.tcn_repo, tcn_weights=a.tcn_weights, threads=a.threads, onnx_file=a.onnx_file,
    )
    if a.model in ("tagger", "tagger-onnx"):
        from .ime import ImeRestorer

        restorer = ImeRestorer(restorer)
    chunks = None
    if a.boundary:
        from .boundary import BoundaryModel, ChunkSuggester, OnnxBoundaryModel

        if a.model not in ("tagger", "tagger-onnx"):
            raise SystemExit("--boundary needs --model tagger or tagger-onnx")
        if a.boundary_onnx:
            bm = OnnxBoundaryModel(a.boundary, onnx_file=a.boundary_onnx, threads=a.threads, use_lexicon=True, unknown_logp=a.unknown_logp)
        else:
            bm = BoundaryModel(a.boundary, threads=a.threads, use_lexicon=True, unknown_logp=a.unknown_logp)
        chunks = ChunkSuggester(bm, restorer)
        chunks.suggest("", "khoidong")
    restorer.restore(["khoi dong"])  # the first call is about 100x slower; do not spend it on the user
    try:
        if a.stdlib:
            raise ImportError
        import uvicorn
    except ImportError:
        serve_stdlib(restorer, chunks, a.host, a.port)
        return
    uvicorn.run(create_app(restorer, chunks), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
