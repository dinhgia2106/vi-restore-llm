import json
import threading
import urllib.error
import urllib.request

import numpy as np

from virestore.models.tagger import topk
from virestore.serve import _Handler


def test_topk_numpy_best_first():
    row = np.array([-3.0, -0.1, -np.inf, -1.0], dtype=np.float32)
    assert topk(row, 2) == [(np.float32(-0.1), 1), (-1.0, 3)]


class FakeRestorer:
    name = "fake"

    def restore(self, inputs):
        return [t.upper() for t in inputs]

    def restore_nbest(self, inputs, k):
        return [[t] * k for t in inputs]


class FakeChunks:
    def suggest(self, context, screen, raw, k):
        return {"candidates": [context + screen + raw][:k], "telex": False}


def _server(chunks):
    from http.server import ThreadingHTTPServer

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler.make(FakeRestorer(), chunks))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_stdlib_server_endpoints():
    srv, url = _server(FakeChunks())
    try:
        with urllib.request.urlopen(url + "/health") as r:
            assert json.loads(r.read()) == {"ok": True, "model": "fake"}
        assert _post(url + "/restore", {"text": "ab"})[1]["text"] == "AB"
        assert _post(url + "/candidates", {"text": "ab", "k": 2})[1]["candidates"] == ["ab", "ab"]
        code, out = _post(url + "/chunk", {"context": "Hôm ", "screen": "nay", "raw": "nay"})
        assert code == 200 and out["candidates"] == ["Hôm naynay"] and out["telex"] is False
        assert _post(url + "/chunk", {"context": ""})[0] == 422
        assert _post(url + "/chunk", {"screen": "a", "k": "x"})[0] == 422
    finally:
        srv.shutdown()


def test_stdlib_server_without_boundary():
    srv, url = _server(None)
    try:
        assert _post(url + "/chunk", {"screen": "a"})[0] == 400
    finally:
        srv.shutdown()
