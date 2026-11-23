"""Adapter for the acausal TCN from github.com/dinhgia2106/Vietnamese-diacritic-restoration.

Clone that repo next to this one and train or copy its weights, then:
    python -m virestore.eval --model tcn --tcn-repo ../Vietnamese-diacritic-restoration \
        --tcn-weights ../Vietnamese-diacritic-restoration/models/best_model.pth
"""

from __future__ import annotations

import sys
from pathlib import Path


class TCNRestorer:
    name = "tcn"

    def __init__(self, repo: str | Path, weights: str | Path | None = None, threads: int | None = None):
        import torch

        if threads:
            torch.set_num_threads(threads)
        self.n_threads = torch.get_num_threads()
        repo = Path(repo).resolve()
        if not (repo / "model_architecture.py").exists():
            raise FileNotFoundError(f"model_architecture.py not found in {repo}")
        sys.path.insert(0, str(repo))
        from model_architecture import VietnameseAccentRestorer  # type: ignore

        weights = Path(weights) if weights else repo / "models" / "best_model.pth"
        if not weights.exists():
            raise FileNotFoundError(f"TCN weights not found: {weights}")
        self.model = VietnameseAccentRestorer(model_path=str(weights))

    def restore(self, inputs: list[str]) -> list[str]:
        return [self._predict(s) for s in inputs]

    def _predict(self, text: str) -> str:
        if len(text) < 2:  # upstream predict() squeezes a one-character batch to a scalar and crashes
            return self.model.predict(text + " ")[: len(text)]
        return self.model.predict(text)
