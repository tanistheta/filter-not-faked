"""Detector wrappers. Every detector maps a list of RGB uint8 images to a
list of P(fake) scores in [0, 1].

- HFDetector: any Hugging Face image-classification model. The fake class is
  found automatically from the label names, or set explicitly.
- HighFreqDetector: a training-free stand-in that scores images by the share
  of high-frequency energy. It exists only to test the pipeline without a GPU
  or internet; never report its numbers.
"""
from __future__ import annotations

import numpy as np

FAKE_HINTS = ("fake", "deepfake", "ai", "artificial", "synthetic", "generated", "spoof")
REAL_HINTS = ("real", "realism", "human", "authentic", "bonafide", "genuine")


class HFDetector:
    def __init__(self, model_id: str, fake_label: str | None = None,
                 device: int | str | None = None, batch_size: int = 32):
        import torch
        from transformers import pipeline

        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.name = model_id.split("/")[-1]
        self.pipe = pipeline("image-classification", model=model_id, device=device)
        self.batch_size = batch_size
        labels = list(self.pipe.model.config.id2label.values())
        self.fake_label = fake_label or self._guess_fake_label(labels)
        print(f"[{self.name}] labels={labels} -> using fake label '{self.fake_label}'")

    @staticmethod
    def _guess_fake_label(labels: list[str]) -> str:
        low = {l: l.lower() for l in labels}
        fakes = [l for l, s in low.items() if any(h in s for h in FAKE_HINTS)
                 and not any(h in s for h in REAL_HINTS)]
        if len(fakes) == 1:
            return fakes[0]
        reals = [l for l, s in low.items() if any(h in s for h in REAL_HINTS)]
        if len(labels) == 2 and len(reals) == 1:
            return next(l for l in labels if l != reals[0])
        raise ValueError(f"Cannot tell which label means fake in {labels}; "
                         f"pass fake_label explicitly.")

    def score(self, images: list[np.ndarray]) -> list[float]:
        from PIL import Image

        pil = [Image.fromarray(im) for im in images]
        outs = self.pipe(pil, top_k=None, batch_size=self.batch_size)
        return [next(o["score"] for o in out if o["label"] == self.fake_label) for out in outs]


class HighFreqDetector:
    """Pipeline-test stand-in: less high-frequency texture -> higher P(fake).
    Smoothing removes texture, so this should rise with smoothing strength,
    which is a useful sanity check that the treatment arms are wired up."""

    name = "highfreq_sanity"

    def score(self, images: list[np.ndarray]) -> list[float]:
        import cv2

        scores = []
        for im in images:
            g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
            lap = cv2.Laplacian(g, cv2.CV_32F)
            hf = float(np.log1p(lap.var()))
            scores.append(float(1 / (1 + np.exp(hf - 5.0))))
        return scores


def load(spec: str):
    """'highfreq' for the sanity detector, else a Hugging Face model id.
    Optional fake label override: 'org/model::LabelName'."""
    if spec == "highfreq":
        return HighFreqDetector()
    model_id, _, label = spec.partition("::")
    return HFDetector(model_id, fake_label=label or None)
