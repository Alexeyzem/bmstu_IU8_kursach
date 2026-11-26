"""Byte histograms, entropy and printable string features."""
import re
from typing import Any

import numpy as np

from . import FeatureType


class ByteHistogram(FeatureType):
    name = "histogram"
    dim = 256

    def raw_features(self, bytez: bytes, lief_binary: Any) -> list[int]:
        return np.bincount(np.frombuffer(bytez, dtype=np.uint8), minlength=256).tolist()

    def process_raw_features(self, raw_obj: list[int]) -> np.ndarray:
        counts = np.asarray(raw_obj, dtype=np.float32)
        total = counts.sum()
        if total == 0:
            return counts
        return counts / total


class ByteEntropyHistogram(FeatureType):
    name = "byteentropy"
    dim = 256

    def __init__(self, step: int = 1024, window: int = 2048) -> None:
        self.step = step
        self.window = window

    def _entropy_bin_counts(self, block: np.ndarray) -> tuple[int, np.ndarray]:
        c = np.bincount(block >> 4, minlength=16)
        p = c.astype(np.float32) / max(len(block), 1)
        wh = np.where(c)[0]
        entropy = np.sum(-p[wh] * np.log2(p[wh])) * 2
        hbin = int(entropy * 2)
        if hbin >= 16:
            hbin = 15
        return hbin, c

    def raw_features(self, bytez: bytes, lief_binary: Any) -> list[int]:
        output = np.zeros((16, 16), dtype=np.int64)
        data = np.frombuffer(bytez, dtype=np.uint8)
        if data.shape[0] < self.window:
            hbin, counts = self._entropy_bin_counts(data)
            output[hbin, :] += counts
        else:
            shape = data.shape[:-1] + (data.shape[-1] - self.window + 1, self.window)
            strides = data.strides + (data.strides[-1],)
            blocks = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)[:: self.step, :]
            for block in blocks:
                hbin, counts = self._entropy_bin_counts(block)
                output[hbin, :] += counts
        return output.flatten().tolist()

    def process_raw_features(self, raw_obj: list[int]) -> np.ndarray:
        counts = np.asarray(raw_obj, dtype=np.float32)
        total = counts.sum()
        if total == 0:
            return counts
        return counts / total


class StringExtractor(FeatureType):
    name = "strings"
    dim = 104

    def __init__(self) -> None:
        self._allstrings = re.compile(b"[\x20-\x7f]{5,}")
        self._paths = re.compile(b"c:\\\\", re.IGNORECASE)
        self._urls = re.compile(b"https?://", re.IGNORECASE)
        self._registry = re.compile(b"HKEY_")
        self._mz = re.compile(b"MZ")

    def raw_features(self, bytez: bytes, lief_binary: Any) -> dict[str, Any]:
        allstrings = self._allstrings.findall(bytez)
        if allstrings:
            lengths = [len(s) for s in allstrings]
            avlength = sum(lengths) / len(lengths)
            shifted = [b - ord(b" ") for b in b"".join(allstrings)]
            counts = np.bincount(shifted, minlength=96)
            csum = counts.sum()
            p = counts.astype(np.float32) / csum
            wh = np.where(counts)[0]
            entropy = float(np.sum(-p[wh] * np.log2(p[wh])))
        else:
            avlength = 0.0
            counts = np.zeros((96,), dtype=np.float32)
            csum = 0
            entropy = 0.0
        return {
            "numstrings": len(allstrings),
            "avlength": avlength,
            "printabledist": counts.tolist(),
            "printables": int(csum),
            "entropy": entropy,
            "paths": len(self._paths.findall(bytez)),
            "urls": len(self._urls.findall(bytez)),
            "registry": len(self._registry.findall(bytez)),
            "MZ": len(self._mz.findall(bytez)),
        }

    def process_raw_features(self, raw_obj: dict[str, Any]) -> np.ndarray:
        divisor = float(raw_obj["printables"]) if raw_obj["printables"] > 0 else 1.0
        return np.hstack(
            [
                raw_obj["numstrings"],
                raw_obj["avlength"],
                raw_obj["printables"],
                np.asarray(raw_obj["printabledist"], dtype=np.float32) / divisor,
                raw_obj["entropy"],
                raw_obj["paths"],
                raw_obj["urls"],
                raw_obj["registry"],
                raw_obj["MZ"],
            ]
        ).astype(np.float32)
