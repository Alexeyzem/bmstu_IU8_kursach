"""Static EMBER feature groups and vector transformations."""
from __future__ import annotations

import hashlib
from typing import Any

import lief
import numpy as np
from pe_static_analyzer.features.groups import FeatureType
from pe_static_analyzer.features.groups.content import ByteEntropyHistogram, ByteHistogram, StringExtractor
from pe_static_analyzer.features.groups.directories import DataDirectories, ExportsInfo, ImportsInfo
from pe_static_analyzer.features.groups.structure import GeneralFileInfo, HeaderFileInfo, SectionInfo

FEATURE_DIM = 2381


class PEFeatureExtractor:
    def __init__(self) -> None:
        self.features: list[FeatureType] = [
            ByteHistogram(),
            ByteEntropyHistogram(),
            StringExtractor(),
            GeneralFileInfo(),
            HeaderFileInfo(),
            SectionInfo(),
            ImportsInfo(),
            ExportsInfo(),
            DataDirectories(),
        ]
        self.dim = sum(feature.dim for feature in self.features)
        if self.dim != FEATURE_DIM:
            raise RuntimeError(f"feature extractor dim {self.dim}; expected {FEATURE_DIM}")

    def raw_features(self, bytez: bytes) -> dict[str, Any]:
        try:
            binary = lief.PE.parse(bytez)
        except Exception:
            binary = None
        raw: dict[str, Any] = {"sha256": hashlib.sha256(bytez).hexdigest()}
        raw.update({feature.name: feature.raw_features(bytez, binary) for feature in self.features})
        return raw

    def process_raw_features(self, raw_obj: dict[str, Any]) -> np.ndarray:
        return np.hstack([feature.process_raw_features(raw_obj[feature.name]) for feature in self.features]).astype(np.float32)

    def feature_vector(self, bytez: bytes) -> np.ndarray:
        return self.process_raw_features(self.raw_features(bytez))


def sorel_signed_log_postprocess(x: np.ndarray) -> np.ndarray:
    out = x.astype(np.float32, copy=True)
    negative = out < 0
    positive = out > 0
    out[negative] = -np.log1p(-out[negative])
    out[positive] = np.log1p(out[positive])
    return out


def is_probably_pe(data: bytes) -> bool:
    if len(data) < 0x40 or data[:2] != b"MZ":
        return False
    pe_offset = int.from_bytes(data[0x3C:0x40], "little", signed=False)
    if pe_offset <= 0 or pe_offset + 4 > len(data):
        return False
    return data[pe_offset : pe_offset + 4] == b"PE\0\0"
