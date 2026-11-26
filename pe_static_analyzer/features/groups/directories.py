"""Imports, exports and PE data directories."""
from typing import Any

import numpy as np
from sklearn.feature_extraction import FeatureHasher

from . import FeatureType
from .structure import enum_name


class ImportsInfo(FeatureType):
    name = "imports"
    dim = 1280

    def raw_features(self, bytez: bytes, lief_binary: Any) -> dict[str, list[str]]:
        imports: dict[str, list[str]] = {}
        if lief_binary is None:
            return imports
        for lib in getattr(lief_binary, "imports", []) or []:
            lib_name = lib.name or ""
            imports.setdefault(lib_name, [])
            for entry in lib.entries:
                try:
                    if entry.is_ordinal:
                        imports[lib_name].append("ordinal" + str(entry.ordinal))
                    else:
                        imports[lib_name].append((entry.name or "")[:10000])
                except Exception:
                    continue
        return imports

    def process_raw_features(self, raw_obj: dict[str, list[str]]) -> np.ndarray:
        libraries = list(set([lib.lower() for lib in raw_obj.keys()]))
        libraries_hashed = FeatureHasher(256, input_type="string").transform([libraries]).toarray()[0]
        imports = [lib.lower() + ":" + entry for lib, entries in raw_obj.items() for entry in entries]
        imports_hashed = FeatureHasher(1024, input_type="string").transform([imports]).toarray()[0]
        return np.hstack([libraries_hashed, imports_hashed]).astype(np.float32)


class ExportsInfo(FeatureType):
    name = "exports"
    dim = 128

    def raw_features(self, bytez: bytes, lief_binary: Any) -> list[str]:
        if lief_binary is None:
            return []
        exports = []
        for export in getattr(lief_binary, "exported_functions", []) or []:
            name = getattr(export, "name", export)
            exports.append(str(name)[:10000])
        return exports

    def process_raw_features(self, raw_obj: list[str]) -> np.ndarray:
        return FeatureHasher(128, input_type="string").transform([raw_obj]).toarray()[0].astype(np.float32)


class DataDirectories(FeatureType):
    name = "datadirectories"
    dim = 30

    def raw_features(self, bytez: bytes, lief_binary: Any) -> list[dict[str, Any]]:
        if lief_binary is None:
            return []
        return [
            {"name": enum_name(directory.type), "size": int(directory.size), "virtual_address": int(directory.rva)}
            for directory in getattr(lief_binary, "data_directories", [])
        ]

    def process_raw_features(self, raw_obj: list[dict[str, Any]]) -> np.ndarray:
        features = np.zeros(30, dtype=np.float32)
        for i in range(15):
            if i < len(raw_obj):
                features[2 * i] = raw_obj[i]["size"]
                features[2 * i + 1] = raw_obj[i]["virtual_address"]
        return features
