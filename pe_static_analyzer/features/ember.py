"""Static EMBER feature groups and vector transformations."""
from __future__ import annotations

import hashlib
import re
from typing import Any

import lief
import numpy as np
from sklearn.feature_extraction import FeatureHasher

FEATURE_DIM = 2381


def enum_name(value: Any) -> str:
    text = str(value)
    return text.split(".")[-1]


class FeatureType:
    name = ""
    dim = 0

    def raw_features(self, bytez: bytes, lief_binary: Any) -> Any:
        raise NotImplementedError

    def process_raw_features(self, raw_obj: Any) -> np.ndarray:
        raise NotImplementedError


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


class GeneralFileInfo(FeatureType):
    name = "general"
    dim = 10

    def raw_features(self, bytez: bytes, lief_binary: Any) -> dict[str, Any]:
        if lief_binary is None:
            return {
                "size": len(bytez),
                "vsize": 0,
                "has_debug": 0,
                "exports": 0,
                "imports": 0,
                "has_relocations": 0,
                "has_resources": 0,
                "has_signature": 0,
                "has_tls": 0,
                "symbols": 0,
            }
        return {
            "size": len(bytez),
            "vsize": int(getattr(lief_binary, "virtual_size", 0) or 0),
            "has_debug": int(bool(getattr(lief_binary, "has_debug", False))),
            "exports": len(getattr(lief_binary, "exported_functions", []) or []),
            "imports": len(getattr(lief_binary, "imported_functions", []) or []),
            "has_relocations": int(bool(getattr(lief_binary, "has_relocations", False))),
            "has_resources": int(bool(getattr(lief_binary, "has_resources", False))),
            "has_signature": int(bool(getattr(lief_binary, "has_signatures", False))),
            "has_tls": int(bool(getattr(lief_binary, "has_tls", False))),
            "symbols": len(getattr(lief_binary, "symbols", []) or []),
        }

    def process_raw_features(self, raw_obj: dict[str, Any]) -> np.ndarray:
        return np.asarray(
            [
                raw_obj["size"],
                raw_obj["vsize"],
                raw_obj["has_debug"],
                raw_obj["exports"],
                raw_obj["imports"],
                raw_obj["has_relocations"],
                raw_obj["has_resources"],
                raw_obj["has_signature"],
                raw_obj["has_tls"],
                raw_obj["symbols"],
            ],
            dtype=np.float32,
        )


class HeaderFileInfo(FeatureType):
    name = "header"
    dim = 62

    def raw_features(self, bytez: bytes, lief_binary: Any) -> dict[str, Any]:
        raw_obj: dict[str, Any] = {
            "coff": {"timestamp": 0, "machine": "", "characteristics": []},
            "optional": {
                "subsystem": "",
                "dll_characteristics": [],
                "magic": "",
                "major_image_version": 0,
                "minor_image_version": 0,
                "major_linker_version": 0,
                "minor_linker_version": 0,
                "major_operating_system_version": 0,
                "minor_operating_system_version": 0,
                "major_subsystem_version": 0,
                "minor_subsystem_version": 0,
                "sizeof_code": 0,
                "sizeof_headers": 0,
                "sizeof_heap_commit": 0,
            },
        }
        if lief_binary is None:
            return raw_obj
        header = lief_binary.header
        optional = lief_binary.optional_header
        raw_obj["coff"]["timestamp"] = int(getattr(header, "time_date_stamps", 0) or 0)
        raw_obj["coff"]["machine"] = enum_name(getattr(header, "machine", ""))
        raw_obj["coff"]["characteristics"] = [enum_name(c) for c in getattr(header, "characteristics_list", [])]
        raw_obj["optional"]["subsystem"] = enum_name(getattr(optional, "subsystem", ""))
        raw_obj["optional"]["dll_characteristics"] = [enum_name(c) for c in getattr(optional, "dll_characteristics_lists", [])]
        raw_obj["optional"]["magic"] = enum_name(getattr(optional, "magic", ""))
        for name in [
            "major_image_version",
            "minor_image_version",
            "major_linker_version",
            "minor_linker_version",
            "major_operating_system_version",
            "minor_operating_system_version",
            "major_subsystem_version",
            "minor_subsystem_version",
            "sizeof_code",
            "sizeof_headers",
            "sizeof_heap_commit",
        ]:
            raw_obj["optional"][name] = int(getattr(optional, name, 0) or 0)
        return raw_obj

    def process_raw_features(self, raw_obj: dict[str, Any]) -> np.ndarray:
        return np.hstack(
            [
                raw_obj["coff"]["timestamp"],
                FeatureHasher(10, input_type="string").transform([[raw_obj["coff"]["machine"]]]).toarray()[0],
                FeatureHasher(10, input_type="string").transform([raw_obj["coff"]["characteristics"]]).toarray()[0],
                FeatureHasher(10, input_type="string").transform([[raw_obj["optional"]["subsystem"]]]).toarray()[0],
                FeatureHasher(10, input_type="string").transform([raw_obj["optional"]["dll_characteristics"]]).toarray()[0],
                FeatureHasher(10, input_type="string").transform([[raw_obj["optional"]["magic"]]]).toarray()[0],
                raw_obj["optional"]["major_image_version"],
                raw_obj["optional"]["minor_image_version"],
                raw_obj["optional"]["major_linker_version"],
                raw_obj["optional"]["minor_linker_version"],
                raw_obj["optional"]["major_operating_system_version"],
                raw_obj["optional"]["minor_operating_system_version"],
                raw_obj["optional"]["major_subsystem_version"],
                raw_obj["optional"]["minor_subsystem_version"],
                raw_obj["optional"]["sizeof_code"],
                raw_obj["optional"]["sizeof_headers"],
                raw_obj["optional"]["sizeof_heap_commit"],
            ]
        ).astype(np.float32)


class SectionInfo(FeatureType):
    name = "section"
    dim = 255

    @staticmethod
    def _properties(section: Any) -> list[str]:
        return [enum_name(c) for c in getattr(section, "characteristics_lists", [])]

    def raw_features(self, bytez: bytes, lief_binary: Any) -> dict[str, Any]:
        if lief_binary is None:
            return {"entry": "", "sections": []}
        entry_section = ""
        try:
            section = lief_binary.section_from_rva(lief_binary.entrypoint - lief_binary.imagebase)
            if section is not None:
                entry_section = section.name
        except Exception:
            entry_section = ""
        if not entry_section:
            for section in lief_binary.sections:
                props = self._properties(section)
                if "MEM_EXECUTE" in props:
                    entry_section = section.name
                    break
        return {
            "entry": entry_section,
            "sections": [
                {
                    "name": section.name,
                    "size": int(section.size),
                    "entropy": float(section.entropy),
                    "vsize": int(section.virtual_size),
                    "props": self._properties(section),
                }
                for section in lief_binary.sections
            ],
        }

    def process_raw_features(self, raw_obj: dict[str, Any]) -> np.ndarray:
        sections = raw_obj["sections"]
        general = [
            len(sections),
            sum(1 for section in sections if section["size"] == 0),
            sum(1 for section in sections if section["name"] == ""),
            sum(1 for section in sections if "MEM_READ" in section["props"] and "MEM_EXECUTE" in section["props"]),
            sum(1 for section in sections if "MEM_WRITE" in section["props"]),
        ]
        section_sizes = [(section["name"], section["size"]) for section in sections]
        section_entropy = [(section["name"], section["entropy"]) for section in sections]
        section_vsize = [(section["name"], section["vsize"]) for section in sections]
        entry_name_hashed = FeatureHasher(50, input_type="string").transform([[raw_obj["entry"]]]).toarray()[0]
        characteristics = [prop for section in sections for prop in section["props"] if section["name"] == raw_obj["entry"]]
        return np.hstack(
            [
                general,
                FeatureHasher(50, input_type="pair").transform([section_sizes]).toarray()[0],
                FeatureHasher(50, input_type="pair").transform([section_entropy]).toarray()[0],
                FeatureHasher(50, input_type="pair").transform([section_vsize]).toarray()[0],
                entry_name_hashed,
                FeatureHasher(50, input_type="string").transform([characteristics]).toarray()[0],
            ]
        ).astype(np.float32)


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
