"""File metadata, PE header and section features."""
from typing import Any

import numpy as np
from sklearn.feature_extraction import FeatureHasher

from . import FeatureType


def enum_name(value: Any) -> str:
    text = str(value)
    return text.split(".")[-1]


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
