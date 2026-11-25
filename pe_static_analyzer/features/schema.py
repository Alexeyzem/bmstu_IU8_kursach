"""Column names and group membership for EMBER-v2 vectors."""
from dataclasses import dataclass

from pe_static_analyzer.features.ember import FEATURE_DIM


DATA_DIRECTORIES = [
    "EXPORT_TABLE",
    "IMPORT_TABLE",
    "RESOURCE_TABLE",
    "EXCEPTION_TABLE",
    "CERTIFICATE_TABLE",
    "BASE_RELOCATION_TABLE",
    "DEBUG",
    "ARCHITECTURE",
    "GLOBAL_PTR",
    "TLS_TABLE",
    "LOAD_CONFIG_TABLE",
    "BOUND_IMPORT",
    "IAT",
    "DELAY_IMPORT_DESCRIPTOR",
    "CLR_RUNTIME_HEADER",
]

@dataclass(frozen=True)
class Feature:
    index: int
    name: str
    group: str


def numbered(prefix: str, count: int) -> list[str]:
    width = max(2, len(str(count - 1)))
    return [f"{prefix}_{i:0{width}d}" for i in range(count)]


def build_feature_schema() -> list[Feature]:
    groups: list[tuple[str, list[str]]] = []
    groups.append(("histogram", numbered("histogram_byte", 256)))
    groups.append(("byteentropy", numbered("byteentropy_bin", 256)))
    groups.append((
        "strings",
        ["strings_numstrings", "strings_avlength", "strings_printables"]
        + numbered("strings_printabledist", 96)
        + ["strings_entropy", "strings_paths", "strings_urls", "strings_registry", "strings_mz"],
    ))
    groups.append((
        "general",
        [
            "general_size",
            "general_vsize",
            "general_has_debug",
            "general_exports",
            "general_imports",
            "general_has_relocations",
            "general_has_resources",
            "general_has_signature",
            "general_has_tls",
            "general_symbols",
        ],
    ))
    groups.append((
        "header",
        ["header_coff_timestamp"]
        + numbered("header_coff_machine_hash", 10)
        + numbered("header_coff_characteristics_hash", 10)
        + numbered("header_optional_subsystem_hash", 10)
        + numbered("header_optional_dll_characteristics_hash", 10)
        + numbered("header_optional_magic_hash", 10)
        + [
            "header_major_image_version",
            "header_minor_image_version",
            "header_major_linker_version",
            "header_minor_linker_version",
            "header_major_operating_system_version",
            "header_minor_operating_system_version",
            "header_major_subsystem_version",
            "header_minor_subsystem_version",
            "header_sizeof_code",
            "header_sizeof_headers",
            "header_sizeof_heap_commit",
        ],
    ))
    groups.append((
        "section",
        [
            "section_count",
            "section_zero_size_count",
            "section_empty_name_count",
            "section_rx_count",
            "section_w_count",
        ]
        + numbered("section_size_hash", 50)
        + numbered("section_entropy_hash", 50)
        + numbered("section_vsize_hash", 50)
        + numbered("section_entry_name_hash", 50)
        + numbered("section_entry_characteristics_hash", 50),
    ))
    groups.append(("imports", numbered("imports_libraries_hash", 256) + numbered("imports_functions_hash", 1024)))
    groups.append(("exports", numbered("exports_hash", 128)))
    data_dir_names: list[str] = []
    for name in DATA_DIRECTORIES:
        lower = name.lower()
        data_dir_names.extend([f"datadirectories_{lower}_size", f"datadirectories_{lower}_rva"])
    groups.append(("datadirectories", data_dir_names))

    features: list[Feature] = []
    for group, names in groups:
        for name in names:
            features.append(Feature(len(features), name, group))
    if len(features) != FEATURE_DIM:
        raise RuntimeError(f"feature schema has {len(features)} entries; expected {FEATURE_DIM}")
    return features
