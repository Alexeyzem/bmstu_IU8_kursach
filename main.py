#!/usr/bin/env python3
"""Небольшой офлайн-интерфейс для статической проверки PE-файлов."""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import sys
import threading
import traceback
from pathlib import Path
from tkinter import BOTH, END, LEFT, NORMAL, DISABLED, RIGHT, Button, Frame, Label, OptionMenu, StringVar, Text, Tk, filedialog, messagebox


ROOT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

SGD_MODEL_PATH = ROOT_DIR / "outputs" / "models" / "baseline_sgd.pkl"
LIGHTGBM_MODEL_PATH = ROOT_DIR / "outputs" / "models" / "lightgbm_pe_full.pkl"
THRESHOLD_SELECTION_PATH = ROOT_DIR / "outputs" / "metrics" / "threshold_selection.json"
VALIDATION_METRICS_PATH = ROOT_DIR / "outputs" / "metrics" / "sorel_validation_metrics.json"
LIGHTGBM_MODES_PATH = ROOT_DIR / "outputs" / "metrics" / "lightgbm_gold_threshold_modes.json"
ALLOWED_SUFFIXES = (".exe", ".dll", ".pe", ".exe.txt", ".dll.txt", ".pe.txt", ".zip", ".zip.txt")
MODE_OPTIONS = {
    "sgd": "SGD baseline",
    "lightgbm_strict": "LightGBM строгий",
    "lightgbm_balanced": "LightGBM сбалансированный",
    "lightgbm_sensitive": "LightGBM чувствительный",
}
DEFAULT_LIGHTGBM_THRESHOLDS = {
    "strict": 0.1223439930122863,
    "balanced": 0.013283536035416748,
    "sensitive": 0.004806249853000717,
}
MODE_DESCRIPTIONS = {
    "sgd": "SGD baseline: честный SOREL-selected threshold; более сбалансирован на текущем gold.",
    "lightgbm_strict": "LightGBM строгий: минимум ложных тревог, но высокий риск пропуска опасных PE.",
    "lightgbm_balanced": "LightGBM сбалансированный: компромисс между FPR и FNR по gold threshold analysis.",
    "lightgbm_sensitive": "LightGBM чувствительный: ловить подозрительные PE; больше ложных тревог.",
}
MODE_LABEL_TO_KEY = {label: key for key, label in MODE_OPTIONS.items()}


def mode_key_from_label(label: str) -> str:
    try:
        return MODE_LABEL_TO_KEY[label]
    except KeyError as exc:
        raise ValueError(f"Неизвестный режим: {label}") from exc


try:
    from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore
except Exception:  # optional drag-and-drop dependency
    DND_FILES = None
    TkinterDnD = None


def is_supported_file(path: Path) -> bool:
    return path.name.lower().endswith(ALLOWED_SUFFIXES)


def is_malwarebazaar_zip(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".zip") or name.endswith(".zip.txt")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_numeric_threshold(value: object) -> float | None:
    if isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0:
        return float(value)
    if isinstance(value, dict):
        preferred = (
            "selected_threshold",
            "best_threshold",
            "operating_threshold",
            "threshold",
        )
        for key in preferred:
            threshold = _find_numeric_threshold(value.get(key))
            if threshold is not None:
                return threshold
    return None


def load_sgd_threshold() -> tuple[float, str]:
    for path in (THRESHOLD_SELECTION_PATH, VALIDATION_METRICS_PATH):
        if not path.exists():
            continue
        try:
            threshold = _find_numeric_threshold(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            threshold = None
        if threshold is not None:
            return threshold, str(path.relative_to(ROOT_DIR))
    return 0.5, "значение по умолчанию"


def load_lightgbm_modes() -> dict[str, dict[str, object]]:
    if not LIGHTGBM_MODES_PATH.exists():
        return {}
    data = json.loads(LIGHTGBM_MODES_PATH.read_text(encoding="utf-8"))
    modes = data.get("modes", [])
    if not isinstance(modes, list):
        return {}
    return {str(item.get("mode")): item for item in modes if isinstance(item, dict) and item.get("mode")}


def resolve_mode(mode_key: str) -> tuple[Path, float, str, str]:
    if mode_key == "sgd":
        threshold, source = load_sgd_threshold()
        return SGD_MODEL_PATH, threshold, source, MODE_DESCRIPTIONS[mode_key]
    if not mode_key.startswith("lightgbm_"):
        raise ValueError(f"Неизвестный режим: {mode_key}")
    mode = mode_key.removeprefix("lightgbm_")
    modes = load_lightgbm_modes()
    mode_payload = modes.get(mode)
    if mode_payload is not None and mode_payload.get("threshold") is not None:
        threshold = float(mode_payload["threshold"])
        source = f"{LIGHTGBM_MODES_PATH.relative_to(ROOT_DIR)}:{mode}"
    else:
        threshold = DEFAULT_LIGHTGBM_THRESHOLDS[mode]
        source = f"встроенный fallback:{mode}"
    return LIGHTGBM_MODEL_PATH, threshold, source, MODE_DESCRIPTIONS[mode_key]


def run_prediction(path: Path, mode_key: str) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    if not path.is_file():
        raise ValueError("Нужно выбрать один файл, а не папку.")
    if not is_supported_file(path):
        raise ValueError("Поддерживаются только .exe/.dll/.pe, defanged .*.txt или MalwareBazaar .zip/.zip.txt.")

    model_path, threshold, threshold_source, mode_description = resolve_mode(mode_key)
    if not model_path.exists():
        raise FileNotFoundError(f"Не найдена модель: {model_path}")

    from predict_pe_file import predict_file

    result = predict_file(path, model_path, is_malwarebazaar_zip(path), threshold)
    result["source_file_sha256"] = file_sha256(path)
    result["source_file_size_bytes"] = path.stat().st_size
    result["threshold_source"] = threshold_source
    result["mode_key"] = mode_key
    result["mode_label"] = MODE_OPTIONS[mode_key]
    result["mode_description"] = mode_description
    result["model_path"] = str(model_path.relative_to(ROOT_DIR))
    return result


def format_result(result: dict[str, object]) -> str:
    probability = float(result["malware_probability"])
    threshold = float(result["threshold"])
    dangerous = probability >= threshold
    verdict = "уязвимо/опасно" if dangerous else "не уязвимо/безопасно"
    return "\n".join(
        [
            f"Результат: {verdict}",
            f"Режим: {result.get('mode_label', '-')} — {result.get('mode_description', '-')}",
            f"Модель: {result.get('model_path', '-')} ({result.get('model_type', '-')})",
            f"Вероятность опасного файла: {probability:.6f}",
            f"Порог: {threshold:.6f} ({result.get('threshold_source', 'неизвестно')})",
            "Хэш и метаданные:",
            f"SHA-256 проверенных PE-байт: {result.get('sha256', '-')}",
            f"SHA-256 выбранного файла: {result.get('source_file_sha256', '-')}",
            f"Размер выбранного файла: {result.get('source_file_size_bytes', '-')} байт",
            f"Режим чтения: {result.get('input_mode', '-')}",
            f"Признаков в модели: {result.get('feature_dim_model', '-')}",
            f"Исходных признаков EMBER-v2: {result.get('feature_dim_original', '-')}",
            "",
            "Безопасность: файл не запускается; выполняется только статический разбор байтов.",
        ]
    )


class PeCheckerApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Статическая проверка PE-файла")
        self.selected_path: Path | None = None
        self.results: queue.Queue[tuple[str, object]] = queue.Queue()
        self.mode_label = StringVar(value=MODE_OPTIONS["sgd"])

        self.status = StringVar(value="Выберите .exe/.dll/.pe, defanged .*.txt или MalwareBazaar .zip/.zip.txt")
        Label(root, textvariable=self.status, anchor="w", justify=LEFT, padx=10, pady=8).pack(fill="x")

        buttons = Frame(root)
        buttons.pack(fill="x", padx=10, pady=4)
        Button(buttons, text="Выбрать файл", command=self.choose_file).pack(side=LEFT)
        Label(buttons, text="Режим:").pack(side=LEFT, padx=(12, 4))
        OptionMenu(buttons, self.mode_label, *MODE_OPTIONS.values()).pack(side=LEFT)
        self.check_button = Button(buttons, text="Проверить", command=self.start_check, state=DISABLED)
        self.check_button.pack(side=RIGHT)

        self.output = Text(root, height=18, width=92, wrap="word")
        self.output.pack(fill=BOTH, expand=True, padx=10, pady=10)
        self.write_output("Файл не выполняется. Архив MalwareBazaar читается в памяти, без распаковки PE на диск.\n")

        if DND_FILES is not None:
            self.output.drop_target_register(DND_FILES)  # type: ignore[attr-defined]
            self.output.dnd_bind("<<Drop>>", self.on_drop)  # type: ignore[attr-defined]
            self.write_output("Можно перетащить файл в это окно.\n")
        else:
            self.write_output("Перетаскивание недоступно: установите tkinterdnd2, либо используйте кнопку выбора файла.\n")

    def write_output(self, text: str) -> None:
        self.output.configure(state=NORMAL)
        self.output.delete("1.0", END)
        self.output.insert(END, text)
        self.output.configure(state=DISABLED)

    def choose_file(self) -> None:
        filename = filedialog.askopenfilename(
            title="Выберите PE-файл или MalwareBazaar ZIP",
            filetypes=[
                ("Поддерживаемые файлы", "*.exe *.dll *.pe *.exe.txt *.dll.txt *.pe.txt *.zip *.zip.txt"),
                ("Все файлы", "*"),
            ],
        )
        if filename:
            self.set_path(Path(filename))

    def on_drop(self, event: object) -> None:
        dropped = self.root.tk.splitlist(getattr(event, "data", ""))
        if dropped:
            self.set_path(Path(dropped[0]))

    def set_path(self, path: Path) -> None:
        self.selected_path = path
        self.status.set(f"Выбран файл: {path}")
        self.check_button.configure(state=NORMAL)
        if not is_supported_file(path):
            self.write_output("Предупреждение: расширение не входит в список поддерживаемых. Проверка может быть отклонена.\n")
        else:
            self.write_output(f"Готово к проверке: {path}\nНажмите «Проверить».\n")

    def start_check(self) -> None:
        if self.selected_path is None:
            messagebox.showwarning("Нет файла", "Сначала выберите файл.")
            return
        self.check_button.configure(state=DISABLED)
        mode_key = mode_key_from_label(self.mode_label.get())
        self.write_output(f"Идёт статическое извлечение признаков и проверка моделью ({MODE_OPTIONS[mode_key]})...\n")
        thread = threading.Thread(target=self._worker, args=(self.selected_path, mode_key), daemon=True)
        thread.start()
        self.root.after(100, self.poll_result)

    def _worker(self, path: Path, mode_key: str) -> None:
        try:
            self.results.put(("ok", run_prediction(path, mode_key)))
        except SystemExit as exc:
            self.results.put(("error", str(exc) or "Проверка остановлена: входные данные не похожи на PE-файл."))
        except (FileNotFoundError, ValueError, RuntimeError, ImportError) as exc:
            self.results.put(("error", str(exc)))
        except Exception as exc:
            self.results.put(("error", f"Непредвиденная ошибка: {exc}\n\n{traceback.format_exc()}"))

    def poll_result(self) -> None:
        try:
            status, payload = self.results.get_nowait()
        except queue.Empty:
            self.root.after(100, self.poll_result)
            return
        self.check_button.configure(state=NORMAL)
        if status == "ok":
            self.write_output(format_result(payload))  # type: ignore[arg-type]
        else:
            self.write_output("Ошибка проверки:\n" + str(payload))


def create_root() -> Tk:
    if TkinterDnD is not None:
        return TkinterDnD.Tk()  # type: ignore[no-any-return]
    return Tk()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Графический офлайн-интерфейс для статической проверки PE-файла.")
    parser.add_argument("--check-imports", action="store_true", help="Проверить импорт main.py без запуска GUI.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check_imports:
        checks = []
        for key in MODE_OPTIONS:
            model, threshold, source, _description = resolve_mode(key)
            checks.append(f"{key}: модель={model.relative_to(ROOT_DIR)}, threshold={threshold:.6f} ({source})")
        print("OK: main.py импортируется; " + "; ".join(checks))
        return 0

    root = create_root()
    PeCheckerApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
