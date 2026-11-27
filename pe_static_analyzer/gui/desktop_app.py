#!/usr/bin/env python3
"""Небольшой офлайн-интерфейс для статической проверки PE-файлов."""

from __future__ import annotations

from pe_static_analyzer.settings import DEFAULT_MODEL
from pe_static_analyzer.core.pe_input import sha256_file
import queue

import threading
import traceback
from pathlib import Path
from tkinter import BOTH, END, LEFT, NORMAL, DISABLED, RIGHT, Button, Frame, Label, StringVar, Text, Tk, filedialog, messagebox


ROOT_DIR = Path(__file__).resolve().parents[2]

ALLOWED_SUFFIXES = (".exe", ".dll", ".pe", ".exe.txt", ".dll.txt", ".pe.txt", ".zip", ".zip.txt")


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


def run_prediction(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    if not path.is_file():
        raise ValueError("Нужно выбрать один файл, а не папку.")
    if not is_supported_file(path):
        raise ValueError("Поддерживаются только .exe/.dll/.pe, defanged .*.txt или MalwareBazaar .zip/.zip.txt.")

    from pe_static_analyzer.prediction.single_file import predict_file

    result = predict_file(path, malwarebazaar_zip=is_malwarebazaar_zip(path))
    result["source_file_sha256"] = sha256_file(path)
    result["source_file_size_bytes"] = path.stat().st_size
    result["threshold_source"] = "фиксированный порог"
    result["model_path"] = str(DEFAULT_MODEL.relative_to(ROOT_DIR))
    return result


def format_result(result: dict[str, object]) -> str:
    probability = float(result["malware_probability"])
    threshold = float(result["threshold"])
    dangerous = probability >= threshold
    verdict = "вредоносный" if dangerous else "легитимный"
    return "\n".join(
        [
            f"Результат: {verdict}",
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

        self.status = StringVar(value="Выберите .exe/.dll/.pe, defanged .*.txt или MalwareBazaar .zip/.zip.txt")
        Label(root, textvariable=self.status, anchor="w", justify=LEFT, padx=10, pady=8).pack(fill="x")

        buttons = Frame(root)
        buttons.pack(fill="x", padx=10, pady=4)
        Button(buttons, text="Выбрать файл", command=self.choose_file).pack(side=LEFT)
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
        self.write_output("Проверка LightGBM на 600 деревьях без заголовка PE…\n")
        thread = threading.Thread(target=self._worker, args=(self.selected_path,), daemon=True)
        thread.start()
        self.root.after(100, self.poll_result)

    def _worker(self, path: Path) -> None:
        try:
            self.results.put(("ok", run_prediction(path)))
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


def main() -> int:
    root = create_root()
    PeCheckerApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
