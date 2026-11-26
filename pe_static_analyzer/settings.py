"""Настройки воспроизводимых опытов. Выберите профиль перед запуском модулей."""
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "examples/lightgbm.txt"


@dataclass(frozen=True)
class Experiment:
    size: int
    seed: int
    work: Path
    report: Path
    parent: Path | None = None


BASELINE = Experiment(100_000, 42, Path("outputs/research"), Path("reports/comparison"))
SMALL_SAMPLE = Experiment(30_000, 42, Path("outputs/research_30k"), Path("reports/volume_30k"),
                          parent=BASELINE.work / "stratified.csv")
REPEAT_43 = Experiment(100_000, 43, Path("outputs/research_43"), Path("reports/replicates/43"))
REPEAT_44 = Experiment(100_000, 44, Path("outputs/research_44"), Path("reports/replicates/44"))
ACTIVE_EXPERIMENT = BASELINE
