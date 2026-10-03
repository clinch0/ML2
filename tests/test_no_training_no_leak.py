"""Жёсткие гарантии: внутри оценки нет обучения и gold не попадает в модель.

Проверка сигнатуры ловит только самый грубый случай. Здесь — разбор тел функций через AST
и проверка изоляции дообучения от пути оценки.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_MODULES = ("btzsc_ru/adapters.py", "btzsc_ru/runner.py", "btzsc_ru/metrics.py",
                "btzsc_ru/sampling.py", "btzsc_ru/reconstruction.py", "btzsc_ru/data.py")
TRAINING_CALLS = {"backward", "zero_grad", "fit"}
TRAINING_NAMES = {"Trainer", "TrainingArguments", "AdamW", "Adam", "SGD",
                  "get_linear_schedule_with_warmup"}


def calls_in(path: Path) -> tuple[set[str], set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    attrs, names = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                attrs.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                names.add(node.func.id)
    return attrs, names


def test_evaluation_path_has_no_training_calls():
    for rel in EVAL_MODULES:
        attrs, names = calls_in(ROOT / rel)
        assert not (attrs & TRAINING_CALLS), f"{rel}: вызовы обучения {attrs & TRAINING_CALLS}"
        assert not (names & TRAINING_NAMES), f"{rel}: обучающие классы {names & TRAINING_NAMES}"


def test_adapters_switch_models_to_eval_mode():
    source = (ROOT / "btzsc_ru/adapters.py").read_text(encoding="utf-8")
    assert ".eval()" in source, "модель должна переводиться в режим оценки"
    assert "no_grad" in source, "градиенты в оценке не нужны"


def test_gold_never_enters_adapter_body():
    from btzsc_ru import adapters

    forbidden = {"gold", "gold_index", "references", "y_true", "labels_true", "answers"}
    for cls_name in ("EncoderAdapter", "LLMAdapter", "DummyAdapter"):
        cls = getattr(adapters, cls_name)
        for method in ("predict", "predict_scores"):
            tree = ast.parse(inspect.getsource(getattr(cls, method)).lstrip())
            used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
            assert not (used & forbidden), f"{cls_name}.{method}: {used & forbidden}"


def test_finetune_is_isolated_from_results_csv():
    source = (ROOT / "btzsc_ru/finetune.py").read_text(encoding="utf-8")
    assert "ft_metrics.csv" in source
    assert "ResultsWriter" not in source, "дообучение не должно писать в общий results.csv"


def test_runner_does_not_import_finetune_in_evaluation():
    from btzsc_ru import runner

    assert "finetune" not in inspect.getsource(runner.evaluate)


def test_gold_is_not_passed_to_adapter():
    from btzsc_ru import runner

    source = inspect.getsource(runner.run_pair)
    start = source.index("adapter.predict(")
    call = source[start:source.index(")", start)]
    assert "gold" not in call, f"gold уходит в адаптер: {call}"
    assert "compute_metrics" in source, "gold используется только при расчёте метрики"


def test_artifacts_record_code_revision():
    """По артефакту должно быть видно, каким кодом он посчитан."""
    from btzsc_ru.io_utils import environment_info

    info = environment_info()
    assert "code_sha256" in info
    assert "git_commit" not in info and "git_dirty" not in info
    assert len(info["code_sha256"]) == 16


def test_comparability_is_recorded_for_every_row():
    """Каждая строка результата обязана нести признак сравнимости с публикацией."""
    from btzsc_ru.io_utils import RESULTS_COLUMNS

    assert "comparable_to_paper" in RESULTS_COLUMNS
    assert "comparability_reason" in RESULTS_COLUMNS


def test_deviations_are_documented_in_code():
    """Расхождения «статья ↔ код авторов ↔ мы» перечислены в коде, а не только в тексте."""
    from btzsc_ru.paper_compat import deviations

    items = deviations()
    assert len(items) >= 5
    for item in items:
        assert {"тема", "статья", "код авторов", "у нас", "влияние"} == set(item)
    темы = {i["тема"] for i in items}
    assert "объём оценки" in темы and "префиксы энкодеров" in темы
    assert any("A.4" in i["статья"] for i in items), "ссылка на приложение A.4 обязательна"
