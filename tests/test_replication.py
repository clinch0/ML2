"""Блок сверки: модели статьи, режим replicate, сравнение с опубликованными числами."""

from __future__ import annotations

import csv
import json
import zipfile

import pytest

from btzsc_ru.config import (
    MODELS_BY_ID,
    PAPER_MODELS,
    PAPER_REFERENCE,
    REPLICATION_DATASET_KEYS,
    RunConfig,
    as_replication,
)
from scripts.check_replication import TOL_OK, compare, render


def test_paper_block_has_five_models_with_reference_numbers():
    assert len(PAPER_MODELS) == 5
    for spec in PAPER_MODELS:
        assert spec.revision and len(spec.revision) == 40, spec.model_id
        assert spec.model_id in MODELS_BY_ID
        ref = PAPER_REFERENCE[spec.model_id]
        from btzsc_ru.config import BTZSC_CORE_KEYS

        assert set(ref) == set(BTZSC_CORE_KEYS), spec.model_id   # эталон по всему ядру
        assert all(0.0 < v < 1.0 for v in ref.values())


def test_paper_models_follow_official_prefix_rules():
    """В btzsc/models/embedding.py префикс получают только e5-*; bge и MiniLM — без префикса."""
    by_id = {s.model_id: s for s in PAPER_MODELS}
    assert by_id["intfloat/e5-base-v2"].query_prefix == "query: "
    assert by_id["intfloat/e5-base-v2"].label_prefix == "passage: "
    assert by_id["intfloat/e5-large-v2"].query_prefix == "query: "
    for mid in ("BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"):
        assert by_id[mid].query_prefix == "", mid
        assert by_id[mid].label_prefix == "", mid


def test_pooling_matches_published_cards():
    by_id = {s.model_id: s for s in PAPER_MODELS}
    assert by_id["sentence-transformers/all-MiniLM-L6-v2"].pooling == "mean"
    assert by_id["intfloat/e5-base-v2"].pooling == "mean"
    assert by_id["BAAI/bge-base-en-v1.5"].pooling == "cls"


def test_reference_numbers_come_from_published_artifacts():
    """Числа статьи не вписаны руками, а читаются из paper_scores/*.json (results.by_dataset)."""
    import json
    from pathlib import Path

    from btzsc_ru.config import PAPER_LEADERBOARD_FILE, PAPER_SCORES_DIR, load_paper_reference

    ref = load_paper_reference()
    assert ref == PAPER_REFERENCE
    path = Path(PAPER_SCORES_DIR) / PAPER_LEADERBOARD_FILE["intfloat/e5-base-v2"]
    published = json.loads(path.read_text(encoding="utf-8"))["results"]["by_dataset"]["imdb"]["macro_f1"]
    assert ref["intfloat/e5-base-v2"]["btzsc_imdb"] == pytest.approx(published, abs=1e-4)


def test_replication_uses_short_text_datasets_for_diagnosis():
    """rottentomatoes и financialphrasebank короткие: отделяют обрезку длины от ошибки в коде."""
    from btzsc_ru.config import DATASETS_BY_KEY

    assert "btzsc_rottentomatoes" in REPLICATION_DATASET_KEYS
    assert "btzsc_financialphrasebank" in REPLICATION_DATASET_KEYS
    for key in ("btzsc_rottentomatoes", "btzsc_financialphrasebank"):
        spec = DATASETS_BY_KEY[key]
        assert spec.hf_id == "btzsc/btzsc" and spec.split == "test"


def test_run_covers_paper_core_and_ru_extension():
    """Дизайн: русские модели проходят ТОТ ЖЕ тест, что модели статьи, плюс русское расширение."""
    from btzsc_ru.config import BTZSC_CORE_KEYS, RU_EXTENSION_KEYS

    cfg = RunConfig()
    assert set(BTZSC_CORE_KEYS) <= set(cfg.dataset_keys), "ядро статьи должно быть в прогоне"
    assert set(RU_EXTENSION_KEYS) <= set(cfg.dataset_keys)
    # ядро покрывает все четыре типа задач статьи
    assert {"btzsc_agnews"} <= set(BTZSC_CORE_KEYS)                       # тема
    assert {"btzsc_imdb", "btzsc_rottentomatoes", "btzsc_financialphrasebank"} <= set(BTZSC_CORE_KEYS)
    assert {"btzsc_emotiondair"} <= set(BTZSC_CORE_KEYS)                  # эмоции
    assert {"btzsc_massive", "btzsc_banking77"} <= set(BTZSC_CORE_KEYS)   # намерения
    # у каждого датасета ядра есть опубликованное число статьи для сравнения
    import json
    from pathlib import Path

    from btzsc_ru.config import PAPER_LEADERBOARD_FILE, PAPER_SCORES_DIR

    published = json.loads(
        (Path(PAPER_SCORES_DIR) / PAPER_LEADERBOARD_FILE["intfloat/e5-base-v2"]).read_text(encoding="utf-8")
    )["results"]["by_dataset"]
    for key in BTZSC_CORE_KEYS:
        assert key.replace("btzsc_", "") in published, key

    # три параллельные EN/RU пары в расширении
    for a, b in (("massive_en", "massive_ru"), ("massive_scenario_en", "massive_scenario_ru"),
                 ("go_emotions_en", "go_emotions_ru")):
        assert a in cfg.dataset_keys and b in cfg.dataset_keys


def test_replication_config_switches_models_and_datasets():
    cfg = as_replication(RunConfig(mode="all"))
    assert cfg.mode == "replicate"
    assert cfg.dataset_keys == REPLICATION_DATASET_KEYS
    assert set(cfg.model_ids) == {s.model_id for s in PAPER_MODELS}
    assert cfg.resolved_n() is None, "сверка идёт на полном тесте, как у авторов"


def test_our_models_are_not_polluted_by_paper_block():
    cfg = RunConfig()
    assert not set(cfg.model_ids) & {s.model_id for s in PAPER_MODELS}
    assert len(cfg.model_ids) == 11, "9 энкодеров + 2 LLM нашего блока"


def make_rows(values: dict[str, float]) -> list[dict]:
    return [
        {"model_id": mid, "dataset_key": "btzsc_agnews", "status": "OK",
         "macro_f1": str(v), "n_samples": "7600"}
        for mid, v in values.items()
    ]


def test_compare_marks_close_numbers_as_matching():
    rows = compare(make_rows({"intfloat/e5-base-v2": 0.755}))
    assert rows[0]["verdict"] == "совпало"
    assert rows[0]["paper_macro_f1"] == pytest.approx(0.7609, abs=1e-3)
    assert abs(rows[0]["diff"]) <= TOL_OK


def test_compare_flags_structural_mismatch():
    rows = compare(make_rows({"intfloat/e5-base-v2": 0.40}))
    assert rows[0]["verdict"] == "расхождение"
    text = render("run9", rows)
    assert "расхождение" in text and "e5-base-v2" in text


def test_render_placeholder_without_data():
    text = render("run9", [])
    assert "ожидается прогон" in text


def test_build_reports_normalises_browser_names(tmp_path, monkeypatch):
    import scripts.build_reports as br

    monkeypatch.setattr(br, "ROOT", tmp_path)
    (tmp_path / "incoming").mkdir()
    raw = tmp_path / "colab_outputs_run1 (3).zip"
    with zipfile.ZipFile(raw, "w") as zf:
        zf.writestr("run_config.json", json.dumps({"mode": "evaluate"}))

    target = br.normalise_name(raw)

    assert target.name == "colab_outputs_run1.zip"
    assert target.parent.name == "incoming"
    assert br.run_id_of(target) == "run1"
    assert raw.exists(), "исходный архив без флага --delete-zip не трогаем"


def test_build_reports_finds_archives_in_both_places(tmp_path, monkeypatch):
    import scripts.build_reports as br

    monkeypatch.setattr(br, "ROOT", tmp_path)
    (tmp_path / "incoming").mkdir()
    (tmp_path / "incoming" / "colab_outputs_a.zip").write_bytes(b"x")
    (tmp_path / "colab_outputs_b.zip").write_bytes(b"x")
    (tmp_path / "something_else.zip").write_bytes(b"x")

    names = {p.name for p in br.find_archives()}

    assert names == {"colab_outputs_a.zip", "colab_outputs_b.zip"}


def test_prefix_policy_matches_paper_code():
    """По умолчанию префиксы — как в коде авторов, а не как в тексте §4."""
    from btzsc_ru.adapters import paper_code_prefixes
    from btzsc_ru.config import RunConfig

    assert RunConfig().prefix_policy == "paper_code"
    assert paper_code_prefixes("intfloat/e5-base-v2") == ("query: ", "passage: ")
    assert paper_code_prefixes("BAAI/bge-base-en-v1.5") == ("", ""), "у BGE в коде авторов префикса нет"
    assert paper_code_prefixes("sentence-transformers/all-MiniLM-L6-v2") == ("", "")
    assert paper_code_prefixes("ai-forever/ru-en-RoSBERTa") == ("", "")
    assert "Instruct:" in paper_code_prefixes("Qwen/Qwen3-Embedding-0.6B")[0]


def test_chat_template_off_by_default():
    """В коде авторов chat template не применяется — у нас тоже."""
    import inspect

    from btzsc_ru.adapters import LLMAdapter
    from btzsc_ru.config import RunConfig

    assert RunConfig().use_chat_template is False
    assert inspect.signature(LLMAdapter.__init__).parameters["use_chat_template"].default is False


def test_llm_prompt_limit_is_not_hardcoded_256():
    """Лимит промпта считается от задачи: 77 вариантов Banking77 в 256 токенов не влезают."""
    import inspect

    from btzsc_ru.adapters import LLMAdapter

    assert inspect.signature(LLMAdapter.__init__).parameters["max_length"].default is None
    src = inspect.getsource(LLMAdapter.predict_scores)
    assert "truncated_prompts" in src, "факт усечения должен фиксироваться"


def test_block_b_has_no_duplicate_origin():
    """BERTA и rubert-mini-frida — две дистилляции FRIDA; обе в блоке Б держать нельзя."""
    from btzsc_ru.config import BLOCK_B_MODEL_IDS

    assert "ai-forever/FRIDA" in BLOCK_B_MODEL_IDS, "сама FRIDA должна быть в блоке"
    distillates = {"sergeyzh/BERTA", "sergeyzh/rubert-mini-frida"}
    assert len(distillates & set(BLOCK_B_MODEL_IDS)) <= 1, "в блоке не больше одного дистиллята FRIDA"
    assert len(BLOCK_B_MODEL_IDS) == 5


def test_block_b_sources_are_distinct():
    """По одной лучшей модели от источника: Сбер, VK, независимый автор, Microsoft, Vikhr."""
    from btzsc_ru.config import BLOCK_B_MODEL_IDS

    owners = [m.split("/")[0] for m in BLOCK_B_MODEL_IDS]
    assert len(owners) == len(set(owners)), f"дубли по источнику: {owners}"


def test_frida_contract_matches_card():
    """FRIDA: CLS-пулинг и префикс categorize: — из карточки, а не из головы."""
    import json
    from pathlib import Path

    from btzsc_ru.config import MODELS_BY_ID

    spec = MODELS_BY_ID["ai-forever/FRIDA"]
    assert spec.pooling == "cls" and spec.query_prefix == "categorize: "
    assert len(spec.revision) == 40

    contracts = json.loads(Path("contracts.json").read_text(encoding="utf-8"))["models"]["ai-forever/FRIDA"]
    assert contracts["pooling"] == spec.pooling
    assert contracts["revision"] == spec.revision
    assert contracts["params_total"] == spec.params
