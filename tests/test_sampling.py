"""Сэмплирование и манифест."""

from btzsc_ru.sampling import (
    build_manifest_rows,
    class_coverage,
    manifest_hash,
    read_manifest,
    stratified_indices,
    write_manifest,
)

VERB = ["класс A", "класс B", "класс C"]
GOLD = [0] * 50 + [1] * 30 + [2] * 20


def test_stratified_is_deterministic():
    a = stratified_indices(GOLD, 20, seed=42)
    b = stratified_indices(GOLD, 20, seed=42)
    assert a == b
    assert len(a) == 20


def test_stratified_covers_all_classes():
    idx = stratified_indices(GOLD, 6, seed=42)
    assert {GOLD[i] for i in idx} == {0, 1, 2}


def test_stratified_none_returns_everything():
    assert stratified_indices(GOLD, None, seed=42) == list(range(len(GOLD)))


def test_manifest_roundtrip(tmp_path):
    idx = stratified_indices(GOLD, 9, seed=42)
    rows = build_manifest_rows("ds", GOLD, VERB, idx)
    path = write_manifest(rows, tmp_path / "sample_manifest.csv")
    again = read_manifest(path)
    assert [r.as_dict() for r in rows] == [r.as_dict() for r in again]
    assert manifest_hash(rows) == manifest_hash(again)
    assert class_coverage(rows)["ds"] == 1.0
    assert rows[0].gold_label == VERB[rows[0].gold_index]


def test_manifest_hash_changes_with_selection():
    a = build_manifest_rows("ds", GOLD, VERB, stratified_indices(GOLD, 9, seed=42))
    b = build_manifest_rows("ds", GOLD, VERB, stratified_indices(GOLD, 12, seed=42))
    assert manifest_hash(a) != manifest_hash(b)
