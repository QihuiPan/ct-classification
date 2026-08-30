from pathlib import Path

from scripts.prepare_covid_ct_md import _age_group, _age_in_years, label_from_directory


def test_class_directory_aliases() -> None:
    assert label_from_directory(Path("COVID-19")) == "covid19"
    assert label_from_directory(Path("COVID-19 Cases")) == "covid19"
    assert label_from_directory(Path("Cap Cases")) == "cap"
    assert label_from_directory(Path("CAP subjects")) == "cap"
    assert label_from_directory(Path("Normal Cases")) == "normal"
    assert label_from_directory(Path("Normal")) == "normal"
    assert label_from_directory(Path("misc")) is None


def test_dicom_age_parsing_and_grouping() -> None:
    assert _age_in_years("045Y") == 45
    assert _age_group(_age_in_years("039Y")) == "under_40"
    assert _age_group(_age_in_years("040Y")) == "40_to_59"
    assert _age_group(_age_in_years("060Y")) == "60_plus"
    assert _age_group(_age_in_years("")) == "unknown"
