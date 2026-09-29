import pandas as pd

from train_model import load_data


def test_dataset_has_no_duplicate_text_after_load():
    data = load_data()
    assert data["text"].duplicated().sum() == 0


def test_dataset_has_no_nulls_after_load():
    data = load_data()
    assert data["text"].isna().sum() == 0
    assert data["label"].isna().sum() == 0


def test_labels_are_only_fake_or_real():
    data = load_data()
    assert set(data["label"].unique()) <= {"FAKE", "REAL"}


def test_raw_csv_actually_has_duplicates_this_guards_the_regression():
    """Sanity check that the source CSV still has the duplicate-row issue
    train_model.load_data() is responsible for removing. If this ever
    fails because the raw CSV was cleaned upstream, the dedup step in
    load_data() becomes a no-op and can be simplified."""
    raw = pd.read_csv("news_dataset.csv")
    raw = raw.dropna()
    assert raw["text"].duplicated().sum() > 0
