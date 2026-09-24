import pandas as pd

from churn.data import FEATURE_COLS, clean, train_test


def test_clean_imputes_total_charges_of_new_customers(raw_df):
    df = clean(raw_df)
    assert df["TotalCharges"].dtype.kind == "f"
    assert df["TotalCharges"].isna().sum() == 0
    assert (df.loc[df["tenure"] == 0, "TotalCharges"] == 0).all()


def test_clean_encodes_target(raw_df):
    df = clean(raw_df)
    assert set(df["Churn"].unique()) <= {0, 1}
    assert df["Churn"].sum() == (raw_df["Churn"] == "Yes").sum()


def test_clean_is_idempotent(raw_df):
    once = clean(raw_df)
    pd.testing.assert_frame_equal(clean(once), once)


def test_train_test_split_is_stratified_and_reproducible(clean_df):
    X_tr, X_te, y_tr, y_te = train_test(clean_df, test_size=0.25)
    assert list(X_tr.columns) == FEATURE_COLS
    assert len(X_te) == round(0.25 * len(clean_df))
    assert abs(y_tr.mean() - y_te.mean()) < 0.03
    assert set(X_tr.index).isdisjoint(X_te.index)
    X_tr2, *_ = train_test(clean_df, test_size=0.25)
    assert X_tr.index.equals(X_tr2.index)
