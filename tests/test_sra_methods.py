import pytest
import pandas as pd
from anndata import AnnData

import adata_hf_datasets.pp.sra as sra_module
from adata_hf_datasets.pp.sra import (
    filter_invalid_sra_ids,
    maybe_add_sra_metadata,
    fetch_sra_metadata,
)


class DummySRAweb:
    def __init__(self, expected_chunks, df_map):
        # expected_chunks: list of lists of IDs
        # df_map: dict mapping ID -> row dict
        self.expected_chunks = expected_chunks
        self.df_map = df_map
        self.calls = []

    def sra_metadata(self, chunk_ids):
        # record call
        self.calls.append(list(chunk_ids))
        # build rows for each id in chunk
        rows = []
        for sid in chunk_ids:
            entry = self.df_map.get(sid)
            if entry:
                rows.append(entry)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def patch_sraweb(monkeypatch):
    # By default, patch SRAweb to DummySRAweb later in individual tests
    monkeypatch.setattr(sra_module, "SRAweb", lambda: DummySRAweb([], {}))
    yield


# Tests for filter_invalid_sra_ids


def test_filter_invalid_missing_column():
    ad = AnnData(obs=pd.DataFrame({"foo": ["SRX1"]}))
    # missing srx_column leads to False
    with pytest.raises(KeyError):
        filter_invalid_sra_ids(ad, srx_column="missing", srs_column=None)


def test_filter_invalid_all_invalid():
    obs = pd.DataFrame({"srx": ["ABC", "XYZ", "123"]})
    ad = AnnData(obs=obs)
    # all invalid SRX -> should return False
    result = filter_invalid_sra_ids(ad, srx_column="srx", srs_column=None)
    assert result is False


def test_filter_invalid_partial_within_tolerance():
    # 5 entries, 1 invalid (20%), tolerance 20% => allowed
    ids = ["SRX1", "SRX2", "SRX3", "INVALID", "SRX5"]
    obs = pd.DataFrame({"srx": ids})
    ad = AnnData(obs=obs)
    result = filter_invalid_sra_ids(ad, srx_column="srx", srs_column=None)
    assert result is True


def test_filter_invalid_partial_exceeds_tolerance():
    # 5 entries, 2 invalid (40%), tolerance 20% => not allowed
    ids = ["SRX1", "INVALID1", "INVALID2", "SRX2", "SRX3"]
    obs = pd.DataFrame({"srx": ids})
    ad = AnnData(obs=obs)
    result = filter_invalid_sra_ids(ad, srx_column="srx", srs_column=None)
    assert result is False


# Tests for maybe_add_sra_metadata


def test_maybe_add_sra_metadata_skips_when_invalid(monkeypatch):
    ad = AnnData(obs=pd.DataFrame(index=["foo", "bar"]))
    # patch filter_invalid to return False
    monkeypatch.setattr(
        sra_module, "filter_invalid_sra_ids", lambda *args, **kwargs: False
    )
    called = {"fetch": False}
    monkeypatch.setattr(
        sra_module,
        "fetch_sra_metadata",
        lambda *args, **kwargs: called.__setitem__("fetch", True),
    )
    maybe_add_sra_metadata(ad, new_cols=["a"], sample_id_key="foo", exp_id_key="bar")
    assert called["fetch"] is False


def test_maybe_add_sra_metadata_calls_fetch(monkeypatch):
    # create adata with valid index and columns
    idx = ["SRX1", "SRX2"]
    obs = pd.DataFrame({"accession": ["SRS1", "SRS2"]}, index=idx)
    ad = AnnData(obs=obs)
    # patch filter_invalid to return True
    monkeypatch.setattr(
        sra_module, "filter_invalid_sra_ids", lambda *args, **kwargs: True
    )
    captured = {}

    def fake_fetch(adata, **kwargs):
        captured["adata"] = adata
        captured["kwargs"] = kwargs

    monkeypatch.setattr(sra_module, "fetch_sra_metadata", fake_fetch)
    maybe_add_sra_metadata(
        ad,
        new_cols=["foo"],
        sample_id_key="accession",
        exp_id_key="experiment_accession",
    )
    assert "adata" in captured
    assert captured["kwargs"]["new_cols"] == ["foo"]


def test_maybe_add_sra_metadata_adds_columns(monkeypatch):
    idx = ["SRX1", "SRX2"]
    obs = pd.DataFrame({"accession": ["SRS1", "SRS2"]}, index=idx)
    ad = AnnData(obs=obs)
    # force filtering to pass
    monkeypatch.setattr(
        sra_module, "filter_invalid_sra_ids", lambda *args, **kwargs: True
    )

    # fake fetch populates a new column
    def fake_fetch(adata, **kwargs):
        adata.obs["new_meta"] = ["val1", "val2"]

    monkeypatch.setattr(sra_module, "fetch_sra_metadata", fake_fetch)
    maybe_add_sra_metadata(
        ad,
        new_cols=["new_meta"],
        sample_id_key="accession",
        exp_id_key="experiment_accession",
    )
    assert "new_meta" in ad.obs.columns
    assert ad.obs["new_meta"].tolist() == ["val1", "val2"]


# Tests for fetch_sra_metadata


def test_fetch_sra_metadata_missing_sample_key():
    ad = AnnData(obs=pd.DataFrame({"experiment_accession": ["SRX1"]}))
    with pytest.raises(ValueError):
        fetch_sra_metadata(ad, sample_id_key="missing")


def test_fetch_sra_metadata_missing_exp_key():
    obs = pd.DataFrame({"accession": ["SRS1"]})
    ad = AnnData(obs=obs)
    with pytest.raises(ValueError):
        fetch_sra_metadata(ad, sample_id_key="accession", exp_id_key="missing")


def test_fetch_sra_metadata_no_unique_ids():
    obs = pd.DataFrame(
        {"accession": [None, None], "experiment_accession": [None, None]}
    )
    ad = AnnData(obs=obs)
    with pytest.raises(ValueError):
        fetch_sra_metadata(
            ad, sample_id_key="accession", exp_id_key="experiment_accession"
        )


def test_fetch_sra_metadata_missing_sra_key(monkeypatch):
    # one valid id, but returned df lacks sample_accession column
    obs = pd.DataFrame({"accession": ["SRS1"], "experiment_accession": ["SRX1"]})
    ad = AnnData(obs=obs)
    # patch SRAweb to return df without sample_accession
    dummy = DummySRAweb(
        expected_chunks=[["SRS1"]],
        df_map={"SRS1": {"experiment_accession": "SRX1", "library_layout": "LAYOUT1"}},
    )
    monkeypatch.setattr(sra_module, "SRAweb", lambda: dummy)
    with pytest.raises(ValueError):
        fetch_sra_metadata(
            ad,
            sample_id_key="accession",
            exp_id_key="experiment_accession",
            new_cols=["library_layout"],
        )


def test_fetch_sra_metadata_chunking_and_merge(monkeypatch):
    # simulate 4 unique samples across 4 experiments
    obs = pd.DataFrame(
        {
            "accession": ["SRS1", "SRS2", "SRS3", "SRS4"],
            "experiment_accession": ["SRX1", "SRX2", "SRX3", "SRX4"],
        }
    )
    ad = AnnData(obs=obs)
    # define df_map entries for each sample
    df_map = {
        "SRS1": {
            "sample_accession": "SRS1",
            "experiment_accession": "SRX1",
            "a": "A1",
            "b": "B1",
        },
        "SRS2": {
            "sample_accession": "SRS2",
            "experiment_accession": "SRX2",
            "a": "A2",
            "b": "B2",
        },
        "SRS3": {
            "sample_accession": "SRS3",
            "experiment_accession": "SRX3",
            "a": "A3",
            "b": "B3",
        },
        "SRS4": {
            "sample_accession": "SRS4",
            "experiment_accession": "SRX4",
            "a": "A4",
            "b": "B4",
        },
    }
    # expect two chunks: first two ids, next two
    dummy = DummySRAweb(
        expected_chunks=[["SRS1", "SRS2"], ["SRS3", "SRS4"]], df_map=df_map
    )
    monkeypatch.setattr(sra_module, "SRAweb", lambda: dummy)
    ad = fetch_sra_metadata(
        ad,
        sample_id_key="accession",
        exp_id_key="experiment_accession",
        new_cols=["a", "b"],
        chunk_size=2,
        fallback="UNK",
    )
    # verify chunks were called correctly
    assert dummy.calls == [["SRS1", "SRS2"], ["SRS3", "SRS4"]]
    # check that a and b columns exist and match
    for col in ["a", "b"]:
        # values match df_map order
        expected = [df_map[s]["{}".format(col)] for s in obs["accession"]]
        assert ad.obs[col].tolist() == expected


def test_fetch_sra_metadata_missing_new_cols(monkeypatch):
    # same as above but request a missing col 'c', expect fallback
    obs = pd.DataFrame({"accession": ["SRS1"], "experiment_accession": ["SRX1"]})
    ad = AnnData(obs=obs)
    df_map = {
        "SRS1": {"sample_accession": "SRS1", "experiment_accession": "SRX1", "a": "A1"}
    }
    dummy = DummySRAweb(expected_chunks=[["SRS1"]], df_map=df_map)
    monkeypatch.setattr(sra_module, "SRAweb", lambda: dummy)
    ad = fetch_sra_metadata(
        ad,
        sample_id_key="accession",
        exp_id_key="experiment_accession",
        new_cols=["a", "c"],
        fallback="NA",
        chunk_size=1,
    )
    assert ad.obs["a"].tolist() == ["A1"]
    # missing 'c', should be fallback
    assert ad.obs["c"].tolist() == ["NA"]
