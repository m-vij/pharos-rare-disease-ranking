import json
import sys

import pytest

from src.split import main, make_split


def test_split_is_deterministic_disjoint_and_complete():
    ids = [f"OMIM:{i}" for i in range(100)] * 2                           # duplicates (a disease with two profiles) count once
    s = make_split(ids, seed=1, tuning_share=0.3)
    assert s == make_split(list(reversed(ids)), seed=1, tuning_share=0.3)
    assert len(s["tuning"]) == 30 and len(s["heldout"]) == 70 and not set(s["tuning"]) & set(s["heldout"])
    assert set(s["tuning"]) | set(s["heldout"]) == set(ids)
    assert s != make_split(ids, seed=2, tuning_share=0.3)


def test_the_split_file_is_never_overwritten(tmp_path, monkeypatch):
    prof = tmp_path / "profiles.json"
    prof.write_text(json.dumps([{"disease_id": f"OMIM:{i}"} for i in range(10)]))
    out = tmp_path / "split.json"
    monkeypatch.setattr(sys, "argv", ["split", "--profiles", str(prof), "--out", str(out)])
    main()
    first = out.read_text()
    with pytest.raises(SystemExit):
        main()
    assert out.read_text() == first
