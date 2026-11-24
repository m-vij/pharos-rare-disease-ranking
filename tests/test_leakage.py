from src.leakage import BUCKETS, bucket_of, by_bucket, removal_by_profile
from src.scorer import make_scorer
from tests.test_scorer import kb


def case(cid, profile, disease, present, pmids=()):
    return {"case_id": cid, "profile_id": profile, "disease_id": disease, "source": "phenopackets",
            "present": present, "exclude_pmids": list(pmids)}


def test_buckets():
    assert [bucket_of(x) for x in (1.0, 0.99, 0.75, 0.74, 0.5, 0.49, 0.25, 0.24, 0.0)] == [
        BUCKETS[0], BUCKETS[1], BUCKETS[1], BUCKETS[2], BUCKETS[2], BUCKETS[3], BUCKETS[3], BUCKETS[4], BUCKETS[4]]


def test_removal_fraction_per_profile_and_recall_by_bucket(tmp_path):
    ann, _ = kb(tmp_path)
    cases = [case("a", "P1", "OMIM:1", ["HP:0000011"], pmids=["PMID:111"]),            # D1 loses its spasm row: 1 of 2 removed
             case("b", "P2", "OMIM:4", ["HP:0000013"], pmids=["PMID:555"]),            # D4 loses its only row: all removed
             case("c", "P3", "OMIM:3", ["HP:0000011"])]                                 # nothing removed
    frac = removal_by_profile(ann, cases)
    assert frac == {"P1": 0.5, "P2": 1.0, "P3": 0.0}
    res = by_bucket(ann, [make_scorer("exact_overlap", ann)], cases, n_boot=20)
    assert {b: v["profiles"] for b, v in res.items()} == {BUCKETS[0]: 1, BUCKETS[2]: 1, BUCKETS[4]: 1}
    assert res[BUCKETS[0]]["results"]["exact_overlap"]["phenopackets"]["R@1"][0] == 0.25      # nothing left: all four diseases tie at zero
