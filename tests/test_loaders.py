import json

from hrec.data import load_amazon_reviews


def test_amazon_2023_schema(tmp_path):
    rev = tmp_path / "Toys.jsonl"
    meta = tmp_path / "meta_Toys.jsonl"
    rows = [{"user_id": f"U{u}", "parent_asin": f"A{i}", "rating": 5.0, "timestamp": 1_600_000_000_000 + u * 1000 + i}
            for u in range(4) for i in range(3)]
    rev.write_text("\n".join(json.dumps(r) for r in rows))
    meta.write_text("\n".join(json.dumps({"parent_asin": f"A{i}", "title": f"Toy {i}", "categories": ["Toys", "Blocks"]})
                              for i in range(3)))
    ds = load_amazon_reviews(rev, meta, min_user=2, min_item=2)
    assert ds.n_users == 4 and ds.n_items == 3
    assert ds.item_meta["genres"].iloc[0] == ["Toys", "Blocks"]
    assert ds.interactions["ts"].min() == 1_600_000_000
