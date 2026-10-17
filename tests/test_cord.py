import json

from underwrite.data.cord import extract_gold


def gt(gt_parse: dict) -> str:
    return json.dumps({"gt_parse": gt_parse, "meta": {"version": "2.0.0"}})


def test_maps_all_three_money_fields():
    gold = extract_gold(
        gt(
            {
                "menu": {"nm": "BASO BIHUN", "price": "43.636"},
                "sub_total": {"subtotal_price": "43.636", "tax_price": "4.364"},
                "total": {"total_price": "48.000", "cashprice": "50.000"},
            }
        )
    )
    assert gold == {"total": "48000.00", "subtotal": "43636.00", "tax": "4364.00"}


def test_missing_sub_total_yields_only_total():
    gold = extract_gold(gt({"menu": {"nm": "PKT AYAM"}, "total": {"total_price": "25.000"}}))
    assert gold == {"total": "25000.00"}


def test_comma_thousands_are_stripped():
    gold = extract_gold(
        gt(
            {
                "sub_total": {"subtotal_price": "259,000", "tax_price": "52,416"},
                "total": {"total_price": "330,000"},
            }
        )
    )
    assert gold == {"total": "330000.00", "subtotal": "259000.00", "tax": "52416.00"}


def test_never_produces_vendor_or_date():
    gold = extract_gold(gt({"total": {"total_price": "10,000"}}))
    assert set(gold) == {"total"}


def test_unparseable_amount_is_omitted():
    assert extract_gold(gt({"total": {"total_price": "N/A"}})) == {}


def test_empty_gt_parse_yields_nothing():
    assert extract_gold(gt({})) == {}


def test_flat_string_amount_is_tolerated():
    assert extract_gold(gt({"total": "25,000"})) == {}
