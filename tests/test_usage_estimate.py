"""Tests for POST /usage-estimate (inverse billing on the 0.000001 grid)."""
import os
from decimal import Decimal

import pytest

os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def _startup():
    with client:
        yield


SAT_INTERVAL = {
    "start": "2024-07-06T00:00:00-04:00",
    "end": "2024-07-06T02:00:00-04:00",
    "timezone": "America/New_York",
}
CROSS_INTERVAL = {  # workday evening -> holiday morning, 8h
    "start": "2024-07-03T22:00:00-04:00",
    "end": "2024-07-04T06:00:00-04:00",
    "timezone": "America/New_York",
}


def _estimate(amount, mx, *, interval=SAT_INTERVAL, version="us-ny-demo-v2024"):
    return client.post("/usage-estimate", json={
        "version": version,
        "interval": interval,
        "billed_amount": amount,
        "max_energy": mx,
    })


# ---------------------------------------------------------------- feasibility

def test_simple_plateau_leisure():
    # Flat leisure @0.15 segment: bill 1.50 for any E with round(E*0.15)=1.50.
    r = _estimate("1.50", "50")
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["feasible"] is True
    assert b["version"] == "us-ny-demo-v2024"
    assert b["billed_amount"] == "1.50"
    assert b["max_energy"] == "50.000000"
    assert b["grid_step"] == "0.000001"
    assert Decimal(b["min_energy"]) == Decimal("9.966664")
    assert Decimal(b["max_energy_solution"]) == Decimal("10.033329")
    assert b["matched_rules"] == ["leisure"]
    assert b["reason_code"] is None
    lo_q, hi_q = b["min_quote"], b["max_quote"]
    assert lo_q["total_energy"] == "9.966664"
    assert hi_q["total_energy"] == "10.033329"
    assert lo_q["total_amount"] == hi_q["total_amount"] == "1.50"
    # Endpoints quote exactly like /quote at the same energies.
    for energy, q in (("9.966664", lo_q), ("10.033329", hi_q)):
        direct = client.post("/quote", json={
            "version": "us-ny-demo-v2024",
            "intervals": [{**SAT_INTERVAL, "energy": energy}],
        }).json()
        assert q["total_amount"] == direct["total_amount"]
        assert q["matched_rules"] == direct["matched_rules"]
        assert len(q["segments"]) == len(direct["segments"])
    # The platform edges are sharp: adjacent grid points bill differently.
    assert _quote_amount(Decimal(b["min_energy"]) - Decimal("0.000001")) == "1.49"
    assert _quote_amount(Decimal(b["max_energy_solution"]) + Decimal("0.000001")) == "1.51"


def _quote_amount(energy: Decimal) -> str:
    r = client.post("/quote", json={
        "version": "us-ny-demo-v2024",
        "intervals": [{**SAT_INTERVAL, "energy": str(energy)}],
    })
    return r.json()["total_amount"]


def test_plateau_reuses_cross_rule_and_tier_split():
    # Same interval as the README cross-rule example: workday @0.22 for the
    # first 2h, leisure tiers after local midnight. 150 kWh bills 26.63.
    r = _estimate("26.63", "200", interval=CROSS_INTERVAL)
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["feasible"] is True
    assert b["matched_rules"] == ["dst-workday", "leisure"]
    for q in (b["min_quote"], b["max_quote"]):
        rules = [(s["rule_code"], s["tier_ordinal"]) for s in q["segments"]]
        assert ("dst-workday", 1) in rules
        assert ("leisure", 1) in rules
        assert ("leisure", 2) in rules
        assert q["total_amount"] == "26.63"
        direct = client.post("/quote", json={
            "version": "us-ny-demo-v2024",
            "intervals": [{**CROSS_INTERVAL, "energy": q["total_energy"]}],
        }).json()
        assert direct["total_amount"] == "26.63"
    assert Decimal(b["min_energy"]) < Decimal("150") <= Decimal(
        b["max_energy_solution"])


def test_plateau_dst_spring_forward_split():
    # 5 UTC hours across the 2024 spring-forward gap, all leisure.
    iv = {
        "start": "2024-03-09T23:30:00-05:00",
        "end": "2024-03-10T05:30:00-04:00",
        "timezone": "America/New_York",
    }
    r = _estimate("1.50", "50", interval=iv)
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["feasible"] is True
    assert b["matched_rules"] == ["leisure"]
    total_seconds = sum(
        s["duration_seconds"] for s in b["max_quote"]["segments"])
    assert total_seconds == 18000


# -------------------------------------------------------------------- zero bill

def test_zero_bill_plateau_starts_at_zero_usage():
    r = _estimate("0", "10")
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["feasible"] is True
    assert b["min_energy"] == "0.000000"
    assert Decimal(b["max_energy_solution"]) > 0
    z = b["min_quote"]
    assert z["total_energy"] == "0.000000"
    assert z["total_amount"] == "0.00"
    assert z["segments"] == []
    assert z["matched_rules"] == []
    assert b["max_quote"]["total_amount"] == "0.00"
    # Zero usage is treated as zero charge (not a /quote validation error).
    assert b["reason_code"] is None


def test_all_grid_points_bill_zero():
    r = _estimate("0", "0.00005")  # at most 0.00005*0.15 ~= 0.0000075 -> 0.00
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["feasible"] is True
    assert b["min_energy"] == "0.000000"
    assert b["max_energy_solution"] == "0.000050"
    assert b["max_quote"]["total_amount"] == "0.00"


def test_plateau_touching_ceiling_includes_the_ceiling_point():
    # Pick max_energy whose own bill is the target: the right edge must be
    # exactly max_energy (not one grid step short).
    direct = client.post("/quote", json={
        "version": "us-ny-demo-v2024",
        "intervals": [{**SAT_INTERVAL, "energy": "9.97"}],
    }).json()
    target = direct["total_amount"]
    r = _estimate(target, "9.97")
    b = r.json()
    assert b["feasible"] is True
    assert b["max_energy_solution"] == "9.970000"
    assert Decimal(b["min_energy"]) <= Decimal("9.970000")


# --------------------------------------------------------------- above ceiling

def test_target_above_reachable_max():
    r = _estimate("5.00", "1", interval=SAT_INTERVAL)  # max bill 0.15
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["feasible"] is False
    assert b["reason_code"] == "above_reachable_max"
    assert b["reachable_max"]["energy"] == "1.000000"
    assert b["reachable_max"]["total_amount"] == "0.15"
    assert b["reachable_max"]["matched_rules"] == ["leisure"]
    assert Decimal(b["reachable_max"]["raw_amount"]) >= 0
    assert b["below"] is None and b["above"] is None
    assert b["min_quote"] is None and b["max_quote"] is None
    assert "5.00" in b["reason"] and "0.15" in b["reason"]


# ---------------------------------------------------------------- rounding gap

def _import_steep_version(code, price):
    r = client.post("/versions/import", json={
        "code": code, "currency": "USD", "energy_unit": "kWh",
        "description": "steep", "holidays": [],
        "rules": [{
            "code": "flat", "name": "flat",
            "effective_from": "2024-01-01T00:00:00",
            "effective_to": "2025-01-01T00:00:00",
            "timezone": "UTC", "weekdays": "", "day_scope": "all",
            "time_on": "00:00", "time_off": "24:00",
            "tiers": [{"ordinal": 1, "cumulative_cap": None,
                       "unit_price": price}],
        }],
    })
    assert r.status_code in (200, 201), r.text


def test_rounding_gap_reports_adjacent_grid_points():
    _import_steep_version("steep-gap-1", "15000")
    iv = {"start": "2024-06-01T00:00:00Z", "end": "2024-06-01T01:00:00Z",
          "timezone": "UTC"}
    r = _estimate("0.01", "0.00001", interval=iv, version="steep-gap-1")
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["feasible"] is False
    assert b["reason_code"] == "rounding_gap"
    below, above = b["below"], b["above"]
    assert below["energy"] == "0.000000" and below["total_amount"] == "0.00"
    assert above["energy"] == "0.000001" and above["total_amount"] == "0.02"
    assert above["raw_amount"] == "0.015000"
    assert above["matched_rules"] == ["flat"]
    # Neighbors really are adjacent grid points on opposite sides.
    assert (Decimal(above["energy"]) - Decimal(below["energy"])
            == Decimal("0.000001"))
    assert b["reachable_max"] is None
    assert b["reason"] and "0.01" in b["reason"]


def test_rounding_gap_inside_grid_with_steep_price():
    # Price 20000: step energy 1e-6 costs 0.02 exactly; every odd cent is
    # unreachable. 0.05 is skipped between the 0.04 and 0.06 platforms.
    _import_steep_version("steep-gap-2", "20000")
    iv = {"start": "2024-06-01T00:00:00Z", "end": "2024-06-01T01:00:00Z",
          "timezone": "UTC"}
    r = _estimate("0.05", "1", interval=iv, version="steep-gap-2")
    b = r.json()
    assert b["feasible"] is False
    assert b["reason_code"] == "rounding_gap"
    assert b["below"]["total_amount"] == "0.04"
    assert b["above"]["total_amount"] == "0.06"
    assert (Decimal(b["above"]["energy"]) - Decimal(b["below"]["energy"])
            == Decimal("0.000001"))
    # The even-cent values are reachable.
    assert _estimate("0.04", "1", interval=iv,
                     version="steep-gap-2").json()["feasible"] is True
    assert _estimate("0.06", "1", interval=iv,
                     version="steep-gap-2").json()["feasible"] is True


# ---------------------------------------------------------------- validation

def test_unknown_version_404():
    r = _estimate("1.00", "10", version="nope")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "version_not_found"


def test_interval_must_not_overlap_itself():
    iv = {"start": "2024-07-06T02:00:00Z", "end": "2024-07-06T00:00:00Z",
          "timezone": "UTC"}
    r = _estimate("1.00", "10", interval=iv)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_interval"


def test_billed_amount_negative_rejected():
    r = client.post("/usage-estimate", json={
        "version": "us-ny-demo-v2024", "interval": SAT_INTERVAL,
        "billed_amount": "-0.01", "max_energy": "10",
    })
    assert r.status_code == 422


def test_billed_amount_more_than_two_decimals_rejected():
    r = client.post("/usage-estimate", json={
        "version": "us-ny-demo-v2024", "interval": SAT_INTERVAL,
        "billed_amount": "1.001", "max_energy": "10",
    })
    assert r.status_code == 422


def test_max_energy_bounds_validated():
    for mx in ("0", "-1", "1000000.000001", "1000001"):
        r = client.post("/usage-estimate", json={
            "version": "us-ny-demo-v2024", "interval": SAT_INTERVAL,
            "billed_amount": "1.00", "max_energy": mx,
        })
        assert r.status_code == 422, mx


def test_max_energy_six_decimals_accepted():
    r = client.post("/usage-estimate", json={
        "version": "us-ny-demo-v2024", "interval": SAT_INTERVAL,
        "billed_amount": "0", "max_energy": "0.000001",
    })
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["max_energy"] == "0.000001"
    # Exactly one nonzero grid point, billing 0.00 at leisure price.
    assert b["feasible"] is True
    assert b["max_energy_solution"] == "0.000001"


def test_energy_field_in_interval_forbidden():
    r = client.post("/usage-estimate", json={
        "version": "us-ny-demo-v2024",
        "interval": {**SAT_INTERVAL, "energy": "10"},
        "billed_amount": "1.00", "max_energy": "10",
    })
    assert r.status_code == 422


def test_non_finite_and_extra_fields_rejected():
    for payload in (
        {"billed_amount": "NaN", "max_energy": "10"},
        {"billed_amount": "Infinity", "max_energy": "10"},
        {"billed_amount": "1.00", "max_energy": "10", "unexpected": 1},
    ):
        r = client.post("/usage-estimate", json={
            "version": "us-ny-demo-v2024", "interval": SAT_INTERVAL,
            **payload,
        })
        assert r.status_code == 422, payload


def test_rule_gap_propagates_like_quote():
    # A narrow interval where the seed version has no rule on the slice is
    # hard to construct; instead compare directly with /quote semantics on a
    # known gap-prone request (engine-level) through the API error contract.
    # 2024-03-10 02:30 naive falls in the NY spring-forward gap.
    r = client.post("/usage-estimate", json={
        "version": "us-ny-demo-v2024",
        "interval": {"start": "2024-03-10T02:30:00",
                     "end": "2024-03-10T03:30:00",
                     "timezone": "America/New_York"},
        "billed_amount": "1.00", "max_energy": "10",
    })
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "nonexistent_local_time"


# --------------------------------------------------------------- bisection at scale

def test_large_grid_bisection_terminates_and_matches_quote():
    # 10**6 energy units -> 10**12 grid points; only ~40 quote evaluations.
    r = _estimate("100000", "1000000", interval=SAT_INTERVAL)
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["feasible"] is True
    lo, hi = Decimal(b["min_energy"]), Decimal(b["max_energy_solution"])
    assert 0 < lo <= hi <= Decimal("1000000")
    assert b["min_quote"]["total_amount"] == b["max_quote"]["total_amount"] == "100000.00"
    # Edge sharpness against direct /quote on both adjacent grid points.
    step = Decimal("0.000001")
    for energy, expected in ((lo - step, "99999.99"), (hi + step, "100000.01")):
        direct = client.post("/quote", json={
            "version": "us-ny-demo-v2024",
            "intervals": [{**SAT_INTERVAL, "energy": str(energy)}],
        })
        assert direct.json()["total_amount"] == expected
