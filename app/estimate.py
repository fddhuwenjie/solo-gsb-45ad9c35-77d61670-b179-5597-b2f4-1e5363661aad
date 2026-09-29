"""Inverse billing: recover the metered usage behind a known bill amount.

Scenario
--------
The meter reading for one time interval is missing, but the rounded total
charge (to cents) is known. This module searches the **discrete energy grid**
of multiples of ``0.000001`` (``Q6``) up to a caller-supplied ``max_energy``
for the grid points whose :func:`pricing.quote` bill rounds
(``ROUND_HALF_UP`` to ``Q2``) to exactly the billed amount.

Assumptions
-----------
* Power is **constant** over the interval, so one positive energy value is
  distributed by the regular quote engine proportionally to elapsed UTC
  seconds — exactly the same slicing, cumulative-tier and rounding semantics
  as ``POST /quote`` (which this module reuses verbatim).
* Zero usage is treated as a zero charge even though ``/quote`` itself rejects
  non-positive energy: grid point 0 is represented by an empty quote.

The rounded bill is a non-decreasing step function of the grid index (prices
are non-negative and per-step energy is one micro-unit), so two
:class:`~decimal.Decimal` bisections over the index locate the **left**
(``k_lo``) and **right** (``k_hi``) edges of the target amount's plateau.
Integer indices keep every midpoint exactly on the grid; no floating point is
involved. Two distinct failure modes are reported:

* ``above_reachable_max`` — even the ``max_energy`` grid point bills below
  the target; its quote is returned.
* ``rounding_gap`` — a grid step skips over the target during cent rounding;
  the adjacent grid points (last below / first above) are returned.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP

from .pricing import (
    Q2,
    Q6,
    PricingError,
    ensure_valid,
    get_zone,
    parse_local_datetime,
    quote,
)
from .schemas import QuoteIntervalIn


@dataclass(frozen=True)
class EstimateInterval:
    start: object          # UTC datetime
    end: object            # UTC datetime
    tz: object             # ZoneInfo


@dataclass(frozen=True)
class GridPoint:
    """One evaluated point of the 0.000001 energy grid."""

    index: int
    energy: object         # Decimal, exact Q6 multiple
    result: object | None  # pricing.QuoteResult; None for the zero point
    total_amount: object   # Decimal, ROUND_HALF_UP quantized to Q2

    @property
    def raw_amount(self):
        if self.result is None:
            return Q2 * 0
        return self.result.total_amount

    @property
    def matched_rules(self) -> list[str]:
        return [] if self.result is None else self.result.matched_rules


@dataclass
class EstimateResult:
    feasible: bool
    target: object                       # Decimal (Q2)
    max_energy: object                   # Decimal (Q6 multiple)
    n_max: int
    left: GridPoint | None = None        # min solution / below / None
    right: GridPoint | None = None       # max solution / above / None
    reachable_max: GridPoint | None = None  # top point when above_reachable_max
    reason_code: str | None = None
    reason: str | None = None


def parse_interval(item) -> EstimateInterval:
    """Validate the ``energy``-less interval with the same grammar as /quote."""
    tz = get_zone(item.timezone)
    start = parse_local_datetime(item.start, tz)
    end = parse_local_datetime(item.end, tz)
    if end <= start:
        raise PricingError(
            "用量区间必须满足 end > start（半开区间）",
            code="invalid_interval", status_code=400,
        )
    return EstimateInterval(start=start, end=end, tz=tz)


def estimate(version, req) -> EstimateResult:
    """Bisect the energy grid for the plateau quoting to ``billed_amount``."""
    # Structural rule errors are a hard failure, identical to /quote; one
    # validation pass is enough because the time split is energy-independent.
    ensure_valid(version)
    iv = parse_interval(req.interval)

    max_energy = req.max_energy.quantize(Q6)
    # The schema guarantees 0 < max_energy <= 10**6 at <= 6 dp, so the
    # quantized value is a positive integral number of micro-units.
    n_max = int((max_energy / Q6).to_integral_value())
    target = req.billed_amount.quantize(Q2)

    def _payload(energy):
        return [QuoteIntervalIn(
            start=iv.start.isoformat().replace("+00:00", "Z"),
            end=iv.end.isoformat().replace("+00:00", "Z"),
            timezone=str(iv.tz),
            energy=energy,
        )]

    cache: dict[int, GridPoint] = {}

    def _point(k: int) -> GridPoint:
        cached = cache.get(k)
        if cached is not None:
            return cached
        energy = Q6 * k
        # validate=False: the version was structurally validated above and
        # rule gaps are time-driven, i.e. identical for every grid point.
        result = None if k == 0 else quote(
            version, _payload(energy), validate=False)
        raw = Q2 * 0 if result is None else result.total_amount
        point = GridPoint(
            index=k, energy=energy, result=result,
            total_amount=raw.quantize(Q2, rounding=ROUND_HALF_UP),
        )
        cache[k] = point
        return point

    # Evaluate the reachable ceiling first: a rule gap there is a rule gap
    # everywhere (slices can only disappear as energy shrinks, never appear).
    top = _point(n_max)

    if top.total_amount < target:
        return EstimateResult(
            feasible=False, target=target, max_energy=max_energy,
            n_max=n_max, reachable_max=top,
            reason_code="above_reachable_max",
            reason=f"目标金额 {target} 高于可达上界：用量上界 "
            f"{max_energy}（0.000001 网格第 {n_max} 点）报价仅为 "
            f"{top.total_amount}，再大的用量超出 max_energy，无法命中",
        )

    # First grid index whose bill >= target (the bill is a monotone
    # non-decreasing step function of the index). top >= target guarantees a
    # match exists inside [0, n_max], so the index is always in range.
    lo, hi = 0, n_max
    while lo < hi:
        mid = (lo + hi) // 2
        if _point(mid).total_amount >= target:
            hi = mid
        else:
            lo = mid + 1
    r = lo

    if _point(r).total_amount != target:
        # The first reachable plateau sits strictly above the target: the
        # adjacent grid points r-1 (last below) and r (first above) frame
        # the rounding jump. (r == 0 only when zero bills above a zero
        # target, which cannot happen, so r - 1 is always valid.)
        below = _point(r - 1)
        above = _point(r)
        reason = (
            f"目标金额 {target} 落在两位小数舍入的跳跃间隙中："
            f"用量 {below.energy} 报 {below.total_amount}"
            f"（未舍入 {below.raw_amount.quantize(Q6)}），"
            f"下一网格点 {above.energy} 报 {above.total_amount}"
            f"（未舍入 {above.raw_amount.quantize(Q6)}），"
            "中间没有任何网格点"
        )
        return EstimateResult(
            feasible=False, target=target, max_energy=max_energy,
            n_max=n_max, left=below, right=above,
            reason_code="rounding_gap", reason=reason,
        )

    # Right edge: last index whose bill == target. If the ceiling grid point
    # is still on the plateau, the edge is n_max itself; otherwise bisect for
    # the first index whose bill > target (guaranteed to exist inside range)
    # and subtract one.
    if top.total_amount == target:
        k_hi = n_max
    else:
        lo, hi = r, n_max
        while lo < hi:
            mid = (lo + hi) // 2
            if _point(mid).total_amount <= target:
                lo = mid + 1
            else:
                hi = mid
        k_hi = lo - 1
    return EstimateResult(
        feasible=True, target=target, max_energy=max_energy,
        n_max=n_max, left=_point(r), right=_point(k_hi),
    )
