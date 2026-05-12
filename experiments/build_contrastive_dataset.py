#!/usr/bin/env python3
"""Build contrastive multi-hop math reasoning dataset.

Generates 500 base problems across 2/3/4-hop depths with three variants each:
  clean       — all intermediate results correct (model generates all responses)
  error_at_1  — wrong value injected at hop 0; subsequent hops use wrong context
  error_at_2  — (3-hop+ only) wrong value injected at hop 1 only

Distribution:
  2-hop: 250 base problems  (arithmetic, percentage, rate_distance, fractions, counting)
  3-hop: 150 base problems  (percentage_chain, compound_rate, multi_stage_geometry)
  4-hop: 100 base problems  (multi_stage_allocation, compound_interest, nested_fractions, multi_stage_geometry)

Error types (cycled for balance):
  off_by_one, wrong_operator, wrong_unit, magnitude_error, wrong_percentage_base

Output: data/raw/contrastive_math_dataset.json
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.common.base_schema import BaseSchema

SEED = 42
OUTPUT_PATH = ROOT / "data" / "raw" / "contrastive_math_dataset.json"
ERROR_TYPES = [
    "off_by_one",
    "wrong_operator",
    "wrong_unit",
    "magnitude_error",
    "wrong_percentage_base",
]

# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class HopSpec(BaseSchema):
    hop_index: int
    prompt: str
    correct_response: str
    correct_value: float
    unit: str
    is_injected: bool = False
    injected_response: Optional[str] = None
    injected_value: Optional[float] = None
    error_type: Optional[str] = None
    propagates_prior_error: bool = False


@dataclass
class TraceSpec(BaseSchema):
    id: str
    base_problem_id: str
    category: str
    subcategory: str
    hop_depth: int
    variant: str
    question: str
    hops: list
    correct_final_answer: str
    injected_error_type: Optional[str] = None
    injected_error_hop: Optional[int] = None
    error_description: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────


def _n(x: float) -> str:
    """Format number: integer if whole, 2 decimal places otherwise."""
    return str(int(x)) if x == int(x) else f"{x:.2f}"


def _m(x: float) -> str:
    """Format as money."""
    return f"${int(x)}" if x == int(x) else f"${x:.2f}"


def _pct(p: float) -> str:
    return f"{int(p)}%" if p == int(p) else f"{p:.1f}%"


def _trace_id(base_problem_id: str, variant: str) -> str:
    payload = f"{base_problem_id}:{variant}"
    return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Hop internal representation (not serialized directly)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _Hop:
    """Internal hop data used to build HopSpec variants."""

    hop_index: int
    prompt: str
    correct_response: str
    correct_value: float
    unit: str
    # Fields needed to compute alternative (wrong) responses
    op_type: str  # mul, div, add, sub, pct, convert, compound
    a: float  # first operand
    b: float  # second operand (or pct for pct/compound ops)
    a_label: str  # formatted first operand for response text
    b_label: str  # formatted second operand for response text
    result_label: str  # formatted correct result for response text
    conclusion: str  # sentence using result (e.g. "The total cost is {r}.")
    is_time_hop: bool = False  # enables wrong_unit injection
    has_pct: bool = False  # enables wrong_percentage_base injection
    pct_val: float = 0.0  # percentage value (for pct/compound ops)


def _build_wrong(hop: _Hop, error_type: str) -> tuple[float, str]:
    """Return (wrong_value, injected_response_text) for the given error type.

    Falls back to magnitude_error if the requested type is not applicable.
    """
    a, b = hop.a, hop.b

    if error_type == "off_by_one":
        wrong_val = hop.correct_value + 1.0
        wrong_result_label = _fmt_like(wrong_val, hop.result_label)
        response = (
            f"{hop.a_label} {_op_sym(hop.op_type)} {hop.b_label} = {wrong_result_label}. "
            + hop.conclusion.format(r=wrong_result_label)
        )
        return wrong_val, response

    if error_type == "wrong_operator" and hop.op_type in ("mul", "div", "add", "sub"):
        wrong_val, op_sym = _wrong_operator_val(hop)
        wrong_result_label = _fmt_like(wrong_val, hop.result_label)
        response = (
            f"{hop.a_label} {op_sym} {hop.b_label} = {wrong_result_label}. "
            + hop.conclusion.format(r=wrong_result_label)
        )
        return wrong_val, response

    if error_type == "wrong_unit" and hop.is_time_hop:
        # Error: report hours when answer should be minutes (÷60) or vice versa
        if hop.unit == "minutes":
            wrong_val = hop.correct_value / 60.0
            wrong_result_label = _fmt_like(wrong_val, hop.result_label)
            response = (
                f"{hop.a_label} {_op_sym(hop.op_type)} {hop.b_label} = {wrong_result_label}. "
                + hop.conclusion.format(r=wrong_result_label)
            )
        else:  # hours → report minutes
            wrong_val = hop.correct_value * 60.0
            wrong_result_label = _fmt_like(wrong_val, hop.result_label)
            response = (
                f"{hop.a_label} {_op_sym(hop.op_type)} {hop.b_label} = {wrong_result_label}. "
                + hop.conclusion.format(r=wrong_result_label)
            )
        return wrong_val, response

    if error_type == "wrong_percentage_base" and hop.has_pct:
        # Use complementary percentage (100 - pct) instead of pct
        wrong_pct = 100.0 - hop.pct_val
        if hop.op_type == "pct":
            wrong_val = a * wrong_pct / 100.0
        else:  # compound: a × (1 - pct/100)
            wrong_val = a * (1.0 - wrong_pct / 100.0)
        wrong_result_label = _fmt_like(wrong_val, hop.result_label)
        wrong_b_label = _pct(wrong_pct)
        response = (
            f"{hop.a_label} × {wrong_b_label} = {wrong_result_label}. "
            + hop.conclusion.format(r=wrong_result_label)
        )
        return wrong_val, response

    # Fallback: magnitude_error (×10)
    wrong_val = hop.correct_value * 10.0
    wrong_result_label = _fmt_like(wrong_val, hop.result_label)
    response = (
        f"{hop.a_label} {_op_sym(hop.op_type)} {hop.b_label} = {wrong_result_label}. "
        + hop.conclusion.format(r=wrong_result_label)
    )
    return wrong_val, response


def _op_sym(op_type: str) -> str:
    return {"mul": "×", "div": "÷", "add": "+", "sub": "−", "pct": "×", "compound": "×", "convert": "×"}.get(
        op_type, "×"
    )


def _wrong_operator_val(hop: _Hop) -> tuple[float, str]:
    """Return (wrong_value, wrong_operator_symbol) by swapping operator."""
    a, b = hop.a, hop.b
    if hop.op_type == "mul":
        return a + b, "+"
    if hop.op_type == "div":
        return a - b if a > b else a + b, ("−" if a > b else "+")
    if hop.op_type == "add":
        return abs(a - b), "−"
    if hop.op_type == "sub":
        return a + b, "+"
    return hop.correct_value * 10.0, "×"


def _fmt_like(val: float, reference: str) -> str:
    """Format val using the same style as reference (money vs plain number)."""
    if reference.startswith("$"):
        return _m(val)
    return _n(val)


# ─────────────────────────────────────────────────────────────────────────────
# Variant builders
# ─────────────────────────────────────────────────────────────────────────────


def _build_clean_hops(raw_hops: list[_Hop]) -> list[HopSpec]:
    return [
        HopSpec(
            hop_index=h.hop_index,
            prompt=h.prompt,
            correct_response=h.correct_response,
            correct_value=h.correct_value,
            unit=h.unit,
        )
        for h in raw_hops
    ]


def _build_error_hops(raw_hops: list[_Hop], inject_at: int, error_type: str) -> list[HopSpec]:
    """Build hops list with an error injected at `inject_at`."""
    result = []
    wrong_val, injected_resp = _build_wrong(raw_hops[inject_at], error_type)

    for h in raw_hops:
        if h.hop_index == inject_at:
            result.append(
                HopSpec(
                    hop_index=h.hop_index,
                    prompt=h.prompt,
                    correct_response=h.correct_response,
                    correct_value=h.correct_value,
                    unit=h.unit,
                    is_injected=True,
                    injected_response=injected_resp,
                    injected_value=wrong_val,
                    error_type=error_type,
                )
            )
        else:
            result.append(
                HopSpec(
                    hop_index=h.hop_index,
                    prompt=h.prompt,
                    correct_response=h.correct_response,
                    correct_value=h.correct_value,
                    unit=h.unit,
                    propagates_prior_error=(h.hop_index > inject_at),
                )
            )
    return result


def _make_traces(
    base_id: str,
    category: str,
    subcategory: str,
    question: str,
    raw_hops: list[_Hop],
    final_answer: str,
    error_type: str,
) -> list[TraceSpec]:
    hop_depth = len(raw_hops)
    traces = []

    # clean
    traces.append(
        TraceSpec(
            id=_trace_id(base_id, "clean"),
            base_problem_id=base_id,
            category=category,
            subcategory=subcategory,
            hop_depth=hop_depth,
            variant="clean",
            question=question,
            hops=_build_clean_hops(raw_hops),
            correct_final_answer=final_answer,
        )
    )

    # error_at_1 (inject at hop 0)
    traces.append(
        TraceSpec(
            id=_trace_id(base_id, "error_at_1"),
            base_problem_id=base_id,
            category=category,
            subcategory=subcategory,
            hop_depth=hop_depth,
            variant="error_at_1",
            question=question,
            hops=_build_error_hops(raw_hops, inject_at=0, error_type=error_type),
            correct_final_answer=final_answer,
            injected_error_type=error_type,
            injected_error_hop=0,
            error_description=f"{error_type} injected at hop 0",
        )
    )

    # error_at_2 (inject at hop 1) — only for 3-hop and 4-hop
    if hop_depth >= 3:
        traces.append(
            TraceSpec(
                id=_trace_id(base_id, "error_at_2"),
                base_problem_id=base_id,
                category=category,
                subcategory=subcategory,
                hop_depth=hop_depth,
                variant="error_at_2",
                question=question,
                hops=_build_error_hops(raw_hops, inject_at=1, error_type=error_type),
                correct_final_answer=final_answer,
                injected_error_type=error_type,
                injected_error_hop=1,
                error_description=f"{error_type} injected at hop 1",
            )
        )

    return traces


# ─────────────────────────────────────────────────────────────────────────────
# Name/item pools
# ─────────────────────────────────────────────────────────────────────────────

NAMES = [
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Karen", "Leo", "Mia", "Noah", "Olivia", "Paul",
    "Quinn", "Rachel", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xander",
    "Yara", "Zoe",
]
ITEMS = ["apple", "orange", "mango", "peach", "plum", "book", "pen", "notebook", "marker", "eraser"]
STORES = ["market", "supermarket", "grocery store", "shop", "store"]
VEHICLES = ["car", "bus", "train", "bicycle", "motorcycle", "truck", "van", "tram"]
FOODS = ["pizza", "pie", "cake", "tart", "quiche", "flatbread", "loaf", "biscuit"]
PRODUCTS = ["jacket", "shirt", "dress", "coat", "pair of shoes", "bag", "hat", "scarf"]
PLACES = ["library", "warehouse", "bookstore", "school", "office", "depot", "storage room", "workshop"]
ROOMS = ["living room", "bedroom", "kitchen", "hallway", "study", "dining room", "office", "studio"]
FACTORIES = ["factory", "plant", "workshop", "facility", "production line", "facility"]


# ─────────────────────────────────────────────────────────────────────────────
# 2-hop generators
# ─────────────────────────────────────────────────────────────────────────────


def _gen_2hop_arithmetic(rng: random.Random, idx: int) -> tuple[str, list[_Hop], str]:
    """Shopping/change: n items × price → bill − cost = change."""
    name = NAMES[idx % len(NAMES)]
    pronoun = "she" if idx % 2 == 0 else "he"
    item = ITEMS[idx % len(ITEMS)]
    store = STORES[idx % len(STORES)]
    price = rng.choice([2, 3, 4, 5, 6, 8, 10, 12, 15])
    n = rng.randint(2, 10)
    cost = n * price
    bill = cost + rng.choice([5, 10, 15, 20, 25])

    question = (
        f"A {store} sells {item}s for {_m(price)} each. "
        f"{name} buys {n} {item}s and pays with a {_m(bill)} bill. "
        f"How much change does {pronoun} receive?"
    )
    hop0_prompt = (
        f"Question: {question}\n\n"
        f"Step 1: How much does {name} pay in total for {n} {item}s at {_m(price)} each?"
    )
    hop0_result = float(cost)
    hop0_result_label = _m(hop0_result)
    hop0_resp = f"{n} × {_m(price)} = {hop0_result_label}. The total cost is {hop0_result_label}."
    hop0 = _Hop(
        hop_index=0, prompt=hop0_prompt,
        correct_response=hop0_resp, correct_value=hop0_result, unit="dollars",
        op_type="mul", a=float(n), b=float(price),
        a_label=str(n), b_label=_m(price), result_label=hop0_result_label,
        conclusion="The total cost is {r}.",
    )

    change = float(bill - cost)
    hop1_prompt = f"Step 2: How much change does {pronoun} receive after paying with a {_m(bill)} bill?"
    hop1_result_label = _m(change)
    hop1_resp = f"{_m(bill)} − {hop0_result_label} = {hop1_result_label}. {name} receives {hop1_result_label} in change."
    hop1 = _Hop(
        hop_index=1, prompt=hop1_prompt,
        correct_response=hop1_resp, correct_value=change, unit="dollars",
        op_type="sub", a=float(bill), b=hop0_result,
        a_label=_m(bill), b_label=hop0_result_label, result_label=hop1_result_label,
        conclusion=f"{name} receives {{r}} in change.",
    )

    return question, [hop0, hop1], hop1_result_label


def _gen_2hop_percentage(rng: random.Random, idx: int) -> tuple[str, list[_Hop], str]:
    """Earnings × hours → total × pct% = savings."""
    name = NAMES[idx % len(NAMES)]
    pronoun = "she" if idx % 2 == 0 else "he"
    pron_poss = "her" if idx % 2 == 0 else "his"
    rate = rng.choice([10, 12, 15, 18, 20, 25])
    hours = rng.choice([4, 6, 7, 8, 9, 10])
    pct = rng.choice([10, 15, 20, 25, 30, 40])
    earnings = float(rate * hours)
    savings = earnings * pct / 100.0

    question = (
        f"{name} earns {_m(rate)} per hour and works {hours} hours today. "
        f"If {pronoun} saves {_pct(pct)} of {pron_poss} earnings, how much does {pronoun} save?"
    )
    hop0_prompt = (
        f"Question: {question}\n\n"
        f"Step 1: How much does {name} earn in total today?"
    )
    hop0_label = _m(earnings)
    hop0_resp = f"{_m(rate)} × {hours} = {hop0_label}. {name} earns {hop0_label} today."
    hop0 = _Hop(
        hop_index=0, prompt=hop0_prompt,
        correct_response=hop0_resp, correct_value=earnings, unit="dollars",
        op_type="mul", a=float(rate), b=float(hours),
        a_label=_m(rate), b_label=str(hours), result_label=hop0_label,
        conclusion=f"{name} earns {{r}} today.",
    )

    hop1_prompt = f"Step 2: How much does {name} save if {pronoun} saves {_pct(pct)} of {pron_poss} total earnings?"
    savings_label = _m(savings)
    hop1_resp = f"{hop0_label} × {_pct(pct)} = {savings_label}. {name} saves {savings_label}."
    hop1 = _Hop(
        hop_index=1, prompt=hop1_prompt,
        correct_response=hop1_resp, correct_value=savings, unit="dollars",
        op_type="pct", a=earnings, b=pct,
        a_label=hop0_label, b_label=_pct(pct), result_label=savings_label,
        conclusion=f"{name} saves {{r}}.",
        has_pct=True, pct_val=pct,
    )

    return question, [hop0, hop1], savings_label


def _gen_2hop_rate_distance(rng: random.Random, idx: int) -> tuple[str, list[_Hop], str]:
    """distance ÷ speed → hours → × 60 = minutes."""
    vehicle = VEHICLES[idx % len(VEHICLES)]
    speed = rng.choice([40, 50, 60, 80, 100, 120])
    hours = rng.choice([1, 2, 3, 4])
    distance = float(speed * hours)
    minutes = float(hours * 60)

    question = (
        f"A {vehicle} needs to travel {_n(distance)} km. "
        f"It moves at {speed} km/h without stopping. "
        f"How many minutes does the trip take?"
    )
    hop0_prompt = (
        f"Question: {question}\n\n"
        f"Step 1: How many hours does the trip take?"
    )
    hop0_val = float(hours)
    hop0_label = _n(hop0_val) + " hour" + ("s" if hours != 1 else "")
    hop0_resp = f"{_n(distance)} ÷ {speed} = {_n(hop0_val)}. The trip takes {hop0_label}."
    hop0 = _Hop(
        hop_index=0, prompt=hop0_prompt,
        correct_response=hop0_resp, correct_value=hop0_val, unit="hours",
        op_type="div", a=distance, b=float(speed),
        a_label=_n(distance), b_label=str(speed), result_label=_n(hop0_val),
        conclusion=f"The trip takes {{r}} hour{'s' if hours != 1 else ''}.",
        is_time_hop=True,
    )

    hop1_prompt = "Step 2: Convert the travel time from hours to minutes."
    minutes_label = _n(minutes) + " minutes"
    hop1_resp = f"{_n(hop0_val)} × 60 = {_n(minutes)}. The trip takes {_n(minutes)} minutes."
    hop1 = _Hop(
        hop_index=1, prompt=hop1_prompt,
        correct_response=hop1_resp, correct_value=minutes, unit="minutes",
        op_type="convert", a=hop0_val, b=60.0,
        a_label=_n(hop0_val), b_label="60", result_label=_n(minutes),
        conclusion="The trip takes {r} minutes.",
        is_time_hop=True,
    )

    return question, [hop0, hop1], minutes_label


def _gen_2hop_fractions(rng: random.Random, idx: int) -> tuple[str, list[_Hop], str]:
    """n people × k slices eaten → (total − eaten) / total = fraction remaining."""
    food = FOODS[idx % len(FOODS)]
    total_slices = rng.choice([8, 10, 12, 16, 20])
    n_people = rng.randint(2, 4)
    k = rng.randint(1, 3)
    eaten = n_people * k
    # Ensure some is left
    while eaten >= total_slices:
        k -= 1
        eaten = n_people * k
    remaining = total_slices - eaten
    # Simplify fraction
    from math import gcd
    g = gcd(remaining, total_slices)
    num, den = remaining // g, total_slices // g
    frac_str = f"{num}/{den}" if num != den else "1"

    question = (
        f"A {food} is cut into {total_slices} equal slices. "
        f"{n_people} people each eat {k} slice{'s' if k > 1 else ''}. "
        f"What fraction of the {food} remains?"
    )
    hop0_prompt = (
        f"Question: {question}\n\n"
        f"Step 1: How many slices are eaten in total?"
    )
    eaten_f = float(eaten)
    hop0_label = _n(eaten_f)
    hop0_resp = f"{n_people} × {k} = {hop0_label}. A total of {hop0_label} slices are eaten."
    hop0 = _Hop(
        hop_index=0, prompt=hop0_prompt,
        correct_response=hop0_resp, correct_value=eaten_f, unit="slices",
        op_type="mul", a=float(n_people), b=float(k),
        a_label=str(n_people), b_label=str(k), result_label=hop0_label,
        conclusion="A total of {r} slices are eaten.",
    )

    hop1_prompt = f"Step 2: What fraction of the {food}'s {total_slices} slices remains?"
    remaining_f = float(remaining)
    hop1_resp = (
        f"{total_slices} − {hop0_label} = {_n(remaining_f)} slices remaining. "
        f"The fraction remaining is {_n(remaining_f)}/{total_slices} = {frac_str}."
    )
    hop1 = _Hop(
        hop_index=1, prompt=hop1_prompt,
        correct_response=hop1_resp, correct_value=remaining_f, unit="slices",
        op_type="sub", a=float(total_slices), b=eaten_f,
        a_label=str(total_slices), b_label=hop0_label, result_label=_n(remaining_f),
        conclusion=f"{{r}} slices remain out of {total_slices}, giving a fraction of {frac_str}.",
    )

    return question, [hop0, hop1], frac_str


def _gen_2hop_counting(rng: random.Random, idx: int) -> tuple[str, list[_Hop], str]:
    """n shelves × items_per_shelf → total − removed = remaining."""
    place = PLACES[idx % len(PLACES)]
    shelves = rng.randint(2, 8)
    per_shelf = rng.choice([10, 12, 15, 20, 25])
    total = shelves * per_shelf
    removed = rng.randint(5, total // 2)

    question = (
        f"A {place} has {shelves} shelves, each holding {per_shelf} items. "
        f"After {removed} items are taken out, how many items remain?"
    )
    hop0_prompt = (
        f"Question: {question}\n\n"
        f"Step 1: How many items are in the {place} in total?"
    )
    total_f = float(total)
    hop0_label = _n(total_f)
    hop0_resp = f"{shelves} × {per_shelf} = {hop0_label}. The {place} holds {hop0_label} items in total."
    hop0 = _Hop(
        hop_index=0, prompt=hop0_prompt,
        correct_response=hop0_resp, correct_value=total_f, unit="items",
        op_type="mul", a=float(shelves), b=float(per_shelf),
        a_label=str(shelves), b_label=str(per_shelf), result_label=hop0_label,
        conclusion=f"The {place} holds {{r}} items in total.",
    )

    remaining = float(total - removed)
    hop1_prompt = f"Step 2: How many items remain after {removed} are taken out?"
    rem_label = _n(remaining)
    hop1_resp = f"{hop0_label} − {removed} = {rem_label}. {rem_label} items remain."
    hop1 = _Hop(
        hop_index=1, prompt=hop1_prompt,
        correct_response=hop1_resp, correct_value=remaining, unit="items",
        op_type="sub", a=total_f, b=float(removed),
        a_label=hop0_label, b_label=str(removed), result_label=rem_label,
        conclusion="{r} items remain.",
    )

    return question, [hop0, hop1], rem_label


# ─────────────────────────────────────────────────────────────────────────────
# 3-hop generators
# ─────────────────────────────────────────────────────────────────────────────


def _gen_3hop_percentage_chain(rng: random.Random, idx: int) -> tuple[str, list[_Hop], str]:
    """original_price → discount → add tax → final price."""
    product = PRODUCTS[idx % len(PRODUCTS)]
    original = float(rng.choice([50, 60, 80, 100, 120, 150, 200]))
    disc_pct = float(rng.choice([10, 15, 20, 25, 30]))
    tax_pct = float(rng.choice([5, 8, 10, 12, 15]))

    discounted = original * (1.0 - disc_pct / 100.0)
    tax_amount = discounted * tax_pct / 100.0
    final = discounted + tax_amount

    question = (
        f"A {product} originally costs {_m(original)}. "
        f"It is discounted by {_pct(disc_pct)}, then a {_pct(tax_pct)} sales tax is applied. "
        f"What is the final price?"
    )
    hop0_prompt = (
        f"Question: {question}\n\n"
        f"Step 1: What is the price of the {product} after the {_pct(disc_pct)} discount?"
    )
    disc_label = _m(discounted)
    hop0_resp = (
        f"{_m(original)} × (1 − {_pct(disc_pct)}) = {_m(original)} × {_n(1 - disc_pct/100)} = {disc_label}. "
        f"The discounted price is {disc_label}."
    )
    hop0 = _Hop(
        hop_index=0, prompt=hop0_prompt,
        correct_response=hop0_resp, correct_value=discounted, unit="dollars",
        op_type="compound", a=original, b=disc_pct,
        a_label=_m(original), b_label=_pct(disc_pct), result_label=disc_label,
        conclusion="The discounted price is {r}.",
        has_pct=True, pct_val=disc_pct,
    )

    hop1_prompt = f"Step 2: How much is the {_pct(tax_pct)} sales tax on the discounted price?"
    tax_label = _m(tax_amount)
    hop1_resp = f"{disc_label} × {_pct(tax_pct)} = {tax_label}. The tax is {tax_label}."
    hop1 = _Hop(
        hop_index=1, prompt=hop1_prompt,
        correct_response=hop1_resp, correct_value=tax_amount, unit="dollars",
        op_type="pct", a=discounted, b=tax_pct,
        a_label=disc_label, b_label=_pct(tax_pct), result_label=tax_label,
        conclusion="The tax is {r}.",
        has_pct=True, pct_val=tax_pct,
    )

    hop2_prompt = "Step 3: What is the final price after adding the tax?"
    final_label = _m(final)
    hop2_resp = f"{disc_label} + {tax_label} = {final_label}. The final price is {final_label}."
    hop2 = _Hop(
        hop_index=2, prompt=hop2_prompt,
        correct_response=hop2_resp, correct_value=final, unit="dollars",
        op_type="add", a=discounted, b=tax_amount,
        a_label=disc_label, b_label=tax_label, result_label=final_label,
        conclusion="The final price is {r}.",
    )

    return question, [hop0, hop1, hop2], final_label


def _gen_3hop_compound_rate(rng: random.Random, idx: int) -> tuple[str, list[_Hop], str]:
    """leg1 time + rest time + leg2 time = total minutes."""
    vehicle = VEHICLES[idx % len(VEHICLES)]
    speed1 = rng.choice([40, 50, 60, 80])
    dist1 = float(speed1 * rng.choice([1, 2]))
    time1_hrs = dist1 / speed1  # exact integer hours
    time1_min = time1_hrs * 60.0

    rest = float(rng.choice([10, 15, 20, 30]))

    speed2 = rng.choice([30, 40, 50, 60])
    dist2 = float(speed2 * rng.choice([1, 2]))
    time2_hrs = dist2 / speed2
    time2_min = time2_hrs * 60.0

    total_min = time1_min + rest + time2_min

    question = (
        f"A {vehicle} travels {_n(dist1)} km at {speed1} km/h, "
        f"takes a {_n(rest)}-minute break, "
        f"then travels {_n(dist2)} km at {speed2} km/h. "
        f"How many minutes does the entire trip take?"
    )
    hop0_prompt = (
        f"Question: {question}\n\n"
        f"Step 1: How many minutes does the first leg ({_n(dist1)} km at {speed1} km/h) take?"
    )
    t1_label = _n(time1_min)
    hop0_resp = (
        f"{_n(dist1)} ÷ {speed1} = {_n(time1_hrs)} hour{'s' if time1_hrs != 1 else ''} "
        f"= {t1_label} minutes. The first leg takes {t1_label} minutes."
    )
    hop0 = _Hop(
        hop_index=0, prompt=hop0_prompt,
        correct_response=hop0_resp, correct_value=time1_min, unit="minutes",
        op_type="mul", a=time1_hrs, b=60.0,
        a_label=_n(time1_hrs), b_label="60", result_label=t1_label,
        conclusion=f"The first leg takes {{r}} minutes.",
        is_time_hop=True,
    )

    hop1_prompt = f"Step 2: How many minutes does the second leg ({_n(dist2)} km at {speed2} km/h) take?"
    t2_label = _n(time2_min)
    hop1_resp = (
        f"{_n(dist2)} ÷ {speed2} = {_n(time2_hrs)} hour{'s' if time2_hrs != 1 else ''} "
        f"= {t2_label} minutes. The second leg takes {t2_label} minutes."
    )
    hop1 = _Hop(
        hop_index=1, prompt=hop1_prompt,
        correct_response=hop1_resp, correct_value=time2_min, unit="minutes",
        op_type="mul", a=time2_hrs, b=60.0,
        a_label=_n(time2_hrs), b_label="60", result_label=t2_label,
        conclusion=f"The second leg takes {{r}} minutes.",
        is_time_hop=True,
    )

    hop2_prompt = f"Step 3: What is the total trip time including the {_n(rest)}-minute break?"
    total_label = _n(total_min)
    hop2_resp = (
        f"{t1_label} + {_n(rest)} + {t2_label} = {total_label}. "
        f"The total trip time is {total_label} minutes."
    )
    hop2 = _Hop(
        hop_index=2, prompt=hop2_prompt,
        correct_response=hop2_resp, correct_value=total_min, unit="minutes",
        op_type="add", a=time1_min + rest, b=time2_min,
        a_label=f"{t1_label} + {_n(rest)}", b_label=t2_label, result_label=total_label,
        conclusion="The total trip time is {r} minutes.",
    )

    return question, [hop0, hop1, hop2], total_label + " minutes"


def _gen_3hop_geometry(rng: random.Random, idx: int) -> tuple[str, list[_Hop], str]:
    """room area − tile coverage → remaining area → cost."""
    room = ROOMS[idx % len(ROOMS)]
    length = float(rng.choice([4, 5, 6, 7, 8, 10]))
    width = float(rng.choice([3, 4, 5, 6]))
    room_area = length * width

    tile_l = float(rng.choice([1, 2]))
    tile_w = float(rng.choice([1, 2]))
    tile_area = tile_l * tile_w
    n_tiles = rng.randint(2, int(room_area / tile_area) - 1)
    covered = float(n_tiles) * tile_area
    remaining = room_area - covered
    rate = float(rng.choice([15, 20, 25, 30, 40, 50]))
    cost = remaining * rate

    question = (
        f"A {room} measures {_n(length)} m × {_n(width)} m. "
        f"It already has {n_tiles} tiles, each {_n(tile_l)} m × {_n(tile_w)} m. "
        f"Flooring for the rest costs {_m(rate)} per m². What is the total flooring cost?"
    )
    hop0_prompt = (
        f"Question: {question}\n\n"
        f"Step 1: What is the total area of the {room}?"
    )
    room_area_label = _n(room_area)
    hop0_resp = f"{_n(length)} × {_n(width)} = {room_area_label} m². The room area is {room_area_label} m²."
    hop0 = _Hop(
        hop_index=0, prompt=hop0_prompt,
        correct_response=hop0_resp, correct_value=room_area, unit="m²",
        op_type="mul", a=length, b=width,
        a_label=_n(length), b_label=_n(width), result_label=room_area_label,
        conclusion="The room area is {r} m².",
    )

    hop1_prompt = f"Step 2: What area is already covered by the {n_tiles} existing tiles?"
    covered_label = _n(covered)
    each_tile_label = _n(tile_area)
    hop1_resp = (
        f"Each tile covers {each_tile_label} m² ({_n(tile_l)} × {_n(tile_w)}). "
        f"{n_tiles} × {each_tile_label} = {covered_label} m². "
        f"The tiles cover {covered_label} m²."
    )
    hop1 = _Hop(
        hop_index=1, prompt=hop1_prompt,
        correct_response=hop1_resp, correct_value=covered, unit="m²",
        op_type="mul", a=float(n_tiles), b=tile_area,
        a_label=str(n_tiles), b_label=each_tile_label, result_label=covered_label,
        conclusion="The tiles cover {r} m².",
    )

    hop2_prompt = f"Step 3: What is the cost to floor the remaining area at {_m(rate)} per m²?"
    remaining_label = _n(remaining)
    cost_label = _m(cost)
    hop2_resp = (
        f"{room_area_label} − {covered_label} = {remaining_label} m² uncovered. "
        f"{remaining_label} × {_m(rate)} = {cost_label}. The flooring cost is {cost_label}."
    )
    hop2 = _Hop(
        hop_index=2, prompt=hop2_prompt,
        correct_response=hop2_resp, correct_value=cost, unit="dollars",
        op_type="mul", a=remaining, b=rate,
        a_label=remaining_label, b_label=_m(rate), result_label=cost_label,
        conclusion="The flooring cost is {r}.",
    )

    return question, [hop0, hop1, hop2], cost_label


# ─────────────────────────────────────────────────────────────────────────────
# 4-hop generators
# ─────────────────────────────────────────────────────────────────────────────


def _gen_4hop_allocation(rng: random.Random, idx: int) -> tuple[str, list[_Hop], str]:
    """budget → spend pct1 → spend pct2 of rest → spend pct3 of that → remainder."""
    orgs = ["company", "school", "hospital", "charity", "government agency"]
    org = orgs[idx % len(orgs)]
    budget = float(rng.choice([1000, 2000, 5000, 10000]))
    p1 = float(rng.choice([20, 25, 30]))
    p2 = float(rng.choice([20, 25, 30]))
    p3 = float(rng.choice([10, 15, 20]))

    after1 = budget * (1 - p1 / 100)
    after2 = after1 * (1 - p2 / 100)
    after3 = after2 * (1 - p3 / 100)

    question = (
        f"A {org} has a budget of {_m(budget)}. "
        f"It spends {_pct(p1)} on salaries, then {_pct(p2)} of the remainder on equipment, "
        f"then {_pct(p3)} of what is left on training. How much budget remains?"
    )
    hop0_prompt = (
        f"Question: {question}\n\n"
        f"Step 1: How much remains after spending {_pct(p1)} on salaries?"
    )
    a1_label = _m(after1)
    hop0_resp = (
        f"{_m(budget)} × (1 − {_pct(p1)}) = {a1_label}. "
        f"After salaries, {a1_label} remains."
    )
    hop0 = _Hop(
        hop_index=0, prompt=hop0_prompt,
        correct_response=hop0_resp, correct_value=after1, unit="dollars",
        op_type="compound", a=budget, b=p1,
        a_label=_m(budget), b_label=_pct(p1), result_label=a1_label,
        conclusion="After salaries, {r} remains.",
        has_pct=True, pct_val=p1,
    )

    hop1_prompt = f"Step 2: How much remains after spending {_pct(p2)} of the remainder on equipment?"
    a2_label = _m(after2)
    hop1_resp = (
        f"{a1_label} × (1 − {_pct(p2)}) = {a2_label}. "
        f"After equipment, {a2_label} remains."
    )
    hop1 = _Hop(
        hop_index=1, prompt=hop1_prompt,
        correct_response=hop1_resp, correct_value=after2, unit="dollars",
        op_type="compound", a=after1, b=p2,
        a_label=a1_label, b_label=_pct(p2), result_label=a2_label,
        conclusion="After equipment, {r} remains.",
        has_pct=True, pct_val=p2,
    )

    hop2_prompt = f"Step 3: How much remains after spending {_pct(p3)} on training?"
    a3_label = _m(after3)
    hop2_resp = (
        f"{a2_label} × (1 − {_pct(p3)}) = {a3_label}. "
        f"After training, {a3_label} remains."
    )
    hop2 = _Hop(
        hop_index=2, prompt=hop2_prompt,
        correct_response=hop2_resp, correct_value=after3, unit="dollars",
        op_type="compound", a=after2, b=p3,
        a_label=a2_label, b_label=_pct(p3), result_label=a3_label,
        conclusion="After training, {r} remains.",
        has_pct=True, pct_val=p3,
    )

    pct_of_budget = after3 / budget * 100.0
    hop3_prompt = "Step 4: What percentage of the original budget remains?"
    pct_label = _n(round(pct_of_budget, 2)) + "%"
    hop3_resp = (
        f"{a3_label} ÷ {_m(budget)} × 100 = {pct_label}. "
        f"{pct_label} of the original budget remains."
    )
    hop3 = _Hop(
        hop_index=3, prompt=hop3_prompt,
        correct_response=hop3_resp, correct_value=pct_of_budget, unit="percent",
        op_type="pct", a=after3, b=100.0 / budget,
        a_label=a3_label, b_label=_m(budget), result_label=pct_label,
        conclusion="{r} of the original budget remains.",
        has_pct=True, pct_val=pct_of_budget,
    )

    return question, [hop0, hop1, hop2, hop3], a3_label


def _gen_4hop_compound_interest(rng: random.Random, idx: int) -> tuple[str, list[_Hop], str]:
    """principal → year1 balance → year2 balance → year3 balance → total interest."""
    principal = float(rng.choice([500, 1000, 2000, 5000]))
    rate = float(rng.choice([5, 8, 10, 12]))

    b1 = principal * (1 + rate / 100)
    b2 = b1 * (1 + rate / 100)
    b3 = b2 * (1 + rate / 100)
    total_interest = b3 - principal

    question = (
        f"{_m(principal)} is invested at {_pct(rate)} annual interest, compounded annually. "
        f"What is the total interest earned after 3 years?"
    )
    hop0_prompt = (
        f"Question: {question}\n\n"
        f"Step 1: What is the balance after year 1?"
    )
    b1_label = _m(b1)
    hop0_resp = f"{_m(principal)} × (1 + {_pct(rate)}) = {b1_label}. Balance after year 1: {b1_label}."
    hop0 = _Hop(
        hop_index=0, prompt=hop0_prompt,
        correct_response=hop0_resp, correct_value=b1, unit="dollars",
        op_type="compound", a=principal, b=rate,
        a_label=_m(principal), b_label=_pct(rate), result_label=b1_label,
        conclusion="Balance after year 1: {r}.",
        has_pct=True, pct_val=rate,
    )

    hop1_prompt = "Step 2: What is the balance after year 2?"
    b2_label = _m(b2)
    hop1_resp = f"{b1_label} × (1 + {_pct(rate)}) = {b2_label}. Balance after year 2: {b2_label}."
    hop1 = _Hop(
        hop_index=1, prompt=hop1_prompt,
        correct_response=hop1_resp, correct_value=b2, unit="dollars",
        op_type="compound", a=b1, b=rate,
        a_label=b1_label, b_label=_pct(rate), result_label=b2_label,
        conclusion="Balance after year 2: {r}.",
        has_pct=True, pct_val=rate,
    )

    hop2_prompt = "Step 3: What is the balance after year 3?"
    b3_label = _m(b3)
    hop2_resp = f"{b2_label} × (1 + {_pct(rate)}) = {b3_label}. Balance after year 3: {b3_label}."
    hop2 = _Hop(
        hop_index=2, prompt=hop2_prompt,
        correct_response=hop2_resp, correct_value=b3, unit="dollars",
        op_type="compound", a=b2, b=rate,
        a_label=b2_label, b_label=_pct(rate), result_label=b3_label,
        conclusion="Balance after year 3: {r}.",
        has_pct=True, pct_val=rate,
    )

    hop3_prompt = "Step 4: What is the total interest earned over the 3 years?"
    interest_label = _m(total_interest)
    hop3_resp = (
        f"{b3_label} − {_m(principal)} = {interest_label}. "
        f"The total interest earned is {interest_label}."
    )
    hop3 = _Hop(
        hop_index=3, prompt=hop3_prompt,
        correct_response=hop3_resp, correct_value=total_interest, unit="dollars",
        op_type="sub", a=b3, b=principal,
        a_label=b3_label, b_label=_m(principal), result_label=interest_label,
        conclusion="The total interest earned is {r}.",
    )

    return question, [hop0, hop1, hop2, hop3], interest_label


def _gen_4hop_nested_fractions(rng: random.Random, idx: int) -> tuple[str, list[_Hop], str]:
    """total widgets → pass QC → premium → exported → air-shipped."""
    factory = FACTORIES[idx % len(FACTORIES)]
    total = float(rng.choice([1000, 2000, 3000, 5000]))
    f1_n, f1_d = rng.choice([(3, 4), (4, 5), (2, 3)])
    f2_n, f2_d = rng.choice([(1, 2), (2, 3), (3, 5)])
    f3_n, f3_d = rng.choice([(1, 2), (3, 4), (2, 5)])
    pct = float(rng.choice([20, 25, 30, 40, 50]))

    passed = total * f1_n / f1_d
    premium = passed * f2_n / f2_d
    exported = premium * f3_n / f3_d
    by_air = exported * pct / 100.0

    question = (
        f"A {factory} produces {_n(total)} widgets per day. "
        f"{f1_n}/{f1_d} pass quality control. "
        f"Of those, {f2_n}/{f2_d} are premium quality. "
        f"Of the premium widgets, {f3_n}/{f3_d} are exported. "
        f"Of the exported widgets, {_pct(pct)} are shipped by air. "
        f"How many widgets are shipped by air?"
    )
    hop0_prompt = (
        f"Question: {question}\n\n"
        f"Step 1: How many widgets pass quality control?"
    )
    passed_label = _n(passed)
    hop0_resp = (
        f"{_n(total)} × {f1_n}/{f1_d} = {passed_label}. "
        f"{passed_label} widgets pass quality control."
    )
    hop0 = _Hop(
        hop_index=0, prompt=hop0_prompt,
        correct_response=hop0_resp, correct_value=passed, unit="widgets",
        op_type="mul", a=total, b=f1_n / f1_d,
        a_label=_n(total), b_label=f"{f1_n}/{f1_d}", result_label=passed_label,
        conclusion="{r} widgets pass quality control.",
    )

    hop1_prompt = f"Step 2: How many of the passing widgets are premium quality?"
    premium_label = _n(premium)
    hop1_resp = (
        f"{passed_label} × {f2_n}/{f2_d} = {premium_label}. "
        f"{premium_label} widgets are premium quality."
    )
    hop1 = _Hop(
        hop_index=1, prompt=hop1_prompt,
        correct_response=hop1_resp, correct_value=premium, unit="widgets",
        op_type="mul", a=passed, b=f2_n / f2_d,
        a_label=passed_label, b_label=f"{f2_n}/{f2_d}", result_label=premium_label,
        conclusion="{r} widgets are premium quality.",
    )

    hop2_prompt = f"Step 3: How many premium widgets are exported?"
    exported_label = _n(exported)
    hop2_resp = (
        f"{premium_label} × {f3_n}/{f3_d} = {exported_label}. "
        f"{exported_label} premium widgets are exported."
    )
    hop2 = _Hop(
        hop_index=2, prompt=hop2_prompt,
        correct_response=hop2_resp, correct_value=exported, unit="widgets",
        op_type="mul", a=premium, b=f3_n / f3_d,
        a_label=premium_label, b_label=f"{f3_n}/{f3_d}", result_label=exported_label,
        conclusion="{r} premium widgets are exported.",
    )

    hop3_prompt = f"Step 4: How many exported widgets are shipped by air?"
    air_label = _n(by_air)
    hop3_resp = (
        f"{exported_label} × {_pct(pct)} = {air_label}. "
        f"{air_label} widgets are shipped by air."
    )
    hop3 = _Hop(
        hop_index=3, prompt=hop3_prompt,
        correct_response=hop3_resp, correct_value=by_air, unit="widgets",
        op_type="pct", a=exported, b=pct,
        a_label=exported_label, b_label=_pct(pct), result_label=air_label,
        conclusion="{r} widgets are shipped by air.",
        has_pct=True, pct_val=pct,
    )

    return question, [hop0, hop1, hop2, hop3], air_label + " widgets"


def _gen_4hop_geometry(rng: random.Random, idx: int) -> tuple[str, list[_Hop], str]:
    """Plot area → build footprint → garden area → garden cost."""
    length = float(rng.choice([20, 25, 30, 40, 50]))
    width = float(rng.choice([10, 15, 20, 25]))
    plot_area = length * width

    b_frac_n, b_frac_d = rng.choice([(1, 2), (1, 3), (2, 5), (3, 5)])
    build_area = plot_area * b_frac_n / b_frac_d

    garden_area = plot_area - build_area
    pct_lawn = float(rng.choice([40, 50, 60]))
    lawn_area = garden_area * pct_lawn / 100.0
    rate = float(rng.choice([5, 8, 10, 12]))
    lawn_cost = lawn_area * rate

    question = (
        f"A plot of land is {_n(length)} m × {_n(width)} m. "
        f"{b_frac_n}/{b_frac_d} of the plot is used for a building. "
        f"Of the remaining garden area, {_pct(pct_lawn)} is lawn. "
        f"Lawn maintenance costs {_m(rate)} per m². What is the annual lawn maintenance cost?"
    )
    hop0_prompt = (
        f"Question: {question}\n\n"
        f"Step 1: What is the total area of the plot?"
    )
    plot_label = _n(plot_area)
    hop0_resp = f"{_n(length)} × {_n(width)} = {plot_label} m². The plot is {plot_label} m²."
    hop0 = _Hop(
        hop_index=0, prompt=hop0_prompt,
        correct_response=hop0_resp, correct_value=plot_area, unit="m²",
        op_type="mul", a=length, b=width,
        a_label=_n(length), b_label=_n(width), result_label=plot_label,
        conclusion="The plot is {r} m².",
    )

    hop1_prompt = f"Step 2: What is the garden area (plot minus the building footprint)?"
    build_label = _n(build_area)
    garden_label = _n(garden_area)
    hop1_resp = (
        f"Building footprint: {plot_label} × {b_frac_n}/{b_frac_d} = {build_label} m². "
        f"Garden area: {plot_label} − {build_label} = {garden_label} m²."
    )
    hop1 = _Hop(
        hop_index=1, prompt=hop1_prompt,
        correct_response=hop1_resp, correct_value=garden_area, unit="m²",
        op_type="sub", a=plot_area, b=build_area,
        a_label=plot_label, b_label=build_label, result_label=garden_label,
        conclusion="The garden area is {r} m².",
    )

    hop2_prompt = f"Step 3: What area of the garden is lawn ({_pct(pct_lawn)} of the garden)?"
    lawn_label = _n(lawn_area)
    hop2_resp = f"{garden_label} × {_pct(pct_lawn)} = {lawn_label} m². The lawn covers {lawn_label} m²."
    hop2 = _Hop(
        hop_index=2, prompt=hop2_prompt,
        correct_response=hop2_resp, correct_value=lawn_area, unit="m²",
        op_type="pct", a=garden_area, b=pct_lawn,
        a_label=garden_label, b_label=_pct(pct_lawn), result_label=lawn_label,
        conclusion="The lawn covers {r} m².",
        has_pct=True, pct_val=pct_lawn,
    )

    hop3_prompt = f"Step 4: What is the annual lawn maintenance cost at {_m(rate)} per m²?"
    cost_label = _m(lawn_cost)
    hop3_resp = f"{lawn_label} × {_m(rate)} = {cost_label}. The annual cost is {cost_label}."
    hop3 = _Hop(
        hop_index=3, prompt=hop3_prompt,
        correct_response=hop3_resp, correct_value=lawn_cost, unit="dollars",
        op_type="mul", a=lawn_area, b=rate,
        a_label=lawn_label, b_label=_m(rate), result_label=cost_label,
        conclusion="The annual cost is {r}.",
    )

    return question, [hop0, hop1, hop2, hop3], cost_label


# ─────────────────────────────────────────────────────────────────────────────
# Dataset assembly
# ─────────────────────────────────────────────────────────────────────────────

# (category, subcategory, hop_depth, count, generator_fn)
GENERATORS = [
    ("math", "arithmetic",           2, 50, _gen_2hop_arithmetic),
    ("math", "percentage",           2, 50, _gen_2hop_percentage),
    ("math", "rate_distance",        2, 50, _gen_2hop_rate_distance),
    ("math", "fractions",            2, 50, _gen_2hop_fractions),
    ("math", "counting",             2, 50, _gen_2hop_counting),
    ("math", "percentage_chain",     3, 50, _gen_3hop_percentage_chain),
    ("math", "compound_rate",        3, 50, _gen_3hop_compound_rate),
    ("math", "multi_stage_geometry", 3, 50, _gen_3hop_geometry),
    ("math", "multi_stage_allocation",   4, 25, _gen_4hop_allocation),
    ("math", "compound_interest",        4, 25, _gen_4hop_compound_interest),
    ("math", "nested_fractions",         4, 25, _gen_4hop_nested_fractions),
    ("math", "multi_stage_geometry_4",   4, 25, _gen_4hop_geometry),
]


def build_dataset() -> dict:
    rng = random.Random(SEED)
    all_traces: list[dict] = []
    base_problems = 0
    error_type_counts: dict[str, int] = {e: 0 for e in ERROR_TYPES}
    error_cycle_idx = 0

    for category, subcategory, hop_depth, count, gen_fn in GENERATORS:
        for local_idx in range(count):
            # Cycle through error types for balance
            error_type = ERROR_TYPES[error_cycle_idx % len(ERROR_TYPES)]
            error_cycle_idx += 1
            error_type_counts[error_type] += 1

            base_id = f"{hop_depth}hop_{subcategory}_{local_idx:04d}"
            try:
                question, raw_hops, final_answer = gen_fn(rng, local_idx)
            except Exception as exc:
                # Skip malformed problems (e.g., fraction generator edge case)
                print(f"  [WARN] skipping {base_id}: {exc}")
                continue

            traces = _make_traces(
                base_id=base_id,
                category=category,
                subcategory=subcategory,
                question=question,
                raw_hops=raw_hops,
                final_answer=final_answer,
                error_type=error_type,
            )
            for t in traces:
                all_traces.append(t.to_dict())
            base_problems += 1

    hop_dist: dict[str, int] = {}
    variant_dist: dict[str, int] = {}
    for t in all_traces:
        k = f"{t['hop_depth']}-hop"
        hop_dist[k] = hop_dist.get(k, 0) + 1
        variant_dist[t["variant"]] = variant_dist.get(t["variant"], 0) + 1

    return {
        "metadata": {
            "description": (
                "Contrastive multi-hop math reasoning dataset for agentic self-monitoring. "
                "Each base problem has clean, error_at_1, and (for 3/4-hop) error_at_2 variants. "
                "Errors are injected via context injection — the assistant turn for the relevant hop "
                "contains a deliberately wrong computation; subsequent hops receive model-generated "
                "responses conditioned on the wrong context."
            ),
            "total_base_problems": base_problems,
            "total_traces": len(all_traces),
            "seed": SEED,
            "hop_distribution": hop_dist,
            "variant_distribution": variant_dist,
            "error_type_distribution": error_type_counts,
            "subcategories": {
                "2-hop": ["arithmetic", "percentage", "rate_distance", "fractions", "counting"],
                "3-hop": ["percentage_chain", "compound_rate", "multi_stage_geometry"],
                "4-hop": [
                    "multi_stage_allocation",
                    "compound_interest",
                    "nested_fractions",
                    "multi_stage_geometry_4",
                ],
            },
        },
        "traces": all_traces,
    }


def main() -> None:
    print("Building contrastive math dataset...")
    dataset = build_dataset()
    meta = dataset["metadata"]
    print(f"  Base problems : {meta['total_base_problems']}")
    print(f"  Total traces  : {meta['total_traces']}")
    print(f"  Hop dist      : {meta['hop_distribution']}")
    print(f"  Variant dist  : {meta['variant_distribution']}")
    print(f"  Error types   : {meta['error_type_distribution']}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"  Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
