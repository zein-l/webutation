"""The basis line was a union written in the grammar of a conjunction.

A live report read "held by: face match 0.99, name, context". No record in that
anchor carried all three: the face came from one page with no context agreement
at all, and every record that agreed on context had no face to compare. A
reader was shown the best of each column and left to assume they described one
thing.
"""

from __future__ import annotations

from app.engine.candidates import AnchorMatch, CandidateDraft
from app.models import GroupingBasis
from app.report import _plain_basis


def match(*, name=None, context=None, face=None, locality=None) -> AnchorMatch:
    return AnchorMatch(
        strength=1.0, name_signal=name, context_signal=context,
        face_signal=face, locality_signal=locality, basis="", admitted=True,
    )


def anchor(**matches) -> CandidateDraft:
    return CandidateDraft(
        name_key="michael petrie",
        grouping_basis=GroupingBasis.anchor,
        is_anchor=True,
        anchor_matches=dict(matches),
    )


def test_evidence_spread_across_records_says_so() -> None:
    """The live shape: a face on one record, context on others, never together."""
    basis = _plain_basis(anchor(
        r1=match(name=0.85, face=0.9939),                 # the face record
        r2=match(name=1.0, context=1.0),                  # context records
        r3=match(name=1.0, context=0.5),
    ))
    assert "spread across 3 records" in basis
    assert "no one record has all of it" in basis
    assert "0.99" in basis


def test_one_record_carrying_everything_is_stated_plainly() -> None:
    basis = _plain_basis(anchor(
        r1=match(name=1.0, context=1.0, face=0.99),
        r2=match(name=1.0, context=1.0, face=0.98),
    ))
    assert "all of it on 2 of 2 records" in basis
    assert "spread" not in basis


def test_a_partial_overlap_counts_the_records_that_have_everything() -> None:
    basis = _plain_basis(anchor(
        r1=match(name=1.0, context=1.0, face=0.99),
        r2=match(name=1.0, context=1.0),
        r3=match(name=1.0),
    ))
    assert "all of it on 1 of 3 records" in basis


def test_a_single_record_anchor_does_not_claim_a_spread() -> None:
    basis = _plain_basis(anchor(r1=match(name=0.85, face=0.99)))
    assert basis.startswith("one record")
    assert "spread" not in basis


def test_locality_is_still_named() -> None:
    """A candidate admitted on its locality must not report "name" as its basis."""
    basis = _plain_basis(anchor(
        r1=match(name=1.0, locality=1.0),
        r2=match(name=1.0),
    ))
    assert "locality" in basis


def test_an_anchor_with_no_matches_says_nothing_it_cannot_support() -> None:
    assert _plain_basis(anchor()) == "matched the subject"


def test_the_face_figure_is_the_best_one_seen() -> None:
    basis = _plain_basis(anchor(
        r1=match(name=1.0, face=0.42),
        r2=match(name=1.0, face=0.99),
    ))
    assert "0.99" in basis and "0.42" not in basis
