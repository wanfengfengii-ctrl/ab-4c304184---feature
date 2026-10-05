"""HTTP API for concatemer decoding.

POST /api/concatemers/decode
    {
      "reference": "8-20 nt circular reference",
      "read": "30-160 nt tandem read",
      "copies": 3-8,                 // consecutive observed fragments
      "max_edits": 0-3,              // per-copy edit budget
      "terminal_mode": "full"|"partial"   // optional, default "full"
    }

With ``terminal_mode="partial"`` the read starts mid-unit and ends mid-unit:
the first fragment aligns to a non-empty proper *suffix* of the common
rotated reference and the last to a non-empty proper *prefix*; middle
fragments still align to whole units.  Successful witnesses carry the two
``terminal_ranges``.

Responses (HTTP 200):
    status == "unique"     -> single optimal explanation
    status == "ambiguous"  -> multiple optima, the first two witnesses
                              (sorted by shift, boundaries, CIGAR, terminal
                              cuts) are shown
Constraint failure:
    HTTP 422 with status == "infeasible" and a locatable ``nearest`` block.
    ``constraint.name`` distinguishes ``segment_length`` (structural),
    ``terminal_range`` (collection truncation is not a legal cyclic cut)
    and ``per_segment_edit_budget`` (noise exceeds the budget).
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from .solver import rotate, solve

app = FastAPI(
    title="Concatemer Decode API",
    version="1.1.0",
    description="Recover a common cut point from noisy tandem barcode reads.",
)

_DNA = set("ACGT")
_TERMINAL_MODES = ("full", "partial")


class DecodeRequest(BaseModel):
    reference: str = Field(..., description="8-20 nt circular reference")
    read: str = Field(..., alias="read", description="30-160 nt tandem read")
    copies: int = Field(..., ge=3, le=8)
    max_edits: int = Field(..., ge=0, le=3)
    terminal_mode: str = Field(
        "full",
        description=(
            "'full' (default): every copy is a whole unit. "
            "'partial': first fragment is a proper suffix and last a proper "
            "prefix of the common rotated reference."
        ),
    )

    model_config = {"populate_by_name": True, "extra": "ignore"}

    @field_validator("reference", "read")
    @classmethod
    def _validate_dna(cls, value: str) -> str:
        seq = value.strip().upper()
        if not seq:
            raise ValueError("sequence must be non-empty")
        bad = sorted({ch for ch in seq if ch not in _DNA})
        if bad:
            raise ValueError(
                f"invalid DNA character(s): {''.join(bad)!r}; only A/C/G/T allowed"
            )
        return seq

    @field_validator("reference")
    @classmethod
    def _check_reference_length(cls, value: str) -> str:
        if not 8 <= len(value) <= 20:
            raise ValueError("reference must be 8 to 20 nt long")
        return value

    @field_validator("read")
    @classmethod
    def _check_read_length(cls, value: str) -> str:
        if not 30 <= len(value) <= 160:
            raise ValueError("read must be 30 to 160 nt long")
        return value

    @field_validator("terminal_mode")
    @classmethod
    def _check_terminal_mode(cls, value: str) -> str:
        mode = value.strip().lower()
        if mode not in _TERMINAL_MODES:
            raise ValueError(
                f"terminal_mode must be one of {_TERMINAL_MODES!r}"
            )
        return mode


class Health(BaseModel):
    status: str
    service: str


@app.get("/health", response_model=Health)
def health() -> Health:
    return Health(status="ok", service="concatemer-decode")


@app.post("/api/concatemers/decode")
def decode(request: DecodeRequest):
    result = solve(
        request.reference,
        request.read,
        request.copies,
        request.max_edits,
        terminal_mode=request.terminal_mode,
    )

    echo = {
        "reference": request.reference,
        "read": request.read,
        "copies": request.copies,
        "max_edits_per_segment": request.max_edits,
        "terminal_mode": request.terminal_mode,
    }
    result["request"] = echo

    if result["status"] == "infeasible":
        return JSONResponse(status_code=422, content=result)

    # Attach the rotated reference used by the optimal explanation(s).
    if result["status"] == "unique":
        witness = result["witness"]
        result["rotated_reference"] = rotate(
            request.reference, witness["shift"]
        )
        result["ordering"] = (
            "objective lexicographically minimizes (total_edits, "
            "max_segment_edits); ties ordered by (shift, boundaries, CIGAR, "
            "terminal cuts)"
        )
    else:
        for witness in result["witnesses"]:
            witness["rotated_reference"] = rotate(
                request.reference, witness["shift"]
            )
        result["ordering"] = (
            "witnesses sorted by (shift, boundaries, CIGAR, terminal cuts); "
            "only the first two are returned"
        )
    return result
