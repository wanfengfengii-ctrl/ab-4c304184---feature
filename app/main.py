"""HTTP API for concatemer decoding.

POST /api/concatemers/decode
    {
      "reference": "8-20 nt circular reference",
      "read": "30-160 nt tandem read",
      "copies": 3-8,                  // number of consecutive observed pieces
      "max_edits": 0-3,               // per-piece edit budget
      "terminal_mode": "full"|"partial"   // optional, defaults to "full"
    }

``terminal_mode`` (optional; omitting it keeps legacy behaviour exactly):

* ``full``    - every piece is one complete rotated copy;
* ``partial`` - the first piece is a non-empty proper suffix and the last a
  non-empty proper prefix of the same rotated reference (the acquisition
  window may cut repeat units at both ends); middle pieces are full copies.
  ``copies`` still counts observed pieces, so it must be at least 2.

Responses (HTTP 200):
    status == "unique"     -> single optimal explanation
    status == "ambiguous"  -> multiple optima, the first two witnesses
                              (sorted by shift, boundaries, CIGAR, and the
                              terminal ranges in partial mode) are shown
Constraint failure:
    HTTP 422 with status == "infeasible" and a locatable ``nearest`` block.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from .solver import TERMINAL_FULL, TERMINAL_PARTIAL, rotate, solve

app = FastAPI(
    title="Concatemer Decode API",
    version="1.1.0",
    description="Recover a common cut point from noisy tandem barcode reads.",
)

_DNA = set("ACGT")
_TERMINAL_MODES = {TERMINAL_FULL, TERMINAL_PARTIAL}


class DecodeRequest(BaseModel):
    reference: str = Field(..., description="8-20 nt circular reference")
    read: str = Field(..., alias="read", description="30-160 nt tandem read")
    copies: int = Field(..., ge=2, le=8)
    max_edits: int = Field(..., ge=0, le=3)
    terminal_mode: str = Field(
        default=TERMINAL_FULL,
        description="'full' requires complete copies; 'partial' allows the "
        "first/last observed piece to be a proper suffix/prefix of one "
        "common rotation.",
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
                f"terminal_mode must be one of {sorted(_TERMINAL_MODES)}"
            )
        return mode

    @model_validator(mode="after")
    def _check_legacy_copy_range(self) -> "DecodeRequest":
        # Omitting terminal_mode (legacy callers) must keep rejecting
        # copies < 3 exactly as before; partial mode allows copies >= 2.
        if self.terminal_mode == TERMINAL_FULL and self.copies < 3:
            raise ValueError(
                "copies must be between 3 and 8 (use terminal_mode=partial "
                "for reads of 2 observed pieces with truncated ends)"
            )
        return self


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
        request.terminal_mode,
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
        if request.terminal_mode == TERMINAL_PARTIAL:
            result["ordering"] = (
                "objective lexicographically minimizes (total_edits, "
                "max_segment_edits); ties ordered by (shift, boundaries, "
                "terminal ranges, CIGAR)"
            )
        else:
            result["ordering"] = (
                "objective lexicographically minimizes (total_edits, "
                "max_segment_edits); ties ordered by (shift, boundaries, CIGAR)"
            )
    else:
        for witness in result["witnesses"]:
            witness["rotated_reference"] = rotate(
                request.reference, witness["shift"]
            )
        if request.terminal_mode == TERMINAL_PARTIAL:
            result["ordering"] = (
                "witnesses sorted by (shift, boundaries, terminal ranges, "
                "CIGAR); only the first two are returned"
            )
        else:
            result["ordering"] = (
                "witnesses sorted by (shift, boundaries, CIGAR); "
                "only the first two are returned"
            )
    return result
