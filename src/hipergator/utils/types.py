"""Shared literal types for the HiPerGator pipeline."""

from typing import Literal


RowsKeys = Literal["table", "total_rows", "shown_rows"]
ScopeType = Literal["latest_encounter", "history"]
PathType = Literal["raw", "cleaned", "metadata", "summary"]

SCOPE_VALUES: tuple[ScopeType, ...] = ("latest_encounter", "history")
