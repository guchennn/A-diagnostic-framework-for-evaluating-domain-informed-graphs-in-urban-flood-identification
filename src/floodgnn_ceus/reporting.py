"""Shared formatting for aggregate analysis reports."""

from __future__ import annotations

import numpy as np
import pandas as pd


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame without its index or an optional tabulate dependency."""

    def render(value: object) -> str:
        if isinstance(value, np.integer):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        return str(value)

    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False)
    ]
    return "\n".join([header, separator, *body])
