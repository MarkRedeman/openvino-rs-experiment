"""Parse class labels from OpenVINO IR model XML files.

Matches the Rust ``labels.rs`` parser — searches ``<rt_info>`` for a
``<labels value="..."/>`` element with space-separated label names.
"""

from __future__ import annotations

import re
from pathlib import Path


def parse_labels_from_model_xml(model_xml: Path | str) -> list[str]:
    """Extract class label names from an OpenVINO IR model XML.

    Returns an ordered list (index 0 = class 0) or an empty list if no
    label metadata is found.
    """
    content = Path(model_xml).read_text()

    rt_start = content.find("<rt_info>")
    if rt_start == -1:
        return []

    rt_section = content[rt_start:]

    # Match <labels value="Clubs Spades Diamonds Hearts" />
    match = re.search(r'<labels\s+value="([^"]*)"', rt_section)
    if not match:
        return []

    labels_str = match.group(1).strip()
    if not labels_str:
        return []

    return labels_str.split()
