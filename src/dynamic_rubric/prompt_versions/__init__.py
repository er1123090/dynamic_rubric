"""Named rubric-generation prompt versions.

``initial_*`` preserves the experiment's original replay prompt. ``onlinerubric_*``
implements the prompt contract published in *Online Rubrics Elicitation from
Pairwise Comparisons* (arXiv:2510.07284v2).
"""

from .initial_rubric_prompt import INITIAL_RUBRIC_CRITERION_INSTRUCTIONS
from .onlinerubric_prompt import (
    ONLINERUBRIC_DEDUP_SYSTEM_PROMPT,
    ONLINERUBRIC_EXTRACTOR_SYSTEM_PROMPT,
    build_onlinerubric_dedup_messages,
    build_onlinerubric_extractor_messages,
)

__all__ = [
    "INITIAL_RUBRIC_CRITERION_INSTRUCTIONS",
    "ONLINERUBRIC_DEDUP_SYSTEM_PROMPT",
    "ONLINERUBRIC_EXTRACTOR_SYSTEM_PROMPT",
    "build_onlinerubric_dedup_messages",
    "build_onlinerubric_extractor_messages",
]
