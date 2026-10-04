"""Prompt used by the initial post-hoc dynamic-rubric implementation.

The explicit ``initial_*`` filename prevents this legacy prompt from being confused
with the paper-faithful OnlineRubrics prompt implementation.
"""

from __future__ import annotations

INITIAL_RUBRIC_CRITERION_INSTRUCTIONS = (
    "Propose at most three single, atomic, positive criteria from these source-blind A/B "
    "pairs. Each criterion must describe exactly one behavior that is judgeable from the "
    "response alone with YES or NO. Criterion text must be 12-500 characters, must not "
    "identify a candidate or source, and must not use conjunctions such as 'and' or 'or'. "
    "Do not copy or summarize a source response. Keep each rationale under 1000 characters. "
    "Pairs: "
)
