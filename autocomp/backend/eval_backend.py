from typing import List

from autocomp.search.prob import Prob
from autocomp.search.code_repo import CodeCandidate

class EvalBackend:
    def __init__(self):
        pass

    def evaluate_code(
        self,
        prob: Prob,
        code_strs: list[str],
        simulator: str,
        candidates: list[CodeCandidate] | None = None,
    ) -> List[dict]:
        pass

    def get_hw_feedback(self, prob: Prob, code_strs: list[str]) -> list[list[str]]:
        """Return per-implementation hardware feedback strings. Default: no feedback."""
        return [[] for _ in code_strs]

    def get_backend_specific_rules(self) -> list[str]:
        """Return backend-specific rule strings for LLM prompts. Default: no rules."""
        return []
