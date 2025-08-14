import json
import re
import logging
from typing import Any, Dict, List, Tuple

from .models import LabTestResult, OrganInjuryResult
from .llm.base import LLMProvider

logger = logging.getLogger(__name__)


class ToxicityDataExtractor:
    """Main class for extracting toxicity data using LLMs"""

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        self.relevance_prompt = self._create_relevance_prompt()
        self.extraction_prompt = self._create_extraction_prompt()

    def _create_relevance_prompt(self) -> str:
        return """You are an expert in analyzing biomedical literature for animal toxicity studies.
Your task is to identify whether the given content contains animal toxicity data.

Look for:
1. Lab test results (AST, ALT, ALP, bilirubin, creatinine, BUN, etc.) with numerical values
2. Organ injury or toxicity observations (liver injury, kidney damage, etc.) with frequencies
3. Side effects or adverse events in animal studies

Important formatting rules:
- Return ONLY a valid JSON object. No code fences, no backticks, no extra text.
- Use double quotes around all keys and string values.
- Do not add trailing commas.

Respond with a JSON object:
{
    "is_relevant": true/false,
    "relevance_score": 0-100,
    "data_types": ["lab_tests", "organ_injury"],
    "reason": "brief explanation"
}"""

    def _create_extraction_prompt(self) -> str:
        return """You are an expert in extracting structured toxicity data from biomedical literature.
Extract the following information from animal toxicity studies:

For Lab Test Results:
- Drug name and dose (with units)
- Lab test name (AST, ALT, etc.)
- Numerical values (mean ± SD format)
- Sample size
- Time point
- Species

For Organ Injury/Toxicity:
- Drug name and dose
- Type of injury or toxicity
- Frequency (X out of Y animals)
- Severity if mentioned
- Time point
- Species

Important formatting rules:
- Return ONLY a valid JSON object. No code fences, no backticks, no extra text.
- Use double quotes around all keys and string values.
- Do not add trailing commas.

Return a JSON object with two arrays:
{
    "lab_tests": [
        {
            "drug": "string",
            "dose": "string",
            "lab_test": "string",
            "value_mean": 0,
            "value_std": 0,
            "value_raw": "string",
            "sample_size": 0,
            "time_point": "string",
            "species": "string",
            "additional_info": "string"
        }
    ],
    "organ_injuries": [
        {
            "drug": "string",
            "dose": "string",
            "injury_type": "string",
            "frequency": 0,
            "total_animals": 0,
            "severity": "string",
            "time_point": "string",
            "species": "string",
            "additional_info": "string"
        }
    ]
}"""

    def assess_relevance(self, content: str, content_type: str = "text") -> Dict[str, Any]:
        prompt = f"""Content Type: {content_type}

Content:
{content[:3000]}

Analyze this content for animal toxicity data."""
        try:
            logger.info(
                "LLM assess_relevance call: provider=%s content_len=%d type=%s",
                self.llm.__class__.__name__,
                len(content),
                content_type,
            )
            # Keep relevance responses short and deterministic to encourage valid JSON
            response = self.llm.generate(prompt, self.relevance_prompt, temperature=0.0, max_tokens=300)
            logger.debug("LLM assess_relevance raw response (truncated): %r", (response or "")[:200])
            try:
                parsed = self._parse_json_response(response)
                return parsed
            except Exception as parse_err:  # attempt heuristic fallback
                logger.warning("JSON parse failed for relevance, using heuristic fallback: %s", parse_err)
                text = (response or "").lower()
                # simple heuristic: look for toxicity markers
                markers = ["ast", "alt", "alkaline phosphatase", "bilirubin", "creatinine", "bun",
                           "hepato", "nephro", "toxicity", "injury", "necrosis", "elevated"]
                score = sum(1 for m in markers if m in text) * 10
                score = max(0, min(100, score))
                is_rel = score >= 50
                return {"is_relevant": is_rel, "relevance_score": score, "data_types": [], "reason": "heuristic fallback"}
        except Exception as e:  # noqa: BLE001
            logger.error("Error assessing relevance: %s", e)
            return {"is_relevant": False, "relevance_score": 0, "reason": str(e)}

    def extract_data(self, content: str, pmid: str, source_location: str) -> Tuple[List[LabTestResult], List[OrganInjuryResult]]:
        prompt = f"""Extract toxicity data from this content:

{content}

Ensure all numerical values are parsed correctly and units are preserved."""
        try:
            logger.info(
                "LLM extract_data call: provider=%s content_len=%d pmid=%s src=%s",
                self.llm.__class__.__name__,
                len(content),
                pmid,
                source_location,
            )
            response = self.llm.generate(prompt, self.extraction_prompt)
            logger.debug("LLM extract_data raw response (truncated): %r", (response or "")[:200])
            data = self._parse_json_response(response)
            lab_results: List[LabTestResult] = []
            for item in data.get("lab_tests", []):
                item["pmid"] = pmid
                item["source_location"] = source_location
                lab_results.append(LabTestResult(**item))
            organ_results: List[OrganInjuryResult] = []
            for item in data.get("organ_injuries", []):
                item["pmid"] = pmid
                item["source_location"] = source_location
                if item.get("frequency") and item.get("total_animals"):
                    item["percentage"] = (item["frequency"] / item["total_animals"]) * 100
                organ_results.append(OrganInjuryResult(**item))
            return lab_results, organ_results
        except Exception as e:  # noqa: BLE001
            logger.error("Error extracting data: %s", e)
            return [], []

    # --- Helpers ---
    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON that may be wrapped in code fences or include extra text.

        Strategy:
        1) Prefer content inside the first ```json ... ``` code block.
        2) Else, extract the largest balanced {...} region.
        3) Attempt json.loads; on failure, retry with a simple cleanup (strip fences/whitespace).
        """
        if not text:
            raise ValueError("Empty response from LLM")

        # Try to extract JSON inside a fenced code block
        fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
        if fence_match:
            json_str = fence_match.group(1)
            try:
                return json.loads(json_str)
            except Exception:
                # Try repair on fenced content
                repaired = self._repair_json_like(json_str)
                return json.loads(repaired)

        # Generic extraction: find a balanced JSON object by scanning braces
        start = text.find('{')
        if start != -1:
            depth = 0
            end = -1
            for i in range(start, len(text)):
                ch = text[i]
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end != -1:
                candidate = text[start:end+1]
                try:
                    return json.loads(candidate)
                except Exception:
                    repaired = self._repair_json_like(candidate)
                    return json.loads(repaired)

        # Last attempt: trim and try direct parse (may raise)
        try:
            return json.loads(text.strip())
        except Exception:
            repaired = self._repair_json_like(text)
            return json.loads(repaired)

    def _repair_json_like(self, s: str) -> str:
        """Heuristically repair JSON-like text:
        - Quote unquoted object keys
        - Quote bareword strings inside arrays
        - Remove code fences/backticks
        Note: This is best-effort for common LLM formatting issues.
        """
        t = s.strip()
        # Remove ASCII control characters that break json.loads
        t = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", t)
        # Remove surrounding code fences if any
        t = re.sub(r"^```(?:json)?", "", t, flags=re.IGNORECASE).strip()
        t = re.sub(r"```$", "", t).strip()

        # Quote unquoted keys: { key: value, another_key: ... } -> { "key": value, "another_key": ... }
        # Handles after {, ,, [ contexts
        t = re.sub(r'([\{\[,]\s*)([A-Za-z_][A-Za-z0-9_\-]*)\s*:', r'\1"\2":', t)

        # Quote bareword strings inside arrays, avoiding true/false/null and numbers
        def quote_barewords_in_arrays(match: re.Match) -> str:
            content = match.group(1)
            # Replace barewords not inside quotes with quoted versions
            def repl(m: re.Match) -> str:
                w = m.group(0)
                lw = w.lower()
                if lw in ("true", "false", "null"):
                    return w
                # numbers
                if re.fullmatch(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", w):
                    return w
                return f'"{w}"'

            content = re.sub(r'(?<!["\\])\b([A-Za-z_][A-Za-z0-9_\-]*)\b(?!["\\])', repl, content)
            return f"[{content}]"

        t = re.sub(r'\[([^\]]*)\]', quote_barewords_in_arrays, t)

        # Remove trailing commas before } or ]
        t = re.sub(r',\s*([}\]])', r'\1', t)
        return t
