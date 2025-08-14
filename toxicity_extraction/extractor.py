import json
import re
import logging
from typing import Any, Dict, List, Tuple

from .models import LabTestResult, OrganInjuryResult
from .llm.base import LLMProvider

logger = logging.getLogger(__name__)


class ToxicityDataExtractor:
    """Main class for extracting toxicity data using LLMs"""

    def __init__(self, llm_provider: LLMProvider, max_tokens: int = 2048):
        self.llm = llm_provider
        self.max_tokens = max_tokens
        self.relevance_prompt = self._create_relevance_prompt()
        self.extraction_prompt = self._create_extraction_prompt()

    def _create_relevance_prompt(self) -> str:
        return """CRITICAL: You MUST return ONLY valid JSON. No extra text, no explanations, no code blocks.

This is a relevance_assessment task. Analyze content for animal toxicity data.

Look for: Lab values (AST, ALT, creatinine, BUN), organ damage (hepatic, renal), toxicity frequencies.

REQUIRED FORMAT (copy exactly, replace values):
{"is_relevant": true, "relevance_score": 85, "data_types": ["lab_tests"], "reason": "Contains AST/ALT data"}

Rules:
- NO backslashes except in strings
- NO code blocks or backticks
- ALL keys and string values must have double quotes
- Score 0-100 based on toxicity data richness
- Test your JSON before responding"""

    def _create_extraction_prompt(self) -> str:
        return """CRITICAL: You MUST return ONLY valid JSON. No extra text, no explanations, no code blocks, no comments.

This is a data extraction task. Extract toxicity data from animal studies and return in the exact format below.

REQUIRED FORMAT (copy exactly, replace values):
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
}

STRICT RULES:
- NO comments or descriptive keys
- NO backslashes except in strings  
- NO code blocks or backticks
- ALL keys and string values must have double quotes
- Numbers without quotes
- Test your JSON before responding"""

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
            # Increased token limit to prevent truncation
            response = self.llm.generate(prompt, self.relevance_prompt, temperature=0.0, max_tokens=self.max_tokens)
            logger.debug("LLM assess_relevance raw response (truncated): %r", (response or "")[:200])
            try:
                parsed = self._parse_json_response(response)
                # Validate the response structure
                if not all(key in parsed for key in ["is_relevant", "relevance_score", "data_types", "reason"]):
                    logger.warning("JSON response missing required keys, using structured fallback")
                    return self._structured_heuristic_assessment(content)
                return parsed
            except Exception as parse_err:
                logger.warning("JSON parse failed for relevance, using structured fallback: %s", parse_err)
                return self._structured_heuristic_assessment(content)
        except Exception as e:  # noqa: BLE001
            logger.error("Error assessing relevance: %s", e)
            return self._structured_heuristic_assessment(content)

    def _structured_heuristic_assessment(self, content: str) -> Dict[str, Any]:
        """Enhanced fallback using regex-based structured extraction."""
        content_lower = content.lower()
        
        # Look for actual lab values and patterns
        lab_patterns = [
            r'\b(ast|alt|alp|alkaline phosphatase|bilirubin|creatinine|bun|ck|creatine kinase)\b[:\s]*(\d+(?:\.\d+)?)',
            r'\b(ast|alt|alp|bilirubin|creatinine|bun|ck)\b.*?(?:increase|elevated|raise|higher)',
            r'(\d+(?:\.\d+)?)\s*(?:mg/dl|μmol/l|u/l|iu/l).*?\b(ast|alt|alp|bilirubin|creatinine|bun|ck)\b'
        ]
        
        # Look for organ damage terms with more specificity
        organ_patterns = [
            r'\b(hepat\w+|liver|renal|kidney|cardiac|lung)\s+(damage|injury|necrosis|toxicity|degeneration)',
            r'\b(necrosis|fibrosis|inflammation|degeneration|lesion).*?(liver|kidney|heart|lung|hepatic|renal)',
            r'\b(acute|chronic|severe|mild|moderate)\s+(hepatotoxicity|nephrotoxicity|cardiotoxicity)',
            r'\bhistopatholog\w+.*?(liver|kidney|heart|lung).*?(damage|injury|change)'
        ]
        
        # Count pattern matches
        lab_score = 0
        for pattern in lab_patterns:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            lab_score += len(matches) * 15  # Higher weight for lab values
        
        organ_score = 0
        for pattern in organ_patterns:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            organ_score += len(matches) * 20  # Higher weight for organ damage
        
        # Look for dosing and treatment information
        dose_patterns = [
            r'\b(\d+(?:\.\d+)?)\s*(mg|g|μg|ng)/kg',
            r'\bdose[d]?\s+(?:of\s+)?(\d+(?:\.\d+)?)',
            r'\btreat(?:ed|ment)\s+(?:with\s+)?.*?(\d+(?:\.\d+)?)\s*(?:mg|g)'
        ]
        dose_score = 0
        for pattern in dose_patterns:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            dose_score += len(matches) * 10
        
        # Animal study indicators
        animal_patterns = [
            r'\b(mice|rats|rabbits|dogs|monkeys|animals)\b',
            r'\b(male|female)\s+(mice|rats|rabbits)',
            r'\b(icr|balb|c57|sprague-dawley|wistar)\b'
        ]
        animal_score = 0
        for pattern in animal_patterns:
            if re.search(pattern, content_lower, re.IGNORECASE):
                animal_score += 5
        
        total_score = min(100, lab_score + organ_score + dose_score + animal_score)
        
        # Determine data types present
        data_types = []
        if lab_score > 0:
            data_types.append("lab_tests")
        if organ_score > 0:
            data_types.append("organ_injury")
        
        is_relevant = total_score >= 30  # Lower threshold for fallback
        
        reason = f"Structured analysis: lab_score={lab_score}, organ_score={organ_score}, dose_score={dose_score}, animal_score={animal_score}"
        
        return {
            "is_relevant": is_relevant,
            "relevance_score": total_score,
            "data_types": data_types,
            "reason": reason
        }

    def _regex_extraction_fallback(self, content: str, pmid: str, source_location: str) -> Tuple[List[LabTestResult], List[OrganInjuryResult]]:
        """Regex-based extraction fallback when JSON parsing fails."""
        logger.info("Using regex extraction fallback for %s", pmid)
        
        lab_results = []
        organ_results = []
        
        # Extract lab test values with more comprehensive patterns
        lab_patterns = [
            # Pattern: "AST (397.80 ± 17.64 U/L)" or similar
            r'\b(ast|alt|alp|alkaline phosphatase|bilirubin|creatinine|bun|ck|creatine kinase)\b[^\d]*?(\d+(?:\.\d+)?)\s*(?:±|±|\+/-)\s*(\d+(?:\.\d+)?)\s*([a-z/]+)',
            # Pattern: "AST: 397.80 U/L" or similar
            r'\b(ast|alt|alp|alkaline phosphatase|bilirubin|creatinine|bun|ck|creatine kinase)\b[:\s]*(\d+(?:\.\d+)?)\s*([a-z/]+)',
            # Pattern: "397.80 ± 17.64 U/L" near lab test names
            r'(\d+(?:\.\d+)?)\s*(?:±|±|\+/-)\s*(\d+(?:\.\d+)?)\s*([a-z/]+).*?\b(ast|alt|alp|bilirubin|creatinine|bun|ck)\b',
        ]
        
        # Extract drug name and dose
        drug_patterns = [
            r'\b(moringa\s+oleifera.*?extract|mohe)\b',
            r'\b(\d+(?:\.\d+)?)\s*(mg|g|μg|ng)/kg',
        ]
        
        # Find drug name
        drug_name = "Unknown"
        for pattern in drug_patterns[:1]:  # First pattern for drug name
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                drug_name = match.group(1).strip()
                break
        
        # Find doses
        doses = []
        for pattern in drug_patterns[1:]:  # Dose patterns
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                doses.append(f"{match[0]} {match[1]}/kg")
        
        # Extract lab values
        for pattern in lab_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                try:
                    if len(match) == 4 and match[0]:  # Has std dev
                        lab_test = match[0].upper()
                        value_mean = float(match[1])
                        value_std = float(match[2]) if match[2] else 0
                        units = match[3]
                    elif len(match) == 3:  # No std dev
                        if len(match) > 3 and match[3]:  # Lab test name at end
                            lab_test = match[3].upper()
                            value_mean = float(match[0])
                            value_std = float(match[1]) if match[1] else 0
                            units = match[2]
                        else:  # Lab test name at start
                            lab_test = match[0].upper()
                            value_mean = float(match[1])
                            value_std = 0
                            units = match[2]
                    else:
                        continue
                    
                    for dose in doses or ["Unknown dose"]:
                        lab_result = LabTestResult(
                            pmid=pmid,
                            source_location=source_location,
                            drug=drug_name,
                            dose=dose,
                            lab_test=lab_test,
                            value_mean=value_mean,
                            value_std=value_std,
                            value_raw=f"{value_mean} ± {value_std} {units}" if value_std else f"{value_mean} {units}",
                            sample_size=0,  # Not extracted by regex
                            time_point="Unknown",
                            species="ICR mice",  # Default from content
                            additional_info="Extracted by regex fallback"
                        )
                        lab_results.append(lab_result)
                except (ValueError, IndexError) as e:
                    logger.debug("Failed to parse lab result: %s", e)
                    continue
        
        # Extract organ injuries
        organ_patterns = [
            r'\b(hepatic|liver|renal|kidney|cardiac|heart|pulmonary|lung)\s+(necrosis|degeneration|injury|toxicity|damage)',
            r'\b(necrosis|degeneration|injury|toxicity|damage)\s+(?:of\s+)?(?:the\s+)?(liver|kidney|heart|lung|hepatic|renal)',
            r'\b(hepatotoxicity|nephrotoxicity|cardiotoxicity)',
        ]
        
        for pattern in organ_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                try:
                    if len(match) == 2:
                        if match[0] in ['hepatic', 'liver', 'renal', 'kidney', 'cardiac', 'heart', 'pulmonary', 'lung']:
                            organ = match[0]
                            injury = match[1]
                        else:
                            injury = match[0]
                            organ = match[1]
                    else:
                        injury = match[0] if match else "toxicity"
                        organ = "liver" if "hepato" in injury else "kidney" if "nephro" in injury else "heart" if "cardio" in injury else "unknown"
                    
                    for dose in doses or ["Unknown dose"]:
                        organ_result = OrganInjuryResult(
                            pmid=pmid,
                            source_location=source_location,
                            drug=drug_name,
                            dose=dose,
                            injury_type=f"{organ} {injury}",
                            frequency=0,  # Not extracted by regex
                            total_animals=0,
                            percentage=None,  # Not available from regex
                            severity="Unknown",
                            time_point="Unknown",
                            species="ICR mice",
                            additional_info="Extracted by regex fallback"
                        )
                        organ_results.append(organ_result)
                except Exception as e:
                    logger.debug("Failed to parse organ injury: %s", e)
                    continue
        
        logger.info("Regex fallback extracted %d lab tests, %d organ injuries", len(lab_results), len(organ_results))
        return lab_results, organ_results

    def extract_data(self, content: str, pmid: str, source_location: str) -> Tuple[List[LabTestResult], List[OrganInjuryResult]]:
        prompt = f"""Extract toxicity data from this content:

{content[:10000]}

Focus on lab test values (AST, ALT, creatinine, etc.) and organ injuries with numerical data."""
        try:
            logger.info(
                "LLM extract_data call: provider=%s content_len=%d pmid=%s src=%s",
                self.llm.__class__.__name__,
                len(content),
                pmid,
                source_location,
            )
            # Increased token limit for extraction to prevent truncation
            response = self.llm.generate(prompt, self.extraction_prompt, temperature=0.0, max_tokens=self.max_tokens)
            logger.debug("LLM extract_data raw response (truncated): %r", (response or "")[:200])
            
            try:
                data = self._parse_json_response(response)
                # Validate the response structure
                if not isinstance(data, dict) or ("lab_tests" not in data and "organ_injuries" not in data):
                    logger.warning("Invalid extraction response structure, using regex fallback")
                    return self._regex_extraction_fallback(content, pmid, source_location)
            except Exception as parse_err:
                logger.warning("JSON parse failed for extraction, using regex fallback: %s", parse_err)
                return self._regex_extraction_fallback(content, pmid, source_location)
            
            lab_results: List[LabTestResult] = []
            for item in data.get("lab_tests", []):
                try:
                    item["pmid"] = pmid
                    item["source_location"] = source_location
                    lab_results.append(LabTestResult(**item))
                except Exception as item_err:
                    logger.debug("Failed to create LabTestResult: %s", item_err)
                    continue
                    
            organ_results: List[OrganInjuryResult] = []
            for item in data.get("organ_injuries", []):
                try:
                    item["pmid"] = pmid
                    item["source_location"] = source_location
                    # Calculate percentage if both frequency and total_animals are available
                    if item.get("frequency") and item.get("total_animals"):
                        item["percentage"] = (item["frequency"] / item["total_animals"]) * 100
                    else:
                        item["percentage"] = None  # Set default percentage
                    organ_results.append(OrganInjuryResult(**item))
                except Exception as item_err:
                    logger.debug("Failed to create OrganInjuryResult: %s", item_err)
                    continue
                    
            return lab_results, organ_results
        except Exception as e:  # noqa: BLE001
            logger.error("Error extracting data: %s", e)
            return self._regex_extraction_fallback(content, pmid, source_location)

    # --- Helpers ---
    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Multi-stage JSON parsing with progressive repair strategies.

        Strategy:
        1) Try original text
        2) Try basic cleanup
        3) Try current repair logic
        4) Try aggressive repair
        5) Try structure reconstruction
        """
        if not text:
            raise ValueError("Empty response from LLM")

        # Multiple parsing attempts with increasing repair levels
        attempts = [
            ("original", text.strip()),
            ("basic_cleanup", self._basic_cleanup(text)),
            ("fence_extraction", self._extract_fenced_json(text)),
            ("balanced_extraction", self._extract_balanced_json(text)),
            ("current_repair", self._repair_json_like(text)),
            ("aggressive_repair", self._aggressive_repair(text)),
        ]
        
        for attempt_name, attempt_text in attempts:
            if not attempt_text:
                continue
            try:
                result = json.loads(attempt_text)
                if attempt_name != "original":
                    logger.debug("JSON parsed successfully using %s method", attempt_name)
                return result
            except Exception as e:
                logger.debug("JSON parsing attempt '%s' failed: %s", attempt_name, str(e)[:100])
                continue
        
        # If all attempts fail, raise the last error with context
        raise ValueError(f"All JSON parsing attempts failed for text: {text[:200]}...")

    def _basic_cleanup(self, text: str) -> str:
        """Basic cleanup of common JSON formatting issues."""
        t = text.strip()
        # Remove control characters
        t = re.sub(r'[\x00-\x1F\x7F]', '', t)
        # Remove code fences
        t = re.sub(r'^```(?:json)?\s*', '', t, flags=re.IGNORECASE).strip()
        t = re.sub(r'\s*```$', '', t).strip()
        return t

    def _extract_fenced_json(self, text: str) -> str:
        """Extract JSON from code fences."""
        fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
        if fence_match:
            return fence_match.group(1)
        return ""

    def _extract_balanced_json(self, text: str) -> str:
        """Extract balanced JSON object by scanning braces."""
        start = text.find('{')
        if start == -1:
            return ""
        
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
            return text[start:end+1]
        return ""

    def _aggressive_repair(self, text: str) -> str:
        """Aggressive repair for severely malformed JSON."""
        t = self._basic_cleanup(text)
        
        # Fix the specific pattern from the log: "key\\\\": value -> "key": value
        t = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*)\\{2,}\\"\\s*:', r'\\"\\1\\":', t)
        
        # Fix excessive escaping patterns common in LLM responses
        # \\\\\\\\\\\\\" -> \"
        t = re.sub(r'\\{2,}"', '"', t)
        # \\\\\\\\[ -> [, \\\\\\\\] -> ]
        t = re.sub(r'\\{2,}([\\[\\]])', r'\\1', t)
        
        # Fix array content with excessive escaping: \\\\\\\\\\\\\"value\\\\\\\\\\\\\" -> \"value\"
        t = re.sub(r'\\{4,}"([^"]+)\\{4,}"', r'"\\1"', t)
        
        # Fix missing quotes on keys (more robust)
        t = re.sub(r'([{\\[,]\\s*)([a-zA-Z_][a-zA-Z0-9_]*)\\s*:', r'\\1"\\2":', t)
        
        # Fix unquoted string values (more comprehensive)
        # Match: "key": unquoted_value -> "key": "unquoted_value"
        t = re.sub(r'("\\w+")\\s*:\\s*([a-zA-Z][a-zA-Z0-9\\s]+)(?=[,}\\]])', r'\\1: "\\2"', t)
        
        # Handle incomplete responses by closing JSON structure
        if t.count('{') > t.count('}'):
            t += '}' * (t.count('{') - t.count('}'))
        if t.count('[') > t.count(']'):
            t += ']' * (t.count('[') - t.count(']'))
        
        # Fix specific unquoted reason values
        t = re.sub(r'"reason"\\s*:\\s*([^",}\\]]+)(?=[,}\\]])', r'"reason": "\\1"', t)
        
        # Remove trailing commas
        t = re.sub(r',\\s*([}\\]])', r'\\1', t)
        
        return t

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
