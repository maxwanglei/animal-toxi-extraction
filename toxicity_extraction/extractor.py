import json
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
            response = self.llm.generate(prompt, self.relevance_prompt)
            return json.loads(response)
        except Exception as e:  # noqa: BLE001
            logger.error("Error assessing relevance: %s", e)
            return {"is_relevant": False, "relevance_score": 0, "reason": str(e)}

    def extract_data(self, content: str, pmid: str, source_location: str) -> Tuple[List[LabTestResult], List[OrganInjuryResult]]:
        prompt = f"""Extract toxicity data from this content:

{content}

Ensure all numerical values are parsed correctly and units are preserved."""
        try:
            response = self.llm.generate(prompt, self.extraction_prompt)
            data = json.loads(response)
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
