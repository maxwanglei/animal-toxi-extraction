"""
Advanced table-specific extraction pipeline inspired by tabular-data repository.
Implements multi-step workflow for better toxicity data extraction from tables.
"""

import logging
import re
from typing import Any, Dict, List, Tuple
import pandas as pd
from io import StringIO

from .models import LabTestResult, OrganInjuryResult
from .llm.base import LLMProvider

logger = logging.getLogger(__name__)


class TableToxicityExtractor:
    """Specialized extractor for table-based toxicity data using multi-step workflow"""
    
    def __init__(self, llm_provider: LLMProvider, max_tokens: int = 3072):
        self.llm = llm_provider
        self.max_tokens = max_tokens
        self.max_retries = 3
        self.initial_wait = 1
        
    def extract_from_table(self, table_content: str, table_caption: str, 
                          pmid: str, source_location: str) -> Tuple[List[LabTestResult], List[OrganInjuryResult]]:
        """
        Multi-step extraction process for tables:
        1. Convert table to structured markdown
        2. Identify relevant rows/columns 
        3. Extract specific data types in focused steps
        4. Validate and combine results
        """
        try:
            # Step 1: Convert to markdown table format
            md_table = self._convert_to_markdown(table_content)
            if not md_table:
                logger.warning("Failed to convert table to markdown: %s", source_location)
                return [], []
            
            # Step 2: Assess table relevance with detailed analysis
            relevance = self._assess_table_relevance(md_table, table_caption)
            if not relevance["is_relevant"]:
                logger.debug("Table not relevant for toxicity data: %s", source_location)
                return [], []
            
            # Step 3: Multi-step extraction based on detected data types
            lab_results = []
            organ_results = []
            
            if "lab_tests" in relevance.get("data_types", []):
                lab_results = self._extract_lab_values_step(md_table, table_caption, pmid, source_location)
                
            if "organ_injury" in relevance.get("data_types", []):
                organ_results = self._extract_organ_injuries_step(md_table, table_caption, pmid, source_location)
            
            # Step 4: If primary extraction fails, try unified extraction
            if not lab_results and not organ_results and relevance["relevance_score"] > 70:
                lab_results, organ_results = self._unified_table_extraction(
                    md_table, table_caption, pmid, source_location
                )
            
            logger.info("Table extraction complete: %d lab tests, %d organ injuries from %s", 
                       len(lab_results), len(organ_results), source_location)
            return lab_results, organ_results
            
        except Exception as e:
            logger.error("Error in table extraction for %s: %s", source_location, e)
            return [], []
    
    def _convert_to_markdown(self, table_content: str) -> str:
        """Convert various table formats to clean markdown"""
        try:
            # If already looks like markdown table
            if "|" in table_content and "---" in table_content:
                return self._clean_markdown_table(table_content)
            
            # Try to parse as CSV-like content and convert to markdown
            if "\t" in table_content or "," in table_content:
                return self._csv_to_markdown(table_content)
                
            # Try to parse as space-separated or pipe-separated
            lines = table_content.strip().split('\n')
            if len(lines) >= 2:
                return self._lines_to_markdown(lines)
                
            return table_content
            
        except Exception as e:
            logger.debug("Table conversion failed: %s", e)
            return table_content
    
    def _clean_markdown_table(self, md_content: str) -> str:
        """Clean and standardize markdown table format"""
        lines = md_content.strip().split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove extra whitespace, ensure proper pipe separation
            if '|' in line:
                parts = [part.strip() for part in line.split('|')]
                # Remove empty parts at start/end
                while parts and not parts[0]:
                    parts.pop(0)
                while parts and not parts[-1]:
                    parts.pop()
                if parts:
                    cleaned_lines.append('| ' + ' | '.join(parts) + ' |')
            elif line.strip():
                cleaned_lines.append(line.strip())
                
        return '\n'.join(cleaned_lines)
    
    def _csv_to_markdown(self, csv_content: str) -> str:
        """Convert CSV-like content to markdown table"""
        try:
            # Try tab-separated first, then comma
            separator = '\t' if '\t' in csv_content else ','
            df = pd.read_csv(StringIO(csv_content), sep=separator)
            
            # Convert to markdown
            md_lines = []
            # Header
            md_lines.append('| ' + ' | '.join(df.columns) + ' |')
            # Separator
            md_lines.append('| ' + ' | '.join(['---'] * len(df.columns)) + ' |')
            # Data rows
            for _, row in df.iterrows():
                md_lines.append('| ' + ' | '.join(str(val) for val in row.values) + ' |')
                
            return '\n'.join(md_lines)
            
        except Exception as e:
            logger.debug("CSV conversion failed: %s", e)
            return csv_content
    
    def _lines_to_markdown(self, lines: List[str]) -> str:
        """Convert space/pipe separated lines to markdown"""
        md_lines = []
        header_processed = False
        
        for line in lines:
            if not line.strip():
                continue
                
            # Split by multiple spaces or pipes
            if '|' in line:
                parts = [p.strip() for p in line.split('|') if p.strip()]
            else:
                parts = [p.strip() for p in re.split(r'\s{2,}', line.strip()) if p.strip()]
            
            if parts:
                md_line = '| ' + ' | '.join(parts) + ' |'
                md_lines.append(md_line)
                
                # Add separator after first line (header)
                if not header_processed:
                    md_lines.append('| ' + ' | '.join(['---'] * len(parts)) + ' |')
                    header_processed = True
                    
        return '\n'.join(md_lines)
    
    def _assess_table_relevance(self, md_table: str, caption: str) -> Dict[str, Any]:
        """Enhanced table relevance assessment with specific toxicity patterns"""
        
        prompt = f"""Analyze this table for animal toxicity data relevance:

TABLE:
{md_table}

CAPTION: {caption}

Assess for:
1. Laboratory test values (ALT, AST, creatinine, BUN, etc.)
2. Organ injury/damage data  
3. Dose-response relationships
4. Animal study context
5. Quantitative toxicity measurements

Score 0-100 based on richness of extractable toxicity data."""

        system_prompt = """You are analyzing a scientific table for animal toxicity data extraction.

Return ONLY valid JSON in this exact format:
{
    "is_relevant": true,
    "relevance_score": 85,
    "data_types": ["lab_tests", "organ_injury"],
    "reason": "Table contains quantitative liver enzyme data with dose groups",
    "detected_elements": {
        "lab_tests": ["ALT", "AST", "creatinine"],
        "doses": ["10 mg/kg", "50 mg/kg"],
        "species": "rats",
        "time_points": ["24h", "7 days"]
    }
}

Rules:
- Score 0-100
- data_types: ["lab_tests"], ["organ_injury"], or both
- Include specific detected elements when possible
- No code blocks or explanations"""

        try:
            response = self.llm.generate(prompt, system_prompt, temperature=0.0, max_tokens=512)
            return self._parse_json_response(response)
        except Exception as e:
            logger.warning("Table relevance assessment failed: %s", e)
            return self._heuristic_table_assessment(md_table, caption)
    
    def _heuristic_table_assessment(self, md_table: str, caption: str) -> Dict[str, Any]:
        """Fallback heuristic assessment for tables"""
        combined_text = (md_table + " " + caption).lower()
        
        # Look for lab test indicators
        lab_indicators = ['alt', 'ast', 'alp', 'bilirubin', 'creatinine', 'bun', 'urea', 'body weight'
                         'albumin', 'protein', 'glucose', 'cholesterol', 'triglyceride']
        lab_score = sum(10 for indicator in lab_indicators if indicator in combined_text)

        # Look for organ injury indicators
        organ_indicators = ['necrosis', 'damage', 'injury', 'toxicity', 'lesion', 'fibrosis', 'body weight decrease',
                           'inflammation', 'degeneration', 'hepatic', 'renal', 'cardiac']
        organ_score = sum(15 for indicator in organ_indicators if indicator in combined_text)
        
        # Look for dose indicators
        dose_patterns = [
            r'\b\d+\s*mg/kg\b', r'\b\d+\s*g/kg\b', r'\b\d+\s*(μg|ug)/kg\b',
            r'\b\d+\s*(mg|μg|ug|g)/(kg|body\s*weight)/(day|d)\b',
            r'\b\d+\s*gy\b', r'\bpfu\b', r'\btcid\s*50\b',
            'dose', 'treatment'
        ]
        dose_score = sum(5 for pattern in dose_patterns 
                        if re.search(pattern, combined_text, re.IGNORECASE))
        
        # Look for animal study indicators
        animal_indicators = ['rats', 'mice', 'rabbits', 'dogs', 'monkeys', 'animals', 'male', 'female', 'fish']
        animal_score = sum(3 for indicator in animal_indicators if indicator in combined_text)
        
        total_score = min(100, lab_score + organ_score + dose_score + animal_score)
        
        data_types = []
        if lab_score > 0:
            data_types.append("lab_tests")
        if organ_score > 0:
            data_types.append("organ_injury")
            
        return {
            "is_relevant": total_score >= 20,
            "relevance_score": total_score,
            "data_types": data_types,
            "reason": f"Heuristic: lab={lab_score}, organ={organ_score}, dose={dose_score}, animal={animal_score}"
        }
    
    def _extract_lab_values_step(self, md_table: str, caption: str, pmid: str, 
                                source_location: str) -> List[LabTestResult]:
        """Focused extraction step for laboratory test values"""
        
        prompt = f"""Extract laboratory test values from this table:

TABLE:
{md_table}

CAPTION: {caption}

Focus on quantitative lab values or biochemical markers like ALT, AST, ALP, bilirubin, creatinine, BUN, body weight, etc.
Extract actual numerical values with their measurement unit when available (e.g., U/L, mg/dL, g/L).
If only qualitative or conclusion text is present (e.g., "significantly increased"), capture it under descriptive_values and leave numerical values null.
Identify dose groups, time points, statistical measures, and treatment names broadly (drug, vaccine, radiation, nanoparticles)."""

        system_prompt = """Extract lab test data and return ONLY valid JSON:

{
    "lab_tests": [
        {
            "drug": "compound name",
            "dose": "10 mg/kg",
            "lab_test": "ALT",
            "unit": "U/L",
            "value_mean": 45.2,
            "value_std": 5.1,
            "value_raw": "45.2 ± 5.1 U/L",
            "descriptive_values": null,
            "sample_size": 8,
            "time_point": "24h",
            "species": "rats",
            "additional_info": "fasted animals"
        }
    ]
}

CRITICAL: Use EXACTLY these field names - drug, dose, lab_test, unit, value_mean, value_std, value_raw, descriptive_values, sample_size, time_point, species, additional_info.
Rules:
- Extract lab tests like ALT, AST, ALP, BUN, creatinine, glucose, body weight, etc.
- Use exact numerical values when available
- If only qualitative change is present, set descriptive_values and keep value_mean/value_std null
- Use "Unknown" for missing text
- Tumor weight is not considered as a lab test results
- No code blocks or explanations"""

        try:
            for attempt in range(self.max_retries):
                try:
                    response = self.llm.generate(prompt, system_prompt, temperature=0.0, max_tokens=self.max_tokens)
                    data = self._parse_json_response(response)
                    
                    lab_results = []
                    for item in data.get("lab_tests", []):
                        try:
                            # Map and validate fields for LabTestResult
                            mapped_item = self._map_lab_test_fields(item)
                            mapped_item["pmid"] = pmid
                            mapped_item["source_location"] = source_location
                            lab_results.append(LabTestResult(**mapped_item))
                        except Exception as e:
                            logger.debug("Failed to create LabTestResult: %s", e)
                            continue
                    
                    return lab_results
                    
                except Exception as e:
                    logger.warning("Lab extraction attempt %d failed: %s", attempt + 1, e)
                    if attempt < self.max_retries - 1:
                        continue
                    else:
                        return []
                        
        except Exception as e:
            logger.error("Lab values extraction failed: %s", e)
            return []
    
    def _extract_organ_injuries_step(self, md_table: str, caption: str, pmid: str,
                                   source_location: str) -> List[OrganInjuryResult]:
        """Focused extraction step for organ injury data"""
        
        prompt = f"""Extract organ injury/damage data from this table:

TABLE:
{md_table}

CAPTION: {caption}

Focus on organ damage, injuries, histopathological findings, toxicity frequencies.
Include descriptive/qualitative findings (e.g., "centrilobular necrosis", "vacuolar degeneration") even if no counts are provided.
Identify treatments broadly (drug, vaccine, radiation, nanoparticles) and capture dose notations (e.g., mg/kg, Gy, PFU)."""

        system_prompt = """Extract organ injury data and return ONLY valid JSON:

{
    "organ_injuries": [
        {
            "drug": "compound name",
            "dose": "50 mg/kg",
            "injury_type": "hepatic necrosis",
            "frequency": 3,
            "total_animals": 10,
            "severity": "mild",
            "time_point": "7 days",
            "species": "rats",
            "descriptive_values": null,
            "additional_info": "focal areas, reversible"
        }
    ]
}

CRITICAL: Use EXACTLY these field names - drug, dose, injury_type, frequency, total_animals, severity, time_point, species, descriptive_values, additional_info.
Rules:
- Extract specific injury types and affected organs
- body weight changes are also relevant
- Include frequency data when available; if counts are not present, set frequency/total_animals to 0 and record qualitative details in descriptive_values
- Use 0 for missing numerical values, "Unknown" for missing text
- No code blocks or explanations"""

        try:
            for attempt in range(self.max_retries):
                try:
                    response = self.llm.generate(prompt, system_prompt, temperature=0.0, max_tokens=self.max_tokens)
                    data = self._parse_json_response(response)
                    
                    organ_results = []
                    for item in data.get("organ_injuries", []):
                        try:
                            # Map and validate fields for OrganInjuryResult
                            mapped_item = self._map_organ_injury_fields(item)
                            mapped_item["pmid"] = pmid
                            mapped_item["source_location"] = source_location
                            
                            # Calculate percentage if available
                            if mapped_item.get("frequency") and mapped_item.get("total_animals"):
                                freq = mapped_item["frequency"]
                                total = mapped_item["total_animals"]
                                if freq != 0 and total != 0:
                                    mapped_item["percentage"] = (freq / total) * 100
                                else:
                                    mapped_item["percentage"] = None
                            else:
                                mapped_item["percentage"] = None
                                
                            organ_results.append(OrganInjuryResult(**mapped_item))
                        except Exception as e:
                            logger.debug("Failed to create OrganInjuryResult: %s", e)
                            continue
                    
                    return organ_results
                    
                except Exception as e:
                    logger.warning("Organ extraction attempt %d failed: %s", attempt + 1, e)
                    if attempt < self.max_retries - 1:
                        continue
                    else:
                        return []
                        
        except Exception as e:
            logger.error("Organ injuries extraction failed: %s", e)
            return []
    
    def _unified_table_extraction(self, md_table: str, caption: str, pmid: str,
                                 source_location: str) -> Tuple[List[LabTestResult], List[OrganInjuryResult]]:
        """Unified extraction as fallback when focused steps fail"""
        
        prompt = f"""Extract all toxicity data from this table:

TABLE:
{md_table}

CAPTION: {caption}

Extract both laboratory test values and organ injury data comprehensively. Include units for lab tests when present, and use descriptive_values when only qualitative descriptions are available. Recognize treatments including drugs, vaccines, radiation, and nanoparticles."""

        system_prompt = """Extract all toxicity data and return ONLY valid JSON:

{
    "lab_tests": [
        {
            "drug": "compound name",
            "dose": "10 mg/kg", 
            "lab_test": "ALT",
            "unit": "U/L",
            "value_mean": 45.2,
            "value_std": 3.1,
            "value_raw": "45.2 ± 3.1 U/L",
            "descriptive_values": null,
            "sample_size": 8,
            "time_point": "24h",
            "species": "rats",
            "additional_info": "vs control p<0.05"
        }
    ],
    "organ_injuries": [
        {
            "drug": "compound name",
            "dose": "50 mg/kg",
            "injury_type": "hepatic necrosis", 
            "frequency": 3,
            "total_animals": 10,
            "severity": "mild",
            "time_point": "7 days",
            "species": "rats",
            "descriptive_values": null,
            "additional_info": "focal areas"
        }
    ]
}

Rules:
- Extract all relevant toxicity data
- For lab tests, include unit when present
- If no numeric values exist, store qualitative/conclusion text in descriptive_values and leave numbers null
- Use "Unknown" for missing text
- No code blocks or explanations"""

        try:
            response = self.llm.generate(prompt, system_prompt, temperature=0.0, max_tokens=self.max_tokens)
            data = self._parse_json_response(response)
            
            lab_results = []
            for item in data.get("lab_tests", []):
                try:
                    mapped_item = self._map_lab_test_fields(item)
                    mapped_item["pmid"] = pmid
                    mapped_item["source_location"] = source_location
                    lab_results.append(LabTestResult(**mapped_item))
                except Exception as e:
                    logger.debug("Failed to create LabTestResult: %s", e)
                    continue
            
            organ_results = []
            for item in data.get("organ_injuries", []):
                try:
                    mapped_item = self._map_organ_injury_fields(item)
                    mapped_item["pmid"] = pmid
                    mapped_item["source_location"] = source_location
                    if mapped_item.get("frequency") and mapped_item.get("total_animals"):
                        mapped_item["percentage"] = (mapped_item["frequency"] / mapped_item["total_animals"]) * 100
                    else:
                        mapped_item["percentage"] = None
                    organ_results.append(OrganInjuryResult(**mapped_item))
                except Exception as e:
                    logger.debug("Failed to create OrganInjuryResult: %s", e)
                    continue
                    
            return lab_results, organ_results
            
        except Exception as e:
            logger.error("Unified table extraction failed: %s", e)
            return [], []
    
    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Enhanced JSON parsing with multiple repair strategies"""
        if not text:
            raise ValueError("Empty response from LLM")

        import json
        
        # Multiple parsing attempts with increasing repair levels
        attempts = [
            ("original", text.strip()),
            ("basic_cleanup", self._basic_cleanup(text)),
            ("fence_extraction", self._extract_fenced_json(text)),
            ("balanced_extraction", self._extract_balanced_json(text)),
            ("aggressive_repair", self._aggressive_repair(text)),
            ("partial_extraction", self._partial_extraction(text)),
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
        
        raise ValueError(f"All JSON parsing attempts failed for text: {text[:200]}...")
    
    def _basic_cleanup(self, text: str) -> str:
        """Basic cleanup of common JSON formatting issues"""
        t = text.strip()
        
        # Handle completely empty responses
        if not t:
            return '{"lab_tests": [], "organ_injuries": []}'
        
        # Remove control characters
        t = re.sub(r'[\x00-\x1F\x7F]', '', t)
        
        # Handle cases where response doesn't start with JSON
        # Look for the first occurrence of a JSON object
        json_start = t.find('{')
        if json_start > 0:
            # There's text before the JSON, remove it
            t = t[json_start:]
        
        # Remove code fences more comprehensively
        t = re.sub(r'^```(?:json)?\s*', '', t, flags=re.IGNORECASE | re.MULTILINE).strip()
        t = re.sub(r'\s*```$', '', t, flags=re.MULTILINE).strip()
        
        # Remove common prefixes that LLMs sometimes add
        prefixes_to_remove = [
            'Here is the extracted data:',
            'The extracted data is:',
            'Based on the table, here is the data:',
            'Here\'s the JSON response:',
            'Response:',
            'JSON:',
        ]
        
        for prefix in prefixes_to_remove:
            if t.lower().startswith(prefix.lower()):
                t = t[len(prefix):].strip()
                # Look for JSON start again after removing prefix
                json_start = t.find('{')
                if json_start > 0:
                    t = t[json_start:]
                break
        
        return t

    def _extract_fenced_json(self, text: str) -> str:
        """Extract JSON from code fences"""
        fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
        if fence_match:
            return fence_match.group(1)
        return ""

    def _extract_balanced_json(self, text: str) -> str:
        """Extract balanced JSON object by scanning braces"""
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
        """Aggressive repair for severely malformed JSON"""
        t = self._basic_cleanup(text)
        
        # Handle severely truncated responses
        if len(t) > 10000 and not t.rstrip().endswith(('}', ']')):
            # Response likely truncated, try to find last complete structure
            lines = t.split('\n')
            for i in range(len(lines) - 1, -1, -1):
                line = lines[i].strip()
                if line.endswith(('}', ']')) and not line.endswith(','):
                    t = '\n'.join(lines[:i+1])
                    break
        
        # Fix excessive escaping patterns
        t = re.sub(r'\\{2,}"', '"', t)
        t = re.sub(r'\\{2,}([\\[\\]])', r'\\1', t)
        
        # Fix missing quotes on keys
        t = re.sub(r'([{\\[,]\\s*)([a-zA-Z_][a-zA-Z0-9_]*)\\s*:', r'\\1"\\2":', t)
        
        # Fix unquoted string values
        t = re.sub(r'("\\w+")\\s*:\\s*([a-zA-Z][a-zA-Z0-9\\s]+)(?=[,}\\]])', r'\\1: "\\2"', t)
        
        # Handle unterminated strings more carefully
        in_string = False
        result = []
        i = 0
        while i < len(t):
            char = t[i]
            if char == '"' and (i == 0 or t[i-1] != '\\'):
                in_string = not in_string
            elif not in_string and char in '{}[]':
                # We're not in a string, so this is structural
                pass
            result.append(char)
            i += 1
        
        # If we ended while in a string, close it
        if in_string:
            result.append('"')
        
        t = ''.join(result)
        
        # Handle incomplete responses by counting structures
        open_braces = t.count('{') - t.count('}')
        open_brackets = t.count('[') - t.count(']')
        
        # Add missing closing characters
        if open_braces > 0:
            t += '}' * open_braces
        if open_brackets > 0:
            t += ']' * open_brackets
        
        # Remove trailing commas
        t = re.sub(r',\\s*([}\\]])', r'\\1', t)
        
        # Fix common structural issues
        t = re.sub(r'\\n\\s*"([^"]+)"\\s*:\\s*"([^"]*)"\\s*$', r'\\n    "\\1": "\\2"\\n}', t)
        
        return t
    
    def _partial_extraction(self, text: str) -> str:
        """Last resort: extract partial data from severely malformed responses"""
        import json
        try:
            # Return minimal valid JSON structure that can be processed
            result = {
                "lab_tests": [],
                "organ_injuries": []
            }
            
            # Try to extract individual items even if overall JSON is broken
            text_lower = text.lower()
            
            # Look for patterns that suggest lab test data
            if any(keyword in text_lower for keyword in ['ast', 'alt', 'creatinine', 'urea', 'glucose', 'cholesterol']):
                # Could potentially extract partial lab test data here
                pass
            
            # Look for patterns that suggest organ injury data
            if any(keyword in text_lower for keyword in ['liver', 'kidney', 'heart', 'lung', 'brain', 'injury', 'damage']):
                # Could potentially extract partial organ injury data here
                pass
            
            return json.dumps(result)
            
        except Exception:
            # Return absolute minimum
            return '{"lab_tests": [], "organ_injuries": []}'
    
    def _map_lab_test_fields(self, item: dict) -> dict:
        """Map LLM response fields to LabTestResult field names"""
        # Field mappings for common variations
        field_mappings = {
            'test': 'lab_test',
            'test_name': 'lab_test',
            'parameter': 'lab_test',
            'biomarker': 'lab_test',
            'mean': 'value_mean',
            'mean_value': 'value_mean',
            'average': 'value_mean',
            'std': 'value_std',
            'std_dev': 'value_std',
            'standard_deviation': 'value_std',
            'stdev': 'value_std',
            'raw': 'value_raw',
            'raw_value': 'value_raw',
            'original': 'value_raw',
            'n': 'sample_size',
            'sample_n': 'sample_size',
            'group_size': 'sample_size',
            'size': 'sample_size',
            'time': 'time_point',
            'timepoint': 'time_point',
            'duration': 'time_point',
            'animal': 'species',
            'animals': 'species',
            'model': 'species',
            'organism': 'species',
            'info': 'additional_info',
            'notes': 'additional_info',
            'comments': 'additional_info',
            'details': 'additional_info',
            'units': 'unit',
            'measure_unit': 'unit',
            'measurement_unit': 'unit',
            'qualitative': 'descriptive_values',
            'description': 'descriptive_values',
            'conclusion': 'descriptive_values',
        }
        
        mapped_item = {}
        
        # Required fields with defaults
        required_defaults = {
            'drug': 'Unknown',
            'dose': 'Unknown', 
            'lab_test': 'Unknown',
            'unit': None,
            'value_mean': None,
            'value_std': None,
            'value_raw': 'Unknown',
            'descriptive_values': None,
            'sample_size': None,
            'time_point': 'Unknown',
            'species': 'Unknown',
            'additional_info': None,
        }
        
        # First, copy exact matches
        for key, value in item.items():
            if key in required_defaults:
                # Convert numeric strings to proper types
                if key in ['value_mean', 'value_std'] and isinstance(value, str):
                    try:
                        mapped_item[key] = float(value) if value.lower() != 'unknown' else None
                    except (ValueError, AttributeError):
                        mapped_item[key] = None
                elif key == 'sample_size' and isinstance(value, str):
                    try:
                        mapped_item[key] = int(value) if value.lower() != 'unknown' else None
                    except (ValueError, AttributeError):
                        mapped_item[key] = None
                else:
                    mapped_item[key] = value
        
        # Then, apply field mappings for non-exact matches
        for key, value in item.items():
            mapped_key = field_mappings.get(key.lower())
            if mapped_key and mapped_key not in mapped_item:
                # Convert types as needed
                if mapped_key in ['value_mean', 'value_std'] and isinstance(value, str):
                    try:
                        mapped_item[mapped_key] = float(value) if value.lower() != 'unknown' else None
                    except (ValueError, AttributeError):
                        mapped_item[mapped_key] = None
                elif mapped_key == 'sample_size' and isinstance(value, str):
                    try:
                        mapped_item[mapped_key] = int(value) if value.lower() != 'unknown' else None
                    except (ValueError, AttributeError):
                        mapped_item[mapped_key] = None
                else:
                    mapped_item[mapped_key] = value
        
        # Fill in missing required fields with defaults
        for key, default_value in required_defaults.items():
            if key not in mapped_item:
                mapped_item[key] = default_value
        
        return mapped_item
    
    def _map_organ_injury_fields(self, item: dict) -> dict:
        """Map LLM response fields to OrganInjuryResult field names"""
        # Field mappings for common variations
        field_mappings = {
            'organ': 'injury_type',  # Common mistake: using 'organ' instead of 'injury_type'
            'injury': 'injury_type',
            'damage': 'injury_type',
            'pathology': 'injury_type',
            'finding': 'injury_type',
            'lesion': 'injury_type',
            'count': 'frequency',
            'affected': 'frequency',
            'number': 'frequency',
            'n_affected': 'frequency',
            'total': 'total_animals',
            'total_n': 'total_animals',
            'group_size': 'total_animals',
            'n_total': 'total_animals',
            'grade': 'severity',
            'level': 'severity',
            'degree': 'severity',
            'intensity': 'severity',
            'time': 'time_point',
            'timepoint': 'time_point',
            'duration': 'time_point',
            'animal': 'species',
            'animals': 'species',
            'model': 'species',
            'organism': 'species',
            'info': 'additional_info',
            'notes': 'additional_info',
            'comments': 'additional_info',
            'details': 'additional_info',
            'qualitative': 'descriptive_values',
            'description': 'descriptive_values',
            'finding_text': 'descriptive_values',
            'conclusion': 'descriptive_values',
        }
        
        mapped_item = {}
        
        # Required fields with defaults
        required_defaults = {
            'drug': 'Unknown',
            'dose': 'Unknown',
            'injury_type': 'Unknown',
            'frequency': 0,
            'total_animals': 0,
            'percentage': None,
            'severity': 'Unknown',
            'time_point': 'Unknown',
            'species': 'Unknown',
            'descriptive_values': None,
            'additional_info': None,
        }
        
        # First, copy exact matches
        for key, value in item.items():
            if key in required_defaults:
                # Convert numeric strings to proper types
                if key in ['frequency', 'total_animals'] and isinstance(value, str):
                    try:
                        mapped_item[key] = int(value) if value.lower() != 'unknown' else 0
                    except (ValueError, AttributeError):
                        mapped_item[key] = 0
                elif key == 'percentage' and isinstance(value, str):
                    try:
                        mapped_item[key] = float(value) if value.lower() != 'unknown' else None
                    except (ValueError, AttributeError):
                        mapped_item[key] = None
                else:
                    mapped_item[key] = value
        
        # Then, apply field mappings for non-exact matches
        for key, value in item.items():
            mapped_key = field_mappings.get(key.lower())
            if mapped_key and mapped_key not in mapped_item:
                # Convert types as needed
                if mapped_key in ['frequency', 'total_animals'] and isinstance(value, str):
                    try:
                        mapped_item[mapped_key] = int(value) if value.lower() != 'unknown' else 0
                    except (ValueError, AttributeError):
                        mapped_item[mapped_key] = 0
                elif mapped_key == 'percentage' and isinstance(value, str):
                    try:
                        mapped_item[mapped_key] = float(value) if value.lower() != 'unknown' else None
                    except (ValueError, AttributeError):
                        mapped_item[mapped_key] = None
                else:
                    mapped_item[mapped_key] = value
        
        # Fill in missing required fields with defaults
        for key, default_value in required_defaults.items():
            if key not in mapped_item:
                mapped_item[key] = default_value
        
        return mapped_item