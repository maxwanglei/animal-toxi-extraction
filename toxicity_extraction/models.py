from dataclasses import dataclass
from typing import Optional


@dataclass
class LabTestResult:
    """Structure for lab test results (AST, ALT, etc.)"""
    pmid: str
    drug: str
    dose: str
    lab_test: str
    unit: Optional[str]
    value_mean: Optional[float]
    value_std: Optional[float]
    value_raw: str  # Original text representation
    descriptive_values: Optional[str]
    sample_size: Optional[int]
    time_point: Optional[str]
    species: Optional[str]
    additional_info: Optional[str]
    source_location: str  # table_id or section


@dataclass
class OrganInjuryResult:
    """Structure for organ injury/toxicity frequency data"""
    pmid: str
    drug: str
    dose: str
    injury_type: str
    frequency: int
    total_animals: int
    percentage: Optional[float]
    severity: Optional[str]
    time_point: Optional[str]
    species: Optional[str]
    descriptive_values: Optional[str]
    additional_info: Optional[str]
    source_location: str
