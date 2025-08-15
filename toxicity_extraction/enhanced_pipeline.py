"""
Enhanced toxicity extraction pipeline with table-aware processing.
Integrates the new table extractor with improved workflow management.
"""

import logging
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from .extractor import ToxicityDataExtractor
from .table_extractor import TableToxicityExtractor
from .parsers.pmc_xml import PMCXMLParser
from .parsers.bioc_parser import BioCParser

logger = logging.getLogger(__name__)


class EnhancedToxicityExtractionPipeline:
    """Enhanced pipeline with table-aware processing and workflow management"""

    def __init__(self, llm_provider, output_dir: str = "./output", max_tokens: int = 3072):
        self.text_extractor = ToxicityDataExtractor(llm_provider, max_tokens=max_tokens)
        self.table_extractor = TableToxicityExtractor(llm_provider, max_tokens=max_tokens)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def process_batch(self, xml_files: List[str], save_intermediate: bool = True, save_individual_json: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process batch of files with enhanced table processing
        
        Args:
            xml_files: List of XML file paths to process
            save_intermediate: Save CSV files for each paper
            save_individual_json: Save detailed JSON results for each paper (useful for SLURM)
        """
        all_lab_results = []
        all_organ_results = []
        processing_stats = {
            "total_files": len(xml_files),
            "table_extractions": 0,
            "text_extractions": 0,
            "failed_files": 0
        }
        
        for i, xml_file in enumerate(xml_files, 1):
            logger.info("Processing file %d/%d: %s", i, len(xml_files), xml_file)
            try:
                lab_df, organ_df, file_stats = self._process_pmc_file_enhanced(xml_file)
                
                # Update processing statistics
                processing_stats["table_extractions"] += file_stats.get("table_extractions", 0)
                processing_stats["text_extractions"] += file_stats.get("text_extractions", 0)
                
                base_name = Path(xml_file).stem
                
                if save_intermediate:
                    lab_df.to_csv(self.output_dir / f"{base_name}_lab_results.csv", index=False)
                    organ_df.to_csv(self.output_dir / f"{base_name}_organ_results.csv", index=False)
                
                if save_individual_json:
                    # Save comprehensive JSON results for each paper
                    individual_result = {
                        "pmid": base_name,
                        "file_path": xml_file,
                        "processing_timestamp": pd.Timestamp.now().isoformat(),
                        "file_stats": file_stats,
                        "lab_tests": lab_df.to_dict('records') if not lab_df.empty else [],
                        "organ_injuries": organ_df.to_dict('records') if not organ_df.empty else [],
                        "summary": {
                            "total_lab_tests": len(lab_df),
                            "total_organ_injuries": len(organ_df),
                            "unique_drugs": list(set(
                                (lab_df['drug'].dropna().unique().tolist() if not lab_df.empty and 'drug' in lab_df.columns else []) + 
                                (organ_df['drug'].dropna().unique().tolist() if not organ_df.empty and 'drug' in organ_df.columns else [])
                            )),
                            "tables_processed": file_stats.get("table_extractions", 0),
                            "text_sections_processed": file_stats.get("text_extractions", 0),
                            "extraction_success": len(lab_df) > 0 or len(organ_df) > 0
                        }
                    }
                    
                    json_path = self.output_dir / f"{base_name}_results.json"
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(individual_result, f, indent=2, ensure_ascii=False)
                    
                all_lab_results.append(lab_df)
                all_organ_results.append(organ_df)
                
            except Exception as e:
                logger.error("Failed to process %s: %s", xml_file, e)
                processing_stats["failed_files"] += 1
                
                # Save error information for failed files when using individual JSON
                if save_individual_json:
                    base_name = Path(xml_file).stem
                    error_result = {
                        "pmid": base_name,
                        "file_path": xml_file,
                        "processing_timestamp": pd.Timestamp.now().isoformat(),
                        "error": str(e),
                        "file_stats": {"error": True},
                        "lab_tests": [],
                        "organ_injuries": [],
                        "summary": {
                            "total_lab_tests": 0,
                            "total_organ_injuries": 0,
                            "unique_drugs": [],
                            "tables_processed": 0,
                            "text_sections_processed": 0,
                            "extraction_success": False,
                            "processing_failed": True
                        }
                    }
                    
                    json_path = self.output_dir / f"{base_name}_results.json"
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(error_result, f, indent=2, ensure_ascii=False)
                        
                continue

        # Combine results
        combined_lab_df = pd.concat(all_lab_results, ignore_index=True) if all_lab_results else pd.DataFrame()
        combined_organ_df = pd.concat(all_organ_results, ignore_index=True) if all_organ_results else pd.DataFrame()
        
        # Save combined results
        combined_lab_df.to_csv(self.output_dir / "combined_lab_results.csv", index=False)
        combined_organ_df.to_csv(self.output_dir / "combined_organ_results.csv", index=False)
        
        # Save processing statistics
        stats_df = pd.DataFrame([processing_stats])
        stats_df.to_csv(self.output_dir / "processing_statistics.csv", index=False)
        
        logger.info(
            "Enhanced extraction complete. Lab tests: %d, Organ injuries: %d. "
            "Stats: %d table extractions, %d text extractions, %d failed files",
            len(combined_lab_df), len(combined_organ_df),
            processing_stats["table_extractions"], 
            processing_stats["text_extractions"],
            processing_stats["failed_files"]
        )
        
        return combined_lab_df, combined_organ_df

    def _detect_xml_format(self, xml_path: str) -> str:
        """Detect whether XML is BioC format or standard JATS/PMC format"""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Check for BioC format markers
            if root.tag == 'collection' and root.find('.//document') is not None:
                return 'bioc'
            # Check for JATS/PMC format markers
            elif 'article' in root.tag.lower() or root.find('.//body') is not None:
                return 'jats'
            else:
                return 'unknown'
        except Exception as e:
            logger.warning("Failed to detect XML format for %s: %s", xml_path, e)
            return 'unknown'

    def _get_parser(self, xml_path: str):
        """Get appropriate parser based on XML format"""
        format_type = self._detect_xml_format(xml_path)
        logger.debug("Detected XML format for %s: %s", xml_path, format_type)
        
        if format_type == 'bioc':
            return BioCParser(xml_path)
        else:
            # Default to PMC parser for JATS and unknown formats
            return PMCXMLParser(xml_path)

    def _process_pmc_file_enhanced(self, xml_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
        """Enhanced processing with separate table and text workflows"""
        parser = self._get_parser(xml_path)
        pmid = parser.pmid
        all_lab_results: List[Dict[str, Any]] = []
        all_organ_results: List[Dict[str, Any]] = []
        file_stats = {"table_extractions": 0, "text_extractions": 0}

        # Priority 1: Process tables with specialized table extractor
        tables = parser.extract_tables()
        logger.debug("%s: found %d tables", xml_path, len(tables))
        
        for table in tables:
            try:
                # Use specialized table extractor
                lab_results, organ_results = self.table_extractor.extract_from_table(
                    table['content'], 
                    table['caption'], 
                    pmid, 
                    f"table_{table['id']}"
                )
                
                if lab_results or organ_results:
                    file_stats["table_extractions"] += 1
                    all_lab_results.extend(asdict(r) for r in lab_results)
                    all_organ_results.extend(asdict(r) for r in organ_results)
                    logger.debug("Table %s: extracted %d lab tests, %d organ injuries", 
                               table['id'], len(lab_results), len(organ_results))
                    
            except Exception as e:
                logger.warning("Table extraction failed for %s: %s", table['id'], e)
                continue

        # Priority 2: Process high-value body sections with text extractor
        sections = parser.extract_body_sections()
        logger.debug("%s: found %d body sections", xml_path, len(sections))
        
        # Focus on results, methods, and discussion sections
        high_value_keywords = [
            'result', 'finding', 'outcome', 'toxicity', 'safety', 'adverse', 'effect',
            'method', 'material', 'procedure', 'protocol', 'analysis',
            'discussion', 'conclusion', 'implication'
        ]
        
        processed_sections = 0
        for i, section in enumerate(sections):
            try:
                # Handle different section formats
                if isinstance(section, dict):
                    section_title = section.get('title', f'Section_{i}')
                    section_content = section.get('content', '')
                elif isinstance(section, str):
                    # BioC parser returns strings, extract title from markdown headers
                    lines = section.split('\n')
                    section_title = lines[0] if lines else f'Section_{i}'
                    if section_title.startswith('##'):
                        section_title = section_title.replace('##', '').strip()
                    section_content = '\n'.join(lines[1:]) if len(lines) > 1 else section
                else:
                    continue
                    
                section_title_lower = section_title.lower()
                
                # Process high-value sections
                if any(keyword in section_title_lower for keyword in high_value_keywords):
                    try:
                        relevance = self.text_extractor.assess_relevance(section_content, "text")
                        if relevance.get("is_relevant", False) and relevance.get("relevance_score", 0) > 40:
                            lab_results, organ_results = self.text_extractor.extract_data(
                                section_content, pmid, f"section_{section_title}"
                            )
                            
                            if lab_results or organ_results:
                                file_stats["text_extractions"] += 1
                                all_lab_results.extend(asdict(r) for r in lab_results)
                                all_organ_results.extend(asdict(r) for r in organ_results)
                                processed_sections += 1
                                
                    except Exception as e:
                        logger.debug("Section extraction failed for %s: %s", section_title, e)
                        continue
            except Exception as e:
                logger.debug("Section processing failed: %s", e)
                continue

        # Priority 3: Fallback processing if minimal data extracted
        if len(all_lab_results) + len(all_organ_results) < 3:
            logger.debug("%s: minimal data extracted, trying fallback methods", xml_path)
            
            # Try full body text
            full_body = parser.extract_full_body_text()
            if full_body:
                try:
                    relevance = self.text_extractor.assess_relevance(full_body, "text")
                    if relevance.get("is_relevant", False) and relevance.get("relevance_score", 0) > 30:
                        lab_results, organ_results = self.text_extractor.extract_data(
                            full_body, pmid, "section_full_body"
                        )
                        if lab_results or organ_results:
                            all_lab_results.extend(asdict(r) for r in lab_results)
                            all_organ_results.extend(asdict(r) for r in organ_results)
                            file_stats["text_extractions"] += 1
                except Exception as e:
                    logger.debug("Full body extraction failed: %s", e)

            # Ultimate fallback: paragraph text
            if len(all_lab_results) + len(all_organ_results) == 0:
                para_text = parser.extract_all_paragraph_text()
                if para_text:
                    try:
                        relevance = self.text_extractor.assess_relevance(para_text, "text")
                        if relevance.get("is_relevant", False) and relevance.get("relevance_score", 0) > 25:
                            lab_results, organ_results = self.text_extractor.extract_data(
                                para_text, pmid, "section_paragraphs"
                            )
                            if lab_results or organ_results:
                                all_lab_results.extend(asdict(r) for r in lab_results)
                                all_organ_results.extend(asdict(r) for r in organ_results)
                                file_stats["text_extractions"] += 1
                    except Exception as e:
                        logger.debug("Paragraph extraction failed: %s", e)

        # Create DataFrames
        lab_df = pd.DataFrame(all_lab_results)
        organ_df = pd.DataFrame(all_organ_results)
        
        # Data quality enhancement
        lab_df = self._enhance_data_quality(lab_df, "lab")
        organ_df = self._enhance_data_quality(organ_df, "organ")
        
        logger.debug("%s: final results - %d lab tests, %d organ injuries", 
                    xml_path, len(lab_df), len(organ_df))
        
        return lab_df, organ_df, file_stats

    def _enhance_data_quality(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Enhance data quality through cleaning and standardization"""
        if df.empty:
            return df
            
        # Standardize drug names
        if 'drug' in df.columns:
            df['drug'] = df['drug'].str.strip()
            df['drug'] = df['drug'].replace('Unknown', 'Unknown compound')
            
        # Standardize species names
        if 'species' in df.columns:
            species_mapping = {
                'mice': 'mouse',
                'rats': 'rat', 
                'rabbits': 'rabbit',
                'dogs': 'dog',
                'icr mice': 'ICR mouse',
                'icr-mice': 'ICR mouse',
                'wistar rats': 'Wistar rat',
                'sprague-dawley rats': 'Sprague-Dawley rat'
            }
            df['species'] = df['species'].str.lower().map(species_mapping).fillna(df['species'])
            
        # Standardize lab test names for lab data
        if data_type == "lab" and 'lab_test' in df.columns:
            lab_mapping = {
                'alt': 'ALT',
                'ast': 'AST', 
                'alp': 'ALP',
                'alkaline phosphatase': 'ALP',
                'creatine kinase': 'CK',
                'ck': 'CK'
            }
            df['lab_test'] = df['lab_test'].str.lower().map(lab_mapping).fillna(df['lab_test'])
            
        # Remove duplicate entries
        if data_type == "lab":
            df = df.drop_duplicates(subset=['pmid', 'drug', 'dose', 'lab_test', 'time_point'], keep='first')
        else:
            df = df.drop_duplicates(subset=['pmid', 'drug', 'dose', 'injury_type', 'time_point'], keep='first')
            
        return df

    def validate_results_enhanced(self, lab_df: pd.DataFrame, organ_df: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced validation with more comprehensive metrics"""
        metrics: Dict[str, Any] = {
            "total_lab_tests": len(lab_df),
            "total_organ_injuries": len(organ_df),
            "unique_drugs": set(),
            "unique_pmids": set(),
            "unique_species": set(),
            "data_quality_metrics": {},
            "extraction_coverage": {},
            "overall_quality_score": 0,
        }

        # Basic counts and unique values
        if not lab_df.empty:
            metrics["unique_drugs"].update(lab_df.get("drug", []).dropna().unique().tolist())
            metrics["unique_pmids"].update(lab_df.get("pmid", []).dropna().unique().tolist())
            metrics["unique_species"].update(lab_df.get("species", []).dropna().unique().tolist())
            
        if not organ_df.empty:
            metrics["unique_drugs"].update(organ_df.get("drug", []).dropna().unique().tolist())
            metrics["unique_pmids"].update(organ_df.get("pmid", []).dropna().unique().tolist())
            metrics["unique_species"].update(organ_df.get("species", []).dropna().unique().tolist())

        # Convert sets to lists for JSON serialization
        metrics["unique_drugs"] = list(metrics["unique_drugs"])
        metrics["unique_pmids"] = list(metrics["unique_pmids"])
        metrics["unique_species"] = list(metrics["unique_species"])

        # Data quality metrics
        lab_quality = self._calculate_data_quality(lab_df)
        organ_quality = self._calculate_data_quality(organ_df)
        
        metrics["data_quality_metrics"] = {
            "lab_tests": lab_quality,
            "organ_injuries": organ_quality
        }
        
        # Extraction coverage metrics
        metrics["extraction_coverage"] = {
            "files_with_lab_data": len(lab_df['pmid'].unique()) if not lab_df.empty else 0,
            "files_with_organ_data": len(organ_df['pmid'].unique()) if not organ_df.empty else 0,
            "total_unique_files": len(set(
                (lab_df['pmid'].unique().tolist() if not lab_df.empty else []) +
                (organ_df['pmid'].unique().tolist() if not organ_df.empty else [])
            ))
        }
        
        # Overall quality score (0-100)
        quality_factors = []
        if lab_quality.get("completeness", 0) > 0:
            quality_factors.append(lab_quality["completeness"])
        if organ_quality.get("completeness", 0) > 0:
            quality_factors.append(organ_quality["completeness"])
        if metrics["extraction_coverage"]["total_unique_files"] > 0:
            coverage_score = (
                (metrics["extraction_coverage"]["files_with_lab_data"] + 
                 metrics["extraction_coverage"]["files_with_organ_data"]) / 
                (2 * metrics["extraction_coverage"]["total_unique_files"])
            ) * 100
            quality_factors.append(coverage_score)
            
        metrics["overall_quality_score"] = round(sum(quality_factors) / len(quality_factors), 2) if quality_factors else 0
        
        return metrics
    
    def _calculate_data_quality(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate data quality metrics for a DataFrame"""
        if df.empty:
            return {"completeness": 0, "numerical_data_ratio": 0}
            
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isna().sum().sum()
        completeness = round(100 * (1 - missing_cells / total_cells), 2)
        
        # Calculate ratio of entries with numerical data
        numerical_ratio = 0
        if 'value_mean' in df.columns:
            numerical_entries = df[df['value_mean'] > 0].shape[0]
            numerical_ratio = round(100 * numerical_entries / len(df), 2) if len(df) > 0 else 0
        elif 'frequency' in df.columns:
            numerical_entries = df[df['frequency'] > 0].shape[0]
            numerical_ratio = round(100 * numerical_entries / len(df), 2) if len(df) > 0 else 0
            
        return {
            "completeness": completeness,
            "numerical_data_ratio": numerical_ratio
        }