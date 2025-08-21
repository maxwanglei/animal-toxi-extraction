import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from .extractor import ToxicityDataExtractor
from .parsers.pmc_xml import PMCXMLParser

logger = logging.getLogger(__name__)


class ToxicityExtractionPipeline:
    """Pipeline for batch processing multiple PMC files"""

    def __init__(self, llm_provider, output_dir: str = "./output", max_tokens: int = 2048):
        self.extractor = ToxicityDataExtractor(llm_provider, max_tokens=max_tokens)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def process_batch(self, xml_files: List[str], save_intermediate: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        start_time = time.perf_counter()
        all_lab_results = []
        all_organ_results = []
        file_timings: List[Dict[str, Any]] = []
        for i, xml_file in enumerate(xml_files, 1):
            logger.info("Processing file %d/%d: %s", i, len(xml_files), xml_file)
            t0 = time.perf_counter()
            base_name = Path(xml_file).stem
            lab_df, organ_df = self._process_pmc_file(xml_file)
            if save_intermediate:
                lab_df.to_csv(self.output_dir / f"{base_name}_lab_results.csv", index=False)
                organ_df.to_csv(self.output_dir / f"{base_name}_organ_results.csv", index=False)
            all_lab_results.append(lab_df)
            all_organ_results.append(organ_df)
            dt = time.perf_counter() - t0
            file_timings.append({
                "pmid": base_name,
                "file_path": xml_file,
                "seconds": round(dt, 3),
                "lab_rows": int(len(lab_df)),
                "organ_rows": int(len(organ_df)),
            })

        combined_lab_df = pd.concat(all_lab_results, ignore_index=True) if all_lab_results else pd.DataFrame()
        combined_organ_df = pd.concat(all_organ_results, ignore_index=True) if all_organ_results else pd.DataFrame()
        combined_lab_df.to_csv(self.output_dir / "combined_lab_results.csv", index=False)
        combined_organ_df.to_csv(self.output_dir / "combined_organ_results.csv", index=False)
        total_time = time.perf_counter() - start_time
        # Save per-file timings
        pd.DataFrame(file_timings).to_csv(self.output_dir / "file_timings.csv", index=False)
        files_per_min = round((len(xml_files) / total_time) * 60.0, 2) if total_time > 0 else 0.0
        # Save processing statistics for consistency with enhanced pipeline
        try:
            _stats = [{
                "total_files": len(xml_files),
                "start_time": None,
                "end_time": None,
                "total_wall_time_sec": round(total_time, 3),
                "avg_time_per_file_sec": round(total_time / len(xml_files), 3) if len(xml_files) else 0.0,
                "files_per_min": files_per_min,
                "table_extractions": None,
                "text_extractions": None,
                "failed_files": None,
            }]
            pd.DataFrame(_stats).to_csv(self.output_dir / "processing_statistics.csv", index=False)
        except Exception:
            pass
        logger.info(
            "Extraction complete in %.2fs (%.2f files/min). Lab tests: %d, Organ injuries: %d",
            total_time,
            files_per_min,
            len(combined_lab_df),
            len(combined_organ_df),
        )
        return combined_lab_df, combined_organ_df

    def _process_pmc_file(self, xml_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        parser = PMCXMLParser(xml_path)
        pmid = parser.pmid
        all_lab_results: List[Dict[str, Any]] = []
        all_organ_results: List[Dict[str, Any]] = []

        # Process tables
        tables = parser.extract_tables()
        logger.debug("%s: found %d tables", xml_path, len(tables))
        for table in tables:
            full_content = (
                f"Table {table['label']}\nCaption: {table['caption']}\n{table['content']}\nFooter: {table['footer']}"
            )
            relevance = self.extractor.assess_relevance(full_content, "table")
            if relevance.get("is_relevant", False) and relevance.get("relevance_score", 0) > 50:
                lab_results, organ_results = self.extractor.extract_data(
                    full_content, pmid, f"table_{table['id']}"
                )
                all_lab_results.extend(asdict(r) for r in lab_results)
                all_organ_results.extend(asdict(r) for r in organ_results)

        # Process body sections
        sections = parser.extract_body_sections()
        logger.debug("%s: found %d body sections", xml_path, len(sections))
        for section in sections:
            if any(k in section['title'].lower() for k in [
                'result', 'toxicity', 'safety', 'adverse', 'effect'
            ]):
                relevance = self.extractor.assess_relevance(section['content'], "text")
                if relevance.get("is_relevant", False) and relevance.get("relevance_score", 0) > 50:
                    lab_results, organ_results = self.extractor.extract_data(
                        section['content'], pmid, f"section_{section['title']}"
                    )
                    all_lab_results.extend(asdict(r) for r in lab_results)
                    all_organ_results.extend(asdict(r) for r in organ_results)

        # Fallback: if nothing extracted, try full body text once
        if not all_lab_results and not all_organ_results:
            full_body = parser.extract_full_body_text()
            logger.debug("%s: fallback full body length=%d", xml_path, len(full_body))
            if full_body:
                relevance = self.extractor.assess_relevance(full_body, "text")
                if relevance.get("is_relevant", False) and relevance.get("relevance_score", 0) > 50:
                    lab_results, organ_results = self.extractor.extract_data(
                        full_body, pmid, "section_full_body"
                    )
                    all_lab_results.extend(asdict(r) for r in lab_results)
                    all_organ_results.extend(asdict(r) for r in organ_results)
            else:
                # Second-chance fallback: concatenate all <p> texts
                para_text = parser.extract_all_paragraph_text()
                logger.debug("%s: paragraph fallback length=%d", xml_path, len(para_text))
                if para_text:
                    relevance = self.extractor.assess_relevance(para_text, "text")
                    if relevance.get("is_relevant", False) and relevance.get("relevance_score", 0) > 50:
                        lab_results, organ_results = self.extractor.extract_data(
                            para_text, pmid, "section_paragraphs"
                        )
                        all_lab_results.extend(asdict(r) for r in lab_results)
                        all_organ_results.extend(asdict(r) for r in organ_results)
                else:
                    # Ultimate fallback: entire document text
                    doc_text = parser.extract_document_text()
                    logger.debug("%s: document fallback length=%d", xml_path, len(doc_text))
                    if doc_text:
                        relevance = self.extractor.assess_relevance(doc_text, "text")
                        if relevance.get("is_relevant", False) and relevance.get("relevance_score", 0) > 50:
                            lab_results, organ_results = self.extractor.extract_data(
                                doc_text, pmid, "section_document"
                            )
                            all_lab_results.extend(asdict(r) for r in lab_results)
                            all_organ_results.extend(asdict(r) for r in organ_results)

        lab_df = pd.DataFrame(all_lab_results)
        organ_df = pd.DataFrame(all_organ_results)
        return lab_df, organ_df

    def validate_results(self, lab_df: pd.DataFrame, organ_df: pd.DataFrame) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "total_lab_tests": len(lab_df),
            "total_organ_injuries": len(organ_df),
            "unique_drugs": set(),
            "unique_pmids": set(),
            "missing_values": {},
            "data_quality_score": 0,
        }

        if not lab_df.empty:
            metrics["unique_drugs"].update(lab_df.get("drug", []).dropna().unique().tolist())
            metrics["unique_pmids"].update(lab_df.get("pmid", []).dropna().unique().tolist())
        if not organ_df.empty:
            metrics["unique_drugs"].update(organ_df.get("drug", []).dropna().unique().tolist())
            metrics["unique_pmids"].update(organ_df.get("pmid", []).dropna().unique().tolist())

        metrics["unique_drugs"] = list(metrics["unique_drugs"])
        metrics["unique_pmids"] = list(metrics["unique_pmids"])

        total_fields = (
            (len(lab_df.columns) * len(lab_df)) + (len(organ_df.columns) * len(organ_df))
        )
        if total_fields > 0:
            missing = lab_df.isna().sum().sum() + organ_df.isna().sum().sum()
            metrics["data_quality_score"] = round(100 * (1 - missing / total_fields), 2)
        return metrics
