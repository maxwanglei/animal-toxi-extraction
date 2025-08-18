"""
Enhanced CLI with options for both original and enhanced pipelines.
Includes new table-aware processing capabilities.
"""

import os
import json
import logging
from logging.handlers import RotatingFileHandler
import argparse

from toxicity_extraction.llm.providers import LocalLLMProvider, make_provider
from toxicity_extraction.pipeline import ToxicityExtractionPipeline
from toxicity_extraction.enhanced_pipeline import EnhancedToxicityExtractionPipeline


def setup_logging(log_file: str, level: str = "INFO") -> None:
    # Ensure directory exists
    log_dir = os.path.dirname(log_file) or "."
    os.makedirs(log_dir, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Avoid duplicate handlers if main() called multiple times
    root.handlers = []

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    root.addHandler(ch)

    # Rotating file handler (verbose)
    fh = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(fh)


def main():
    parser = argparse.ArgumentParser(
        description="Animal toxicity data extraction from PMC XML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with enhanced pipeline
  python -m scripts.enhanced_cli --xml data/*.xml --out ./output --enhanced

  # Use original pipeline for comparison
  python -m scripts.enhanced_cli --xml data/*.xml --out ./output_original

  # Enhanced pipeline with custom token limit
  python -m scripts.enhanced_cli --xml data/*.xml --out ./output --enhanced --max-tokens 4096

  # Meta Llama with enhanced table processing
  python -m scripts.enhanced_cli --provider meta-llama --model "Llama-4-Maverick-17B-128E-Instruct-FP8" \\
    --xml data/*.xml --out ./output --enhanced --api-key YOUR_KEY --base-url YOUR_URL
        """
    )
    
    # Required arguments
    parser.add_argument("--xml", nargs="+", required=True, help="Paths to PMC XML files")
    parser.add_argument("--out", default="./toxicity_output", help="Output directory")
    
    # Pipeline selection
    parser.add_argument(
        "--enhanced", 
        action="store_true", 
        help="Use enhanced pipeline with table-aware processing (recommended)"
    )
    
    # Model configuration
    parser.add_argument("--model-path", default="/path/to/llama-model", help="Local LLM model path")
    parser.add_argument(
        "--provider",
        choices=["meta-llama", "gemini", "vertex-ai", "local", "langchain"],
        default=os.environ.get("PROVIDER", "meta-llama"),
        help="LLM provider to use",
    )
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME"), help="Model name")
    parser.add_argument("--api-key", default=os.environ.get("LLAMA_API_KEY"), help="API key for the provider")
    parser.add_argument("--base-url", default=os.environ.get("LLAMA_BASE_URL"), help="Base URL for API")
    
    # Vertex AI specific
    parser.add_argument("--project-id", default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument("--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))
    
    # Gemini API specific
    parser.add_argument("--google-api-key", default=os.environ.get("GOOGLE_API_KEY"))
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature (0.0 for deterministic)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--frequency-penalty", type=float, default=1.0, help="Frequency penalty")
    parser.add_argument("--max-tokens", type=int, default=3072, help="Maximum tokens per request")
    
    # Processing options
    parser.add_argument("--save-intermediate", action="store_true", help="Save intermediate results per file")
    parser.add_argument("--save-individual-json", action="store_true", 
                       help="Save detailed JSON results for each paper (useful for SLURM array jobs)")
    parser.add_argument("--compare-pipelines", action="store_true", 
                       help="Run both pipelines and compare results (outputs to separate directories)")
    
    # Logging
    parser.add_argument("--log-file", default=None, help="Path to write logs; defaults to <out>/run.log")
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"), 
                       help="Console log level (INFO/DEBUG/WARNING/ERROR)")
    
    args = parser.parse_args()

    # Validation
    if args.provider != "local" and not args.model:
        parser.error("--model is required (or set MODEL_NAME) for remote providers (meta-llama, gemini, vertex-ai)")

    # Initialize logging
    log_file = args.log_file or os.path.join(args.out, "run.log")
    setup_logging(log_file, args.log_level)
    logger = logging.getLogger(__name__)
    logger.info("Logging to %s (level=%s)", log_file, args.log_level.upper())
    logger.info("Provider selected: %s", args.provider)
    if args.enhanced:
        logger.info("Using enhanced pipeline with table-aware processing")
    else:
        logger.info("Using original pipeline")

    # Build the appropriate provider
    provider = build_provider(args, parser)

    # Run pipeline(s)
    if args.compare_pipelines:
        run_comparison(args, provider, logger)
    else:
        run_single_pipeline(args, provider, logger)


def build_provider(args, parser):
    """Build the appropriate LLM provider based on arguments"""
    provider_name = args.provider
    provider_kwargs = {}
    
    if provider_name in ("meta-llama", "llama", "llama-meta"):
        api_key = args.api_key
        if not api_key:
            parser.error("--api-key or LLAMA_API_KEY env var is required for provider meta-llama")
        provider_kwargs = dict(
            model=args.model,
            api_key=api_key,
            base_url=args.base_url,
            temperature=args.temperature,
            top_p=args.top_p,
            frequency_penalty=args.frequency_penalty,
            max_tokens=args.max_tokens,
        )
    elif provider_name == "gemini":
        api_key = args.google_api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            parser.error("--google-api-key or GOOGLE_API_KEY env var is required for provider gemini")
        provider_kwargs = dict(
            api_key=api_key,
            model=args.model,
        )
    elif provider_name == "vertex-ai":
        if not args.project_id:
            parser.error("--project-id or GOOGLE_CLOUD_PROJECT env var is required for provider vertex-ai")
        provider_kwargs = dict(
            project_id=args.project_id,
            location=args.location,
            model=args.model,
        )
    elif provider_name == "langchain":
        api_key = args.api_key or os.environ.get("LLAMA_API_KEY")
        if not api_key:
            parser.error("--api-key or LLAMA_API_KEY env var is required for provider langchain")
        provider_kwargs = dict(
            api_key=api_key,
            model=args.model,
            base_url=args.base_url,
        )
    elif provider_name == "local":
        return LocalLLMProvider(model_path=args.model_path)
    else:
        provider_kwargs = dict(model=args.model)

    return make_provider(provider_name, **provider_kwargs)


def run_single_pipeline(args, provider, logger):
    """Run a single pipeline (either original or enhanced)"""
    if args.enhanced:
        pipeline = EnhancedToxicityExtractionPipeline(
            provider, 
            output_dir=args.out, 
            max_tokens=args.max_tokens
        )
        logger.info("Processing %d files with enhanced pipeline", len(args.xml))
        lab_df, organ_df = pipeline.process_batch(
            args.xml, 
            save_intermediate=args.save_intermediate,
            save_individual_json=args.save_individual_json
        )
        metrics = pipeline.validate_results_enhanced(lab_df, organ_df)
    else:
        pipeline = ToxicityExtractionPipeline(
            provider, 
            output_dir=args.out, 
            max_tokens=args.max_tokens
        )
        logger.info("Processing %d files with original pipeline", len(args.xml))
        lab_df, organ_df = pipeline.process_batch(args.xml, save_intermediate=args.save_intermediate)
        metrics = pipeline.validate_results(lab_df, organ_df)

    # Print results
    print(f"\\n{'='*60}")
    print(f"EXTRACTION RESULTS ({'Enhanced' if args.enhanced else 'Original'} Pipeline)")
    print(f"{'='*60}")
    print(f"Total lab tests extracted: {len(lab_df)}")
    print(f"Total organ injuries extracted: {len(organ_df)}")
    print(f"Unique drugs found: {len(metrics.get('unique_drugs', []))}")
    print(f"Files processed: {len(metrics.get('unique_pmids', []))}")
    
    if args.enhanced and 'overall_quality_score' in metrics:
        print(f"Overall quality score: {metrics['overall_quality_score']}/100")
        
    print("\\nDetailed metrics:")
    print(json.dumps(metrics, indent=2))
    
    # Save metrics
    metrics_file = os.path.join(args.out, "extraction_metrics.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_file)


def run_comparison(args, provider, logger):
    """Run both pipelines and compare results"""
    logger.info("Running comparison between original and enhanced pipelines")
    
    # Prepare output directories
    original_out = args.out + "_original"
    enhanced_out = args.out + "_enhanced"
    os.makedirs(original_out, exist_ok=True)
    os.makedirs(enhanced_out, exist_ok=True)
    
    # Run original pipeline
    logger.info("Running original pipeline...")
    original_pipeline = ToxicityExtractionPipeline(
        provider, 
        output_dir=original_out, 
        max_tokens=args.max_tokens
    )
    orig_lab_df, orig_organ_df = original_pipeline.process_batch(args.xml, save_intermediate=True)
    orig_metrics = original_pipeline.validate_results(orig_lab_df, orig_organ_df)
    
    # Run enhanced pipeline
    logger.info("Running enhanced pipeline...")
    enhanced_pipeline = EnhancedToxicityExtractionPipeline(
        provider, 
        output_dir=enhanced_out, 
        max_tokens=args.max_tokens
    )
    enh_lab_df, enh_organ_df = enhanced_pipeline.process_batch(
        args.xml, 
        save_intermediate=True,
        save_individual_json=args.save_individual_json
    )
    enh_metrics = enhanced_pipeline.validate_results_enhanced(enh_lab_df, enh_organ_df)
    
    # Compare results
    comparison = {
        "original_pipeline": {
            "lab_tests": len(orig_lab_df),
            "organ_injuries": len(orig_organ_df),
            "unique_drugs": len(orig_metrics.get('unique_drugs', [])),
            "data_quality_score": orig_metrics.get('data_quality_score', 0),
            "metrics": orig_metrics
        },
        "enhanced_pipeline": {
            "lab_tests": len(enh_lab_df),
            "organ_injuries": len(enh_organ_df),
            "unique_drugs": len(enh_metrics.get('unique_drugs', [])),
            "overall_quality_score": enh_metrics.get('overall_quality_score', 0),
            "metrics": enh_metrics
        },
        "improvements": {
            "lab_tests_increase": len(enh_lab_df) - len(orig_lab_df),
            "organ_injuries_increase": len(enh_organ_df) - len(orig_organ_df),
            "drugs_increase": len(enh_metrics.get('unique_drugs', [])) - len(orig_metrics.get('unique_drugs', [])),
        }
    }
    
    # Print comparison
    print(f"\\n{'='*80}")
    print("PIPELINE COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"{'Metric':<25} {'Original':<15} {'Enhanced':<15} {'Improvement':<15}")
    print(f"{'-'*80}")
    print(f"{'Lab tests':<25} {len(orig_lab_df):<15} {len(enh_lab_df):<15} {comparison['improvements']['lab_tests_increase']:<15}")
    print(f"{'Organ injuries':<25} {len(orig_organ_df):<15} {len(enh_organ_df):<15} {comparison['improvements']['organ_injuries_increase']:<15}")
    print(f"{'Unique drugs':<25} {len(orig_metrics.get('unique_drugs', [])):<15} {len(enh_metrics.get('unique_drugs', [])):<15} {comparison['improvements']['drugs_increase']:<15}")
    
    if 'overall_quality_score' in enh_metrics:
        quality_improvement = enh_metrics['overall_quality_score'] - orig_metrics.get('data_quality_score', 0)
        print(f"{'Quality score':<25} {orig_metrics.get('data_quality_score', 0):<15.1f} {enh_metrics['overall_quality_score']:<15.1f} {quality_improvement:<15.1f}")
    
    print("\\nDetailed comparison:")
    print(json.dumps(comparison, indent=2))
    
    # Save comparison
    comparison_file = os.path.join(args.out, "pipeline_comparison.json")
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2)
    logger.info("Comparison results saved to %s", comparison_file)
    
    # Determine recommendation
    total_improvement = (comparison['improvements']['lab_tests_increase'] + 
                        comparison['improvements']['organ_injuries_increase'])
    
    print(f"\\n{'='*80}")
    if total_improvement > 0:
        print("RECOMMENDATION: Enhanced pipeline shows significant improvements!")
        print(f"Total additional extractions: {total_improvement}")
    elif total_improvement == 0:
        print("RECOMMENDATION: Both pipelines performed similarly.")
    else:
        print("RECOMMENDATION: Original pipeline performed slightly better.")
        print("Consider adjusting enhanced pipeline parameters.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()