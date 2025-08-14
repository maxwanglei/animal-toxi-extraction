import os
import json
import logging
from logging.handlers import RotatingFileHandler
import argparse

from toxicity_extraction.llm.providers import LocalLLMProvider, make_provider
from toxicity_extraction.pipeline import ToxicityExtractionPipeline

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
    parser = argparse.ArgumentParser(description="Animal toxicity data extraction from PMC XML")
    parser.add_argument("--xml", nargs="+", required=True, help="Paths to PMC XML files")
    parser.add_argument("--out", default="./toxicity_output", help="Output directory")
    parser.add_argument("--model-path", default="/path/to/llama-model", help="Local LLM model path")
    parser.add_argument("--provider", default=os.environ.get("PROVIDER", "meta-llama"))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME"))
    parser.add_argument("--base-url", default=os.environ.get("LLAMA_BASE_URL"))
    # Vertex AI specific
    parser.add_argument("--project-id", default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument("--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))
    # Gemini API specific
    parser.add_argument("--google-api-key", default=os.environ.get("GOOGLE_API_KEY"))
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--frequency-penalty", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--log-file", default=None, help="Path to write logs; defaults to <out>/run.log")
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"), help="Console log level (INFO/DEBUG/...) ")
    args = parser.parse_args()

    if args.provider != "local" and not args.model:
        parser.error("--model is required (or set MODEL_NAME) for remote providers (meta-llama, gemini, vertex-ai)")

    # Initialize logging as early as possible
    log_file = args.log_file or os.path.join(args.out, "run.log")
    setup_logging(log_file, args.log_level)
    logging.getLogger(__name__).info("Logging to %s (level=%s)", log_file, args.log_level.upper())

    # Build the appropriate provider with the right credentials
    provider_name = args.provider
    provider_kwargs = {}
    if provider_name in ("meta-llama", "llama", "llama-meta"):
        provider_kwargs = dict(
            model=args.model,
            api_key=os.environ.get("LLAMA_API_KEY"),
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
    elif provider_name == "local":
        provider_kwargs = dict(model_path=args.model_path)
    else:
        # Pass through generic fields if any custom providers are added later
        provider_kwargs = dict(model=args.model)

    provider = make_provider(provider_name, **provider_kwargs)

    # If user explicitly selects local, use LocalLLMProvider; otherwise use the constructed provider
    llm = provider if provider_name != "local" else LocalLLMProvider(model_path=args.model_path)
    pipeline = ToxicityExtractionPipeline(llm, output_dir=args.out)
    lab_df, organ_df = pipeline.process_batch(args.xml)

    metrics = pipeline.validate_results(lab_df, organ_df)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
