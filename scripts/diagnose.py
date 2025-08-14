import os
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("diagnose")


def diagnose_meta_llama(model: str, base_url: str | None, api_key: str | None) -> int:
    try:
        from openai import OpenAI
    except Exception as e:
        logger.error("openai package not installed: %s", e)
        return 1

    api_key = api_key or os.environ.get("LLAMA_API_KEY")
    base_url = base_url or os.environ.get("LLAMA_BASE_URL", "https://api.llama.com/compat/v1/")
    if not api_key:
        logger.error("LLAMA_API_KEY is not set")
        return 2
    if not model:
        logger.error("--model (MODEL_NAME) is required")
        return 2

    client = OpenAI(api_key=api_key, base_url=base_url)
    logger.info("Sending test request: base_url=%s model=%s", base_url, model)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a concise assistant."},
                  {"role": "user", "content": "Reply with the word PONG."}],
        max_tokens=8,
        temperature=0.0,
    )
    text = resp.choices[0].message.content
    usage = getattr(resp, "usage", None)
    logger.info("Response id=%s text=%r", getattr(resp, "id", None), text)
    if usage:
        logger.info(
            "Usage prompt_tokens=%s completion_tokens=%s total_tokens=%s",
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
            getattr(usage, "total_tokens", None),
        )
    else:
        logger.info("No usage field present on response")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Diagnose provider connectivity and accounting")
    parser.add_argument("--provider", required=True, choices=["meta-llama", "gemini", "vertex-ai"], help="Which provider to test")
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME"))
    parser.add_argument("--base-url", default=os.environ.get("LLAMA_BASE_URL"))
    parser.add_argument("--google-api-key", default=os.environ.get("GOOGLE_API_KEY"))
    parser.add_argument("--project-id", default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument("--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))
    args = parser.parse_args()

    if args.provider == "meta-llama":
        code = diagnose_meta_llama(args.model, args.base_url, os.environ.get("LLAMA_API_KEY"))
        raise SystemExit(code)

    # Stubs for other providers if needed later
    if args.provider == "gemini":
        try:
            import google.generativeai as genai
        except Exception as e:
            logger.error("google-generativeai not installed: %s", e)
            raise SystemExit(1)
        if not args.google_api_key:
            logger.error("--google-api-key or GOOGLE_API_KEY is required for gemini")
            raise SystemExit(2)
        genai.configure(api_key=args.google_api_key)
        model = genai.GenerativeModel(args.model or "gemini-1.5-pro")
        logger.info("Sending Gemini test request: model=%s", args.model)
        resp = model.generate_content("Reply with the word PONG.")
        logger.info("Gemini text=%r", getattr(resp, "text", None))
        raise SystemExit(0)

    if args.provider == "vertex-ai":
        try:
            from google.cloud import aiplatform
            from vertexai.generative_models import GenerativeModel
        except Exception as e:
            logger.error("google-cloud-aiplatform/vertexai not installed: %s", e)
            raise SystemExit(1)
        if not args.project_id:
            logger.error("--project-id or GOOGLE_CLOUD_PROJECT is required for vertex-ai")
            raise SystemExit(2)
        aiplatform.init(project=args.project_id, location=args.location)
        model = GenerativeModel(args.model or "gemini-1.5-pro")
        logger.info("Sending Vertex AI test request: project=%s location=%s model=%s", args.project_id, args.location, args.model)
        resp = model.generate_content("Reply with the word PONG.")
        logger.info("Vertex text=%r", getattr(resp, "text", None))
        raise SystemExit(0)


if __name__ == "__main__":
    main()
