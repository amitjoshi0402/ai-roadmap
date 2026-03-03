import os
import time
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI


@dataclass
class RunConfig:
    model: str = "gpt-4.1-mini"
    temperature: float = 0.2
    max_output_tokens: int = 250
    instructions: Optional[str] = None  # "system" style guidance


def run_prompt(client: OpenAI, prompt: str, cfg: RunConfig) -> str:
    resp = client.responses.create(
        model=cfg.model,
        input=prompt,
        instructions=cfg.instructions,
        temperature=cfg.temperature,
        max_output_tokens=cfg.max_output_tokens,
    )
    return resp.output_text.strip()


def print_block(title: str, text: str):
    line = "-" * 80
    print(f"\n{line}\n{title}\n{line}\n{text}\n")


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found. Did you create .env and activate venv?")

    client = OpenAI(api_key=api_key)

    prompt = "Explain what an API is in 2 sentences."

    configs = [
        RunConfig(
            temperature=0.0,
            instructions="You are a concise technical writer. Use plain English.",
        ),
        RunConfig(
            temperature=0.8,
            instructions="You are a friendly teacher. Use a simple analogy.",
        ),
        RunConfig(
            temperature=0.2,
            max_output_tokens=60,
            instructions="Write as bullet points only. No intro line.",
        ),
    ]

    for i, cfg in enumerate(configs, start=1):
        t0 = time.time()
        out = run_prompt(client, prompt, cfg)
        dt_ms = int((time.time() - t0) * 1000)

        print_block(
            f"Run #{i} | model={cfg.model} | temp={cfg.temperature} | max_out={cfg.max_output_tokens} | {dt_ms}ms",
            out,
        )


if __name__ == "__main__":
    main()
