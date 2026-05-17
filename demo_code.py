import os
import json
import argparse
import textwrap
import hashlib
import re
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


def build_roses_system_prompt() -> str:
    # ROSES: Role, Objective, Steps, Examples, Style
    return textwrap.dedent("""
        ROLE: You are a requirements engineer specializing in semantic relations between requirements.
        OBJECTIVE: Given two requirement texts (Text1 and Text2), classify their relation.
        STEPS:
        1) Read Text1 and Text2 carefully.
        2) Decide if they are semantically the SAME requirement (Duplicate), CONTRADICT each other (Conflict),
           or are unrelated/compatible (Neutral).
        3) Output exactly one label from: Conflict, Duplicate, Neutral.
        EXAMPLES (label only):
        - Text1: 'The UAV shall send the Pilot real-time information about malfunctions that impact the mission.'
          + Text2: 'The UAV flight range shall exceed 20 miles.' -> Neutral
        - Text1: 'Up to 4 viewers' + Text2: 'Up to 2 viewers' -> Conflict
        - Text1: 'secure connection required' + Text2: 'communications must be secure against hacking' -> Duplicate
        STYLE: Output ONLY the single label, no punctuation, no extra words.
        """).strip()


def _sanitize_for_filename(value: str) -> str:
    value = value.strip()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", value)
    return value[:180] if len(value) > 180 else value


def _prompt_id(system_prompt: str) -> str:
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:12]


def load_processed_indices(path: str) -> set[int]:
    processed: set[int] = set()
    if not os.path.exists(path):
        return processed
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("row_index"), int):
                processed.add(obj["row_index"])
    return processed


def _call_openai(
    *, api_key: str, model: str, system_prompt: str, text1: str, text2: str
) -> str:
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Text1: {text1}\n" f"Text2: {text2}\n\n" "Return the label now."
                ),
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()


def _call_litellm(*, model: str, system_prompt: str, text1: str, text2: str) -> str:
    # LiteLLM uses OpenAI-style messages and returns OpenAI-style responses.
    from litellm import completion  # type: ignore

    if "/" not in model:
        # Common case: use OpenAI via LiteLLM.
        model = f"openai/{model}"

    response: Any = completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Text1: {text1}\n" f"Text2: {text2}\n\n" "Return the label now."
                ),
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify requirement pairs with an LLM"
    )
    parser.add_argument(
        "--backend",
        choices=["openai", "litellm"],
        default="openai",
        help="LLM backend to use (default: openai)",
    )
    parser.add_argument(
        "--csv", default="train.csv", help="Input CSV path (default: train.csv)"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional explicit output JSONL path (overrides --pred-dir)",
    )
    parser.add_argument(
        "--pred-dir",
        default="predictions",
        help="Directory to store per-model/per-prompt prediction files (default: predictions)",
    )
    parser.add_argument(
        "--model", default="gpt-4o", help="Model name (default: gpt-4o)"
    )
    parser.add_argument(
        "--prompt-name",
        default="roses_v1",
        help="Logical prompt name used in filenames/metadata (default: roses_v1)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of rows from CSV to process (default: 3)",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    df = (
        pd.read_csv(args.csv, usecols=["Text1", "Text2", "Class"])
        .head(args.limit)
        .reset_index(drop=True)
    )

    system_prompt = build_roses_system_prompt()
    prompt_id = _prompt_id(system_prompt)

    csv_name = Path(args.csv).name
    backend_safe = _sanitize_for_filename(args.backend)
    model_safe = _sanitize_for_filename(args.model)
    prompt_name_safe = _sanitize_for_filename(args.prompt_name)

    pred_dir = Path(args.pred_dir)
    prompts_dir = pred_dir / "prompts"
    pred_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = (
            pred_dir
            / f"{csv_name}__{backend_safe}__{model_safe}__{prompt_name_safe}__{prompt_id}.jsonl"
        )

    prompt_path = prompts_dir / f"{prompt_name_safe}__{prompt_id}.txt"
    if not prompt_path.exists():
        prompt_path.write_text(system_prompt + "\n", encoding="utf-8")

    processed = load_processed_indices(str(out_path))

    with open(out_path, "a", encoding="utf-8") as out:
        for row_index in range(len(df)):
            row = df.iloc[row_index]
            if row_index in processed:
                continue

            text1 = str(row["Text1"]) if row["Text1"] is not None else ""
            text2 = str(row["Text2"]) if row["Text2"] is not None else ""
            true_label = str(row["Class"]) if row["Class"] is not None else ""

            if args.backend == "openai":
                if not api_key:
                    raise RuntimeError(
                        "OPENAI_API_KEY is missing (required for --backend openai)."
                    )
                pred = _call_openai(
                    api_key=api_key,
                    model=args.model,
                    system_prompt=system_prompt,
                    text1=text1,
                    text2=text2,
                )
            else:
                pred = _call_litellm(
                    model=args.model,
                    system_prompt=system_prompt,
                    text1=text1,
                    text2=text2,
                )

            record = {
                "source_csv": args.csv,
                "row_index": row_index,
                "text1": text1,
                "text2": text2,
                "true_label": true_label,
                "pred_label": pred,
                "backend": args.backend,
                "model": args.model,
                "prompt_name": args.prompt_name,
                "prompt_id": prompt_id,
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            os.fsync(out.fileno())

            print(f"row={row_index} true={true_label} pred={pred}")


if __name__ == "__main__":
    main()
