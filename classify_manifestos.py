"""Classify manifesto quasi-sentences as left / neutral / right with the OpenAI API.

Part of the bachelor's thesis "How well does GPT reconstruct parties' left-right
positions compared to MARPOR's RILE?" (David Hinssen, Universitaet Tuebingen, 2025).

Input : a CSV with one (quasi-)sentence per row in the column `text_en`.
Output: the same rows plus a column `orientation` in {left, right, neutral}.

Sentences are sent to the model in batches; the model answers in JSON, which is
parsed back into a dataframe and merged onto the input by row id. Rows that could
not be classified are left empty for manual review rather than silently dropped.

Usage:
    export OPENAI_API_KEY=sk-...
    python classify_manifestos.py --input data/SPDGER2013.csv --output results/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from openai import OpenAI

# --------------------------------------------------------------------------- #
# Label definitions
#
# Condensed from MARPOR's 56 categories. Three design decisions behind this
# wording (see thesis section 4.7): the definitions are kept short so the model
# is not overwhelmed with detail; "neutral" is framed as a full third option
# rather than a fallback, which otherwise absorbs ambiguous sentences; and
# contested categories such as 305 "Political Authority" are excluded, since
# they are a documented source of error in RILE (Mikhaylov et al. 2012).
# --------------------------------------------------------------------------- #

DEFINITIONS = """LEFT
- Anti-imperialism | end colonial rule, self-determination
- Anti-war | reduce military spending, abolish conscription
- Commitment to peace | cease-fires, diplomatic negotiations
- International co-operation | empower UN, development aid, global governance, international courts
- Deepen democracy | citizen participation, separation of powers, rule of law, free elections
- Consumer protection | break monopolies, social market economy, antitrust
- State economic steering | price controls, minimum wage, rent caps, state-owned industry
- Comprehensive welfare state | universal healthcare, social housing, child-care, pensions
- Expand education | free universities, more teachers, school investment, lifelong learning
- Support organised labour | fair wages, collective bargaining, job security
- Social-justice ethos | equitable treatment, anti-discrimination, cultural diversity
- Reject patriotism/nationalism as guiding values

RIGHT
- Rearmament | higher defence spending, modernised forces
- Freedom from state coercion | broad deregulation
- Free-market capitalism | strong property rights, laissez-faire, entrepreneurship
- Pro-business incentives | lower corporate taxes, investment incentives, subsidies, venture capital
- Fiscal restraint | balanced budgets, sound currency, deficit reduction, debt brakes
- Limit social spending | curb welfare dependence, subsidiarity, means testing, shrink state
- Promote national pride | patriotism, national symbols, sovereignty, security-first doctrine
- Traditional morals/family values | religious principles, oppose abortion, protect marriage
- Tough law enforcement | more police, stricter sentencing, zero tolerance, expanded prisons

NEUTRAL
- references to specific countries or the EU (positive or negative)
- general statements on international cooperation
- proposals to keep or change the constitution
- calls for decentralisation or centralisation
- remarks on governmental efficiency or competence
- broad economic goals (growth, employment) without distributional stance
- Keynesian demand management (public or private)
- technology or infrastructure development plans
- environmental protection measures
- publicly funded cultural or leisure facilities"""


def build_messages(batch: pd.DataFrame, language: str, document_type: str, source: str) -> list[dict]:
    """Assemble the three-message prompt for one batch.

    The output format is JSON rather than a table or plain text: it survives
    parsing reliably and converts straight into a dataframe. Each line carries
    its row id so answers can be merged back even when the model reorders or
    drops rows.
    """
    system = (
        f"You are an expert in political-text classification. "
        f"Analyze {language} {document_type} from {source}. "
        "For each input sentence think step by step but do not show the thoughts. "
        "According to the definitions provided below, decide whether each sentence is "
        "'left', 'right', or 'neutral'. "
        "Each input line begins with an id. Return JSON: "
        '{"records":[{"id":"...","orientation":"left|right|neutral"}, ...]}.'
    )
    instructions = (
        DEFINITIONS
        + "\n\n"
        + '0001 || CURR: [TARGET] ... [/TARGET]\n'
        + '{"records":[{"id":"0001","orientation":"neutral"}]}\n'
        + f"You will now receive {len(batch)} lines. Return one record per line."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": instructions},
        {"role": "user", "content": "\n".join(batch["prompt_line"])},
    ]


def classify(
    documents: pd.DataFrame,
    client: OpenAI,
    model: str,
    batch_size: int,
    temperature: float,
    language: str,
    document_type: str,
    source: str,
) -> pd.DataFrame:
    """Send the corpus to the model batch by batch and collect the labels."""
    classification = pd.DataFrame(columns=["id", "orientation"])
    start_time = datetime.now()
    print("Start time:", start_time.strftime("%H:%M:%S"), flush=True)

    for i in range(0, len(documents), batch_size):
        batch = documents.iloc[i : i + batch_size]
        response = client.chat.completions.create(
            model=model,
            messages=build_messages(batch, language, document_type, source),
            temperature=temperature,
            max_completion_tokens=16384,
            response_format={"type": "json_object"},
        )
        choice = response.choices[0]
        print(
            f"batch {i}-{i + len(batch)}",
            "| finish_reason:", choice.finish_reason,
            "| prompt_tokens:", response.usage.prompt_tokens,
            "| completion_tokens:", response.usage.completion_tokens,
            flush=True,
        )

        try:
            records = json.loads(choice.message.content)["records"]
        except (json.JSONDecodeError, KeyError) as err:
            # A malformed batch costs at most `batch_size` rows: they stay
            # unlabelled and surface in the "not classified" count below.
            print(f"  JSON error in batch {i}-{i + len(batch)}: {err}", file=sys.stderr)
            print("  ", choice.message.content[:400], file=sys.stderr)
            continue

        classification = pd.concat(
            [classification, pd.DataFrame.from_records(records, columns=["id", "orientation"])],
            ignore_index=True,
        )

        done = round((i + len(batch)) / len(documents) * 100)
        print(f"  {done}% complete", flush=True)

    elapsed = datetime.now() - start_time
    print(f"Done in {elapsed}.", flush=True)
    return classification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="CSV with a `text_en` column")
    parser.add_argument("--output", default=Path("."), type=Path, help="directory for the result CSV")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model id")
    parser.add_argument("--batch-size", default=300, type=int, help="quasi-sentences per API call")
    parser.add_argument("--temperature", default=0.2, type=float, help="decoding temperature")
    parser.add_argument("--language", default="English")
    parser.add_argument("--document-type", default="(quasi)sentences")
    parser.add_argument("--source", default="political manifestos")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=api_key)

    documents = pd.read_csv(args.input, header=0, encoding="utf-8")
    if "text_en" not in documents.columns:
        sys.exit(f"{args.input} has no column `text_en` (found: {list(documents.columns)}).")

    documents["id"] = documents.index.astype(str)
    documents["prompt_line"] = documents["id"] + " || CURR: [TARGET] " + documents["text_en"] + " [/TARGET]"

    classification = classify(
        documents,
        client=client,
        model=args.model,
        batch_size=args.batch_size,
        temperature=args.temperature,
        language=args.language,
        document_type=args.document_type,
        source=args.source,
    )

    result = documents[["id", "text_en"]].merge(classification, on="id", how="left")
    result["orientation"] = result["orientation"].fillna("")

    missing = int((result["orientation"] == "").sum())
    print(
        f"{missing} sentences were not classified - left empty for manual review"
        if missing
        else "All sentences classified"
    )

    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / f"Reasoning{args.input.stem}.csv"
    result.to_csv(out_path, index=False)
    print("Written:", out_path)


if __name__ == "__main__":
    main()
