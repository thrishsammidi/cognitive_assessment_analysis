import os
import json
import time
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel
from typing import Literal

INPUT = "questions.csv"
PROMPT = "bloom_prompt.txt"
OUTPUT = "bloom_results.csv"

BATCH_SIZE = 40
MODEL = "gpt-5.6-terra"

client = OpenAI(api_key='')

class Classification(BaseModel):
    cognitive_process: Literal[
        "Remember", "Understand", "Apply",
        "Analyze", "Evaluate", "Create"
    ]
    knowledge_type: Literal[
        "Factual", "Conceptual",
        "Procedural", "Metacognitive"
    ]
    rationale: str
    evidence: str
    confidence: float


class Output(BaseModel):
    classifications: list[Classification]


def classify(questions, prompt):

    for attempt in range(3):
        try:
            response = client.beta.chat.completions.parse(
                model=MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Classify exactly {len(questions)} questions "
                            "in the same order:\n\n"
                            + json.dumps(questions, ensure_ascii=False)
                        )
                    }
                ],
                response_format=Output
            )

            result = response.choices[0].message.parsed

            if len(result.classifications) != len(questions):
                raise ValueError("Wrong number of classifications.")

            return result.classifications

        except Exception as e:
            print(f"Retry {attempt + 1}: {e}")
            time.sleep(3)

    raise RuntimeError("Batch failed after 3 attempts.")


# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------

df = pd.read_csv(INPUT)

with open(PROMPT, "r", encoding="utf-8") as f:
    prompt = f.read()

results = []

for start in range(0, len(df), BATCH_SIZE):

    batch = df.iloc[start:start + BATCH_SIZE]
    questions = batch["Question_Text"].astype(str).tolist()

    print(f"Processing {start + 1}-{start + len(batch)}...")

    classifications = classify(questions, prompt)

    for (_, row), result in zip(batch.iterrows(), classifications):

        results.append({
            **row.to_dict(),
            "Cognitive_Process": result.cognitive_process,
            "Knowledge_Type": result.knowledge_type,
            "Rationale": result.rationale,
            "Evidence": result.evidence,
            "Confidence": result.confidence
        })

pd.DataFrame(results).to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8"
)

print(f"\nDone. Saved to {OUTPUT}")
