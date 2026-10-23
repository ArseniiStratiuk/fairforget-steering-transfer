"""Classify saved answers with the pinned Jev choice API."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter


class JevJudge:
    requires_greedy_replay = False

    def __init__(self, revision, rubric, api_key=None):
        from typesafe_sdk import TypeSafeClient

        criteria = rubric.get("criteria")
        if not criteria or set(criteria) != set(rubric["labels"]):
            raise ValueError("Jev requires a description for every label")
        key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise ValueError("Set TYPESAFE_API_KEY before loading Jev")
        self.client = TypeSafeClient(api_key=key, model=revision, timeout=60)
        self.rubric = rubric
        self.instruction = rubric["instruction"]
        if rubric.get("examples"):
            examples = json.dumps(rubric["examples"], ensure_ascii=True)
            self.instruction += "\nLabeled examples:\n" + examples
        self.specification = {
            "provider": "typesafe",
            "model": revision,
            "rubric": rubric,
            "sdk": "typesafe-sdk==0.7.5",
        }

    def judge_batch(self, records):
        with ThreadPoolExecutor(max_workers=min(4, max(1, len(records)))) as workers:
            return list(workers.map(self.judge_response, records))

    def judge_response(self, record):
        from typesafe_sdk import Choice

        started = perf_counter()
        result = self.client.system_one(
            state={"question": record["prompt"], "answer": record["response"]},
            questions={
                "judgment": Choice(
                    instructions=self.instruction,
                    criteria=self.rubric["criteria"],
                )
            },
        )
        decision = result.choices["judgment"]
        if decision.choice not in self.rubric["labels"]:
            raise ValueError("Jev returned a label outside the rubric")
        return {
            "label": decision.choice,
            "status": "success",
            "probabilities": decision.probabilities,
            "confidence": decision.confidence,
            "model": result.model,
            "request_id": result.request_id,
            "usage": result.usage.model_dump(),
            "latency_seconds": perf_counter() - started,
        }

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.client.close()
