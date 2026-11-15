"""Exchange judge requests through shared storage when GPUs have no network."""

import json
import time
from pathlib import Path

from .result_storage import content_hash, write_json


class QueuedJudge:
    def __init__(self, judge, directory, timeout_seconds=600):
        self.specification = judge.specification
        self.directory = Path(directory)
        self.timeout_seconds = timeout_seconds

    def judge_all(self, record, rubrics):
        request = {
            "record": record,
            "rubrics": rubrics,
            "specification": self.specification,
        }
        identity = content_hash(request)
        request_path = self.directory / "requests" / f"{identity}.json"
        response_path = self.directory / "responses" / f"{identity}.json"
        write_json(request_path, request)
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            if response_path.exists():
                response = json.loads(response_path.read_text())
                if response["request_hash"] != identity:
                    raise ValueError("Queued judge response identity differs")
                if "error" in response:
                    raise RuntimeError(response["error"])
                return response["result"]
            time.sleep(1)
        raise TimeoutError(f"Judge worker did not answer request {identity}")


def serve_requests(directory, judge):
    directory = Path(directory)
    print("Judge worker ready", flush=True)
    while True:
        for request_path in sorted((directory / "requests").glob("*.json")):
            response_path = directory / "responses" / request_path.name
            if response_path.exists():
                continue
            request = json.loads(request_path.read_text())
            identity = content_hash(request)
            if identity != request_path.stem:
                raise ValueError("Queued request checksum differs")
            if request["specification"] != judge.specification:
                raise ValueError("Queued judge configuration differs")
            try:
                result = judge.judge_all(request["record"], request["rubrics"])
                response = {"request_hash": identity, "result": result}
            except Exception as error:
                response = {"request_hash": identity, "error": str(error)}
            write_json(response_path, response)
            print(identity, "failed" if "error" in response else "complete", flush=True)
        time.sleep(1)
