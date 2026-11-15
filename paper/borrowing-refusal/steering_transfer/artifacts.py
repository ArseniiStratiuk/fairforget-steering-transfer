"""Load checksum-pinned experiment artifacts; bulk files stay outside Git."""

import hashlib
import json
import shutil

from huggingface_hub import hf_hub_download


def load_artifacts(root, *groups):
    """Verify local files, downloading missing files from a pinned HF release."""
    releases = json.loads((root / "configs/reviewed_experiments.json").read_text())
    for group in groups:
        release = releases[group]
        for item in release["files"]:
            path = root / item["local_path"]
            if not path.exists():
                if not release["revision"]:
                    raise FileNotFoundError(
                        f"{path}: artifact release is prepared locally but not published."
                    )
                cached = hf_hub_download(
                    release["repo_id"],
                    item["filename"],
                    repo_type=release["repo_type"],
                    revision=release["revision"],
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cached, path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Artifact checksum mismatch: {path}")
