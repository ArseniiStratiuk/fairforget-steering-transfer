"""Execute a research notebook with a private kernel using this interpreter."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


def main():
    import nbformat
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-model-steps", action="store_true")
    arguments = parser.parse_args()
    source = arguments.notebook.resolve()
    root = next(path for path in source.parents if (path / "pyproject.toml").exists())
    notebook = nbformat.read(source, as_version=4)
    environment = os.environ.copy()
    environment["RUN_RESEARCH_GPU"] = "1" if arguments.execute_model_steps else "0"
    with tempfile.TemporaryDirectory(prefix="steering-kernel-") as temporary:
        kernels = Path(temporary)
        kernel_directory = kernels / "steering-research"
        kernel_directory.mkdir()
        specification = {
            "argv": [
                sys.executable,
                "-m",
                "ipykernel_launcher",
                "-f",
                "{connection_file}",
            ],
            "display_name": "Steering research",
            "language": "python",
        }
        (kernel_directory / "kernel.json").write_text(json.dumps(specification))
        manager = KernelManager(
            kernel_name="steering-research",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]),
        )
        client = NotebookClient(
            notebook,
            km=manager,
            timeout=None,
            resources={"metadata": {"path": str(root)}},
        )
        try:
            client.execute(env=environment, cleanup_kc=True)
        finally:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            nbformat.write(notebook, arguments.output)


if __name__ == "__main__":
    main()
