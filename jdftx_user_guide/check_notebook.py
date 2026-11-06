"""Check that the user-guide notebook runs, and that JDFTx accepts the input files it writes.

Runs every code cell of jdftx_io_user_guide.ipynb, in order, failing on any exception or warning. The cells run in an
empty temporary folder that sits next to a copy of the repository's JDFTx test files, laid out as in the repository, so
the notebook's relative paths work and nothing it writes lands in the repository. If a JDFTx executable is given, also
dry-runs (jdftx -n) every .in file the notebook wrote. Needs only the Python environment the notebook uses (no Jupyter).

Usage: python check_notebook.py [path/to/jdftx]
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
EXAMPLES = Path("test-files/io/jdftx")  # the notebook's example files, relative to the repository


def main() -> int:
    """Run the notebook's code cells, then dry-run the input files they wrote; return the exit status."""
    jdftx = sys.argv[1] if len(sys.argv) > 1 else None
    notebook = json.loads((HERE / "jdftx_io_user_guide.ipynb").read_text())
    cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(REPO / EXAMPLES, Path(tmp) / EXAMPLES)
        work = Path(tmp) / HERE.name
        work.mkdir()
        namespace: dict = {}
        cwd = Path.cwd()
        try:
            os.chdir(work)
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                for i, source in enumerate(cells, start=1):
                    try:
                        exec(compile(source, f"<code cell {i}>", "exec"), namespace)  # noqa: S102
                    except Exception as error:
                        print(f"Code cell {i} failed: {type(error).__name__}: {error}\n{source}")
                        return 1
        finally:
            os.chdir(cwd)
        print(f"All {len(cells)} code cells ran without errors or warnings.")
        written = sorted(p for p in work.iterdir() if p.suffix == ".in")
        if jdftx is None:
            print(f"Wrote {[p.name for p in written]}; give a JDFTx executable to dry-run them.")
            return 0
        failed = 0
        for path in written:
            result = subprocess.run([jdftx, "-n", "-i", path.name, "-o", f"{path.stem}.out"], cwd=work, check=False)
            print(f"{path.name}: {'accepted' if result.returncode == 0 else 'REFUSED'} by JDFTx")
            failed += result.returncode != 0
        return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
