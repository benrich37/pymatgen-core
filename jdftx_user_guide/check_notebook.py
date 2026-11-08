"""Check that the user-guide notebook runs, and that JDFTx accepts the input files it writes.

Runs every code cell of jdftx_io_user_guide.ipynb, in order, failing on any exception or warning. The cells run in an
empty temporary folder that sits next to a copy of the repository's JDFTx test files, laid out as in the repository, so
the notebook's relative paths work and nothing it writes lands in the repository. If a JDFTx executable is given, also
dry-runs (jdftx -n) every .in file the notebook wrote, and shows the end of JDFTx's output for any file it refuses.
Needs only the Python environment the notebook uses (no Jupyter).

Usage: python check_notebook.py [path/to/jdftx]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
EXAMPLES = Path("test-files/io/jdftx")  # the notebook's example files, relative to the repository
NOTEBOOK = HERE / "jdftx_io_user_guide.ipynb"


def main() -> int:
    """Run the notebook's code cells, then dry-run the input files they wrote; return the exit status."""
    parser = argparse.ArgumentParser(description="Run the user-guide notebook and dry-run the input files it writes.")
    parser.add_argument("jdftx", nargs="?", help="JDFTx executable (a path or a command) for the dry runs")
    args = parser.parse_args()
    jdftx = None
    if args.jdftx is not None:
        found = shutil.which(args.jdftx)
        if found is None:
            parser.error(f"{args.jdftx!r} is not an executable file or a command on PATH")
        jdftx = Path(found).resolve()  # absolute, because JDFTx runs in the temporary folder

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
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
                    except Exception:
                        print(f"Code cell {i} of {len(cells)} failed:\n{source}\n")
                        traceback.print_exc(file=sys.stdout)
                        return 1
        finally:
            os.chdir(cwd)
        print(f"All {len(cells)} code cells ran without errors or warnings.")

        written = sorted(path for path in work.iterdir() if path.suffix == ".in")
        if jdftx is None:
            print(f"Wrote {[path.name for path in written]}; give a JDFTx executable to dry-run them.")
            return 0
        refused = 0
        for path in written:
            out = path.with_suffix(".out")
            # JDFTx's output can hold bytes that aren't valid UTF-8, hence errors="replace"
            result = subprocess.run(
                [jdftx, "-n", "-i", path.name, "-o", out.name],
                cwd=work,
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
            )
            if result.returncode == 0:
                print(f"{path.name}: accepted by JDFTx")
                continue
            refused += 1
            log = out.read_text(encoding="utf-8", errors="replace") if out.exists() else ""
            log = log or result.stdout + result.stderr  # no output file, or an empty one
            print(f"{path.name}: REFUSED by JDFTx (exit status {result.returncode}). End of its output:")
            print("\n".join(f"    {line}" for line in log.splitlines()[-10:]))
        return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
