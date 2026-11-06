# User guide for `pymatgen.io.jdftx`: plan

Started 2026-11-06. Issue numbers refer to `JDFTX_OPEN_ISSUES.md` at the repo root.

## Goal and readers

For someone new to pymatgen's JDFTx support. They know Python and the basics of pymatgen (`Structure`), and may be
new to JDFTx's input language. Within about an hour they should be able to read, edit and write input files, build one
from a structure, read the results of a run, follow its steps, and restart it.

## Two documents

| | Role | Status |
|---|---|---|
| `jdftx_io_user_guide.ipynb` | Hands-on tour, run top to bottom on pymatgen's JDFTx test files; JDFTx not needed | 8 sections drafted; all 18 code cells run, and JDFTx (`b3c54e1d`) accepts the 3 files it writes |
| `jdftx_io_user_guide.md` | Reference: the same topics in more depth, plus tables of value forms and units, restarts, limitations | Outline with drafted sections; *(to write)* marks parts waiting on fixes |

Both follow the same order, and every snippet in the Markdown guide should match a notebook cell, so the two stay
consistent. `check_notebook.py` runs the notebook's code cells in an empty temporary folder, next to a copy of the test
files laid out as in the repository, failing on any error or warning, and dry-runs the input files they write:
`python check_notebook.py [path/to/jdftx]`.

## Example data

For now the notebook reads pymatgen-core's JDFTx test files in `test-files/io/jdftx/` (`EXAMPLES` in its first code
cell), relative to this folder, instead of bundled copies. That keeps them out of the pre-commit hooks: `test-files/` is
excluded, and codespell would otherwise flag misspelled words in the messages JDFTx prints to its out files. Published
outside this repository, the guide needs its own copies again, excluded from codespell (see decision 2).

All small, and checked for personal paths and host names (none).

| File, in `test-files/io/jdftx/` | What it is | Used for |
|---|---|---|
| `test_jdftx_in_files/h2o.in` | Written for the guide (no test uses it): water in LinearPCM water with 0.5 M NaF | Reading and editing inputs |
| `test_jdftx_out_files/eopt_scf_ionmin.out` | H₂ geometry optimization, 3 steps | Results, steps, trajectory, restart |
| `test_jdftx_out_files/species_names.out` | Spin-polarized stretched H₂, species `H1`/`H2` | Charges, magnetic moments, species names |
| `test_jdftx_calc_dirs/H2_nbands_multiplier/` | Out file, `eigenvals`, `bandProjections` | Dump files |
| `test_jdftx_out_files/nolowdin_md.out` | Short MD run | Planned MD section |
| `test_jdftx_out_files/pcm_params.out` | Solvated run with `pcm-params` | Planned solvation section |

## What the guide waits on

These issues show up in the guide's examples, or force a workaround in them. Suggested order of fixing:

1. **47:** restarts need `write_file(strip_nan=True)`; with the default, JDFTx refuses every MD restart.
2. **59 (with 45):** settings given as dictionaries are stored before they're checked, so the guide can only promise
   that mistakes are caught for values given as strings.
3. **24, 30, 31:** `etype`, `efermi`/`mu`, `homo`/`lumo`/`is_metal` can be mislabelled or in Hartree. The spin example
   shows `etype` `'F'` and a Hartree `efermi` today, so section 5 leaves eigenvalue statistics out for now.
4. **9:** thermostat and barostat state, for the MD and restart sections.
5. **48:** results can't be saved and loaded with `dumpfn`/`loadfn`, so there's no "saving results" section yet.
6. **49:** `JdftxInputSet` is broken, so input sets aren't covered yet.
7. **53, 54:** decide which output attributes are the documented core before writing section 5 in full; otherwise the
   guide documents attributes that are about to be deprecated.
8. **34-36, 51:** for the dump-file section beyond eigenvalues.

## Polish before publishing

- Site properties hold numpy scalars, which print as `np.float64(0.05)`.
- Written files have double and trailing spaces (56).
- An unknown command's error names an internal list: "The elec-cuttoff tag is not in MASTER_TAG_LIST".
- The first code cell starts with `from __future__ import annotations`, which ruff adds to every Python file and
  notebook in this repository. It means nothing to a reader; drop it when publishing elsewhere.

## Sections to add later

- Molecular dynamics: velocities, thermostat and barostat, restarts (after 9; the NPT run from issue 9 is a candidate
  example).
- Solvation settings: `fluid`, `pcm-variant`, solvents and ions (with `pcm_params.out`).
- Input sets: `JdftxInputSet` and `BaseJdftxSet.yaml` (after 49).
- Saving and loading results (after 48).
- Possibly: running JDFTx from Python, as optional cells that skip when JDFTx isn't installed.

## Decisions for you

1. **Where it will be published.** This repo (pymatgen-core has no docs folder), pymatgen's documentation site,
   pymatgen's example notebooks (matgenb), or JDFTx's own site. This decides the Markdown flavour, how the example data
   is hosted, and the links.
2. **Example data, once published:** the notebook reads the repository's test files for now. Published elsewhere, it
   needs bundled copies (about 110 KB, excluded from codespell) or a download of the files it uses.
3. **Running JDFTx:** whether the notebook should include optional cells that run it.
