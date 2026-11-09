# JDFTx input and output files in pymatgen

> **Draft.** Sections marked *(to write)* wait on fixes listed in `PLAN.md`. The hands-on version of this guide is
> `jdftx_io_user_guide.ipynb`, which runs on pymatgen-core's JDFTx test files (`test-files/io/jdftx/`).

`pymatgen.io.jdftx` reads and writes the files of [JDFTx](https://jdftx.org), a plane-wave DFT code built around joint
density-functional theory (implicit and classical solvation). It lets you:

- read, edit and write JDFTx input files, and build them from a pymatgen `Structure`;
- read JDFTx's output file: structures, energies, forces, charges, eigenvalue statistics, every step of an optimization
  or molecular-dynamics run;
- write an input file that restarts a finished run;
- read the binary dump files (eigenvalues, band projections) into numpy arrays and a pymatgen `BandStructure`.

## Contents

1. [The classes at a glance](#1-the-classes-at-a-glance)
2. [Input files](#2-input-files)
3. [How settings are represented](#3-how-settings-are-represented)
4. [Building inputs from a structure](#4-building-inputs-from-a-structure)
5. [Output files](#5-output-files)
6. [Optimization and molecular-dynamics steps](#6-optimization-and-molecular-dynamics-steps)
7. [Restarting a run](#7-restarting-a-run)
8. [Dump files and band structures](#8-dump-files-and-band-structures)
9. [Units and coordinates](#9-units-and-coordinates)
10. [Warnings, errors and known limitations](#10-warnings-errors-and-known-limitations)

## 1. The classes at a glance

| You have | Use | You get |
|---|---|---|
| a JDFTx input file | `JDFTXInfile.from_file(path)` | a dictionary of commands |
| a pymatgen `Structure` | `JDFTXInfile.from_structure(structure)` | the `lattice` and `ion` commands |
| JDFTx's output file (`jdftx -o out`) | `JDFTXOutfile.from_file(path)` | results of the run(s) in it |
| a calculation folder with dump files | `JDFTXOutputs.from_calc_dir(folder, store_vars=[...])` | the output file plus eigenvalues, projections |

An output file can hold several runs (JDFTx appends to an existing `-o` file). `JDFTXOutfile.slices` has one
`JDFTXOutfileSlice` per run, and the attributes of the `JDFTXOutfile` itself describe the last run. Within a run,
`jstrucs` holds one `JOutStructure` per ionic step.

## 2. Input files

```python
from pymatgen.io.jdftx.inputs import JDFTXInfile

jif = JDFTXInfile.from_file("h2o.in")
jif["elec-cutoff"]                                 # {'Ecut': 20.0, 'EcutRho': 100.0}
jif["elec-cutoff"] = {"Ecut": 25, "EcutRho": 125}  # change a setting
jif["elec-smearing"] = "Fermi 0.01"                # or give it as written in the file
jif.append_tag("fluid-cation", {"name": "K+", "concentration": 0.1})
jif.append_tag("fluid-anion", {"name": "Cl-", "concentration": 0.1})  # JDFTx refuses a charged electrolyte
del jif["ionic-minimize"]
jif.write_file("h2o_edited.in")
```

- `include` lines are read in place, as JDFTx does; an included file is looked for as given and then next to the main
  file.
- Comments (`#`) and line continuations (`\`) are handled; they aren't kept when the file is written again.
- Commands are written in a fixed order, grouped by topic, not in the order of the original file.
- A value given as a string is read with JDFTx's rules, so a misspelled option or a missing parameter is reported
  straight away. *(To write after issue 59: values given as dictionaries are only partly checked today.)*
- Combine two input files with `+`. Repeatable commands (`ion`, `fluid-cation`, ...) are added together; any other
  command takes its value from the right-hand file.
- `get_differing_tags(other)` lists the commands two inputs set differently, counting JDFTx's defaults.

## 3. How settings are represented

Each key of a `JDFTXInfile` is a JDFTx command, and parameter names are JDFTx's own (see `jdftx -t`, or the
[command reference](https://jdftx.org/Commands.html)). Values stay in JDFTx's units.

| In the file | In pymatgen |
|---|---|
| `elec-ex-corr gga-PBE` | `"gga-PBE"` |
| `converge-empty-states yes` | `True` |
| `elec-cutoff 20 100` | `{"Ecut": 20.0, "EcutRho": 100.0}` |
| `ionic-minimize nIterations 10` | `{"nIterations": 10}` |
| `ion H 0 0 1.5 1` | `[{"species-id": "H", "x0": 0.0, "x1": 0.0, "x2": 1.5, "moveScale": 1.0}]` |
| `ion-species GBRV/$ID_pbe.uspp` | `["GBRV/$ID_pbe.uspp"]` |
| `fluid-cation Na+ 0.5` | `[{"name": "Na+", "concentration": 0.5}]` |
| `dump End State ElecDensity` | `[{"End": {"State": True, "ElecDensity": True}}]` |
| `initial-magnetic-moments H 1` | `{"species-values": [{"species-id": "H", "values": [1.0]}]}` |

Commands that JDFTx accepts more than once are always lists, even when given once; a single value can still be
assigned and is wrapped in a list. Some commands take one of several forms (for example `elec-ex-corr` with one
functional, or an exchange and a correlation functional); each form has its own representation. *(To write: a short
list of the forms new users meet most: `lattice`, `coulomb-interaction`, `ion` constraints, `dump`.)*

## 4. Building inputs from a structure

```python
from pymatgen.io.jdftx.inputs import JDFTXInfile

structure_part = JDFTXInfile.from_structure(structure)  # lattice and ions, Å converted to bohr
settings = JDFTXInfile.from_dict({"ion-species": "GBRV/$ID_pbe.uspp", "elec-cutoff": {"Ecut": 20, "EcutRho": 100}})
(structure_part + settings).write_file("calc.in")
```

- Positions are written in lattice coordinates unless `write_cart_coords=True`.
- `selective_dynamics` (per atom, which may move) and `velocities` can be passed, or are taken from site properties.
- JDFTx names a species after its pseudopotential file, so two kinds of the same element (e.g. `Fe1.upf` and `Fe2.upf`
  for an antiferromagnet) have different names. pymatgen keeps such names in the `jdftx_species` site property; the
  element is the longest element symbol the name starts with.
- *(To write: magnetic moments, constraints, `JdftxInputSet` and `BaseJdftxSet.yaml`, after issue 49.)*

## 5. Output files

```python
from pymatgen.io.jdftx.outputs import JDFTXOutfile

out = JDFTXOutfile.from_file("jdftx.out")
out.structure     # final structure (pymatgen Structure, Å)
out.e             # final energy, eV
out.ecomponents   # energy terms, eV
out.forces        # eV/Å, one row per atom
out.infile        # the run's settings, as JDFTx printed them
```

- Lowdin charges and magnetic moments are site properties of the structure (`charges`, `magmom`).
- *(To write after issues 24, 30 and 31: the energy type `etype`, and the eigenvalue statistics `homo`, `lumo`,
  `efermi`/`mu`, `is_metal`. Today they can be mislabelled or in the wrong unit.)*
- *(To write after issue 53: which of the many other attributes are worth knowing.)*

## 6. Optimization and molecular-dynamics steps

`out.jstrucs` holds one structure per ionic step, each with its energy, forces, stress and charges; `out.trajectory`
returns the same steps as a pymatgen `Trajectory`. For molecular dynamics, each step also carries velocities (Å/fs, the
`velocities` site property), kinetic energy and temperature.

*(To write after issue 9: thermostat and barostat state.)*

## 7. Restarting a run

```python
restart = out.to_jdftxinfile()
restart.write_file("restart.in", strip_nan=True)
```

The restart keeps the run's settings, with the final positions written exactly as JDFTx printed them (and, for MD, the
final velocities and thermostat state). *(To write after issues 9, 41 and 47: what is carried over, and why
`strip_nan=True` is needed today.)*

## 8. Dump files and band structures

```python
from pymatgen.io.jdftx.outputs import JDFTXOutputs

outputs = JDFTXOutputs.from_calc_dir("calc_folder", store_vars=["eigenvals", "bandProjections"])
outputs.eigenvals        # eV, shape (spins x k-points, bands)
outputs.bandProjections  # shape (spins x k-points, bands, orbitals)
outputs.orb_label_list   # orbital of each projection, e.g. "H#1(s)"
```

Asking for `"bandstructure"` builds a pymatgen `BandStructure` from the eigenvalues, the k-points and, if present, the
projections. *(To write after issues 34-36 and 51.)*

## 9. Units and coordinates

| Quantity | In JDFTx files and `JDFTXInfile` | In pymatgen objects from out files |
|---|---|---|
| lengths, positions | bohr (or lattice coordinates) | Å |
| energies | Hartree | eV |
| forces | Hartree/bohr | eV/Å |
| stress | Hartree/bohr³ | eV/Å³ |
| velocities | bohr per Hartree time unit | Å/fs |

`JDFTXInfile` values are never converted: they are what JDFTx reads. `JDFTXInfile.structure` and everything read
from out files is in pymatgen's units.

## 10. Warnings, errors and known limitations

- Reading an out file never stops on something pymatgen doesn't recognize: it warns and skips it.
- An input value that doesn't fit its command's syntax raises `ValueError` (`KeyError` for an unknown command).
  pymatgen doesn't make every check JDFTx makes (for example that `EcutRho` is at least 4 `Ecut`), so JDFTx can still
  refuse a file pymatgen accepts; `jdftx -n -i file.in` checks a file without running it.
- *(To write: the main limitations a user can hit, drawn from the open-issue list once it settles.)*
