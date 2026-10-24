from __future__ import annotations

import re
from itertools import combinations
from time import process_time
from timeit import Timer
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pymatgen.core.units import Ha_to_eV, bohr_to_ang
from pymatgen.io.jdftx._output_utils import read_outfile_slices
from pymatgen.io.jdftx.inputs import JDFTXInfile
from pymatgen.io.jdftx.jdftxoutfileslice import JDFTXOutfileSlice
from pymatgen.io.jdftx.outputs import JDFTXOutfile, _jof_atr_from_last_slice

from .inputs_test_utils import aimd_infile_fname
from .outputs_test_utils import (
    eopt_elecmin_outfile_path,
    eopt_scf_ionmin_outfile_path,
    eopt_scf_outfile_path,
    etot_etype_outfile_known_simple,
    etot_etype_outfile_path,
    ex_out_files_dir,
    example_aimd_outfile_known,
    example_aimd_outfile_known_site_properties,
    example_aimd_outfile_path,
    example_igp_aimd_outfile_known,
    example_igp_aimd_outfile_known_site_properties,
    example_igp_aimd_outfile_path,
    example_ionmin_outfile_known,
    example_ionmin_outfile_known_simple,
    example_ionmin_outfile_path,
    example_latmin_outfile_known,
    example_latmin_outfile_known_simple,
    example_latmin_outfile_path,
    example_sp_outfile_known,
    example_sp_outfile_known_simple,
    example_sp_outfile_path,
    example_vib_modes_known,
    example_vib_nrg_components,
    example_vib_outfile_path,
    jdftxoutfile_fromfile_matches_known,
    jdftxoutfile_fromfile_matches_known_simple,
    jdftxoutfile_matches_known,
    jdftxoutfile_matches_known_simple,
    noeigstats_outfile_known_simple,
    noeigstats_outfile_path,
    nolowdin_ionmin_outfile_path,
    nolowdin_latmin_outfile_path,
    nolowdin_md_outfile_path,
    partial_lattice_init_outfile_known_lattice,
    partial_lattice_init_outfile_path,
    problem2_outfile_known_simple,
    problem2_outfile_path,
    problem3_outfile_path,
    vdw_d2_outfile_path,
)
from .shared_test_utils import assert_same_value

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("filename", "known"),
    [
        (example_sp_outfile_path, example_sp_outfile_known),
        (example_latmin_outfile_path, example_latmin_outfile_known),
        (example_ionmin_outfile_path, example_ionmin_outfile_known),
    ],
)
def test_JDFTXOutfile_fromfile(filename: Path, known: dict):
    jdftxoutfile_fromfile_matches_known(filename, known)


def _parse_time(filename: Path, skim_levels: tuple[str, ...] | None = None) -> float:
    """Best-of-n CPU time of parsing an out file.

    CPU time leaves out time spent waiting on other processes (e.g. other pytest-xdist workers), and taking the
    fastest of several parses discards the ones slowed by interruptions, since noise can only add time.
    """
    timer = Timer(lambda: JDFTXOutfile.from_file(filename, skim_levels=skim_levels), timer=process_time)
    return min(timer.repeat(repeat=5, number=1))


@pytest.mark.parametrize(
    ("filename", "known", "known_simple"),
    [
        (example_sp_outfile_path, example_sp_outfile_known, example_sp_outfile_known_simple),
        (example_latmin_outfile_path, example_latmin_outfile_known, example_latmin_outfile_known_simple),
        (example_ionmin_outfile_path, example_ionmin_outfile_known, example_ionmin_outfile_known_simple),
    ],
)
def test_JDFTXOutfile_skimming(filename: Path, known: dict, known_simple: dict):
    jof_full = JDFTXOutfile.from_file(filename)
    time_full = _parse_time(filename)
    # Count how minimum fewer objects can be read at each depth level
    nless_outslices = len(jof_full.slices) - 1
    nless_last_geom = len(jof_full.slices[-1].jstrucs.slices) - 1
    nless_last_elec = len(jof_full.slices[-1].jstrucs.slices[-1].elecmindata.slices) - 1
    skim_level_options = ["outfile", "geom", "elec"]
    # Don't run outfile depth skimming test if only one slice
    if not nless_outslices > 0:
        skim_level_options.remove("outfile")
    # Generate all combinations of skim level depths to test
    skim_levelss: list[tuple[str, ...]] = []
    for r in range(1, len(skim_level_options) + 1):
        skim_levelss.extend(combinations(skim_level_options, r))
    levels_time_tested = 0
    time_save_logs = []

    for skim_levels in skim_levelss:
        jof_skim = JDFTXOutfile.from_file(filename, skim_levels=skim_levels)
        # Check that the skimmed version matches the full version where it should
        jdftxoutfile_matches_known(jof_skim, known)
        jdftxoutfile_matches_known_simple(jof_skim, known_simple)
        # Make sure skim levels are correctly applied
        if "outfile" in skim_levels:
            assert len(jof_skim.slices) == 1
        if "geom" in skim_levels:
            for oslice in jof_skim.slices:
                assert len(oslice.jstrucs.slices) == 1
        if "elec" in skim_levels:
            for oslice in jof_skim.slices:
                for gslice in oslice.jstrucs.slices:
                    assert len(gslice.elecmindata.slices) == 1
        # Make sure less objects are being read before testing speedup
        less_outfile_slices_read = nless_outslices if "outfile" in skim_levels else 0
        less_geom_slices_read = nless_last_geom if "geom" in skim_levels else 0
        less_elec_slices_read = nless_last_elec if "elec" in skim_levels else 0
        # Skipping only electronic steps does not save enough time to be detectable over noise in timing, so only
        # test the speedup when whole out file slices or geometry steps are skipped
        if less_outfile_slices_read + less_geom_slices_read > 0:
            skim_time = _parse_time(filename, skim_levels)
            time_save_info = [
                filename.name,
                skim_levels,
                time_full,
                skim_time,
                less_outfile_slices_read,
                less_geom_slices_read,
                less_elec_slices_read,
            ]
            assert skim_time < time_full, time_save_info
            levels_time_tested += 1
            time_save_logs.append(time_save_info)
    if nless_outslices + nless_last_geom > 0:
        assert levels_time_tested > 0, "No skim levels tested - likely an error in test code."
    print(time_save_logs)


@pytest.mark.parametrize(
    ("filename", "known"),
    [
        (example_sp_outfile_path, example_sp_outfile_known_simple),
        (example_latmin_outfile_path, example_latmin_outfile_known_simple),
        (example_ionmin_outfile_path, example_ionmin_outfile_known_simple),
        (noeigstats_outfile_path, noeigstats_outfile_known_simple),
        (problem2_outfile_path, problem2_outfile_known_simple),
        (etot_etype_outfile_path, etot_etype_outfile_known_simple),
    ],
)
def test_JDFTXOutfile_fromfile_simple(filename: Path, known: dict):
    jdftxoutfile_fromfile_matches_known_simple(filename, known)


@pytest.mark.parametrize(
    ("filename", "latknown"),
    [
        (partial_lattice_init_outfile_path, partial_lattice_init_outfile_known_lattice),
    ],
)
def test_JDFTXOutfile_default_struc_inheritance(filename: Path, latknown: dict):
    jof = JDFTXOutfile.from_file(filename)
    lattice = jof.lattice
    for i in range(3):
        for j in range(3):
            assert pytest.approx(lattice[i][j]) == latknown[f"{i}{j}"]


# Make sure all possible exceptions are caught when none_on_error is True
@pytest.mark.parametrize(("ex_outfile_path"), [(partial_lattice_init_outfile_path)])
def test_none_on_partial(ex_outfile_path: Path):
    texts = read_outfile_slices(str(ex_outfile_path))
    texts0 = texts[:-1]

    slices = [
        JDFTXOutfileSlice._from_out_slice(text, [], [], is_bgw=False, none_on_error=False)
        for i, text in enumerate(texts0)
    ]
    outfile1 = JDFTXOutfile(slices=slices)
    slices.append(None)
    outfile2 = JDFTXOutfile(slices=slices)
    assert isinstance(outfile2, JDFTXOutfile)
    for var in _jof_atr_from_last_slice:
        assert_same_value(
            getattr(outfile1, var),
            getattr(outfile2, var),
        )


@pytest.mark.parametrize(
    ("example_vib_outfile_path", "example_vib_modes_known", "example_vib_nrg_components"),
    [
        (example_vib_outfile_path, example_vib_modes_known, example_vib_nrg_components),
    ],
)
def test_vib_parse(
    example_vib_outfile_path: Path, example_vib_modes_known: list[dict], example_vib_nrg_components: dict
):
    """
    Test that the vibration modes are parsed correctly from the outfile.
    """
    jdftxoutfile = JDFTXOutfile.from_file(example_vib_outfile_path)
    assert_same_value(jdftxoutfile.slices[-1].vibrational_modes, example_vib_modes_known)
    assert_same_value(jdftxoutfile.slices[-1].vibrational_energy_components, example_vib_nrg_components)


@pytest.mark.parametrize(
    ("aimd_outfile_path", "aimd_outfile_known", "aimd_outfile_known_site_properties"),
    [
        (example_aimd_outfile_path, example_aimd_outfile_known, example_aimd_outfile_known_site_properties),
        (example_igp_aimd_outfile_path, example_igp_aimd_outfile_known, example_igp_aimd_outfile_known_site_properties),
    ],
)
def test_aimd_parse(aimd_outfile_path: Path, aimd_outfile_known: dict, aimd_outfile_known_site_properties: dict):
    """
    Test that the AIMD properties are parsed correctly from the outfile.
    """
    jdftxoutfile = JDFTXOutfile.from_file(aimd_outfile_path)
    assert jdftxoutfile.slices[-1].is_md
    for key, val in aimd_outfile_known.items():
        assert hasattr(jdftxoutfile.slices[-1].jstrucs, key)
        assert_same_value(getattr(jdftxoutfile.slices[-1].jstrucs, key), val)
        assert hasattr(jdftxoutfile.slices[-1], key)
        assert_same_value(getattr(jdftxoutfile.slices[-1], key), val)
        assert hasattr(jdftxoutfile, key)
        assert_same_value(getattr(jdftxoutfile, key), val)
    for sp, val in aimd_outfile_known_site_properties.items():
        assert sp in jdftxoutfile.structure.site_properties
        assert_same_value(jdftxoutfile.structure.site_properties[sp], val)


def test_aimd_velocity_units():
    # The kinetic energy from the parsed velocities (angstrom/fs) matches the kinetic energy JDFTx reports
    jdftxoutfile = JDFTXOutfile.from_file(example_aimd_outfile_path)
    structure = jdftxoutfile.structure
    masses = np.array([site.specie.atomic_mass for site in structure])  # amu
    velocities = np.array(structure.site_properties["velocities"])  # angstrom/fs
    amu_ang2_per_fs2_to_ev = 1.66053906892e-27 * 1e10 / 1.602176634e-19  # CODATA amu and e
    ke = 0.5 * np.sum(masses[:, None] * velocities**2) * amu_ang2_per_fs2_to_ev
    assert ke == pytest.approx(jdftxoutfile.ke, rel=1e-4)


@pytest.mark.parametrize(
    ("outfile_path", "infile_path", "outfile_known"),
    [
        (example_aimd_outfile_path, aimd_infile_fname, example_aimd_outfile_known),
    ],
)
def test_outfile_to_infile(outfile_path: Path, infile_path: Path, outfile_known: dict):
    """
    Test that the JDFTXOutfileSlice.to_infile() method works correctly.
    """
    jdftxoutfile: JDFTXOutfile = JDFTXOutfile.from_file(outfile_path)
    JDFTXInfile.from_file(infile_path)
    new_infile: JDFTXInfile = jdftxoutfile.to_jdftxinfile()
    old_struc = jdftxoutfile.structure
    new_struc = new_infile.to_pmg_structure(new_infile, fill_site_properties=True)
    old_coords = old_struc.cart_coords
    new_coords = new_struc.cart_coords
    assert_same_value(old_coords, new_coords)
    old_velocities = old_struc.site_properties.get("velocities", None)
    new_velocities = new_struc.site_properties.get("velocities", None)
    [ov / nv for ov, nv in zip(old_velocities, new_velocities, strict=False)]
    assert_same_value(old_velocities, new_velocities)
    old_thermostat_velocity = jdftxoutfile.slices[-1].jstrucs.thermostat_velocity
    new_thermostat_velocity = np.array([new_infile["thermostat-velocity"][f"v{i + 1}"] for i in range(3)])
    assert_same_value(old_thermostat_velocity, new_thermostat_velocity)

    # TODO: Do the old TODO related to filling out default input tags so the robust comparison below
    # can be used instead of the above.
    # assert new_infile.is_comparable_to(jdftxinfile, exclude_tags=["ion", "thermostat-velocity", "dump"])


def test_etot_bug_handling():
    outfile = JDFTXOutfile.from_file(problem3_outfile_path)
    assert len(outfile.slices) == 1
    assert len(outfile.slices[-1].jstrucs.slices) == 7
    # The first ionic step in this outfile has the etot bug
    assert outfile.slices[-1].jstrucs.slices[0].etype == "F"
    assert "F" not in outfile.slices[-1].jstrucs.slices[0].ecomponents
    assert_same_value(outfile.slices[-1].jstrucs.slices[0].e, -3.018425965201787 * Ha_to_eV)


@pytest.mark.parametrize(
    ("outfile_path", "eopt_type", "converged_reason", "geom_converged"),
    [
        (eopt_elecmin_outfile_path, "ElecMinimize", "|Delta F|<1.000000e-08 for 2 iters", True),
        (eopt_scf_outfile_path, "SCF", "|Delta E|<1.000000e-08 for 2 iters", True),
        # Stopped after 2 ionic steps, each converged electronically
        (eopt_scf_ionmin_outfile_path, "SCF", "|Delta E|<1.000000e-08 for 2 iters", False),
    ],
    ids=["elecmin", "scf", "scf_ionmin"],
)
def test_eopt_types(outfile_path: Path, eopt_type: str, converged_reason: str, geom_converged: bool):
    """Electronic steps are read for each electronic optimizer (logged as "ElecMinimize: Iter:" or "SCF: Cycle:")."""
    outfile = JDFTXOutfile.from_file(outfile_path, none_slice_on_error=False)
    jstrucs = outfile.slices[-1].jstrucs
    assert outfile.eopt_type == eopt_type
    assert outfile.converged is geom_converged
    for jstruc in jstrucs:
        assert jstruc.elecmindata.opt_type == eopt_type
        assert jstruc.elecmindata.converged
        assert jstruc.elecmindata.converged_reason == converged_reason
    # Every step JDFTx printed is parsed, with its step number, energy label and energy
    printed = [
        (int(m["nstep"]), m["etype"], float(m["e"]))
        for m in re.finditer(
            rf"^{re.escape(eopt_type)}: (?:Iter|Cycle):\s+(?P<nstep>\d+)\s+(?P<etype>F|G|Etot): (?P<e>\S+)",
            outfile_path.read_text(),
            flags=re.MULTILINE,
        )
    ]
    parsed = [(step.nstep, step.etype, step.e / Ha_to_eV) for jstruc in jstrucs for step in jstruc.elecmindata]
    assert [p[:2] for p in parsed] == [p[:2] for p in printed]
    assert [p[2] for p in parsed] == pytest.approx([p[2] for p in printed], rel=0, abs=1e-12)
    # SCF steps have no line-minimization data
    last_step = jstrucs[-1].elecmindata[-1]
    assert (last_step.grad_k is None) is (eopt_type == "SCF")
    assert last_step.t_s is not None


def test_eopt_types_agree():
    # Both electronic optimizers converge the same calculation to the same free energy
    elecmin = JDFTXOutfile.from_file(eopt_elecmin_outfile_path, none_slice_on_error=False)
    scf = JDFTXOutfile.from_file(eopt_scf_outfile_path, none_slice_on_error=False)
    assert scf.e == pytest.approx(elecmin.e, rel=0, abs=1e-8 * Ha_to_eV)
    assert scf.elec_nstep > 0
    assert elecmin.elec_nstep > 0


@pytest.mark.parametrize(
    ("outfile_path", "geom_opt_type", "geom_converged"),
    [
        (nolowdin_ionmin_outfile_path, "IonicMinimize", True),
        (nolowdin_md_outfile_path, "IonicDynamics", False),
        (nolowdin_latmin_outfile_path, "LatticeMinimize", False),
    ],
    ids=["ionmin", "md", "latmin"],
)
def test_no_lowdin_steps(outfile_path: Path, geom_opt_type: str, geom_converged: bool):
    """Every geometry step is kept when JDFTx skips the Lowdin block (no pseudopotential has atomic orbitals)."""
    text = outfile_path.read_text()
    assert "Lowdin" not in text
    outfile = JDFTXOutfile.from_file(outfile_path, none_slice_on_error=False)
    jstrucs = outfile.slices[-1].jstrucs
    assert jstrucs.opt_type == geom_opt_type
    assert outfile.converged is geom_converged
    # One structure per printed geometry step, with its step number and energy (MD prints PE with 6 decimals)
    step_key = "Step" if geom_opt_type == "IonicDynamics" else "Iter"
    printed = [
        (int(m["nstep"]), float(m["e"]))
        for m in re.finditer(
            rf"^{geom_opt_type}: {step_key}:\s+(?P<nstep>\d+)\s+(?:Etot|PE):\s+(?P<e>\S+)", text, flags=re.MULTILINE
        )
    ]
    assert [jstruc.nstep for jstruc in jstrucs] == [p[0] for p in printed]
    assert [jstruc.e / Ha_to_eV for jstruc in jstrucs] == pytest.approx([p[1] for p in printed], rel=0, abs=1e-6)
    # Each structure has the positions printed for its step, and its own converged electronic minimization
    printed_posns = [
        np.array([line.split()[2:5] for line in block.splitlines()], dtype=float) * bohr_to_ang
        for block in re.findall(r"^# Ionic positions in cartesian coordinates:\n((?:ion .*\n)+)", text, re.MULTILINE)
    ]
    assert len(printed_posns) == len(jstrucs)
    for jstruc, posns in zip(jstrucs, printed_posns, strict=True):
        assert jstruc.cart_coords == pytest.approx(posns)
        assert len(jstruc.elecmindata)
        assert jstruc.elecmindata.converged


@pytest.mark.parametrize("smearing", [True, False])
def test_unavailable_eigstats(tmp_path: Path, smearing: bool):
    # Without empty bands (e.g. an insulator with the default number of bands), JDFTx prints the LUMO, mu and gaps
    # as "unavailable"; the slice must still be parsed
    text = example_sp_outfile_path.read_text()
    for key in ["LUMO", "mu  ", "HOMO-LUMO gap", "Optical gap  "]:
        text = re.sub(rf"^(\s*{re.escape(key)}): .*$", r"\1: unavailable", text, flags=re.MULTILINE)
    if not smearing:
        text = re.sub(r"^elec-smearing .*\n", "", text, flags=re.MULTILINE)
    (tmp_path / "jdftx.out").write_text(text)
    outfile = JDFTXOutfile.from_file(tmp_path / "jdftx.out", none_slice_on_error=False)
    reference = JDFTXOutfile.from_file(example_sp_outfile_path)
    assert len(outfile.slices) == 1
    assert outfile.lumo is None
    assert outfile.egap is None
    assert outfile.homo == pytest.approx(reference.homo)
    # The Fermi level falls back to the mu printed during the electronic minimization
    assert outfile.efermi == pytest.approx(outfile.mu)
    if smearing:
        # The lumo filling needs the (unavailable) lumo energy, so metallicity is unknown
        assert outfile.homo_filling == pytest.approx(reference.homo_filling)
        assert outfile.lumo_filling is None
        assert outfile.is_metal is None
    else:
        # Without smearing the fillings don't depend on the band energies
        assert outfile.lumo_filling == 0
        assert outfile.is_metal is False


def _printed_final_ions(text: str) -> tuple[str, list[list[str]]] | None:
    """Coordinate system and ion lines of the last positions block JDFTx printed (None if there isn't one)."""
    if "# Ionic positions in " not in text:
        return None
    header, *lines = text.rsplit("# Ionic positions in ", maxsplit=1)[-1].splitlines()
    return header.split()[0], [line.split() for line in lines if line.startswith("ion ")]


def _check_restart_matches_out_file(text: str, restart: JDFTXInfile, infile: JDFTXInfile, lattice_changed: bool):
    printed_ions = _printed_final_ions(text)
    if printed_ions is None:  # No positions were printed, so the input's ions are kept
        assert restart["ion"] == infile["ion"]
        return
    # The ions are exactly as JDFTx printed them, as if the final positions block was copied into a new input
    coords_type, printed = printed_ions
    assert restart.get("coords-type", "Lattice").lower() == coords_type.lower()
    assert len(restart["ion"]) == len(printed)
    for ion, row in zip(restart["ion"], printed, strict=True):
        assert ion["species-id"] == row[1]
        assert [ion[f"x{k}"] for k in range(3)] == [float(x) for x in row[2:5]]
        i_move_scale = 5
        if "v" in row:
            i = row.index("v")
            assert [ion["v"][f"vx{k}"] for k in range(3)] == [float(x) for x in row[i + 1 : i + 4]]
            i_move_scale = i + 4
        assert ion["moveScale"] == float(row[i_move_scale])
    # An unchanged lattice keeps the input's own lattice and latt-scale tags
    if not lattice_changed:
        assert restart["lattice"] == infile["lattice"]
        assert restart.get("latt-scale") == infile.get("latt-scale")


@pytest.mark.parametrize("outfile_path", sorted(ex_out_files_dir.glob("*.out")), ids=lambda path: path.name)
def test_to_jdftxinfile_matches_out_file(outfile_path: Path):
    outfile = JDFTXOutfile.from_file(outfile_path)
    restart = JDFTXInfile.from_str(str(outfile.to_jdftxinfile()))
    infile = outfile.slices[-1].infile
    lattice_changed = not np.array_equal(infile.structure.lattice.matrix, outfile.structure.lattice.matrix)
    _check_restart_matches_out_file(outfile_path.read_text(), restart, infile, lattice_changed)


def test_to_jdftxinfile_keeps_fractional_move_scale(tmp_path: Path):
    # A moveScale below 1 must not become 0 (a fixed ion) in the restart input
    text = re.sub(r"^(ion \S+(?: +\S+){3}) 1$", r"\1 0.5", example_ionmin_outfile_path.read_text(), flags=re.MULTILINE)
    (tmp_path / "jdftx.out").write_text(text)
    outfile = JDFTXOutfile.from_file(tmp_path / "jdftx.out")
    # The fixed ions (moveScale 0) are left as they are
    assert {ion["moveScale"] for ion in outfile.slices[-1].infile["ion"]} == {0.0, 0.5}
    restart = JDFTXInfile.from_str(str(outfile.to_jdftxinfile()))
    assert {ion["moveScale"] for ion in restart["ion"]} == {0.0, 0.5}
    _check_restart_matches_out_file(text, restart, outfile.slices[-1].infile, lattice_changed=False)


def test_to_jdftxinfile_keeps_d2_dispersion():
    # JDFTx echoes DFT-D2 with the default scale as a bare "van-der-waals", which must not be dropped
    outfile = JDFTXOutfile.from_file(vdw_d2_outfile_path)
    assert outfile.infile["van-der-waals"] == 0.0
    restart = JDFTXInfile.from_str(str(outfile.to_jdftxinfile()))
    assert restart["van-der-waals"] == 0.0


@pytest.mark.parametrize("latt_scale", ["2 2 1", "1.5 0.75 2"])
def test_to_jdftxinfile_with_latt_scale(tmp_path: Path, latt_scale: str):
    # A supercell made with latt-scale is not scaled a second time in the restart input
    text = example_aimd_outfile_path.read_text().replace("latt-scale 1 1 1 ", f"latt-scale {latt_scale} ", 1)
    (tmp_path / "jdftx.out").write_text(text)
    outfile = JDFTXOutfile.from_file(tmp_path / "jdftx.out")
    restart = JDFTXInfile.from_str(str(outfile.to_jdftxinfile()))
    _check_restart_matches_out_file(text, restart, outfile.slices[-1].infile, lattice_changed=False)
