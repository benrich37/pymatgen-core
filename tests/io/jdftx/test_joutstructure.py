"""Tests for the JOutStructure class."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pytest import approx

from pymatgen.core import Lattice, Structure
from pymatgen.core.units import bohr_to_ang
from pymatgen.io.jdftx.inputs import JDFTXInfile, JDFTXStructure
from pymatgen.io.jdftx.joutstructure import JOutStructure, _parse_posn_line
from pymatgen.util.testing import TEST_FILES_DIR

from .outputs_test_utils import ex_jstruc_slice1 as ex_slice1
from .outputs_test_utils import ex_jstruc_slice1_known, ex_jstruc_slice2_known
from .outputs_test_utils import ex_jstruc_slice2 as ex_slice2

ex_files_dir = Path(TEST_FILES_DIR) / "io" / "jdftx" / "example_files"


@pytest.mark.parametrize(
    ("eslice", "eknowns"), [(ex_slice1, ex_jstruc_slice1_known), (ex_slice2, ex_jstruc_slice2_known)]
)
def test_jstructure(eslice: list[str], eknowns: dict):
    """Test the JOutStructure class.

    Args:
        eslice: The text slice to test the JOutStructure class.
        eknowns: The known values to test against.
    """
    jst = JOutStructure._from_text_slice(eslice, opt_type=eknowns["opt_type"])
    assert jst.nstep == eknowns["nstep"]
    assert jst.etype == eknowns["etype"]
    assert approx(eknowns["E"]) == jst.e
    assert jst.ecomponents["Eewald"] == approx(eknowns["Eewald"])
    assert jst.ecomponents["EH"] == approx(eknowns["EH"])
    assert jst.ecomponents["Eloc"] == approx(eknowns["Eloc"])
    assert jst.ecomponents["Enl"] == approx(eknowns["Enl"])
    assert jst.ecomponents["EvdW"] == approx(eknowns["EvdW"])
    assert jst.ecomponents["Exc"] == approx(eknowns["Exc"])
    assert jst.ecomponents["Exc_core"] == approx(eknowns["Exc_core"])
    assert jst.ecomponents["KE"] == approx(eknowns["KE"])
    assert jst.ecomponents["Etot"] == approx(eknowns["Etot"])
    assert jst.ecomponents["TS"] == approx(eknowns["TS"])
    assert jst.ecomponents["F"] == approx(eknowns["F"])
    assert jst.elecmindata[0].mu == approx(eknowns["mu0"])
    assert jst.elecmindata[-1].mu == approx(eknowns["mu-1"])
    assert approx(eknowns["E0"]) == jst.elecmindata[0].e
    assert approx(eknowns["E-1"]) == jst.elecmindata[-1].e
    assert len(jst.elecmindata) == eknowns["nEminSteps"]
    assert len(jst.forces) == eknowns["nAtoms"]
    assert len(jst.cart_coords) == eknowns["nAtoms"]
    assert jst.elecmindata.converged_reason == eknowns["EconvReason"]
    assert jst.elecmindata.converged == eknowns["conv"]
    assert jst.lattice.matrix[0, 0] == approx(eknowns["cell_00"])
    assert jst.strain[0, 0] == approx(eknowns["strain_00"])
    assert jst.stress[0, 0] == approx(eknowns["stress_00"])
    for i in range(3):
        assert jst.cart_coords[0][i] == approx(eknowns["posn0"][i])
        assert jst.forces[0][i] == approx(eknowns["force0"][i])
        assert jst.cart_coords[-1][i] == approx(eknowns["posn-1"][i])
        assert jst.forces[-1][i] == approx(eknowns["force-1"][i])
    assert jst.charges[0] == approx(eknowns["ox0"])
    assert jst.magnetic_moments[0] == approx(eknowns["mag0"])
    assert jst.charges[-1] == approx(eknowns["ox-1"])
    assert jst.magnetic_moments[-1] == approx(eknowns["mag-1"])


@pytest.mark.parametrize(
    ("eslice", "eknowns"), [(ex_slice1, ex_jstruc_slice1_known), (ex_slice2, ex_jstruc_slice2_known)]
)
def test_jstructure_structure(eslice: list[str], eknowns: dict):
    jst = JOutStructure._from_text_slice(eslice, opt_type=eknowns["opt_type"])
    assert isinstance(jst.structure, Structure)
    assert not isinstance(jst.structure, JOutStructure)


@pytest.mark.parametrize(
    ("line", "constraint_type", "constraint_vector", "group_names"),
    [
        ("ion H 0.1 0.2 0.3 1", None, None, None),
        ("ion H 0.1 0.2 0.3 1 Linear 1 0 0", "Linear", [1, 0, 0], None),
        ("ion H 0.1 0.2 0.3 v 0.01 0.02 0.03 1 Planar 0 0 1", "Planar", [0, 0, 1], None),
        # A HyperPlane constraint always gives a list of vectors and a list of group names
        ("ion H 0.1 0.2 0.3 1 HyperPlane 1 0 0", "HyperPlane", [[1, 0, 0]], [None]),
        ("ion H 0.1 0.2 0.3 1 HyperPlane 1 0 0 g1", "HyperPlane", [[1, 0, 0]], ["g1"]),
        (
            "ion H 0.1 0.2 0.3 1 HyperPlane 1 0 0 g1 HyperPlane 0 1 0 g2",
            "HyperPlane",
            [[1, 0, 0], [0, 1, 0]],
            ["g1", "g2"],
        ),
    ],
)
def test_parse_posn_line_constraints(line, constraint_type, constraint_vector, group_names):
    *_, parsed_type, parsed_vector, parsed_groups = _parse_posn_line(line)
    assert parsed_type == constraint_type
    assert parsed_groups == group_names
    if constraint_vector is None:
        assert parsed_vector is None
    else:
        np.testing.assert_allclose(parsed_vector, constraint_vector)


def test_posns_constraints_are_cartesian():
    # Lattice vectors are the columns of r (in bohr), in a non-orthogonal cell
    r = np.array([[10.0, 2.0, 1.0], [0.0, 9.0, 3.0], [0.0, 0.0, 8.0]])
    jst = JOutStructure(lattice=Lattice(r.T * bohr_to_ang), species=[], coords=[])
    posns_lines = [
        "# Ionic positions in lattice coordinates:",
        "ion H 0.1 0.2 0.3 1 Linear 1 0 0",
        "ion O 0.4 0.5 0.6 1 HyperPlane 0 1 1 g1",
        "ion H 0.7 0.8 0.9 1",
    ]
    jst._parse_posns_lines(posns_lines, [])
    # Constraints on only some sites are kept, with vectors converted to Cartesian like the positions are
    assert jst.constraint_types == ["Linear", "HyperPlane", None]
    assert jst.group_names == [None, ["g1"], None]
    np.testing.assert_allclose(jst.constraint_vectors[0], r @ [1, 0, 0])
    np.testing.assert_allclose([r.T @ v for v in jst.constraint_vectors[1]], [[0, 1, 1]])
    assert jst.constraint_vectors[2] is None


def test_posns_constraints_to_infile():
    # Constraints on every site are kept as site properties, so they carry over to a new input file
    jst = JOutStructure(lattice=Lattice.cubic(10.0 * bohr_to_ang), species=[], coords=[])
    posns_lines = [
        "# Ionic positions in lattice coordinates:",
        "ion H 0.1 0.2 0.3 1 Linear 1 0 0",
        "ion O 0.4 0.5 0.6 1 Planar 0 1 1",
    ]
    jst._parse_posns_lines(posns_lines, [])
    jif = JDFTXInfile.from_structure(jst)
    assert [ion["constraint type"] for ion in jif["ion"]] == ["Linear", "Planar"]
    jstruct = JDFTXStructure.from_jdftxinfile(jif)
    np.testing.assert_allclose(jstruct.constraint_vectors, jst.constraint_vectors)
