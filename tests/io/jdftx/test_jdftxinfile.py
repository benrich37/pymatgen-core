from __future__ import annotations

import re
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from pymatgen.core.structure import Site, Structure
from pymatgen.core.units import bohr_to_ang
from pymatgen.io.jdftx.inputs import (
    JDFTXInfile,
    JDFTXStructure,
    _infile_to_pmg_lattice,
    _remove_at_path,
    clean_infile_of_nans,
    selective_dynamics_site_prop_to_jdftx_interpretable,
)
from pymatgen.io.jdftx.jdftxinfile_default_inputs import antoinePvap, default_inputs
from pymatgen.io.jdftx.jdftxinfile_master_format import get_tag_object

from .inputs_test_utils import (
    assert_equiv_jdftxstructure,
    assert_idential_jif,
    ex_in_files_dir,
    ex_infile1_fname,
    ex_infile1_knowns,
    ex_infile2_fname,
    ex_infile3_fname,
)
from .shared_test_utils import assert_same_value

if TYPE_CHECKING:
    from collections.abc import Callable

    from pymatgen.io.jdftx.generic_tags import MultiformatTag, TagContainer
    from pymatgen.util.typing import PathLike


def test_jdftxinfile_structuregen():
    jif = JDFTXInfile.from_file(ex_infile1_fname)
    jdftxstruct = jif.to_jdftxstructure(jif)
    assert isinstance(jdftxstruct, JDFTXStructure)


@pytest.mark.parametrize(
    ("infile_fname", "bool_func"),
    [
        (ex_infile1_fname, lambda jif: all(jif["kpoint-folding"][x] == 1 for x in jif["kpoint-folding"])),
        (ex_infile1_fname, lambda jif: jif["elec-n-bands"]["n"] == 15),
    ],
)
def test_JDFTXInfile_known_lambda(infile_fname: str, bool_func: Callable[[JDFTXInfile], bool]):
    jif = JDFTXInfile.from_file(infile_fname)
    assert bool_func(jif)


def JDFTXInfile_self_consistency_tester(jif: JDFTXInfile, tmp_path: PathLike):
    """Create an assortment of JDFTXinfile created from the same data but through different methods, and test that
    they are all equivalent through "assert_idential_jif" """
    dict_jif = jif.as_dict()
    # # Commenting out tests with jif2 due to the list representation asserted
    jif2 = JDFTXInfile.get_dict_representation(JDFTXInfile._from_dict(dict_jif))
    str_list_jif = jif.get_text_list()
    str_jif = "\n".join(str_list_jif)
    jif3 = JDFTXInfile.from_str(str_jif)
    tmp_fname = tmp_path / "tmp.in"
    jif.write_file(tmp_fname)
    jif4 = JDFTXInfile.from_file(tmp_fname)
    jifs = [jif, jif2, jif3, jif4]
    for i in range(len(jifs)):
        for j in range(i + 1, len(jifs)):
            assert_idential_jif(jifs[i], jifs[j])


def test_JDFTXInfile_from_dict(tmp_path) -> None:
    jif = JDFTXInfile.from_file(ex_infile1_fname)
    jif_dict = jif.as_dict()
    # Test that dictionary can be modified and that _from_dict will fix set values
    jif_dict["elec-cutoff"] = 20
    jif2 = JDFTXInfile.from_dict(jif_dict)
    JDFTXInfile_self_consistency_tester(jif2, tmp_path)


@pytest.mark.parametrize("infile_fname", [ex_infile3_fname, ex_infile1_fname, ex_infile2_fname])
def test_JDFTXInfile_self_consistency_fromfile(infile_fname: PathLike, tmp_path) -> None:
    """Test that JDFTXInfile objects with different assortments of tags survive inter-conversion done within
    "JDFTXInfile_self_consistency_tester"""
    jif = JDFTXInfile.from_file(infile_fname)
    JDFTXInfile_self_consistency_tester(jif, tmp_path)


@pytest.mark.parametrize(
    ("val_key", "val"),
    [
        ("lattice", np.eye(3)),
        ("fluid-solvent", "H2O 0.5"),
        ("fluid-solvent", "H2O"),
        ("latt-scale", "1 1 1"),
        ("latt-scale", ["1", "1", "1"]),
        ("latt-scale", [1, 1, 1]),
        ("latt-scale", {"s0": 1, "s1": 1, "s2": 1}),
        ("elec-cutoff", {"Ecut": 20.0, "EcutRho": 100.0}),
        ("elec-cutoff", "20 100"),
        ("elec-cutoff", [20, 100]),
        ("elec-cutoff", 20),
    ],
)
def test_JDFTXInfile_set_values(val_key: str, val: Any, tmp_path) -> None:
    """Test value setting for various tags"""
    jif = JDFTXInfile.from_file(ex_infile1_fname)
    jif[val_key] = val
    # Test that the JDFTXInfile object is still consistent
    JDFTXInfile_self_consistency_tester(jif, tmp_path)


@pytest.mark.parametrize(
    ("val_key", "val"),
    [
        ("fluid-solvent", "H2O"),
        ("dump", "End DOS"),
        ("dump", "End DOS BandEigs"),
        ("dump-interval", "Electronic 1"),
        ("ion", "Fe 1 1 1 0"),
    ],
)
def test_JDFTXInfile_append_values(val_key: str, val: Any, tmp_path) -> None:
    """Test the append_tag method"""
    jif = JDFTXInfile.from_file(ex_infile1_fname)
    val_old = None if val_key not in jif else deepcopy(jif[val_key])
    jif.append_tag(val_key, val)
    val_new = jif[val_key]
    assert val_old != val_new
    # Test that the append_tag does not break the JDFTXInfile object
    JDFTXInfile_self_consistency_tester(jif, tmp_path)


def test_JDFTXInfile_expected_exceptions():
    jif = JDFTXInfile.from_file(ex_infile1_fname)
    with pytest.raises(KeyError):
        jif["barbie"] = "ken"
    # non-repeating tags raise value-errors when appended
    tag = "initial-state"
    with pytest.raises(ValueError, match=re.escape(f"The tag '{tag}' cannot be repeated and thus cannot be appended")):
        jif.append_tag(tag, "$VAR")
    # Phonon and Wannier tags raise value-errors at _preprocess_line
    with pytest.raises(ValueError, match="Phonon functionality has not been added!"):
        jif._preprocess_line("phonon idk")
    with pytest.raises(ValueError, match="Wannier functionality has not been added!"):
        jif._preprocess_line("wannier idk")
    # Tags not in MASTER_TAG_LIST raise value-errors at _preprocess_line
    err_str = f"The barbie tag in {['barbie', 'ken allan']} is not in MASTER_TAG_LIST and is not a comment, "
    "something is wrong with this input data!"
    with pytest.raises(ValueError, match=re.escape(err_str)):
        jif._preprocess_line("barbie ken allan")
    # include tags raise warnings if the file cannot be found
    _filename = "barbie"
    err_str = f"The include file {_filename} ({_filename}) does not exist!"
    with pytest.warns(UserWarning, match=re.escape(err_str)):
        JDFTXInfile.from_str(f"include {_filename}\n", dont_require_structure=True)
    # If it does exist, no error should be raised
    filename = ex_in_files_dir / "barbie"
    err_str = f"The include file {_filename} ({filename}) does not exist!"
    # str(err_str) # What does this do?
    # If the wrong parent_path is given for a file that does exist, error
    with pytest.warns(UserWarning, match=re.escape(err_str)):
        JDFTXInfile.from_str(f"include {_filename}\n", path_parent=ex_in_files_dir, dont_require_structure=True)
    # JDFTXInfile cannot be constructed without lattice and ion tags
    with pytest.raises(ValueError, match="This input file is missing required structure tags"):
        JDFTXInfile.from_str("dump End DOS\n")
    # "barbie" here is supposed to be "list-to-dict" or "dict-to-list"
    with pytest.raises(ValueError, match="Conversion type barbie is not 'list-to-dict' or 'dict-to-list'"):
        jif._needs_conversion("barbie", ["ken"])
    # Setting tags with unfixable values immediately raises an error
    tag = "exchange-params"
    value = {"blockSize": 1, "nOuterVxx": "barbie"}
    err_str = str(f"The {tag} tag with value:\n{value}\ncould not be fixed!")
    with pytest.raises(ValueError, match=re.escape(err_str)):
        # Implicitly tests validate_tags
        jif[tag] = value
    # `copy` will end up trying to validate the tags, so we need to remove the offending tag first
    jif.pop(tag)
    jif2 = jif.copy()
    # Setting tags with unfixable values through "update" side-steps the error, but will raise it once
    # "validate_tags" is inevitably called
    jif2.update({tag: value})
    with pytest.raises(ValueError, match=re.escape(err_str)):
        jif2.validate_tags(try_auto_type_fix=True)
    # The inevitable error can be reduced to a warning if you tell it not to try to fix the values
    with pytest.warns(UserWarning, match="The exchange-params tag with value"):
        jif2.validate_tags(try_auto_type_fix=False)
    # Setting a non-string tag raises an error within the JDFTXInfile object
    err_str = str(f"{1.2} is not a string!")
    with pytest.raises(TypeError, match=err_str):
        jif[1.2] = 3.4


def test_JDFTXInfile_strip_structure():
    jif = JDFTXInfile.from_file(ex_infile1_fname)
    structural_tags = ["lattice", "ion", "coords-type"]
    assert all(tag in jif for tag in structural_tags)
    jif.strip_structure_tags()
    assert all(tag not in jif for tag in structural_tags)


def test_JDFTXInfile_niche_cases():
    jif = JDFTXInfile.from_file(ex_infile1_fname)
    tag_object, tag, value = jif._preprocess_line("dump-only")
    assert value == ("")
    tag = "elec-ex-corr"
    tag_object = get_tag_object(tag)
    value = "gga"
    params = jif.as_dict()
    err_str = f"The '{tag}' tag appears multiple times in this input when it should not!"
    with pytest.raises(ValueError, match=err_str):
        jif._store_value(params, tag_object, tag, value)
    struct = jif.to_pmg_structure(jif)
    assert isinstance(struct, Structure)
    noneout = jif.validate_tags(return_list_rep=True)
    assert noneout is None
    jif["fluid-solvent"] = {"name": "H2O", "concentration": 0.5}
    assert len(jif["fluid-solvent"]) == 1
    jif.append_tag("fluid-solvent", {"name": "H2O", "concentration": 0.5})
    assert len(jif["fluid-solvent"]) == 2


def test_JDFTXInfile_add_method():
    """Test the __add__ method"""
    # No new values are being assigned in jif2, so jif + jif2 should be the same as jif
    # Since the convenience of this method would be lost if the user has to pay special attention to duplicating
    # repeatable values, repeatable tags are not append to each other
    jif = JDFTXInfile.from_file(ex_infile1_fname)
    jif2 = jif.copy()
    assert jif2 is not jif  # Testing robustness of copy method while we are at it
    jif3 = jif + jif2
    assert_idential_jif(jif, jif3)
    # If a tag is repeated, the values must be the same since choice of value is ambiguous
    key = "elec-ex-corr"
    val_old = deepcopy(jif[key])
    val_new = "lda"
    assert val_old != val_new
    jif2[key] = val_new
    jif4 = jif + jif2
    assert_same_value(jif4[key], val_new)  # Make sure addition chooses second value for non-repeatable tags
    del jif4
    jif2.append_tag("dump", "Fluid State")
    jif.append_tag("dump", "Fluid Berry")
    jif4 = jif + jif2
    assert len(jif4["dump"]) == len(jif2["dump"]) + 1
    assert {"Fluid": {"State": True}} in jif4["dump"]
    assert {"Fluid": {"State": True}} not in jif["dump"]
    assert {"Fluid": {"Berry": True}} in jif4["dump"]
    assert {"Fluid": {"Berry": True}} not in jif2["dump"]
    # Normal expected behavior
    key_add = "target-mu"
    val_add = 0.5
    assert key_add not in jif
    jif2 = jif.copy()
    jif2[key_add] = val_add
    jif3 = jif + jif2
    assert jif3[key_add]["mu"] == pytest.approx(val_add)


@pytest.mark.parametrize(("infile_fname", "knowns"), [(ex_infile1_fname, ex_infile1_knowns)])
def test_JDFTXInfile_knowns_simple(infile_fname: PathLike, knowns: dict):
    """Test that known values that can be tested with assert_same_value are correct"""
    jif = JDFTXInfile.from_file(infile_fname)
    for key, val in knowns.items():
        assert_same_value(jif[key], val)


def test_jdftxstructure():
    """Test the JDFTXStructure object associated with the JDFTXInfile object"""
    jif = JDFTXInfile.from_file(ex_infile2_fname)
    jstruct = jif.to_jdftxstructure(jif)
    assert isinstance(jstruct, JDFTXStructure)
    struc_str = str(jstruct)
    assert isinstance(struc_str, str)
    newstruct = JDFTXStructure.from_str(struc_str)
    assert isinstance(newstruct, JDFTXStructure)
    # Double checking I got the column/row order right
    assert_same_value(jstruct.structure.lattice, newstruct.structure.lattice)
    assert jstruct.natoms == 16
    with open(ex_infile2_fname) as f:
        lines = list.copy(list(f))
    # Test different ways of creating a JDFTXStructure object create the same object if data is the same
    data = "\n".join(lines)
    struct2 = JDFTXStructure.from_str(data)
    assert_equiv_jdftxstructure(jstruct, struct2)
    struct3 = JDFTXStructure.from_dict(jstruct.as_dict())
    assert_equiv_jdftxstructure(jstruct, struct3)


def test_disordered_structures():
    """Test that disordered structures are handled correctly"""
    jif = JDFTXInfile.from_file(ex_infile2_fname)
    struct = jif.structure
    # Making a disordered structure by replacing some sites with partial occupancies
    struct.sites[1] = Site(species={"Si": 0.5, "Ge": 0.5}, coords=struct.sites[1].frac_coords)
    with pytest.raises(
        ValueError, match="Disordered structure with partial occupancies cannot be converted into JDFTXStructure!"
    ):
        JDFTXStructure(structure=struct)


def test_jdftxstructure_infile_site_properties_conversion():
    """Test conversions related to going back and forth between JDFTXInfile and JDFTXStructure/Structure"""
    jif = JDFTXInfile.from_file(ex_infile2_fname)
    jif["ion"][0]["moveScale"] = 0
    jif["ion"][1]["moveScale"] = 1
    for i in range(len(jif["ion"])):
        jif["ion"][i]["v"] = {"vx0": 0.0, "vx1": 0.0, "vx2": 0.0}
    jif["ion"][3]["v"] = {"vx0": 1.0, "vx1": 2.0, "vx2": 3.0}
    jstruct = jif.to_jdftxstructure(jif)
    struct = jif.structure
    # moveScale is stored as a per-direction selective dynamics site property
    assert struct.site_properties["selective_dynamics"][:2] == [[False] * 3, [True] * 3]
    jif_from_jstruc = JDFTXInfile.from_jdftxstructure(jstruct)
    # Since velocities and moveScale are compatible to be stored as site properties, exporting to
    # `Structure` will save them in `Structure.site_properties`. Initializing a new JDFTXInfile
    # from this `Structure` should yield the same result as `jif_from_jstruc`.
    jif_from_struc = JDFTXInfile.from_structure(struct)
    # Providing a site property as an argument will override the site property in the Structure, however
    # the format for 'selective_dynamics' in `Structure.site_properties` is not the same as that expected
    # by JDFTXInfile.
    # Testing here that the wrong format can be properly mapped in the JDFTx 'moveScale' format.
    jif_from_struc_2 = JDFTXInfile.from_structure(
        struct, selective_dynamics=struct.site_properties["selective_dynamics"]
    )
    # Providing selective dynamics in the JDFTx expected format should also work
    jif_from_struc_3 = JDFTXInfile.from_structure(
        struct,
        selective_dynamics=selective_dynamics_site_prop_to_jdftx_interpretable(
            struct.site_properties["selective_dynamics"]
        ),
    )
    for jif2 in [jif_from_jstruc, jif_from_struc, jif_from_struc_2, jif_from_struc_3]:
        assert isinstance(jif2, JDFTXInfile)
        assert_idential_jif(jif["ion"], jif2["ion"])


def test_pmg_struc():
    jif = JDFTXInfile.from_file(ex_infile2_fname)
    struc1 = jif.to_pmg_structure(jif)
    struc2 = jif.structure
    for s in [struc1, struc2]:
        assert isinstance(s, Structure)
    assert_idential_jif(struc1.as_dict(), struc2.as_dict())


def test_jdftxtructure_naming():
    """Test the naming of the JDFTXStructure object.

    Test to make sure reading from a Structure with labels not exactly matching the element names
    (ie Si0, Si1, or Si+2) will still be read correctly.
    """
    struct = Structure.from_file(ex_in_files_dir / "Si.cif")
    jstruct = JDFTXStructure(structure=struct)
    JDFTXInfile.from_jdftxstructure(jstruct)
    JDFTXInfile.from_structure(struct)


@pytest.mark.parametrize(
    ("value_str", "expected_dict"),
    [
        (
            "H 1.0 1.0 1.0 0",
            {
                "species-id": "H",
                "x0": 1.0,
                "x1": 1.0,
                "x2": 1.0,
                "moveScale": 0,
            },
        ),
        (
            "H 1.0 1.0 1.0 v 1.0 1.0 1.0 0",
            {
                "species-id": "H",
                "x0": 1.0,
                "x1": 1.0,
                "x2": 1.0,
                "v": {
                    "vx0": 1.0,
                    "vx1": 1.0,
                    "vx2": 1.0,
                },
                "moveScale": 0,
            },
        ),
        (
            "H 1.0 1.0 1.0 1 Linear 1.0 1.0 1.0",
            {
                "species-id": "H",
                "x0": 1.0,
                "x1": 1.0,
                "x2": 1.0,
                "moveScale": 1,
                "constraint type": "Linear",
                "d0": 1.0,
                "d1": 1.0,
                "d2": 1.0,
            },
        ),
        (
            "H 1.0 1.0 1.0 1 HyperPlane 1.0 1.0 1.0 g1 HyperPlane 1.0 1.0 1.0 g1",
            {
                "species-id": "H",
                "x0": 1.0,
                "x1": 1.0,
                "x2": 1.0,
                "moveScale": 1,
                "HyperPlane": [
                    {
                        "d0": 1.0,
                        "d1": 1.0,
                        "d2": 1.0,
                        "group": "g1",
                    },
                    {
                        "d0": 1.0,
                        "d1": 1.0,
                        "d2": 1.0,
                        "group": "g1",
                    },
                ],
            },
        ),
    ],
)
def test_ion_reading(value_str: str, expected_dict: dict):
    ion_tag: MultiformatTag = get_tag_object("ion")
    i = ion_tag.get_format_index_for_str_value("ion", value_str)
    tag_object: TagContainer = ion_tag.format_options[i]
    parsed_tag = tag_object.read("ion", value_str)
    assert_same_value(parsed_tag, expected_dict)


@pytest.mark.parametrize(
    ("expected_out", "stored_dict"),
    [
        (
            "ion H 1.000000000000 1.000000000000 1.000000000000 0",
            {
                "species-id": "H",
                "x0": 1.0,
                "x1": 1.0,
                "x2": 1.0,
                "moveScale": 0,
            },
        ),
        (
            "ion H 1.000000000000 1.000000000000 1.000000000000 v 1.000000000000 1.000000000000 1.000000000000 0",
            {
                "species-id": "H",
                "x0": 1.0,
                "x1": 1.0,
                "x2": 1.0,
                "v": {
                    "vx0": 1.0,
                    "vx1": 1.0,
                    "vx2": 1.0,
                },
                "moveScale": 0,
            },
        ),
        (
            "ion H 1.000000000000 1.000000000000 1.000000000000 1 Linear 1.000000000000 1.000000000000 1.000000000000",
            {
                "species-id": "H",
                "x0": 1.0,
                "x1": 1.0,
                "x2": 1.0,
                "moveScale": 1,
                "constraint type": "Linear",
                "d0": 1.0,
                "d1": 1.0,
                "d2": 1.0,
            },
        ),
        (
            "ion H 1.000000000000 1.000000000000 1.000000000000 1 HyperPlane 1.000000000000 1.000000000000 "
            "1.000000000000 g1 HyperPlane 1.000000000000 1.000000000000 1.000000000000 g1",
            {
                "species-id": "H",
                "x0": 1.0,
                "x1": 1.0,
                "x2": 1.0,
                "moveScale": 1,
                "HyperPlane": [
                    {
                        "d0": 1.0,
                        "d1": 1.0,
                        "d2": 1.0,
                        "group": "g1",
                    },
                    {
                        "d0": 1.0,
                        "d1": 1.0,
                        "d2": 1.0,
                        "group": "g1",
                    },
                ],
            },
        ),
    ],
)
def test_ion_writing(expected_out: str, stored_dict: dict):
    ion_tag: MultiformatTag = get_tag_object("ion")
    i, _ = ion_tag._determine_format_option("ion", stored_dict)
    tag_object: TagContainer = ion_tag.format_options[i]
    output = tag_object.write("ion", stored_dict)
    assert_same_value(output.strip().split(), expected_out.strip().split())


@pytest.mark.parametrize(
    ("lattice_type", "value_str"),
    [
        ("Rhombohedral", "Rhombohedral 1.0 1.0"),
        ("Triclinic", "Triclinic 1.0 1.0 1.0 1.0 1.0 1.0"),
        ("Hexagonal", "Hexagonal 1.0 1.0"),
        ("Cubic", "Body-Centered Cubic 1.0"),
        ("Cubic", "Cubic 1.0"),
    ],
)
def test_lattice_reading(lattice_type: str, value_str: str):
    mft_lattice_tag = get_tag_object("lattice")
    assert mft_lattice_tag is not None
    i = mft_lattice_tag.get_format_index_for_str_value("lattice", value_str)
    tag_object = mft_lattice_tag.format_options[i]
    parsed_tag = tag_object.read("lattice", value_str)
    assert lattice_type in parsed_tag


@pytest.mark.parametrize(
    ("value_str"),
    [
        ("Rhombohedral 1.0 1.0"),
        ("Triclinic 1.0 1.0 1.0 1.0 1.0 1.0"),
        ("Hexagonal 1.0 1.0"),
        ("Body-Centered Cubic 1.0"),
        ("Cubic 1.0"),
        ("Orthorhombic 1.0 1.0 1.0"),
        ("Base-Centered Orthorhombic 1.0 1.0 1.0"),
        ("Monoclinic 1.0 1.0 1.0 1.0"),
        ("Base-Centered Monoclinic 1.0 1.0 1.0 1.0"),
        ("Tetragonal 1.0 1.0"),
        ("Body-Centered Tetragonal 1.0 1.0"),
    ],
)
def test_lattice_writing(value_str: str):
    mft_lattice_tag = get_tag_object("lattice")
    assert mft_lattice_tag is not None
    i = mft_lattice_tag.get_format_index_for_str_value("lattice", value_str)
    tag_object = mft_lattice_tag.format_options[i]
    parsed_tag = tag_object.read("lattice", value_str)
    output = tag_object.write("lattice", parsed_tag)
    assert_same_value(
        ("lattice " + value_str).strip().split(),
        output.strip().split(),
    )


@pytest.mark.parametrize(
    ("value_str"),
    [
        ("Rhombohedral 1.0 1.0"),
        ("Triclinic 1.0 1.0 1.0 1.0 1.0 1.0"),
        ("Hexagonal 1.0 1.0"),
        ("Body-Centered Cubic 1.0"),
        ("Cubic 1.0"),
        ("Orthorhombic 1.0 1.0 1.0"),
        ("Base-Centered Orthorhombic 1.0 1.0 1.0"),
        ("Monoclinic 1.0 1.0 1.0 1.0"),
        ("Base-Centered Monoclinic 1.0 1.0 1.0 1.0"),
        ("Tetragonal 1.0 1.0"),
        ("Body-Centered Tetragonal 1.0 1.0"),
    ],
)
def test_jdftxstructure_lattice_conversion(value_str: str):
    test_vars = ["a", "b", "c", "alpha", "beta", "gamma"]
    conv_vars = ["a", "b", "c"]
    mft_lattice_tag = get_tag_object("lattice")
    assert mft_lattice_tag is not None
    i = mft_lattice_tag.get_format_index_for_str_value("lattice", value_str)
    tag_object = mft_lattice_tag.format_options[i]
    parsed_tag = tag_object.read("lattice", value_str)
    infile = JDFTXInfile.from_str("lattice " + value_str + "\n ion H 0.0 0.0 0.0 0", dont_require_structure=True)
    if "modification" in parsed_tag:
        with pytest.raises(NotImplementedError):
            _ = infile.to_pmg_structure(infile)
    else:
        structure = infile.to_pmg_structure(infile)
        for var in test_vars:
            if var in parsed_tag:
                should_be = parsed_tag[var]
                if var in conv_vars:
                    should_be *= bohr_to_ang
                assert_same_value(float(getattr(structure.lattice, var)), float(should_be))


def test_jdftxinfile_comparison():
    jif1 = JDFTXInfile.from_file(ex_infile1_fname)
    jif2 = JDFTXInfile.from_file(ex_infile2_fname)
    assert len(jif1.get_differing_tags(jif2))  # At least one tag should be different
    jif1copy = jif1.copy()
    assert not len(jif1.get_differing_tags(jif1copy))  # No tags should be different
    default_test_tag = "davidson-band-ratio"
    default_test_val = default_inputs[default_test_tag]
    jif1[default_test_tag] = default_test_val
    assert not len(jif1.get_differing_tags(jif1copy))  # Even though jif1copy doesn't have the tag,
    # it won't be recognized as a difference since it is in the default_inputs and matches the default value
    jif1copy["elec-n-bands"] = 20001
    assert len(jif1.get_filtered_differing_tags(jif1copy))  # Change in elec-n-bands should be recognized
    assert not len(
        jif1.get_filtered_differing_tags(jif1copy, exclude_tags=["elec-n-bands"])
    )  # Specific tags can be filtered out
    assert not len(
        jif1.get_filtered_differing_tags(jif1copy, exclude_tag_categories=["electronic"])
    )  # Tag categories can be filtered out


def test_antoine_pvap():
    assert_same_value(antoinePvap(298, 7.31549, 1794.88, -34.764), 1.06736e-10)


def test_nan_stripping():
    jif = JDFTXInfile.from_file(ex_infile1_fname)
    # Single optional subtag being nan should only remove that subtag
    jif["elec-cutoff"]["EcutRho"] = np.nan
    jif = clean_infile_of_nans(jif)
    assert "EcutRho" not in jif["elec-cutoff"]
    # Single required subtag being nan should remove the whole tag
    jif["latt-move-scale"]["s0"] = np.nan
    jif = clean_infile_of_nans(jif)
    assert "latt-move-scale" not in jif
    jif.pop("fluid-solvent")
    jif.read_line("fluid-solvent H2O 55.338 ScalarEOS epsBulk nan")
    assert "epsBulk" in jif["fluid-solvent"][0]
    jif = clean_infile_of_nans(jif)
    assert "epsBulk" not in jif["fluid-solvent"][0]
    print(jif)


def _species_values(*groups: tuple[str, list[float]]) -> dict:
    return {"species-values": [{"species-id": sp, "values": vals} for sp, vals in groups]}


@pytest.mark.parametrize(
    ("magmom_value", "oxstate_value", "magmom_expected", "oxstate_expected"),
    [
        (
            "Mn 3 -3 3",
            "Mn +4 O -2",
            _species_values(("Mn", [3.0, -3.0, 3.0])),
            _species_values(("Mn", [4.0]), ("O", [-2.0])),
        ),
        (
            "Mn 3.0 0 0 3 180 90 O 0 0 0",
            "Mn 4",
            _species_values(("Mn", [3.0, 0.0, 0.0, 3.0, 180.0, 90.0]), ("O", [0.0, 0.0, 0.0])),
            _species_values(("Mn", [4.0])),
        ),
        (
            "Mn 1e-1 -2.5E+0 O 0",
            "Mn +4.0 O -2.0",
            _species_values(("Mn", [0.1, -2.5]), ("O", [0.0])),
            _species_values(("Mn", [4.0]), ("O", [-2.0])),
        ),
    ],
)
def test_species_values_tags_infile_roundtrip(
    magmom_value: str, oxstate_value: str, magmom_expected: dict, oxstate_expected: dict
):
    text = f"spintype z-spin\ninitial-magnetic-moments {magmom_value}\ninitial-oxidation-state {oxstate_value}\n"
    first = JDFTXInfile.from_str(text, dont_require_structure=True)
    assert first["initial-magnetic-moments"] == magmom_expected
    assert first["initial-oxidation-state"] == oxstate_expected
    # Writing and re-reading gives back the same data (the text itself may differ, e.g. "3" is written as "3.0")
    second = JDFTXInfile.from_str(str(first), dont_require_structure=True)
    assert second["initial-magnetic-moments"] == magmom_expected
    assert second["initial-oxidation-state"] == oxstate_expected
    assert first.is_comparable_to(second)


def test_initial_magnetic_moments_comparison():
    base = "spintype z-spin\ninitial-magnetic-moments "
    jif = JDFTXInfile.from_str(base + "Mn 3 -3 O 0", dont_require_structure=True)
    # Species groups can come in any order
    assert jif.is_comparable_to(JDFTXInfile.from_str(base + "O 0 Mn 3 -3", dont_require_structure=True))
    # Values within a group are per-ion, so their order matters
    assert not jif.is_comparable_to(JDFTXInfile.from_str(base + "Mn -3 3 O 0", dont_require_structure=True))
    assert not jif.is_comparable_to(JDFTXInfile.from_str(base + "Mn 3 -3 O 1", dont_require_structure=True))


def test_initial_magnetic_moments_setitem():
    jif = JDFTXInfile.from_str("spintype z-spin\n", dont_require_structure=True)
    jif["initial-magnetic-moments"] = "Fe 2 -2 O 0"
    assert jif["initial-magnetic-moments"] == _species_values(("Fe", [2.0, -2.0]), ("O", [0.0]))
    # Integer values given in dict form are converted to floats
    jif["initial-magnetic-moments"] = _species_values(("Fe", [2, -2]))
    assert jif["initial-magnetic-moments"] == _species_values(("Fe", [2.0, -2.0]))
    assert all(isinstance(v, float) for v in jif["initial-magnetic-moments"]["species-values"][0]["values"])
    assert "initial-magnetic-moments Fe 2.0 -2.0 " in str(jif)


ION_LINES = "ion Mn 0.0 0.0 0.0 1\nion Mn 0.5 0.5 0.5 1\nion O 0.25 0.25 0.25 1\n"


@pytest.mark.parametrize(
    "text",
    [
        "spintype z-spin\n",
        "spintype z-spin\ninitial-magnetic-moments Mn 3 -3 O 0\n" + ION_LINES,
        "spintype z-spin\ninitial-magnetic-moments Mn 3 -3\n",
        "spintype vector-spin\ninitial-magnetic-moments Mn 3 0 0 3 180 0\n",
        "spintype spin-orbit\ninitial-magnetic-moments Mn 3 0 0 3 180 0 O 0 0 0\n" + ION_LINES,
    ],
)
def test_validate_initial_magnetic_moments_valid(text: str):
    JDFTXInfile.from_str(text, dont_require_structure=True).validate_initial_magnetic_moments()


@pytest.mark.parametrize(
    ("text", "error_match"),
    [
        ("initial-magnetic-moments Mn 3\n", "spintype tag must be specified"),
        ("spintype spin-orbit\ninitial-magnetic-moments Mn 3 -3\n", "must have a multiple of 3 values, got 2"),
        ("spintype z-spin\ninitial-magnetic-moments Fe 3\n" + ION_LINES, "Fe in initial-magnetic-moments tag is not"),
        ("spintype z-spin\ninitial-magnetic-moments Mn 3 O 0\n" + ION_LINES, "but there are 2 ions"),
        ("spintype vector-spin\ninitial-magnetic-moments Mn 3 0 0\n" + ION_LINES, "values for 1 ions, but there are 2"),
    ],
)
def test_validate_initial_magnetic_moments_invalid(text: str, error_match: str):
    jif = JDFTXInfile.from_str(text, dont_require_structure=True)
    with pytest.raises(ValueError, match=error_match):
        jif.validate_initial_magnetic_moments()


def _u_channel(species: str, orb_desc: str, u_minus_j: float, vexts: list[tuple[int, float]] | None = None) -> dict:
    channel: dict[str, Any] = {"species": species, "orbDesc": orb_desc, "UminusJ": u_minus_j}
    if vexts is not None:
        channel["Vext"] = [{"atom": atom, "V": v} for atom, v in vexts]
    return channel


@pytest.mark.parametrize(
    ("text", "expected_channels"),
    [
        ("add-U Mn d 0.06\n", [_u_channel("Mn", "d", 0.06)]),
        ("add-U Mn d 0.06 O p 0.1\n", [_u_channel("Mn", "d", 0.06), _u_channel("O", "p", 0.1)]),
        # How JDFTx echoes add-U in out files: one channel per continuation line
        ("add-U \tMn d 0.06 \\\n\tO p 0.1\n", [_u_channel("Mn", "d", 0.06), _u_channel("O", "p", 0.1)]),
        ("add-U Mn d 0.06 Vext 1 0.1 Vext 2 -0.1\n", [_u_channel("Mn", "d", 0.06, [(1, 0.1), (2, -0.1)])]),
        (
            "add-U Mn d 0.06 Vext 1 0.1 O 2p 0.1 Vext 3 0.2\n",
            [_u_channel("Mn", "d", 0.06, [(1, 0.1)]), _u_channel("O", "2p", 0.1, [(3, 0.2)])],
        ),
        (
            "add-U Mn d 0.06 O 2p 0.1 Vext 1 0.1\n",
            [_u_channel("Mn", "d", 0.06), _u_channel("O", "2p", 0.1, [(1, 0.1)])],
        ),
    ],
)
def test_add_u_read_write_roundtrip(text: str, expected_channels: list[dict]):
    jif = JDFTXInfile.from_str(text, dont_require_structure=True)
    assert jif["add-U"] == {"U-channels": expected_channels}
    written = str(jif)
    # JDFTx only accepts add-U once, so every channel must be written on a single add-U line
    assert written.count("add-U") == 1
    jif2 = JDFTXInfile.from_str(written, dont_require_structure=True)
    assert jif2["add-U"] == jif["add-U"]
    assert jif.is_comparable_to(jif2)


@pytest.mark.parametrize(
    ("text", "error_match"),
    [
        ("add-U Mn d 0.06\nadd-U O p 0.1\n", "The 'add-U' tag appears multiple times"),
        ("add-U Mn d\n", "The subtag 'U-channels' for tag 'add-U' is not optional"),
        ("add-U Mn d 0.06 Vext 1\n", "The subtag 'U-channels' for tag 'add-U' is not optional"),
        ("add-U Vext 1 0.1\n", "The subtag 'U-channels' for tag 'add-U' is not optional"),
        ("add-U Mn d 0.06 O p\n", re.escape("were not processed: ['O', 'p']")),
    ],
)
def test_add_u_read_errors(text: str, error_match: str):
    with pytest.raises(ValueError, match=error_match):
        JDFTXInfile.from_str(text, dont_require_structure=True)


def test_add_u_comparison():
    def from_str(value: str) -> JDFTXInfile:
        return JDFTXInfile.from_str(f"add-U {value}\n", dont_require_structure=True)

    jif = from_str("Mn d 0.06 Vext 1 0.1 Vext 2 -0.1 O p 0.1")
    # Neither channel order nor Vext order changes the input
    assert jif.is_comparable_to(from_str("O p 0.1 Mn d 0.06 Vext 2 -0.1 Vext 1 0.1"))
    assert not jif.is_comparable_to(from_str("Mn d 0.06 Vext 1 0.1 Vext 2 -0.1 O p 0.2"))
    # A Vext belongs to the channel it follows
    assert not jif.is_comparable_to(from_str("Mn d 0.06 Vext 1 0.1 O p 0.1 Vext 2 -0.1"))


def test_add_u_setitem():
    # Setting a value runs type validation, which must accept channels that differ in having Vext entries
    jif = JDFTXInfile.from_str("spintype z-spin\n", dont_require_structure=True)
    expected = {"U-channels": [_u_channel("Mn", "d", 0.06), _u_channel("O", "p", 0.1, [(1, 0.1)])]}
    jif["add-U"] = "Mn d 0.06 O p 0.1 Vext 1 0.1"
    assert jif["add-U"] == expected
    jif["add-U"] = {"U-channels": [_u_channel("Mn", "d", 0.06), _u_channel("O", "p", 0.1, [(1, 0.1)])]}
    assert jif["add-U"] == expected


def _ion_lines(*ions: tuple[str, str]) -> str:
    return "".join(f"ion {species} {x0} 0.0 0.0 1\n" for species, x0 in ions)


@pytest.mark.parametrize(
    ("text", "expected_species"),
    [
        # A nan in a non-optional ion subtag removes that ion
        (_ion_lines(("Mn", "0.0"), ("Fe", "nan"), ("O", "0.5")), ["Mn", "O"]),
        # Several nan ions are all removed, without later indices shifting onto the wrong ion
        (_ion_lines(("Mn", "0.0"), ("Fe", "nan"), ("Co", "nan"), ("O", "0.5")), ["Mn", "O"]),
        (_ion_lines(("Fe", "nan"), ("Mn", "0.0"), ("Co", "nan")), ["Mn"]),
        # Two nans in the same ion remove only that ion
        ("ion Mn nan nan 0.0 1\nion O 0.5 0.5 0.5 1\n", ["O"]),
        # If every ion is removed, the ion tag is removed instead of being left empty
        (_ion_lines(("Mn", "nan"), ("O", "nan")), None),
    ],
)
def test_nan_stripping_repeated_tag_entries(text: str, expected_species: list[str] | None):
    jif = JDFTXInfile.from_str("spintype z-spin\n" + text, dont_require_structure=True)
    cleaned = clean_infile_of_nans(jif)
    if expected_species is None:
        assert "ion" not in cleaned
    else:
        assert [ion["species-id"] for ion in cleaned["ion"]] == expected_species
    assert cleaned["spintype"] == "z-spin"
    # The original infile is left untouched
    assert len(jif["ion"]) == text.count("ion ")


def test_remove_at_path():
    params: dict[str, Any] = {"a": [1.0, 2.0, 3.0], "b": {"c": [4.0], "d": 5.0}}
    _remove_at_path(params, ["a", 1])
    assert params["a"] == [1.0, 3.0]
    _remove_at_path(params, ["b", "c", 0])  # emptied list is removed from its parent
    assert params["b"] == {"d": 5.0}
    _remove_at_path(params, ["b", "missing", 0])  # paths that no longer exist are ignored
    _remove_at_path(params, ["a", 5])
    assert params == {"a": [1.0, 3.0], "b": {"d": 5.0}}


@pytest.mark.parametrize(
    "lattice_value",
    [
        "Cubic 10.0",
        "Tetragonal 4.0 6.0",
        "Hexagonal 4.0 6.0",
        "Orthorhombic 4.0 5.0 6.0",
        "Rhombohedral 5.0 70.0",
        "Monoclinic 4.0 5.0 6.0 100.0",
        "Triclinic 4.0 5.0 6.0 80.0 85.0 95.0",
    ],
)
def test_named_lattice_latt_scale(lattice_value: str):
    scale = np.array([2.0, 0.5, 1.8897])
    unscaled = _infile_to_pmg_lattice(JDFTXInfile.from_str(f"lattice {lattice_value}\n", dont_require_structure=True))
    scaled = _infile_to_pmg_lattice(
        JDFTXInfile.from_str(
            f"lattice {lattice_value}\nlatt-scale {scale[0]} {scale[1]} {scale[2]}\n", dont_require_structure=True
        )
    )
    # Lattice vector k is scaled by s_k, so its length scales and the angles between vectors are unchanged
    np.testing.assert_allclose(scaled.abc, np.array(unscaled.abc) * scale)
    np.testing.assert_allclose(scaled.angles, unscaled.angles)


def test_latt_scale_named_and_matrix_lattices_agree():
    named = JDFTXInfile.from_str("lattice Cubic 10.0\nlatt-scale 2 1 1.5\n", dont_require_structure=True)
    matrix = JDFTXInfile.from_str(
        "lattice \\\n 10 0 0 \\\n 0 10 0 \\\n 0 0 10\nlatt-scale 2 1 1.5\n", dont_require_structure=True
    )
    np.testing.assert_allclose(_infile_to_pmg_lattice(named).abc, _infile_to_pmg_lattice(matrix).abc)
    np.testing.assert_allclose(_infile_to_pmg_lattice(named).abc, np.array([20.0, 10.0, 15.0]) * bohr_to_ang)


def test_latt_scale_keeps_fractional_values():
    # e.g. an angstrom-to-bohr conversion factor must not be truncated to 1
    jif = JDFTXInfile.from_str("latt-scale 1.8897 1.8897 1.8897\n", dont_require_structure=True)
    assert jif["latt-scale"] == {"s0": 1.8897, "s1": 1.8897, "s2": 1.8897}
    assert JDFTXInfile.from_str(str(jif), dont_require_structure=True)["latt-scale"] == jif["latt-scale"]


def test_line_continuation():
    # Whitespace around a "\" collapses to a single space, and comments are stripped before lines are joined
    joined = JDFTXInfile.from_str(
        "lattice \\\n 10 0 0  \\  # comment\n0 10 0\\\n0 0 10\nelec-cutoff 20\n\\\nelec-ex-corr gga\n",
        dont_require_structure=True,
    )
    single = JDFTXInfile.from_str(
        "lattice 10 0 0 0 10 0 0 0 10\nelec-cutoff 20\nelec-ex-corr gga\n", dont_require_structure=True
    )
    assert joined == single


@pytest.mark.parametrize(
    ("selective_dynamics", "expected"),
    [
        ([[True, True, True], [True, False, True]], [1, 0]),
        (np.array([[1, 1, 1], [0, 0, 0]]), [1, 0]),
        ([(1, 1, 1), np.array([0.0, 1.0, 1.0]), [], True, 0.0, None], [1, 0, 1, 1, 0, 0]),
    ],
)
def test_selective_dynamics_site_prop_to_jdftx_interpretable(selective_dynamics: Any, expected: list[int]):
    converted = selective_dynamics_site_prop_to_jdftx_interpretable(selective_dynamics)
    np.testing.assert_array_equal(converted, expected)
    assert converted.dtype.kind == "i"
