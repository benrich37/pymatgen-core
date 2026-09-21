from __future__ import annotations

from pymatgen.io.jdftx.inputs import JDFTXInfile

# from .shared_test_utils import assert_same_value

sample_infile = "add-U Mn d 1.4"

infile = JDFTXInfile.from_str(sample_infile, dont_require_structure=True)
print(infile["add-U"])
