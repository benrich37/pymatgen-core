from __future__ import annotations

from pymatgen.io.jdftx.inputs import JDFTXInfile

# from .shared_test_utils import assert_same_value

sample_infile = "add-U Mn d 0.06 \n initial-magnetic-moments Mn 3.0"

infile = JDFTXInfile.from_str("add-U Mn d 0.06 \n initial-magnetic-moments Mn 3.0", dont_require_structure=True)
print(infile["add-U"])

print(infile)
