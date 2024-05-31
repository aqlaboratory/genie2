"""
Residue constants used in Genie 2.
"""
from collections import OrderedDict


# Mapping from one-letter residue name to three-letter residue name
RESTYPE_1_TO_3 = OrderedDict({
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL',
})

# Mapping from three-letter residue name to one-letter residue name
RESTYPE_3_TO_1 = {v: k for k, v in RESTYPE_1_TO_3.items()}

# List of residue names
RESTYPES = list(RESTYPE_1_TO_3.keys())

# Mapping from one-letter residue name to its correponding index in the list of residue names
RESTYPE_ORDER = {restype: i for i, restype in enumerate(RESTYPES)}
