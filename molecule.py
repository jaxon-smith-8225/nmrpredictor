"""
molecule.py
-----------
Define and build the target molecule for NMR prediction.

Currently set up for acetone (propan-2-one), the initial NMR target.
Swap out `mol.atom` and `mol.basis` here to change molecules or basis sets.
"""

from pyscf import gto
import numpy as np


def build_acetone(basis: str = "sto-3g", verbose: int = 0) -> gto.Mole:
    """
    Build and return a PySCF Mole object for acetone.

    Parameters
    ----------
    basis   : str  — basis set name (e.g. 'sto-3g', '6-31g*', 'cc-pVDZ')
    verbose : int  — PySCF verbosity level (0 = silent)

    Returns
    -------
    mol : pyscf.gto.Mole  (already built)
    """
    mol = gto.Mole()
    mol.atom = """
        C   -1.2678   0.0000   0.0000
        C    0.0000   0.6984   0.0000
        C    1.2678   0.0000   0.0000
        O    0.0000   1.9018   0.0000
        H   -1.2678  -0.6532   0.8941
        H   -1.2678  -0.6532  -0.8941
        H   -2.1906   0.6532   0.0000
        H    1.2678  -0.6532  -0.8941
        H    1.2678  -0.6532   0.8941
        H    2.1906   0.6532   0.0000
    """
    mol.basis   = basis
    mol.charge  = 0
    mol.spin    = 0
    mol.verbose = verbose
    mol.build()
    return mol


def print_mol_info(mol: gto.Mole) -> None:
    """Print a brief summary of the molecule."""
    print(f"Basis functions: {mol.nao}")
    print(f"Electrons:       {mol.nelectron}")
    print(f"ERI tensor size: {mol.nao**4 * 8 / 1e6:.1f} MB")


if __name__ == "__main__":
    mol = build_acetone()
    print_mol_info(mol)