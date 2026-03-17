"""
tests/test_rhf.py  —  RHF SCF correctness tests
-------------------------------------------------

Checks
------
  test_scf_converges         — SCF returns a result (not None)
  test_electron_count        — Tr(PS) equals the number of electrons
  test_energy_range          — energy is physically plausible for acetone/STO-3G
  test_orbital_energies_ordered — orbital energies are non-decreasing
  test_homo_lumo_gap_positive   — HOMO < LUMO (no accidental degeneracy)
  test_density_idempotent    — P S P = 2P for a pure RHF state (idempotency)
  test_compare_pyscf         — energy matches PySCF reference within 1 µHartree
"""

import numpy as np
import pytest
from pyscf import scf as pyscf_scf


# ── basic convergence ──────────────────────────────────────────────────────────

def test_scf_converges(rhf_result):
    """SCF must return (E, C, epsilon, P), not None."""
    assert rhf_result is not None


def test_electron_count(mol, rhf_result):
    """Tr(PS) should equal the total number of electrons."""
    _, _, _, P = rhf_result
    S = mol.intor("int1e_ovlp")
    assert abs(np.trace(P @ S) - mol.nelectron) < 1e-6


# ── energy sanity ──────────────────────────────────────────────────────────────

def test_energy_range(rhf_result):
    """
    Acetone/STO-3G RHF energy should sit between −190 and −185 Hartree.
    Flags gross errors (wrong geometry, wrong molecule, wrong basis).
    """
    E, *_ = rhf_result
    assert -190.0 < E < -185.0, f"Energy {E:.4f} outside expected range"


def test_compare_pyscf(mol, rhf_result):
    """Energy must match PySCF's own RHF within 1 µHartree."""
    E, *_ = rhf_result
    mf = pyscf_scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    assert abs(E - mf.e_tot) < 1e-6, (
        f"Energy deviation from PySCF: {E - mf.e_tot:+.2e} Hartree"
    )


# ── orbital structure ──────────────────────────────────────────────────────────

def test_orbital_energies_ordered(rhf_result):
    """Orbital energies from eigh must be non-decreasing."""
    _, _, epsilon, _ = rhf_result
    assert np.all(np.diff(epsilon) >= -1e-10), "Orbital energies not sorted"


def test_homo_lumo_gap_positive(mol, rhf_result):
    """HOMO energy must be strictly below LUMO energy."""
    _, _, epsilon, _ = rhf_result
    n_occ = mol.nelectron // 2
    assert epsilon[n_occ - 1] < epsilon[n_occ], "Negative HOMO–LUMO gap"


# ── density matrix properties ─────────────────────────────────────────────────

def test_density_idempotent(mol, rhf_result):
    """
    For a pure RHF state, PSP = 2P (idempotency in the AO basis).
    Tests that the density matrix represents a proper single Slater determinant.
    """
    _, _, _, P = rhf_result
    S = mol.intor("int1e_ovlp")
    residual = P @ S @ P - 2 * P
    assert np.max(np.abs(residual)) < 1e-8, (
        f"Density not idempotent: max |PSP - 2P| = {np.max(np.abs(residual)):.2e}"
    )
