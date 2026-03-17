"""
tests/test_ks_dft.py  —  KS-LDA SCF correctness tests
-------------------------------------------------------

Checks
------
  test_scf_converges          — SCF returns a result (not None)
  test_electron_count         — Tr(PS) equals the number of electrons
  test_density_integrates_to_N — ∫ρ dr ≈ N_elec on the numerical grid
  test_energy_below_rhf       — LDA energy < RHF (correlation lowers energy)
  test_compare_pyscf          — energy matches PySCF LDA reference within 1 mHartree
  test_lda_exchange_negative  — Slater ε_x ≤ 0 everywhere (correct sign)
  test_vwn_correlation_negative — VWN5 ε_c ≤ 0 (correlation is attractive)
  test_exc_energy_negative    — total E_xc < 0
"""

import numpy as np
import pytest
from pyscf import dft as pyscf_dft

from ks_dft import (
    lda_exchange, vwn_correlation, lda_xc,
    eval_density,
)


# ── basic convergence ──────────────────────────────────────────────────────────

def test_scf_converges(ks_result):
    assert ks_result is not None


def test_electron_count(mol, ks_result):
    """Tr(PS) should equal the total number of electrons."""
    _, _, _, P = ks_result
    S = mol.intor("int1e_ovlp")
    assert abs(np.trace(P @ S) - mol.nelectron) < 1e-6


# ── density on the grid ────────────────────────────────────────────────────────

def test_density_integrates_to_N(mol, rhf_density, grid, ao_values):
    """
    The numerical integral of ρ over the Becke–Lebedev grid should equal
    the electron count.  Uses the RHF density as an input-independent check
    of the grid and AO evaluator.
    """
    _, weights = grid
    rho = eval_density(rhf_density, ao_values)
    N   = np.dot(weights, rho)
    assert abs(N - mol.nelectron) < 1e-4, (
        f"∫ρ dr = {N:.6f}, expected {mol.nelectron}"
    )


# ── energy comparisons ────────────────────────────────────────────────────────

def test_energy_below_rhf(rhf_result, ks_result):
    """
    LDA captures correlation, so E_LDA must be lower than E_RHF.
    (True for virtually all molecules with a standard basis.)
    """
    E_rhf, *_ = rhf_result
    E_ks,  *_ = ks_result
    assert E_ks < E_rhf, (
        f"Expected E_LDA < E_RHF, got {E_ks:.6f} vs {E_rhf:.6f}"
    )


def test_compare_pyscf(mol, ks_result):
    """Energy must match PySCF's LDA/VWN reference within 1 mHartree."""
    E_ks, *_ = ks_result
    mf = pyscf_dft.RKS(mol)
    mf.xc          = "lda,vwn"
    mf.grids.level = 3
    mf.verbose     = 0
    mf.kernel()
    assert abs(E_ks - mf.e_tot) < 1e-3, (
        f"LDA deviation from PySCF: {E_ks - mf.e_tot:+.2e} Hartree"
    )


# ── functional correctness ────────────────────────────────────────────────────

@pytest.fixture
def sample_rho():
    """A range of physically reasonable densities (a.u.)."""
    return np.array([1e-5, 0.001, 0.01, 0.1, 0.5, 1.0])


def test_lda_exchange_negative(sample_rho):
    """Slater exchange energy density must be ≤ 0."""
    eps_x, _ = lda_exchange(sample_rho)
    assert np.all(eps_x <= 0), f"Positive ε_x found: {eps_x}"


def test_vwn_correlation_negative(sample_rho):
    """VWN5 correlation energy density must be ≤ 0."""
    eps_c, _ = vwn_correlation(sample_rho)
    assert np.all(eps_c <= 0), f"Positive ε_c found: {eps_c}"


def test_exc_energy_negative(sample_rho):
    """Combined LDA ε_xc must be ≤ 0 for all positive densities."""
    eps_xc, _ = lda_xc(sample_rho)
    assert np.all(eps_xc <= 0)


def test_zero_density_handled():
    """lda_xc must return zero (not NaN/inf) when ρ ≈ 0."""
    rho = np.array([0.0, 1e-20])
    eps_xc, v_xc = lda_xc(rho)
    assert np.all(np.isfinite(eps_xc))
    assert np.all(np.isfinite(v_xc))
