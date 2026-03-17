"""
tests/test_giao.py  —  GIAO Stage 1 integral correctness tests
---------------------------------------------------------------

Each mathematical property that must hold analytically is expressed as
its own test, making it easy to pinpoint exactly which integral and which
symmetry breaks when something goes wrong.

Symmetry properties tested
--------------------------
  Overlap derivative S^(α)
    test_overlap_deriv_antisymmetric   — S_d[α] + S_d[α].T ≈ 0
    test_overlap_deriv_diagonal_zero   — same-centre terms vanish

  Core Hamiltonian derivative h^(α)
    test_hcore_deriv_antisymmetric     — h_d[α] + h_d[α].T ≈ 0
    test_hcore_equals_T_plus_V         — h_d = T_d + V_d

  PSO integrals
    test_pso_antisymmetric             — pso[K,α] + pso[K,α].T ≈ 0  (heavy atoms)

  Diamagnetic integrals
    test_dia_mn_symmetric              — dia[K,α,α] = dia[K,α,α].T
    test_dia_ab_symmetric              — dia[K,α,β] = dia[K,β,α]

  Density-matrix traces
    test_trace_P_Sd_vanishes           — Tr[P · S_d[α]] = 0
"""

import numpy as np
import pytest

from giao import (
    giao_overlap_deriv,
    giao_hcore_deriv,
    giao_pso,
    giao_diamagnetic,
)


AXES  = [0, 1, 2]
NAMES = ["x", "y", "z"]

TOL_ANTISYM = 1e-10
TOL_TRACE   = 1e-8


# ── fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def S_d(mol):
    return giao_overlap_deriv(mol)


@pytest.fixture(scope="module")
def hcore_derivs(mol):
    return giao_hcore_deriv(mol)   # (h_d, T_d, V_d)


@pytest.fixture(scope="module")
def pso(mol):
    return giao_pso(mol)


@pytest.fixture(scope="module")
def dia(mol):
    return giao_diamagnetic(mol)


@pytest.fixture(scope="module")
def heavy_atoms(mol):
    """Indices and symbols of non-hydrogen atoms."""
    return [(K, mol.atom_symbol(K))
            for K in range(mol.natm)
            if mol.atom_symbol(K) != "H"]


# ── overlap derivative ────────────────────────────────────────────────────────

@pytest.mark.parametrize("α,name", list(zip(AXES, NAMES)))
def test_overlap_deriv_antisymmetric(S_d, α, name):
    """S_d[α] must satisfy S + S.T = 0 (purely imaginary, antisymmetric)."""
    err = np.max(np.abs(S_d[α] + S_d[α].T))
    assert err < TOL_ANTISYM, (
        f"S_d[{name}] not antisymmetric: max|S+S.T| = {err:.2e}"
    )


@pytest.mark.parametrize("α,name", list(zip(AXES, NAMES)))
def test_overlap_deriv_diagonal_zero(S_d, α, name):
    """
    Same-centre matrix elements ⟨χ_μ|(R_μ−R_μ)×r|χ_μ⟩ vanish identically.
    """
    err = np.max(np.abs(np.diag(S_d[α])))
    assert err < TOL_ANTISYM, (
        f"S_d[{name}] diagonal not zero: max|diag| = {err:.2e}"
    )


# ── core Hamiltonian derivative ───────────────────────────────────────────────

@pytest.mark.parametrize("α,name", list(zip(AXES, NAMES)))
def test_hcore_deriv_antisymmetric(hcore_derivs, α, name):
    """h_d[α] must be antisymmetric."""
    h_d, _, _ = hcore_derivs
    err = np.max(np.abs(h_d[α] + h_d[α].T))
    assert err < TOL_ANTISYM, (
        f"h_d[{name}] not antisymmetric: max|h+h.T| = {err:.2e}"
    )


@pytest.mark.parametrize("α,name", list(zip(AXES, NAMES)))
def test_hcore_equals_T_plus_V(hcore_derivs, α, name):
    """h_d must equal T_d + V_d component-wise."""
    h_d, T_d, V_d = hcore_derivs
    err = np.max(np.abs(h_d[α] - (T_d[α] + V_d[α])))
    assert err < TOL_ANTISYM, (
        f"h_d[{name}] ≠ T_d + V_d: max deviation = {err:.2e}"
    )


# ── PSO integrals ─────────────────────────────────────────────────────────────

def test_pso_antisymmetric(pso, heavy_atoms):
    """pso[K,α] must be antisymmetric for every heavy atom and every axis."""
    for K, sym in heavy_atoms:
        for α, name in zip(AXES, NAMES):
            err = np.max(np.abs(pso[K, α] + pso[K, α].T))
            assert err < TOL_ANTISYM, (
                f"pso[K={K}({sym}),{name}] not antisymmetric: max = {err:.2e}"
            )


# ── diamagnetic integrals ─────────────────────────────────────────────────────

def test_dia_mn_symmetric(dia, heavy_atoms):
    """Diagonal α=β blocks of dia must be symmetric in μ,ν."""
    for K, sym in heavy_atoms:
        for α, name in zip(AXES, NAMES):
            err = np.max(np.abs(dia[K, α, α] - dia[K, α, α].T))
            assert err < TOL_ANTISYM, (
                f"dia[K={K}({sym}),{name}{name}] not symmetric in μ,ν: {err:.2e}"
            )


def test_dia_ab_symmetric(dia, heavy_atoms):
    """Diamagnetic tensor must be symmetric under α↔β exchange."""
    for K, sym in heavy_atoms:
        err = max(
            np.max(np.abs(dia[K, α, β] - dia[K, β, α]))
            for α in AXES for β in AXES
        )
        assert err < TOL_ANTISYM, (
            f"dia[K={K}({sym})] not symmetric under α↔β: max = {err:.2e}"
        )


# ── density-matrix traces ──────────────────────────────────────────────────────

@pytest.mark.parametrize("α,name", list(zip(AXES, NAMES)))
def test_trace_P_Sd_vanishes(rhf_density, S_d, α, name):
    """
    Tr[P · S_d[α]] = 0.
    The imaginary part of the norm of any physical state vanishes.
    """
    tr = abs(np.einsum("mn,mn->", rhf_density, S_d[α]))
    assert tr < TOL_TRACE, (
        f"Tr[P·S_d[{name}]] = {tr:.2e}, expected ≈ 0"
    )
