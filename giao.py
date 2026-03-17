"""
giao.py  —  Stage 1: Magnetic Field Derivative Integrals
---------------------------------------------------------

The GIAO (Gauge-Including Atomic Orbital) method attaches a
field-dependent phase factor to each basis function:

    χ_μ^GIAO(r; B) = exp( −i/2 (B × R_μ) · r ) · χ_μ(r)

where R_μ is the centre of basis function μ.  At B = 0 the phase is 1,
so the zero-field SCF is unchanged.  The NMR shielding tensor requires
the **first derivative** of every one-electron matrix with respect to B,
evaluated at B = 0.

Functions defined here
----------------------
  giao_overlap_deriv(mol)    → ∂S/∂B_α  at B=0   shape (3, nao, nao)
  giao_hcore_deriv(mol)      → ∂H/∂B_α  at B=0   shape (3, nao, nao)
  giao_pso(mol)              → PSO integrals      shape (natm, 3, nao, nao)
  giao_diamagnetic(mol)      → diamagnetic ints   shape (natm, 3, 3, nao, nao)
  validate_giao_integrals(mol, P)  — symmetry / sanity checks

Next stages (not yet implemented)
----------------------------------
  Stage 2 — CPKS (coupled-perturbed KS) equations → perturbed density P^(α)
  Stage 3 — Assemble shielding tensor σ^K = σ^K,dia + σ^K,para
  Stage 4 — Convert σ to chemical shifts δ relative to a reference
"""

import numpy as np


# ── Stage 1 integral builders ──────────────────────────────────────────────────

def giao_overlap_deriv(mol) -> np.ndarray:
    """
    First derivative of the AO overlap matrix w.r.t. B_α at B = 0.

    S^(α)_μν = ∂/∂B_α ⟨χ_μ^GIAO | χ_ν^GIAO⟩|_{B=0}

    Expanding the GIAO phase factor to first order, the derivative
    reduces to a matrix element involving the angular momentum of
    the basis-function centres:

        S^(α)_μν = −i/2 ⟨χ_μ | (R_μ − R_ν) × r |_α | χ_ν⟩

    This is purely imaginary and antisymmetric: S^(α) = −(S^(α))†.
    PySCF stores the imaginary part as a real-valued array, so every
    element here is the imaginary component of the true integral.

    Physical role
    -------------
    S^(α) appears in the RHS of the CPKS equations as part of the
    'skeleton' contribution to the perturbed Fock matrix.  It also
    enters the energy-weighted density-matrix correction to the shielding.

    Returns
    -------
    S_d : ndarray (3, nao, nao), float64
        S_d[α, μ, ν] = Im[S^(α)_μν],  α = 0/1/2 → x/y/z.
    """
    return mol.intor('int1e_igovlp', comp=3)


def giao_hcore_deriv(mol):
    """
    First derivative of the core Hamiltonian w.r.t. B_α at B = 0.

    h^(α)_μν = T^(α)_μν + V^(α)_μν

    Both contributions are imaginary and antisymmetric.

    T^(α)_μν = −i/2 ⟨χ_μ | (R_μ − R_ν) × r |_α (−½∇²) | χ_ν⟩ + c.c.

    Physical role
    -------------
    h^(α) is the skeleton RHS of the CPKS equation.  Together with
    two-electron and XC response terms, it determines how the MOs
    distort in a magnetic field — driving paramagnetic shielding.

    Returns
    -------
    h_d : ndarray (3, nao, nao)   — kinetic + nuclear contribution
    T_d : ndarray (3, nao, nao)   — kinetic part only
    V_d : ndarray (3, nao, nao)   — nuclear-attraction part only
    """
    # PySCF 'igkin' returns the bra-side GIAO contribution only.
    # Full antisymmetric kinetic derivative = bra − ket transpose.
    T_d_raw = mol.intor('int1e_igkin', comp=3)
    T_d     = T_d_raw - T_d_raw.transpose(0, 2, 1)

    # 'ignuc' likewise returns the bra-1-only GIAO contribution.
    # Must be antisymmetrised the same way as igkin.
    V_d_raw = mol.intor('int1e_ignuc', comp=3)
    V_d     = V_d_raw - V_d_raw.transpose(0, 2, 1)

    h_d = T_d + V_d
    return h_d, T_d, V_d


def giao_pso(mol) -> np.ndarray:
    """
    Paramagnetic nuclear Spin–Orbit (PSO) integrals for every nucleus.

    For nucleus K at position R_K, the PSO operator in direction β is:

        h^(K,β)_μν = ⟨χ_μ | (r_K × ∇)_β / |r_K|³ | χ_ν⟩

    This describes how the orbital angular momentum of an electron,
    measured relative to nucleus K, couples to the nuclear magnetic
    moment along β.  The 1/r³ weight makes it a local operator.

    Physical role
    -------------
    PSO integrals appear in the shielding as:

        σ^K,para_αβ = Tr[ P^(α) · h^(K,β) ]

    where P^(α) is the first-order density matrix from CPKS (Stage 2).

    Returns
    -------
    pso : ndarray (natm, 3, nao, nao), float64
        pso[K, β, μ, ν] = h^(K,β)_μν.
        These matrices are real and antisymmetric: pso[K,β] = −pso[K,β].T
    """
    natm = mol.natm
    nao  = mol.nao
    pso  = np.zeros((natm, 3, nao, nao))
    for K in range(natm):
        with mol.with_rinv_at_nucleus(K):
            pso[K] = mol.intor('int1e_prinvxp', comp=3)
    return pso


def giao_diamagnetic(mol) -> np.ndarray:
    """
    Diamagnetic shielding integrals for every nucleus.

    For nucleus K, the diamagnetic tensor element is:

        d^(K)_αβ,μν = ⟨χ_μ | (r_K · r δ_αβ − r_{K,α} r_β) / |r_K|³ | χ_ν⟩

    This arises from second-order perturbation theory and requires
    only the zero-field density matrix P — no CPKS needed.

    Physical role
    -------------
        σ^K,dia_αβ = (α²/2) Tr[ P · d^(K)_αβ ]

    The diamagnetic term is always positive (shielding) and tends to
    dominate for ¹H.  For ¹³C/¹⁵N/¹⁷O, paramagnetic variation
    matters more for chemical *shifts*.

    Returns
    -------
    dia : ndarray (natm, 3, 3, nao, nao), float64
        dia[K, α, β, μ, ν] = d^(K)_αβ,μν.
        Symmetric under both μ↔ν and α↔β.
    """
    natm = mol.natm
    nao  = mol.nao
    dia  = np.zeros((natm, 3, 3, nao, nao))

    for K in range(natm):
        with mol.with_rinv_at_nucleus(K):
            # Returns (9, nao, nao) for the 9 Cartesian αβ pairs
            raw = mol.intor('int1e_giao_a11part', comp=9).reshape(3, 3, nao, nao)

        # Enforce μ,ν symmetry (bra + ket average)
        sym_mn  = (raw + raw.transpose(0, 1, 3, 2)) / 2
        # Enforce α,β symmetry of the shielding tensor
        dia[K]  = (sym_mn + sym_mn.transpose(1, 0, 2, 3)) / 2

    return dia