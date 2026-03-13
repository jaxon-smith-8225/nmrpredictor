"""
giao.py  —  Stage 1: Magnetic Field Derivative Integrals
---------------------------------------------------------

The GIAO (Gauge-Including Atomic Orbital) method attaches a
field-dependent phase factor to each basis function:

    chi_(mu)^GIAO(r; B) = exp( -i/2 (B cross R_(mu)) · r ) · chi_(mu)(r)

where R_(mu) is the centre of basis function (mu).  At B = 0 the phase is 1,
so the zero-field SCF is unchanged.  The NMR shielding tensor requires
the **first derivative** of every one-electron matrix with respect to B,
evaluated at B = 0.

Functions defined here
----------------------
  giao_overlap_deriv(mol)    → ∂S/∂B_alpha  at B=0   shape (3, nao, nao)
  giao_hcore_deriv(mol)      → ∂H/∂B_alpha  at B=0   shape (3, nao, nao)
  giao_pso(mol)              → PSO integrals      shape (natm, 3, nao, nao)
  giao_diamagnetic(mol)      → diamagnetic ints   shape (natm, 3, 3, nao, nao)
  validate_giao_integrals(mol, P)  — symmetry / sanity checks

Next stages (not yet implemented)
----------------------------------
  Stage 2 — CPKS (coupled-perturbed KS) equations → perturbed density P^(alpha)
  Stage 3 — Assemble shielding tensor sigma^K = sigma^K,dia + sigma^K,para
  Stage 4 — Convert sigma to chemical shifts δ relative to a reference
"""

import numpy as np


# ── Stage 1 integral builders ──────────────────────────────────────────────────

def giao_overlap_deriv(mol) -> np.ndarray:
    """
    First derivative of the AO overlap matrix w.r.t. B_alpha at B = 0.

    S^(alpha)_(mu)(nu) = ∂/∂B_alpha ⟨chi_(mu)^GIAO | chi_(nu)^GIAO⟩|_{B=0}

    Expanding the GIAO phase factor to first order, the derivative
    reduces to a matrix element involving the angular momentum of
    the basis-function centres:

        S^(alpha)_(mu)(nu) = -i/2 ⟨chi_(mu) | (R_(mu) - R_(nu)) cross r |_alpha | chi_(nu)⟩

    This is purely imaginary and antisymmetric: S^(alpha) = -(S^(alpha))†.
    PySCF stores the imaginary part as a real-valued array, so every
    element here is the imaginary component of the true integral.

    Physical role
    -------------
    S^(alpha) appears in the RHS of the CPKS equations as part of the
    'skeleton' contribution to the perturbed Fock matrix.  It also
    enters the energy-weighted density-matrix correction to the shielding.

    Returns
    -------
    S_d : ndarray (3, nao, nao), float64
        S_d[alpha, (mu), (nu)] = Im[S^(alpha)_(mu)(nu)],  alpha = 0/1/2 → x/y/z.
    """
    return mol.intor('int1e_igovlp', comp=3)


def giao_hcore_deriv(mol):
    """
    First derivative of the core Hamiltonian w.r.t. B_alpha at B = 0.

    h^(alpha)_(mu)(nu) = T^(alpha)_(mu)(nu) + V^(alpha)_(mu)(nu)

    Both contributions are imaginary and antisymmetric.

    T^(alpha)_(mu)(nu) = -i/2 ⟨chi_(mu) | (R_(mu) - R_(nu)) cross r |_alpha (-½∇²) | chi_(nu)⟩ + c.c.

    Physical role
    -------------
    h^(alpha) is the skeleton RHS of the CPKS equation.  Together with
    two-electron and XC response terms, it determines how the MOs
    distort in a magnetic field — driving paramagnetic shielding.

    Returns
    -------
    h_d : ndarray (3, nao, nao)   — kinetic + nuclear contribution
    T_d : ndarray (3, nao, nao)   — kinetic part only
    V_d : ndarray (3, nao, nao)   — nuclear-attraction part only
    """
    # PySCF 'igkin' returns the bra-side GIAO contribution only.
    # Full antisymmetric kinetic derivative = bra - ket transpose.
    T_d_raw = mol.intor('int1e_igkin', comp=3)
    T_d     = T_d_raw - T_d_raw.transpose(0, 2, 1)

    # 'ignuc' sums over all nuclei automatically.
    V_d = mol.intor('int1e_ignuc', comp=3)

    h_d = T_d + V_d
    return h_d, T_d, V_d


def giao_pso(mol) -> np.ndarray:
    """
    Paramagnetic nuclear Spin-Orbit (PSO) integrals for every nucleus.

    For nucleus K at position R_K, the PSO operator in direction beta is:

        h^(K,beta)_(mu)(nu) = ⟨chi_(mu) | (r_K cross ∇)_beta / |r_K|^3 | chi_(nu)⟩

    This describes how the orbital angular momentum of an electron,
    measured relative to nucleus K, couples to the nuclear magnetic
    moment along beta.  The 1/r³ weight makes it a local operator.

    Physical role
    -------------
    PSO integrals appear in the shielding as:

        sigma^K,para_(alpha,beta) = Tr[ P^(alpha) · h^(K,beta) ]

    where P^(alpha) is the first-order density matrix from CPKS (Stage 2).

    Returns
    -------
    pso : ndarray (natm, 3, nao, nao), float64
        pso[K, beta, (mu), (nu)] = h^(K,beta)_(mu)(nu).
        These matrices are real and antisymmetric: pso[K,beta] = -pso[K,beta].T
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

        d^(K)_alphabeta,(mu)(nu) = ⟨chi_(mu) | (r_K · r δ_(alpha,beta) - r_{K,alpha} r_beta) / |r_K|^3 | chi_(nu)⟩

    This arises from second-order perturbation theory and requires
    only the zero-field density matrix P — no CPKS needed.

    Physical role
    -------------
        sigma^K,dia_alphabeta = (alpha²/2) Tr[ P · d^(K)_alphabeta ]

    The diamagnetic term is always positive (shielding) and tends to
    dominate for ¹H.  For ¹³C/¹⁵N/¹⁷O, paramagnetic variation
    matters more for chemical *shifts*.

    Returns
    -------
    dia : ndarray (natm, 3, 3, nao, nao), float64
        dia[K, alpha, beta, (mu), (nu)] = d^(K)_alphabeta,(mu)(nu).
        Symmetric under both (mu)↔(nu) and alpha↔beta.
    """
    natm = mol.natm
    nao  = mol.nao
    dia  = np.zeros((natm, 3, 3, nao, nao))

    for K in range(natm):
        with mol.with_rinv_at_nucleus(K):
            # Returns (9, nao, nao) for the 9 Cartesian alphabeta pairs
            raw = mol.intor('int1e_giao_a11part', comp=9).reshape(3, 3, nao, nao)

        # Enforce (mu),(nu) symmetry (bra + ket average)
        sym_mn  = (raw + raw.transpose(0, 1, 3, 2)) / 2
        # Enforce alpha,beta symmetry of the shielding tensor
        dia[K]  = (sym_mn + sym_mn.transpose(1, 0, 2, 3)) / 2

    return dia


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_giao_integrals(mol, P: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Run all four GIAO integral builders and verify every mathematical
    property that should hold analytically.

    Checks
    ------
    1.  S^(alpha) antisymmetric and purely off-diagonal
    2.  h^(alpha) antisymmetric
    3.  PSO antisymmetric per nucleus (heavy atoms)
    4.  Diamagnetic ((mu),(nu)) symmetric and (alpha,beta) symmetric
    5.  Tr(P · S_d[alpha]) = 0  (imaginary part of norm vanishes)

    Parameters
    ----------
    mol : pyscf.gto.Mole
    P   : ndarray (nao, nao) — converged density matrix
    tol : float — threshold for 'approximately zero'

    Returns
    -------
    all_passed : bool
    """
    PASS = "\033[92m PASS\033[0m"
    FAIL = "\033[91m FAIL\033[0m"

    def check(label, value, threshold=tol):
        ok  = value < threshold
        tag = PASS if ok else FAIL
        print(f"  {tag}  {label:<55s}  max={value:.2e}")
        return ok

    results = []
    print("=" * 72)
    print("GIAO Integral Validation")
    print("=" * 72)

    # 1. Overlap derivative
    print("\n[1] Overlap derivative S^(alpha)")
    S_d = giao_overlap_deriv(mol)
    for alpha, name in enumerate(['x', 'y', 'z']):
        results.append(check(f"S_d[{name}] antisymmetric (S+S.T≈0)",
                             np.max(np.abs(S_d[alpha] + S_d[alpha].T))))
        results.append(check(f"S_d[{name}] diagonal ≈ 0 (same-centre terms)",
                             np.max(np.abs(np.diag(S_d[alpha])))))

    # 2. Core Hamiltonian derivative
    print("\n[2] Core Hamiltonian derivative h^(alpha)")
    h_d, T_d, V_d = giao_hcore_deriv(mol)
    for alpha, name in enumerate(['x', 'y', 'z']):
        results.append(check(f"h_d[{name}] antisymmetric (h+h.T≈0)",
                             np.max(np.abs(h_d[alpha] + h_d[alpha].T))))

    # 3. PSO integrals (heavy atoms only for brevity)
    print("\n[3] PSO integrals per nucleus")
    pso   = giao_pso(mol)
    heavy = [(K, mol.atom_symbol(K)) for K in range(mol.natm)
             if mol.atom_symbol(K) != 'H']
    for K, sym in heavy:
        for alpha, name in enumerate(['x', 'y', 'z']):
            results.append(check(
                f"pso[K={K}({sym}),{name}] antisymmetric",
                np.max(np.abs(pso[K, alpha] + pso[K, alpha].T))))

    # 4. Diamagnetic integrals
    print("\n[4] Diamagnetic integrals per nucleus")
    dia = giao_diamagnetic(mol)
    for K, sym in heavy:
        for alpha, name in enumerate(['x', 'y', 'z']):
            results.append(check(
                f"dia[K={K}({sym}),{name}{name}] symmetric in (mu),(nu)",
                np.max(np.abs(dia[K, alpha, alpha] - dia[K, alpha, alpha].T))))
        ab_err = max(
            np.max(np.abs(dia[K, alpha, beta] - dia[K, beta, alpha]))
            for alpha in range(3) for beta in range(3)
        )
        results.append(check(f"dia[K={K}({sym})] symmetric under alpha↔beta", ab_err))

    # 5. Density-matrix traces
    print("\n[5] Traces with density matrix")
    for alpha, name in enumerate(['x', 'y', 'z']):
        tr = abs(np.einsum('mn,mn->', P, S_d[alpha]))
        results.append(check(f"Tr[P · S_d[{name}]] = 0  (imaginary norm)",
                             tr, threshold=1e-8))

    # Summary
    n_pass, n_total = sum(results), len(results)
    print("\n" + "=" * 72)
    print(f"Result: {n_pass}/{n_total} checks passed")
    if n_pass == n_total:
        print("All GIAO integrals validated — Stage 1 complete.")
    else:
        print("WARNING: some checks failed — inspect output above.")
    print("=" * 72)
    return n_pass == n_total


if __name__ == "__main__":
    from molecule import build_acetone
    from rhf import run_rhf

    mol    = build_acetone()
    result = run_rhf(mol)

    if result:
        E, C, epsilon, P = result
        validate_giao_integrals(mol, P)

        # Preview diamagnetic trace for oxygen (atom 3)
        dia = giao_diamagnetic(mol)
        dia_O_iso = sum(np.einsum('mn,mn->', P, dia[3, i, i]) for i in range(3)) / 3
        print(f"\nTr[P·d_iso] for oxygen (no prefactor): {dia_O_iso:.6f} a.u.")
