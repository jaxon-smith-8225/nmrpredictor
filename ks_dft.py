"""
ks_dft.py
---------
Kohn-Sham DFT SCF implementation, extending the RHF code in rhf.py.

The SCF loop, DIIS, orthogonalizer, and density matrix are **unchanged**
from RHF.  The only modification is the Fock build: the -1/2K exchange
term is replaced by the exchange-correlation potential V_xc evaluated
on a numerical integration grid.

Implemented here
----------------
  Grid & density
    build_grid        — Becke-Lebedev numerical integration grid
    eval_aos          — AO basis functions evaluated at grid points
    eval_density      — electron density (rho)(r) at grid points

  LDA functionals   (local density approximation)
    lda_exchange      — Slater exchange  ε_x = C_x (rho)^(1/3)
    vwn_correlation   — VWN5 correlation  (Vosko, Wilk & Nusair 1980)
    lda_xc            — combined LDA: lda_exchange + vwn_correlation

  KS Fock builder
    build_vxc_matrix  — V_xc matrix by numerical quadrature
    build_fock_ks     — KS Fock matrix F = H_core + J + V_xc
    compute_energy_ks — KS total energy

  SCF driver
    scf_ks            — KS-DFT SCF loop with DIIS
    run_ks_lda        — convenience wrapper for LDA on a Mole object
"""

import numpy as np
from pyscf import dft
from pyscf.dft import numint

from rhf import (
    nuclear_repulsion, orthogonalizer,
    initial_guess, make_density, DIIS,
)


# ── Grid & density ─────────────────────────────────────────────────────────────

def build_grid(mol, level: int = 3):
    """
    Generate a Becke-Lebedev numerical integration grid.

    PySCF handles Becke partitioning and Lebedev angular quadrature
    internally.  Level 3 is a good balance of accuracy vs. cost.

    Returns
    -------
    coords  : ndarray (n_grid, 3)  — grid coordinates in Bohr
    weights : ndarray (n_grid,)    — quadrature weights
    """
    grids         = dft.gen_grid.Grids(mol)
    grids.level   = level
    grids.build()
    print(f"Grid points: {len(grids.weights):,}")
    return grids.coords, grids.weights


def eval_aos(mol, coords: np.ndarray) -> np.ndarray:
    """
    Evaluate all AO basis functions at every grid point.

    Returns ao of shape (n_grid, n_ao) where ao[g, (mu)] = χ_(mu)(r_g).
    """
    return numint.eval_ao(mol, coords, deriv=0)   # values only


def eval_density(P: np.ndarray, ao: np.ndarray) -> np.ndarray:
    """
    Compute the electron density at each grid point.

    (rho)(r_g) = Sum_(mu)(nu)) P_(mu)(nu)) χ_(mu)(r_g) χ_(nu))(r_g)

    Returns rho of shape (n_grid,).
    """
    return np.einsum('gm,mn,gn->g', ao, P, ao)


# ── LDA exchange-correlation functionals ──────────────────────────────────────

def lda_exchange(rho: np.ndarray):
    """
    Slater exchange functional (Dirac 1930).

    Energy density : ε_x = C_x · (rho)^(1/3)
    Potential      : v_x = (4/3) · ε_x   [= deltaE_x/delta(rho)]

    C_x = -(3/4)(3/π)^(1/3)

    Returns
    -------
    eps_x, v_x : ndarray (n_grid,) each
    """
    Cx   = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0)
    mask = rho > 1e-15
    rs   = np.where(mask, rho, 1e-15)

    eps_x = Cx * rs ** (1.0 / 3.0)
    v_x   = (4.0 / 3.0) * eps_x

    return np.where(mask, eps_x, 0.0), np.where(mask, v_x, 0.0)


def vwn_correlation(rho: np.ndarray):
    """
    VWN5 correlation functional (Vosko, Wilk & Nusair 1980).

    Fitted to quantum Monte Carlo data for the uniform electron gas.

    Returns
    -------
    eps_c, v_c : ndarray (n_grid,) each
        v_c = ε_c + (rho) · dε_c/d(rho)  (functional derivative)
    """
    A, x0, b, c = 0.0310907, -0.10498, 3.72744, 12.9352

    mask     = rho > 1e-15
    rho_safe = np.where(mask, rho, 1e-15)

    rs = (3.0 / (4.0 * np.pi * rho_safe)) ** (1.0 / 3.0)   # Wigner-Seitz radius
    x  = np.sqrt(rs)
    X  = x**2 + b * x + c
    X0 = x0**2 + b * x0 + c
    Q  = np.sqrt(4 * c - b**2)

    eps_c = A * (
        np.log(x**2 / X)
        + (2 * b / Q) * np.arctan(Q / (2 * x + b))
        - (b * x0 / X0) * (
            np.log((x - x0)**2 / X)
            + (2 * (2 * x0 + b) / Q) * np.arctan(Q / (2 * x + b))
        )
    )

    # Potential: v_c = ε_c + (rho) · dε_c/d(rho)
    # Chain rule: d/d(rho) = (drs/d(rho))(d/drs),  drs/d(rho) = -rs/(3(rho))
    #             d/drs = (1/2x) d/dx
    dX_dx   = 2 * x + b
    deps_dx = A * (
        2 / x - dX_dx / X
        - (b * x0 / X0) * (2 / (x - x0) - dX_dx / X)
    )
    v_c = eps_c - (rs / 3.0) * deps_dx * (1.0 / (2.0 * x))

    return np.where(mask, eps_c, 0.0), np.where(mask, v_c, 0.0)


def lda_xc(rho: np.ndarray):
    """
    Full LDA: Slater exchange + VWN5 correlation.

    Returns (eps_xc, v_xc) at each grid point.
    """
    eps_x, v_x = lda_exchange(rho)
    eps_c, v_c = vwn_correlation(rho)
    return eps_x + eps_c, v_x + v_c


# ── KS Fock builder ───────────────────────────────────────────────────────────

def build_vxc_matrix(P, ao, weights, xc_func):
    """
    Assemble the V_xc matrix by numerical integration.

    V_xc,(mu)(nu)) = ∫ χ_(mu)(r) v_xc((rho)(r)) χ_(nu))(r) dr
            ≈ Sum_g w_g χ_(mu)(r_g) v_xc((rho)(r_g)) χ_(nu))(r_g)

    E_xc = ∫ (rho)(r) ε_xc((rho)(r)) dr  ≈  Sum_g w_g (rho)(r_g) ε_xc((rho)(r_g))

    Returns
    -------
    Vxc  : ndarray (nao, nao)
    E_xc : float
    """
    rho           = eval_density(P, ao)
    eps_xc, v_xc  = xc_func(rho)
    E_xc          = np.dot(weights, rho * eps_xc)
    weighted_ao   = ao * (weights * v_xc)[:, np.newaxis]   # (n_grid, nao)
    Vxc           = ao.T @ weighted_ao                      # (nao, nao)
    return Vxc, E_xc


def build_fock_ks(H_core, P, ERI, ao, weights, xc_func):
    """
    Kohn-Sham Fock (KS matrix) builder.

    F_(mu)(nu)) = H_(mu)(nu)) + J_(mu)(nu)) + V_xc,(mu)(nu))

    Compare with RHF:
      F_(mu)(nu)) = H_(mu)(nu)) + J_(mu)(nu)) - 1/2K_(mu)(nu))

    Returns
    -------
    F    : ndarray (nao, nao)  — KS matrix
    J    : ndarray (nao, nao)  — Coulomb matrix (needed for energy)
    E_xc : float               — XC energy
    """
    J          = np.einsum('ls,mnls->mn', P, ERI)
    Vxc, E_xc = build_vxc_matrix(P, ao, weights, xc_func)
    F          = H_core + J + Vxc
    return F, J, E_xc


def compute_energy_ks(P, H_core, J, E_xc, Vnn) -> float:
    """
    KS-DFT total energy.

    E = Tr[P·H_core]  +  1/2 Tr[P·J]  +  E_xc  +  V_nn

    where:
      Tr[P·H_core]  — one-electron energy (kinetic + nuclear attraction)
      1/2 Tr[P·J]     — classical Coulomb repulsion (1/2 avoids double-counting)
      E_xc          — exchange-correlation energy from grid
      V_nn          — nuclear repulsion
    """
    E_1e   = np.einsum('mn,mn->', P, H_core)
    E_coul = 0.5 * np.einsum('mn,mn->', P, J)
    return E_1e + E_coul + E_xc + Vnn


# ── KS SCF driver ─────────────────────────────────────────────────────────────

def scf_ks(mol, xc_func, ao, coords, weights,
           max_cycles=100, e_tol=1e-8, d_tol=1e-6):
    """
    Kohn-Sham DFT SCF loop.

    Identical flow to scf_loop_diis in rhf.py — only the Fock build
    and energy expression differ.  DIIS, orthogonalizer, and density
    matrix are reused without modification.

    Parameters
    ----------
    mol      : pyscf Mole object
    xc_func  : callable  rho → (eps_xc, v_xc)  e.g. lda_xc
    ao       : ndarray (n_grid, nao)  — AO values on grid (precomputed)
    coords   : ndarray (n_grid, 3)   — grid coordinates
    weights  : ndarray (n_grid,)     — grid weights

    Returns
    -------
    (E, C, epsilon, P) on convergence, or None.
    """
    S      = mol.intor('int1e_ovlp')
    T      = mol.intor('int1e_kin')
    V_nuc  = mol.intor('int1e_nuc')
    ERI    = mol.intor('int2e')
    H_core = T + V_nuc
    Vnn    = nuclear_repulsion(mol)
    X      = orthogonalizer(S)
    n_occ  = mol.nelectron // 2

    C, epsilon = initial_guess(H_core, X)
    P          = make_density(C, n_occ)
    diis       = DIIS(max_vecs=8)
    E_old      = 0.0

    print(f"{'Cycle':>6}  {'Energy (Hartree)':>18}  {'deltaE':>12}  "
          f"{'RMS deltaP':>12}  {'E_xc':>12}  {'Status':>10}")
    print("-" * 82)

    for cycle in range(1, max_cycles + 1):
        F, J, E_xc = build_fock_ks(H_core, P, ERI, ao, weights, xc_func)
        E           = compute_energy_ks(P, H_core, J, E_xc, Vnn)
        F           = diis.update(F, P, S)

        F_prime          = X.T @ F @ X
        epsilon, C_prime = np.linalg.eigh(F_prime)
        C                = X @ C_prime
        P_new            = make_density(C, n_occ)

        delta_E   = E - E_old
        rms_dP    = np.sqrt(np.mean((P_new - P) ** 2))
        converged = (abs(delta_E) < e_tol) and (rms_dP < d_tol)

        print(f"{cycle:>6}  {E:>18.8f}  {delta_E:>12.2e}  "
              f"{rms_dP:>12.2e}  {E_xc:>12.6f}  "
              f"{'CONVERGED' if converged else '':>10}")

        if converged:
            break

        P     = P_new
        E_old = E
    else:
        print(f"\nWARNING: SCF did not converge in {max_cycles} cycles")
        return None

    print(f"\nFinal KS-LDA energy : {E:.8f} Hartree")
    return E, C, epsilon, P


def run_ks_lda(mol, grid_level: int = 3):
    """
    Convenience wrapper: build grid, run KS-LDA, return results.

    Returns (E, C, epsilon, P) or None.
    """
    coords, weights = build_grid(mol, level=grid_level)
    ao              = eval_aos(mol, coords)
    print(f"AO array shape: {ao.shape}   (n_grid × n_ao)")
    return scf_ks(mol, lda_xc, ao, coords, weights)


if __name__ == "__main__":
    from pyscf import dft as pyscf_dft
    from molecule import build_acetone
    from rhf import run_rhf

    mol = build_acetone()

    # Run KS-LDA
    result_ks = run_ks_lda(mol)

    # Compare with PySCF reference and RHF
    result_rhf = run_rhf(mol)
    mf_ref        = pyscf_dft.RKS(mol)
    mf_ref.xc     = 'lda,vwn'
    mf_ref.grids.level = 3
    mf_ref.verbose = 0
    mf_ref.kernel()

    E_ks,  *_ = result_ks
    E_rhf, *_ = result_rhf

    print(f"\n{'Method':<22} {'Energy (Hartree)':>18}")
    print("-" * 42)
    print(f"{'Your RHF':<22} {E_rhf:>18.8f}")
    print(f"{'Your KS-LDA':<22} {E_ks:>18.8f}")
    print(f"{'PySCF KS-LDA (ref)':<22} {mf_ref.e_tot:>18.8f}")
    print(f"\nYour LDA vs PySCF ref : {E_ks - mf_ref.e_tot:>+.2e} Hartree")
    print(f"RHF vs LDA (corr. E)  : {E_ks - E_rhf:>+.6f} Hartree")
    print("\nThe KS-LDA energy is lower than RHF because DFT captures")
    print("electron correlation that HF misses entirely.")
