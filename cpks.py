"""
cpks.py  —  Stage 2: Coupled-Perturbed Kohn-Sham (CPKS) Equations
------------------------------------------------------------------

Solver strategy
---------------
We treat the CPKS equation as a standard linear system A·u = −r and
solve it directly via LU factorization (unconditionally stable).

Notation / sign conventions
----------------------------
All quantities are real-valued.  The true perturbed density is P^(α) = i·p,
the true perturbed MO coefficients are U_ai = i·u (u real).  Imaginary
factors are tracked in comments and cancel consistently.

Bug history
-----------
  1. Factor in perturbed_density_ao was 4, correct is 2.

  2. int1e_ignuc needs antisymmetrization (same as int1e_igkin):
     both return the bra-1-only GIAO contribution; V_d = V_raw - V_raw.T.

  3. build_cpks_rhs_2e was computing J and K contributions that
     silently reduce to zero for symmetric P.  The correct formula for
     the full 2e GIAO skeleton (bra + ket contributions) is:

       J^(α)_μν = Σ_λσ P_λσ ∂(μν|λσ)/∂B_α

     The full derivative includes both electron-1 and electron-2 GIAO
     phases.  With int2e_ig1 storing only the bra-1 (electron-1) term:

       ∂(μν|λσ)/∂B_α = (ig1[μ,ν,λ,σ] − ig1[ν,μ,λ,σ])   ← bra antisym
                      + (ig1[λ,σ,μ,ν] − ig1[σ,λ,μ,ν])   ← ket via permutation

     The ket terms T3 and T4 cancel by P-symmetry when contracted,
     leaving:
       J^(α) = T1 − T1.T   where T1_μν = Σ_λσ P_λσ ig1[μ,ν,λ,σ]

     Similarly for K:
       K^(α) = K_T1 − K_T2 + K_T1.T − K_T2.T
               K_T1_μν = Σ_λσ P_λσ ig1[μ,λ,ν,σ]
               K_T2_μν = Σ_λσ P_λσ ig1[λ,μ,ν,σ]

     The Fock 2e GIAO skeleton: G^(α) = J^(α) − ½K^(α)

Functions
---------
  build_cpks_rhs_1e   — 1e skeleton: (T_d + V_d)_ai − ε_i S_d_ai
  build_cpks_rhs_2e   — 2e GIAO skeleton (corrected)
  build_cpks_rhs      — full RHS = 1e + 2e
  build_cpks_matrix   — orbital Hessian A
  solve_cpks          — direct LU solver
  solve_uncoupled     — SOS approximation (debugging only)
  perturbed_density_ao — p^(α) in AO basis from u
"""

import numpy as np
import scipy.linalg
from giao import giao_overlap_deriv


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ao_to_mo_ai(M_ao, C_vir, C_occ):
    """Virtual–occupied block of M in MO basis.  Shape: (n_vir, n_occ)."""
    return C_vir.T @ M_ao @ C_occ


# ── RHS builders ───────────────────────────────────────────────────────────────

def build_cpks_rhs_1e(mol, C, epsilon, n_occ):
    """
    One-electron skeleton RHS.

    r^(1e,α)_{ai} = (T_d + V_d)_{ai} − ε_i · S_d_{ai}

    T_d and V_d are both antisymmetrised (bra − bra.T):
    int1e_igkin and int1e_ignuc both return bra-1-only contributions.

    Returns
    -------
    r_1e : (3, n_vir, n_occ)
    """
    C_occ   = C[:, :n_occ]
    C_vir   = C[:, n_occ:]
    eps_occ = epsilon[:n_occ]

    S_d = giao_overlap_deriv(mol)             # (3, nao, nao), already antisymmetric

    T_raw = mol.intor('int1e_igkin', comp=3)  # bra-1 only → antisymmetrise
    T_d   = T_raw - T_raw.transpose(0, 2, 1)

    V_raw = mol.intor('int1e_ignuc', comp=3)  # bra-1 only → antisymmetrise
    V_d   = V_raw - V_raw.transpose(0, 2, 1)

    h_d = T_d + V_d

    n_vir = C_vir.shape[1]
    r = np.zeros((3, n_vir, n_occ))
    for alpha in range(3):
        h_ai     = _ao_to_mo_ai(h_d[alpha], C_vir, C_occ)
        S_ai     = _ao_to_mo_ai(S_d[alpha], C_vir, C_occ)
        r[alpha] = h_ai - S_ai * eps_occ[np.newaxis, :]

    return r


def build_cpks_rhs_2e(mol, C, P, n_occ, ERI_ig1):
    """
    Two-electron GIAO skeleton contribution to the CPKS RHS.

    int2e_ig1 stores the bra-1-only (electron-1) GIAO derivative:
        ig1[α, μ, ν, λ, σ]  =  bra-1 contribution to ∂(μν|λσ)/∂B_α

    The full antisymmetric 2e GIAO derivative is:
        ∂(μν|λσ)/∂B_α = (ig1[μ,ν,λ,σ] − ig1[ν,μ,λ,σ])
                       + (ig1[λ,σ,μ,ν] − ig1[σ,λ,μ,ν])

    When contracted with symmetric P, the ket terms (T3, T4) cancel,
    and the Fock 2e GIAO skeleton simplifies to:

        G^(α) = J^(α) − ½K^(α)

        J^(α) = T1 − T1.T
        K^(α) = K_T1 − K_T2 + K_T1.T − K_T2.T

        T1_μν   = Σ_λσ P_λσ ig1[μ,ν,λ,σ]
        K_T1_μν = Σ_λσ P_λσ ig1[μ,λ,ν,σ]
        K_T2_μν = Σ_λσ P_λσ ig1[λ,μ,ν,σ]

    Returns
    -------
    r_2e : (3, n_vir, n_occ)
    """
    C_occ = C[:, :n_occ]
    C_vir = C[:, n_occ:]
    n_vir = C_vir.shape[1]

    r_2e = np.zeros((3, n_vir, n_occ))

    for alpha in range(3):
        eri_a = ERI_ig1[alpha]   # (nao, nao, nao, nao)

        # J contribution
        T1 = np.einsum('ls,mnls->mn', P, eri_a)
        J  = T1 - T1.T

        # K contribution
        K_T1 = np.einsum('ls,mlns->mn', P, eri_a)
        K_T2 = np.einsum('ls,lmns->mn', P, eri_a)
        K    = K_T1 - K_T2 + K_T1.T - K_T2.T

        G = J - 0.5 * K

        r_2e[alpha] = _ao_to_mo_ai(G, C_vir, C_occ)

    return r_2e


def build_cpks_rhs(mol, C, P, epsilon, n_occ, ERI_ig1=None):
    """Full CPKS RHS = r_1e + r_2e."""
    r = build_cpks_rhs_1e(mol, C, epsilon, n_occ)

    if ERI_ig1 is not None:
        r += build_cpks_rhs_2e(mol, C, P, n_occ, ERI_ig1)
    else:
        print("  [WARNING] 2e GIAO skeleton omitted — heavy-atom shieldings "
              "will be inaccurate.")

    return r


# ── CPKS matrix ────────────────────────────────────────────────────────────────

def build_cpks_matrix(C, epsilon, n_occ, ERI):
    """
    Orbital Hessian A_{ai,bj} = (ε_a−ε_i)δ_{ab}δ_{ij} + K_{ai,bj}.

    K_{ai,bj} = 4(ai|bj) − (ab|ij) − (aj|bi)
    """
    C_occ   = C[:, :n_occ]
    C_vir   = C[:, n_occ:]
    eps_occ = epsilon[:n_occ]
    eps_vir = epsilon[n_occ:]
    n_vir   = len(eps_vir)
    nov     = n_vir * n_occ

    ERI_mn_bj = np.einsum('mnls,lb,sj->mnbj', ERI, C_vir, C_occ)
    mo_aibj   = np.einsum('mnbj,ma,ni->aibj', ERI_mn_bj, C_vir, C_occ)

    ERI_mn_ij = np.einsum('mnls,li,sj->mnij', ERI, C_occ, C_occ)
    mo_abij   = np.einsum('mnij,ma,nb->abij', ERI_mn_ij, C_vir, C_vir)

    ERI_mn_bi = np.einsum('mnls,lb,si->mnbi', ERI, C_vir, C_occ)
    mo_ajbi   = np.einsum('mnbi,ma,nj->ajbi', ERI_mn_bi, C_vir, C_occ)

    K = (4.0 * mo_aibj
         - mo_abij.transpose(0, 2, 1, 3)
         - mo_ajbi.transpose(0, 3, 2, 1))

    A = K.reshape(nov, nov)

    denom = (eps_vir[:, np.newaxis] - eps_occ[np.newaxis, :]).ravel()
    A[np.arange(nov), np.arange(nov)] += denom

    return A


# ── Solvers ────────────────────────────────────────────────────────────────────

def solve_uncoupled(mol, C, P, epsilon, n_occ, ERI_ig1=None):
    """SOS approximation u = −r / (ε_a − ε_i).  Debugging use only."""
    eps_occ = epsilon[:n_occ]
    eps_vir = epsilon[n_occ:]
    denom   = eps_vir[:, np.newaxis] - eps_occ[np.newaxis, :]
    r       = build_cpks_rhs(mol, C, P, epsilon, n_occ, ERI_ig1)
    return -r / denom[np.newaxis, :, :]


def solve_cpks(mol, C, P, epsilon, n_occ, ERI=None, ERI_ig1=None):
    """
    Direct CPKS solver: LU-factorize A once, solve A·u = −r for α=x,y,z.

    Parameters
    ----------
    ERI     : (nao,)*4 standard ERIs
    ERI_ig1 : (3, nao, nao, nao, nao) GIAO 2e derivative (int2e_ig1, comp=3)

    Returns
    -------
    u : (3, n_vir, n_occ)
    """
    if ERI is None:
        print("  Computing ERI tensor …")
        ERI = mol.intor('int2e')

    n_vir = C.shape[1] - n_occ
    nov   = n_vir * n_occ

    r = build_cpks_rhs(mol, C, P, epsilon, n_occ, ERI_ig1)

    print(f"\n  Building CPKS A matrix ({nov}×{nov}) …")
    A    = build_cpks_matrix(C, epsilon, n_occ, ERI)
    cond = np.linalg.cond(A)
    print(f"  Condition number of A : {cond:.3e}")

    lu, piv = scipy.linalg.lu_factor(A)

    u = np.zeros((3, n_vir, n_occ))
    print(f"\n  {'α':>3}  {'‖Au+r‖_max':>14}")
    print("  " + "-" * 20)
    for alpha in range(3):
        rhs      = -r[alpha].ravel()
        u_flat   = scipy.linalg.lu_solve((lu, piv), rhs)
        u[alpha] = u_flat.reshape(n_vir, n_occ)
        res      = np.max(np.abs(A @ u_flat + r[alpha].ravel()))
        print(f"  {alpha:>3}  {res:>14.3e}")

    return u


# ── Perturbed AO density ───────────────────────────────────────────────────────

def perturbed_density_ao(C, u, n_occ):
    """
    Real part of first-order AO density: p^(α)_μν = 2 Σ_{ai} u_ai (C_μa C_νi − C_μi C_νa).

    P^(α) = i·p^(α).  Factor is 2 (double occupation), NOT 4.

    Returns
    -------
    p : (3, nao, nao), antisymmetric per component
    """
    C_occ = C[:, :n_occ]
    C_vir = C[:, n_occ:]
    nao   = C.shape[0]

    p = np.zeros((3, nao, nao))
    for alpha in range(3):
        contrib  = C_vir @ u[alpha] @ C_occ.T
        p[alpha] = 2.0 * (contrib - contrib.T)

    return p