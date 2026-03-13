"""
rhf.py
------
Restricted Hartree-Fock (RHF) SCF implementation from scratch.

Pipeline
--------
1.  nuclear_repulsion   — constant V_nn for fixed geometry
2.  orthogonalizer      — X = S^(-1/2)  via eigendecomposition
3.  initial_guess       — diagonalise H_core in orthogonal basis
4.  make_density        — P_(mu)(nu) = 2 Sum(C_(mu)i C_(nu)i)  (occupied MOs only)
5.  build_fock          — F = H_core + J - 1/2K  (full 4-index ERI)
6.  compute_energy      — E = 1/2 Tr[P(H+F)] + V_nn
7.  DIIS                — Pulay extrapolation for fast SCF convergence
8.  scf_loop_diis       — main SCF driver
"""

import numpy as np


# ── Utilities ──────────────────────────────────────────────────────────────────

def nuclear_repulsion(mol) -> float:
    """
    Nuclear repulsion energy (constant for a fixed geometry).

    V_nn = Sum_{i<j} Z_i Z_j / r_ij   (atomic units, distances in Bohr)
    """
    Vnn     = 0.0
    coords  = mol.atom_coords()    # Bohr
    charges = mol.atom_charges()
    for i in range(mol.natm):
        for j in range(i + 1, mol.natm):
            r = np.linalg.norm(coords[i] - coords[j])
            Vnn += charges[i] * charges[j] / r
    return Vnn


def orthogonalizer(S: np.ndarray) -> np.ndarray:
    """
    Compute the orthogonalising transformation X = S^(-1/2).

    Uses eigendecomposition: S = U s U*  →  X = U s^(-1/2) U*

    Raises
    ------
    ValueError if the basis is nearly linearly dependent
    (eigenvalue < 1e-8).
    """
    eigenvalues, U = np.linalg.eigh(S)
    if np.any(eigenvalues < 1e-8):
        raise ValueError(
            f"Near-linear dependence in basis: min eigenvalue = {eigenvalues.min():.2e}"
        )
    s_inv_sqrt = np.diag(eigenvalues ** -0.5)
    return U @ s_inv_sqrt @ U.T


# ── Density matrix ─────────────────────────────────────────────────────────────

def initial_guess(H_core: np.ndarray, X: np.ndarray):
    """
    Guess MO coefficients by diagonalising the core Hamiltonian.

    Returns
    -------
    C       : ndarray (nao, nao)  — MO coefficients in AO basis
    epsilon : ndarray (nao,)      — orbital energies
    """
    F_prime          = X.T @ H_core @ X
    epsilon, C_prime = np.linalg.eigh(F_prime)
    C                = X @ C_prime
    return C, epsilon


def make_density(C: np.ndarray, n_occ: int) -> np.ndarray:
    """
    Build the closed-shell density matrix.

    P_(mu))(nu) = 2 (sigma)ᵢ^{n_occ} C_(mu))i C_(nu)i
    """
    C_occ = C[:, :n_occ]
    return 2.0 * C_occ @ C_occ.T


# ── Fock matrix ────────────────────────────────────────────────────────────────

def build_fock(H_core: np.ndarray, P: np.ndarray, ERI: np.ndarray) -> np.ndarray:
    """
    Build the Fock matrix.

    F_(mu)(nu) = H_(mu))(nu)  +  (sigma)_(lambda)(sigma) P_(lambda)(sigma) [((mu))(nu)|(lambda)(sigma)) - 1/2((mu))(lambda)|(nu)(sigma))]
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                       Coulomb J        Exchange K

    ERI has shape ((mu)), (nu), (lambda), (sigma)) with PySCF's ((mu))(nu)|(lambda)(sigma)) ordering.
    """
    J = np.einsum('ls,mnls->mn', P, ERI)   # Coulomb
    K = np.einsum('ls,mlns->mn', P, ERI)   # Exchange
    return H_core + J - 0.5 * K


def compute_energy(P: np.ndarray, H_core: np.ndarray,
                   F: np.ndarray, Vnn: float) -> float:
    """
    Total RHF energy.

    E = 1/2 Tr[P(H_core + F)] + V_nn

    The 1/2 avoids double-counting electron-electron repulsion,
    since it appears once in each electron's Fock operator.
    """
    E_elec = 0.5 * np.einsum('mn,mn->', P, H_core + F)
    return E_elec + Vnn


# ── DIIS accelerator ───────────────────────────────────────────────────────────

class DIIS:
    """
    Pulay DIIS accelerator for SCF convergence.

    Stores the last `max_vecs` Fock matrices and their commutator error
    vectors e = FPS - SPF, then extrapolates an improved Fock matrix
    each cycle by solving a small least-squares problem.
    """

    def __init__(self, max_vecs: int = 8):
        self.max_vecs      = max_vecs
        self.fock_history  = []
        self.error_history = []

    def update(self, F: np.ndarray, P: np.ndarray, S: np.ndarray) -> np.ndarray:
        """
        Compute error e = FPS - SPF, store (F, e), return extrapolated F.
        Falls back to the plain F if the B matrix is singular.
        """
        e = F @ P @ S - S @ P @ F

        self.fock_history.append(F.copy())
        self.error_history.append(e.copy())

        if len(self.fock_history) > self.max_vecs:
            self.fock_history.pop(0)
            self.error_history.pop(0)

        n = len(self.fock_history)
        if n < 2:
            return F

        # Build B_ij = <e_i | e_j>  with Lagrange row/column
        B = np.zeros((n + 1, n + 1))
        for i in range(n):
            for j in range(n):
                B[i, j] = np.einsum('mn,mn->', self.error_history[i],
                                               self.error_history[j])
        B[-1, :-1] = -1.0
        B[:-1, -1] = -1.0

        rhs      = np.zeros(n + 1)
        rhs[-1]  = -1.0

        try:
            coeffs = np.linalg.solve(B, rhs)
        except np.linalg.LinAlgError:
            return F  # singular B → fall back

        F_diis = sum(coeffs[i] * self.fock_history[i] for i in range(n))
        return F_diis


# ── SCF driver ─────────────────────────────────────────────────────────────────

def scf_loop_diis(H_core, S, ERI, X, Vnn, n_occ,
                  max_cycles=100, e_tol=1e-8, d_tol=1e-6):
    """
    Run the RHF SCF loop with DIIS acceleration.

    Parameters
    ----------
    H_core     : ndarray (nao, nao)  — core Hamiltonian T + V_nuc
    S          : ndarray (nao, nao)  — AO overlap matrix
    ERI        : ndarray (nao,)*4    — two-electron repulsion integrals
    X          : ndarray (nao, nao)  — orthogonalising matrix S^(-1/2)
    Vnn        : float               — nuclear repulsion energy
    n_occ      : int                 — number of doubly occupied MOs
    max_cycles : int                 — iteration ceiling
    e_tol      : float               — energy convergence threshold (Hartree)
    d_tol      : float               — density-matrix RMS convergence threshold

    Returns
    -------
    (E, C, epsilon, P)  on convergence, or None if SCF fails.
    """
    C, epsilon = initial_guess(H_core, X)
    P          = make_density(C, n_occ)
    diis       = DIIS(max_vecs=8)
    E_old      = 0.0

    print(f"{'Cycle':>6}  {'Energy (Hartree)':>18}  {'ΔE':>12}  {'RMS ΔP':>12}  {'Status':>10}")
    print("-" * 68)

    for cycle in range(1, max_cycles + 1):
        F         = build_fock(H_core, P, ERI)
        F         = diis.update(F, P, S)
        E         = compute_energy(P, H_core, F, Vnn)

        F_prime          = X.T @ F @ X
        epsilon, C_prime = np.linalg.eigh(F_prime)
        C                = X @ C_prime
        P_new            = make_density(C, n_occ)

        delta_E   = E - E_old
        rms_dP    = np.sqrt(np.mean((P_new - P) ** 2))
        converged = (abs(delta_E) < e_tol) and (rms_dP < d_tol)

        print(f"{cycle:>6}  {E:>18.8f}  {delta_E:>12.2e}  {rms_dP:>12.2e}  "
              f"{'CONVERGED' if converged else '':>10}")

        if converged:
            break

        P     = P_new
        E_old = E
    else:
        print(f"WARNING: SCF did not converge in {max_cycles} cycles")
        return None

    print(f"\nFinal RHF energy:  {E:.8f} Hartree")
    return E, C, epsilon, P


# ── Convenience runner ─────────────────────────────────────────────────────────

def run_rhf(mol):
    """
    Build all integrals and run RHF SCF for `mol`.

    Returns (E, C, epsilon, P) or None.
    """
    S      = mol.intor('int1e_ovlp')
    T      = mol.intor('int1e_kin')
    V      = mol.intor('int1e_nuc')
    ERI    = mol.intor('int2e')
    H_core = T + V
    Vnn    = nuclear_repulsion(mol)
    X      = orthogonalizer(S)
    n_occ  = mol.nelectron // 2

    return scf_loop_diis(H_core, S, ERI, X, Vnn, n_occ)


if __name__ == "__main__":
    from molecule import build_acetone
    mol    = build_acetone()
    result = run_rhf(mol)
    if result:
        E, C, epsilon, P = result
        S = mol.intor('int1e_ovlp')
        n_occ = mol.nelectron // 2
        print(f"\nTr(PS) = {np.trace(P @ S):.8f}  (should be {mol.nelectron})")
        print("\nOrbital energies (Hartree):")
        for i, e in enumerate(epsilon):
            occ = "occ" if i < n_occ else "vir"
            print(f"  MO {i+1:2d} ({occ}): {e:10.5f}")
