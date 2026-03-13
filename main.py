"""
main.py  —  NMR Predictor (GIAO) Pipeline Driver
-------------------------------------------------

Runs the full pipeline in order:

  Step 1  Molecule setup           (molecule.py)
  Step 2  RHF SCF                  (rhf.py)
  Step 3  KS-LDA SCF               (ks_dft.py)
  Step 4  GIAO Stage 1 integrals   (giao.py)

  Steps 5+ (coming soon)
    Stage 2  CPKS equations → perturbed density P^(alpha)
    Stage 3  Assemble shielding tensor sigma^K_(alpha,beta)
    Stage 4  Convert sigma → chemical shifts δ

Usage
-----
    python main.py

Each module can also be run standalone for faster iteration:
    python rhf.py
    python ks_dft.py
    python giao.py
"""

import numpy as np
from pyscf import dft as pyscf_dft

from molecule import build_acetone, print_mol_info
from rhf import run_rhf
from ks_dft import run_ks_lda, eval_density, eval_aos, build_grid
from giao import validate_giao_integrals


def main():
    # ── Step 1: Molecule ─────────────────────────────────────────────────────
    print("=" * 68)
    print("STEP 1  Molecule Setup")
    print("=" * 68)
    mol = build_acetone(basis='sto-3g')
    print_mol_info(mol)

    # ── Step 2: RHF ──────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("STEP 2  Restricted Hartree-Fock (RHF)")
    print("=" * 68)
    result_rhf = run_rhf(mol)
    if result_rhf is None:
        raise RuntimeError("RHF SCF failed to converge.")
    E_rhf, C_rhf, eps_rhf, P_rhf = result_rhf

    S     = mol.intor('int1e_ovlp')
    n_occ = mol.nelectron // 2
    print(f"\nTr(PS) = {np.trace(P_rhf @ S):.8f}  (should be {mol.nelectron})")
    print("\nOrbital energies (Hartree):")
    for i, e in enumerate(eps_rhf):
        occ = "occ" if i < n_occ else "vir"
        print(f"  MO {i+1:2d} ({occ}): {e:10.5f}")

    # ── Step 3: KS-LDA ───────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("STEP 3  Kohn-Sham DFT (LDA = Slater exchange + VWN5 correlation)")
    print("=" * 68)
    result_ks = run_ks_lda(mol, grid_level=3)
    if result_ks is None:
        raise RuntimeError("KS-LDA SCF failed to converge.")
    E_ks, C_ks, eps_ks, P_ks = result_ks

    # Validate density integrates to N
    coords, weights = build_grid(mol, level=3)
    ao              = eval_aos(mol, coords)
    rho             = eval_density(P_rhf, ao)
    N_check         = np.dot(weights, rho)
    print(f"\n∫ρ dr = {N_check:.6f}   (should be {mol.nelectron})")
    print(f"Error : {abs(N_check - mol.nelectron):.2e} electrons")

    # Cross-check against PySCF reference
    mf_ref             = pyscf_dft.RKS(mol)
    mf_ref.xc          = 'lda,vwn'
    mf_ref.grids.level = 3
    mf_ref.verbose     = 0
    mf_ref.kernel()

    print(f"\n{'Method':<22} {'Energy (Hartree)':>18}")
    print("-" * 42)
    print(f"{'Your RHF':<22} {E_rhf:>18.8f}")
    print(f"{'Your KS-LDA':<22} {E_ks:>18.8f}")
    print(f"{'PySCF KS-LDA (ref)':<22} {mf_ref.e_tot:>18.8f}")
    print(f"\nYour LDA vs PySCF ref : {E_ks - mf_ref.e_tot:>+.2e} Hartree")
    print(f"RHF vs LDA (corr. E)  : {E_ks - E_rhf:>+.6f} Hartree")
    print("\nKS-LDA energy is lower than RHF — DFT captures electron correlation.")

    # ── Step 4: GIAO Stage 1 ─────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("STEP 4  GIAO Stage 1 — Magnetic Field Derivative Integrals")
    print("=" * 68)
    all_ok = validate_giao_integrals(mol, P_rhf)
    if not all_ok:
        print("WARNING: GIAO validation failed — check giao.py output.")

    print("\nPipeline complete.  Next: implement CPKS (Stage 2).")


if __name__ == "__main__":
    main()