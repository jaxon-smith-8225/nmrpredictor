"""
main.py  —  NMR Predictor (GIAO) Pipeline Driver
"""

import numpy as np
from pyscf import dft as pyscf_dft

from molecule import build_acetone, print_mol_info
from rhf import run_rhf
from ks_dft import run_ks_lda, eval_density, eval_aos, build_grid
from giao import giao_diamagnetic
from nmr_shielding import run_nmr


def main():
    print("=" * 68)
    print("STEP 1  Molecule Setup")
    print("=" * 68)
    mol = build_acetone(basis='sto-3g')
    print_mol_info(mol)

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

    print("\n" + "=" * 68)
    print("STEP 3  Kohn-Sham DFT (LDA)")
    print("=" * 68)
    result_ks = run_ks_lda(mol, grid_level=3)
    if result_ks is None:
        raise RuntimeError("KS-LDA SCF failed to converge.")
    E_ks, C_ks, eps_ks, P_ks = result_ks

    mf_ref             = pyscf_dft.RKS(mol)
    mf_ref.xc          = 'lda,vwn'
    mf_ref.grids.level = 3
    mf_ref.verbose     = 0
    mf_ref.kernel()
    print(f"Your KS-LDA vs PySCF ref : {E_ks - mf_ref.e_tot:>+.2e} Hartree")

    print("\n" + "=" * 68)
    print("STEP 4  GIAO Stage 1 — Magnetic Field Derivative Integrals")
    print("=" * 68)
    dia = giao_diamagnetic(mol)
    dia_O_iso = sum(np.einsum('mn,mn->', P_rhf, dia[3, i, i]) for i in range(3)) / 3
    print(f"Tr[P·d_iso] for oxygen (no prefactor): {dia_O_iso:.6f} a.u.")

    # Precompute ERIs once
    print("\nPrecomputing ERI and GIAO-ERI tensors …")
    ERI     = mol.intor('int2e')
    ERI_ig1 = mol.intor('int2e_ig1', comp=3)   # 2e GIAO skeleton (bra-1 only)

    # NMR prediction — validate=True keeps PySCF comparison until we verify
    print("\n\n" + "=" * 68)
    print("NMR PREDICTION  (GIAO-RHF / STO-3G)")
    print("=" * 68)
    sigma_ppm, sigma_iso, delta = run_nmr(
        mol, C_rhf, eps_rhf, P_rhf, n_occ,
        ERI=ERI,
        ERI_ig1=ERI_ig1,
        use_cpks=True,
        print_tensor=True,
        validate=True,   # set False once results match reference
    )

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()