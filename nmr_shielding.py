"""
nmr_shielding.py  —  Stages 3 & 4: Shielding Tensor & Chemical Shifts
----------------------------------------------------------------------

Stage 3 — Shielding tensor assembly
    σ^K_αβ = σ^K,dia_αβ + σ^K,para_αβ

    Diamagnetic: σ^K,dia_αβ = (1/(2c²)) Tr[ P · d^K_αβ ]
    Paramagnetic: σ^K,para_αβ = (1/c²) Tr[ p^(α) · pso^(K,β) ]

Stage 4 — Chemical shifts
    δ^K = σ_ref − σ^K_iso  (ppm)
"""

import numpy as np
from giao import giao_pso, giao_diamagnetic
from cpks import solve_cpks, solve_uncoupled, perturbed_density_ao


ALPHA_FINE          = 7.2973525693e-3
C_LIGHT             = 1.0 / ALPHA_FINE
SHIELDING_PREFACTOR = 1.0 / C_LIGHT**2

# Reference shieldings (σ_ref, ppm) used to compute chemical shifts:
#   δ^K = σ_ref − σ^K_iso
#
# *** THESE VALUES ARE BASIS-SET SPECIFIC ***
# The numbers below were computed at the GIAO-RHF/STO-3G level.
# Using them with any other basis set will give wrong chemical shifts.
# Pass custom_refs to run_nmr / chemical_shift to override per-run.
#
# Common alternatives (GIAO-RHF):
#   Basis      ¹H(TMS)   ¹³C(TMS)   ¹⁵N(NH₃)   ¹⁷O(H₂O)
#   STO-3G      30.7      188.1      -135.8       -344.0
#   6-31G*      31.8      194.5      -135.9       -333.9
#   cc-pVDZ     31.5      193.2      -136.4       -330.1
#
_REFERENCE_STO3G = {
    1:  ("TMS  ¹H",   30.7),
    6:  ("TMS  ¹³C", 188.1),
    7:  ("NH₃  ¹⁵N", -135.8),
    8:  ("H₂O  ¹⁷O", -344.0),
}

_REFERENCE_6_31Gstar = {
    1:  ("TMS  ¹H",   31.8),
    6:  ("TMS  ¹³C", 194.5),
    7:  ("NH₃  ¹⁵N", -135.9),
    8:  ("H₂O  ¹⁷O", -333.9),
}

_REFERENCE_ccpVDZ = {
    1:  ("TMS  ¹H",   31.5),
    6:  ("TMS  ¹³C", 193.2),
    7:  ("NH₃  ¹⁵N", -136.4),
    8:  ("H₂O  ¹⁷O", -330.1),
}

# Lookup by basis set name — extend as needed
_REFERENCE_BY_BASIS = {
    'sto-3g':   _REFERENCE_STO3G,
    '6-31g*':   _REFERENCE_6_31Gstar,
    '6-31g(d)': _REFERENCE_6_31Gstar,
    'cc-pvdz':  _REFERENCE_ccpVDZ,
}

def get_references(basis: str = 'sto-3g', custom_refs: dict = None) -> dict:
    """
    Return the reference shielding table for a given basis set.

    Parameters
    ----------
    basis       : str  — basis set name (case-insensitive)
    custom_refs : dict — overrides, keyed by atomic number Z

    Returns
    -------
    refs : dict  {Z: (label, sigma_ref_ppm)}
    """
    key  = basis.lower().replace(' ', '')
    refs = dict(_REFERENCE_BY_BASIS.get(key, _REFERENCE_STO3G))
    if key not in _REFERENCE_BY_BASIS:
        import warnings
        warnings.warn(
            f"No built-in reference shieldings for basis '{basis}'. "
            f"Falling back to STO-3G values — chemical shifts will be wrong "
            f"unless you supply custom_refs.",
            UserWarning, stacklevel=3,
        )
    if custom_refs:
        refs.update(custom_refs)
    return refs


# ── Stage 3 ────────────────────────────────────────────────────────────────────

def assemble_shielding(mol, P, p):
    """
    σ^K_αβ = σ^K,dia + σ^K,para for all nuclei.

    Returns sigma_dia (natm,3,3), sigma_para (natm,3,3), sigma (natm,3,3)
    — all in atomic units (multiply by 1e6 for ppm).
    """
    natm   = mol.natm
    pso    = giao_pso(mol)
    dia_ao = giao_diamagnetic(mol)

    sigma_dia  = np.zeros((natm, 3, 3))
    sigma_para = np.zeros((natm, 3, 3))

    for K in range(natm):
        for alpha in range(3):
            for beta in range(3):
                sigma_dia[K, alpha, beta] = (
                    0.5 * SHIELDING_PREFACTOR
                    * np.einsum('mn,mn->', P, dia_ao[K, alpha, beta])
                )
                sigma_para[K, alpha, beta] = (
                    SHIELDING_PREFACTOR
                    * np.einsum('mn,mn->', p[alpha], pso[K, beta])
                )

    sigma = sigma_dia + sigma_para
    return sigma_dia, sigma_para, sigma


def sigma_to_ppm(sigma):
    return sigma * 1e6


def isotropic(sigma_ppm):
    return np.trace(sigma_ppm, axis1=1, axis2=2) / 3.0


def chemical_shift(sigma_iso_ppm, Z_list, basis: str = 'sto-3g', custom_refs=None):
    refs  = get_references(basis, custom_refs)
    delta = np.zeros(len(sigma_iso_ppm))
    for K, (s, Z) in enumerate(zip(sigma_iso_ppm, Z_list)):
        _, sigma_ref = refs.get(int(Z), ("—", 0.0))
        delta[K] = sigma_ref - s
    return delta


# ── PySCF reference (optional, for validation) ─────────────────────────────────

def pyscf_reference_nmr(mol):
    """
    Run PySCF's built-in GIAO-RHF NMR.  Returns isotropic shieldings (ppm).
    Called only when validate=True in run_nmr.
    """
    try:
        from pyscf.prop import nmr as pyscf_nmr
        from pyscf import scf
    except ImportError:
        from pyscf.nmr import rhf as pyscf_nmr
        from pyscf import scf

    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()

    try:
        nmr_obj = pyscf_nmr.RHF(mf)
    except AttributeError:
        nmr_obj = pyscf_nmr.NMR(mf)

    nmr_obj.verbose = 0
    tensors  = nmr_obj.kernel()
    ref_iso  = np.trace(tensors, axis1=1, axis2=2) / 3.0
    return ref_iso


# ── Pretty-printer ─────────────────────────────────────────────────────────────

def print_nmr_table(mol, sigma_dia_ppm, sigma_para_ppm, sigma_ppm,
                    sigma_iso, delta, ref_iso=None, include_tensor=False):
    print()
    print("=" * 84)
    print("  NMR Shielding Summary")
    print("=" * 84)

    hdr = (f"  {'#':>3}  {'Elem':>5}  {'σ_dia':>10}  {'σ_para':>10}"
           f"  {'σ_iso':>10}  {'δ (ppm)':>10}")
    if ref_iso is not None:
        hdr += f"  {'σ_iso ref':>10}  {'error':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for K in range(mol.natm):
        sym  = mol.atom_pure_symbol(K)
        d_K  = isotropic(sigma_dia_ppm[K:K+1])[0]
        p_K  = isotropic(sigma_para_ppm[K:K+1])[0]
        line = (f"  {K+1:>3}  {sym:>5}  {d_K:>10.2f}  {p_K:>10.2f}"
                f"  {sigma_iso[K]:>10.2f}  {delta[K]:>10.2f}")
        if ref_iso is not None:
            err  = sigma_iso[K] - ref_iso[K]
            line += f"  {ref_iso[K]:>10.2f}  {err:>8.2f}"
        print(line)

    print("  " + "-" * (len(hdr) - 2))

    if ref_iso is not None:
        rms = np.sqrt(np.mean((sigma_iso - ref_iso)**2))
        mae = np.mean(np.abs(sigma_iso - ref_iso))
        print(f"\n  RMS error vs reference: {rms:.2f} ppm")
        print(f"  MAE error vs reference: {mae:.2f} ppm")

    if include_tensor:
        print("\n  Full shielding tensors (ppm):")
        for K in range(mol.natm):
            print(f"\n  Nucleus {K+1} ({mol.atom_pure_symbol(K)}):")
            for row in sigma_ppm[K]:
                print("    " + "  ".join(f"{v:9.2f}" for v in row))

    print("=" * 84)


# ── Pipeline driver ────────────────────────────────────────────────────────────

def run_nmr(mol, C, epsilon, P, n_occ,
            ERI=None, ERI_ig1=None, use_cpks=True,
            basis: str = None, custom_refs=None, print_tensor=False,
            validate=True):
    """
    Full NMR pipeline: CPKS → σ tensor → chemical shifts.

    Parameters
    ----------
    ERI_ig1    : (3, nao, nao, nao, nao) from mol.intor('int2e_ig1', comp=3)
    basis      : str — basis set name used for the SCF (e.g. 'sto-3g', '6-31g*').
                 Selects the matching reference shieldings for chemical shifts.
                 Defaults to mol.basis if not supplied.
    validate   : if True, also run PySCF's GIAO-RHF as a reference (recommended
                 until results are verified correct)
    """
    if basis is None:
        basis = mol.basis if isinstance(mol.basis, str) else 'sto-3g'
    if ERI is None:
        ERI = mol.intor('int2e')

    # Stage 2 — CPKS
    print("\n" + "=" * 68)
    print("STAGE 2  CPKS — Perturbed MO Coefficients")
    print("=" * 68)

    if use_cpks:
        u = solve_cpks(mol, C, P, epsilon, n_occ, ERI, ERI_ig1)
    else:
        u = solve_uncoupled(mol, C, P, epsilon, n_occ, ERI_ig1)

    p = perturbed_density_ao(C, u, n_occ)
    asym = max(np.max(np.abs(p[a] + p[a].T)) for a in range(3))
    print(f"\n  Antisymmetry ‖p+p.T‖_max = {asym:.2e}")

    # Stage 3 — Shielding tensor
    print("\n" + "=" * 68)
    print("STAGE 3  Shielding Tensor Assembly")
    print("=" * 68)
    s_dia_au, s_para_au, s_au = assemble_shielding(mol, P, p)
    s_dia_ppm  = sigma_to_ppm(s_dia_au)
    s_para_ppm = sigma_to_ppm(s_para_au)
    s_ppm      = sigma_to_ppm(s_au)
    s_iso      = isotropic(s_ppm)
    print(f"  Assembled tensors for {mol.natm} nuclei.")

    # Stage 4 — Chemical shifts
    print("\n" + "=" * 68)
    print("STAGE 4  Chemical Shifts")
    print("=" * 68)
    charges = mol.atom_charges().tolist()
    delta   = chemical_shift(s_iso, charges, basis=basis, custom_refs=custom_refs)

    # Optional PySCF validation
    ref_iso = None
    if validate:
        print("\n  Running PySCF GIAO-RHF reference …")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref_iso = pyscf_reference_nmr(mol)

    print_nmr_table(mol, s_dia_ppm, s_para_ppm, s_ppm, s_iso, delta,
                    ref_iso=ref_iso, include_tensor=print_tensor)

    return s_ppm, s_iso, delta