"""
conftest.py  —  shared pytest fixtures
---------------------------------------
Fixtures are session-scoped so the expensive SCF calculations run only
once per `pytest` invocation, regardless of how many test modules import them.
"""

import pytest
import sys
import os

# Make the project root importable when tests are run from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from molecule import build_acetone
from rhf import run_rhf
from ks_dft import run_ks_lda, build_grid, eval_aos


@pytest.fixture(scope="session")
def mol():
    """Acetone molecule with STO-3G basis."""
    return build_acetone(basis="sto-3g")


@pytest.fixture(scope="session")
def rhf_result(mol):
    """Converged RHF result: (E, C, epsilon, P)."""
    result = run_rhf(mol)
    assert result is not None, "RHF SCF failed to converge"
    return result


@pytest.fixture(scope="session")
def rhf_density(rhf_result):
    """Convenience: converged RHF density matrix P."""
    _, _, _, P = rhf_result
    return P


@pytest.fixture(scope="session")
def ks_result(mol):
    """Converged KS-LDA result: (E, C, epsilon, P)."""
    result = run_ks_lda(mol, grid_level=3)
    assert result is not None, "KS-LDA SCF failed to converge"
    return result


@pytest.fixture(scope="session")
def grid(mol):
    """Numerical integration grid (coords, weights)."""
    return build_grid(mol, level=3)


@pytest.fixture(scope="session")
def ao_values(mol, grid):
    """AO basis functions evaluated on the grid."""
    coords, _ = grid
    return eval_aos(mol, coords)
