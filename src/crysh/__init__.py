"""CrysH — Crystal Hierarchy: a multi-layer crystal knowledge mapper.

Layers (see README):

    L0 validity        L1 dimensionality   L2 morphology
    L3 local building blocks (m_i)         L4 motif-network architecture

The public API is assembled in :mod:`crysh` from the layer modules; the package is
import-safe without optional dependencies (pandas / pymatgen / matplotlib are only
needed by the ``tables`` and ``research`` extras).
"""

from __future__ import annotations

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
