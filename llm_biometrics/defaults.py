"""
Default hyper-parameters, in one torch-free module.

Keeping these out of :mod:`llm_biometrics.extract` lets the scoring, analysis
and plotting code run in a numpy-only environment -- only *extraction* needs
torch and transformers.
"""

#: Singular directions retained per weight matrix when extracting subspaces.
#: The paper's cached bases use 512; Sec. 5 and Fig. 8 discuss k = 256.  The
#: score is insensitive to this within a wide range because Eq. (8) reads the
#: least-aligned tail, but it is not identical -- see docs/reproducing.md.
DEFAULT_TOP_K = 512

#: ``J`` in Eq. (8): least-aligned singular directions averaged per layer.
DEFAULT_J = 3

#: ``K_layer`` in Alg. A2: least-aligned layers averaged per component.
DEFAULT_N_BOTTOM = 3
