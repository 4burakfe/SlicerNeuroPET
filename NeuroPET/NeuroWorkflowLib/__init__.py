"""
NeuroWorkflowLib: the step-by-step workflow shared by the Neuro modules (Slicer / Qt code).

    base        logic (normalization, atlas regions) and widget (steps, toolbar, Data / Normalization / Regions
                pages, embedded PET4D panels) of DynamicNeuroPET and StaticNeuroPET

Pure numpy code that can be tested without Slicer stays in NeuroPETLib.
"""

from NeuroPETLib import requireBaseExtension

requireBaseExtension()     # PET4D (DICOMGatedDynamicPETLib) importable before any file here is imported

# in reload order (a file before the files that import it)
MODULES = ("base",)


def reloadModules():
    """Developer reload: this package's files again (Slicer reloads only a module's own file)."""
    import importlib
    import sys
    importlib.reload(sys.modules[__name__])
    for name in MODULES:
        importlib.reload(importlib.import_module(__name__ + "." + name))
