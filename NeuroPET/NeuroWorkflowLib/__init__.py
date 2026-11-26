"""
NeuroWorkflowLib: the step-by-step workflow shared by the Neuro modules (Slicer / Qt code).

    base        logic (normalization, atlas regions) and widget (steps, toolbar, Data / Normalization / Regions
                pages, embedded PET4D panels) of DynamicNeuroPET and StaticNeuroPET
    scenestate  the workflow saved with the scene: the step and module shown, the layouts known at startup, the
                workflow opened again on its step when a saved scene is loaded
    database    the normal database: folder and list, the pick on the Normalization page, the Regions page locked
                to its region settings, the regions step views
    compare     the Dynamic Neuro Review/Compare step: the parametric maps in the Static Neuro review
    dynamicdb   a patient's parametric maps and region fits added to a dynamic normal database
    dynreport   the Dynamic Neuro reports: Modeling (model, TACs, frames, results) and Compare (PDF / DICOM)
    dyntoolbar  the Modeling step's toolbar row (report, screenshots, view settings of the Modeling layout)

Pure numpy code that can be tested without Slicer stays in NeuroPETLib.
"""

from NeuroPETLib import requireBaseExtension

requireBaseExtension()     # PET4D (DICOMGatedDynamicPETLib) importable before any file here is imported

# in reload order (a file before the files that import it)
MODULES = ("scenestate", "base", "database", "compare", "dynamicdb", "dynreport", "dyntoolbar")


def reloadModules():
    """Developer reload: this package's files again (Slicer reloads only a module's own file)."""
    import importlib
    import sys
    importlib.reload(sys.modules[__name__])
    for name in MODULES:
        importlib.reload(importlib.import_module(__name__ + "." + name))
