"""
Dynamic Neuro Receptor PET (3D Slicer module) - workflow app

A step-by-step workflow for dynamic receptor PET of the brain, built on the other PET4D modules:

    1. Data                 the patient's dynamic PET series (DICOM, or a BIDS 4D NIfTI: a file or a subject /
                            session folder, with its CT / MR) and its CT / MR (the contrast, T1 / T2 / PD / CT, read
                            from the DICOM tags, a BIDS name or the .json sidecar); name, ID, age and sex (editable);
                            optional Gaussian smoothing of the series (off by default) before the next steps
    2. Motion correction    the Dynamic PET Motion Correction panel, on the series of step 1 (iterative engine by
                            default, BRAINSFit selectable)
    3. CT/MR registration   the Dynamic PET to CT/MR Registration panel, the series to the CT / MR (iterative rigid
                            only in this workflow)
    4. Normalization        to a template space (default: MNI ICBM 152 2009c; the iterative engine, one affine stage
                            by default; BRAINSFit and deformable passes selectable): the PET registered to the
                            template, or the CT / MR normalized and the same transform applied to the PET (default
                            when the study has a CT / MR). Checked in the registration layout (PET / fusion /
                            template, checkerboard, manual correction), then hardened (Overwrite by default).
    5. Regions              a segmentation of the space (CerebrA with pons, CerebrA PET, or one of the user's) in a
                            grouping (its regions, broad regions, Broad (Compatible)), left and right separate; the
                            reference region a region of the same atlas with the sides merged (default: the
                            cerebellar grey matter) or the centrum semiovale of the space (white matter, for tracers
                            binding in all grey matter). The label maps of the groupings are made on first use.
    6. Modeling             the Dynamic PET Modeling panel, ready with the normalized series, the regions, SRTM and
                            the reference region of step 5

Steps 2, 3 and 6 show the panels of those modules inside this one; the series each step makes (a motion corrected or
registered series saved separately) becomes the series of the next step. The template spaces and segmentations come
from the atlas library (NeuroPETLib/neuro.py): the extension's Resources/Atlas and the user's ~/NeuroPET/Atlas.
"""


import logging
import os
import sys

import ctk
import numpy as np
import qt
import vtk
import slicer
from slicer.ScriptedLoadableModule import (ScriptedLoadableModule, ScriptedLoadableModuleWidget,
                                           ScriptedLoadableModuleLogic, ScriptedLoadableModuleTest)

_MODULE_DIR = os.path.dirname(os.path.abspath(globals().get("__file__", "") or "."))
if globals().get("__file__") and _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

# The logic and the widget of the workflow are in NeuroWorkflowLib/base.py (shared with the Static Neuro module);
# every name of it is imported here, so 'from DynamicNeuroPET import ...' keeps working.
from NeuroWorkflowLib.base import (VTKObservationMixin, requireBaseExtension, dynamicviews, reg, resampling,
                                   hardenDialog, credits, neuro, normaldb, MergenBanner, PREFIX, REG_PREFIX,
                                   MC_PREFIX, NEURO_PREFIX, ATLAS_FILE_ATTRIBUTE, ATLAS_STAMP_ATTRIBUTE,
                                   NORMALIZED_SUFFIX, REGIONS_NAME, REFERENCE_SUFFIX, SETTINGS_SPACE,
                                   SETTINGS_REGION_ATLAS, GROW_MM, ERODE_REFERENCE_MM, WM_PROBABILITY_LIMIT,
                                   DEFAULT_TARGET, DEFAULT_MODEL, METHOD_PET, METHOD_ANATOMY, METHOD_PET_TEMPLATE,
                                   METHODS, PET_TEMPLATE_ATTRIBUTE, PET_TEMPLATE_COLORMAP, NORM_TEMPLATE_PET_COLOR_ID,
                                   NORM_PATIENT_PET_COLOR_ID, NORMALIZATION_STEPS, NORMALIZATION_WORKING_MM,
                                   NORMALIZATION_BSPLINE_GRID, NORMALIZATION_SAMPLING,
                                   NORMALIZATION_CONVERGENCE_FACTOR, NORMALIZATION_BRAINSFIT, NORMALIZATION_ENGINE,
                                   NORMALIZATION_ITERATIVE_LINEAR, NORMALIZATION_ITERATIVE_DEFORMABLE,
                                   NORMALIZATION_ITERATIVE_SPACING_MM, NORMALIZATION_ITERATIVE_QUALITY,
                                   NORMALIZATION_ITERATIVE_GRID_MM, NORMALIZATION_BRAIN_MASK,
                                   NORMALIZATION_INITIALIZATION, CT_BRAIN_WORKING_MM, CT_SKULL_STRIP, CT_CLIP_HU,
                                   CT_BONE_HU, CT_BONE_VALUE, CT_BRAIN_MIN_ML, NORMALIZATION_GRIDS, STEP_DATA,
                                   STEP_MC, STEP_REG, STEP_NORM, STEP_REGIONS, STEP_MODEL, STEPS, TOOLBAR_OBJECT_NAME,
                                   ICON_PATH, STATE_DONE, STATE_READY, STATE_WAITING, STATE_COLORS, _vtkMatrix,
                                   _ijkToRas, _percentiles, defaultReferenceName, NeuroWorkflowLogic, _badge,
                                   _badgePixmap, _button, _growIntoCombo, NeuroWorkflowWidget)  # noqa: E402,F401


class DynamicNeuroPET(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Mergen - Neuro PET - Dynamic"
        parent.categories = ["Nuclear Medicine"]
        parent.dependencies = ["DICOMGatedDynamicPET", "DynamicPETEditor", "DynamicPETMotionCorrection",
                               "DynamicPETRegistration", "DynamicPETModeling"]     # the PET4D extension
        parent.contributors = ["Burak Demir"]
        parent.helpText = (
            "Workflow for dynamic receptor PET of the brain: <b>1. Data</b> (dynamic series from DICOM or a BIDS "
            "NIfTI, the CT/MR, patient, age and sex, optional smoothing), <b>2. Motion correction</b>, <b>3. CT/MR "
            "registration</b> (rigid), <b>4. Normalization</b> to a template space (default MNI ICBM 152 2009c), "
            "<b>5. Regions</b> (a segmentation of the space, default CerebrA) and <b>6. Modeling</b> (SRTM with the "
            "cerebellar grey matter as the reference by default). Steps 2, 3 and 6 show the panels of the PET4D "
            "modules. The extension brings the MNI 2009c space and the CerebrA segmentations; more spaces and "
            "segmentations go in your atlas folder (~/NeuroPET/Atlas, 'Open atlas folder' in step 4).")
        parent.acknowledgementText = (
            "Part of Mergen (Neuro PET/SPECT Analysis), built on the PET4D (DICOM Gated / Dynamic PET) extension. "
            "MNI ICBM 152 nonlinear symmetric 2009c template: Fonov "
            "VS et al., NeuroImage 2011;54:313-327 and NeuroImage 2009;47:S102 (Copyright (C) 1993-2004 Louis "
            "Collins, McConnell Brain Imaging Centre, Montreal Neurological Institute, McGill University). CerebrA: "
            "Manera AL et al., Scientific Data 2020;7:237 (CC0). Registration by the iterative multi-resolution "
            "engine (SimpleITK) or BRAINSFit.")


class DynamicNeuroPETLogic(NeuroWorkflowLogic):
    """The workflow's logic (NeuroWorkflowLib.base.NeuroWorkflowLogic): normalization and regions; usable without the
    widget (see there)."""


class DynamicNeuroPETWidget(NeuroWorkflowWidget):
    """The workflow's widget (NeuroWorkflowLib.base.NeuroWorkflowWidget) with the Dynamic Neuro steps."""

    MODULE_NAME = "DynamicNeuroPET"
    LOGIC_CLASS = DynamicNeuroPETLogic

    def onReload(self):
        """Developer reload: the NeuroWorkflowLib files reloaded first (Slicer reloads only this file)."""
        import NeuroWorkflowLib
        NeuroWorkflowLib.reloadModules()
        ScriptedLoadableModuleWidget.onReload(self)


def _testFolder(prefix="tmp"):
    """A temporary folder of a test, removed when Python (Slicer) exits."""
    import atexit
    import shutil
    import tempfile
    folder = tempfile.mkdtemp(prefix=prefix)
    atexit.register(shutil.rmtree, folder, True)
    return folder


# ---------------------------------------------------------------------------
# Test: CerebrA labels into a segmentation; contrast; a synthetic normalization
# ---------------------------------------------------------------------------
class DynamicNeuroPETTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_Regions()
        self.setUp()
        self.test_Widget()

    def test_Regions(self):
        folder, cache = _testFolder(), _testFolder()
        previous = {v: os.environ.get(v) for v in (neuro.USER_FOLDER_VARIABLE, neuro.GENERATED_FOLDER_VARIABLE)}
        os.environ[neuro.USER_FOLDER_VARIABLE] = folder
        os.environ[neuro.GENERATED_FOLDER_VARIABLE] = cache
        try:
            self._testRegions(folder)
        finally:
            for variable, value in previous.items():
                if value is None:
                    os.environ.pop(variable, None)
                else:
                    os.environ[variable] = value

    def _testRegions(self, folder):
        # a segmentation of the user's atlas folder: a label volume, its table and a groups table
        segmentationFolder = os.path.join(folder, neuro.SEGMENTATION_FOLDER, "TestRegions")
        os.makedirs(segmentationFolder)
        with open(os.path.join(segmentationFolder, neuro.SEGMENTATION_MANIFEST), "w") as f:
            f.write('{"name": "Test regions", "space": "TestSpace"}')
        with open(os.path.join(segmentationFolder, neuro.SEGMENTATION_CSV_NAME), "w") as f:
            f.write("Mindboggle ID;Label Name;RH Label;LH Labels;Notes;Dice Kappa\n"
                    "51;Putamen;21;72;;0,87\n45;Cerebellum Gray Matter;46;97;;0,83\n444;Pons;103;104;;0\n")
        with open(os.path.join(segmentationFolder, neuro.GROUPS_CSV_NAME), "w") as f:
            f.write("Group;Structures\nStriatum;Putamen\nCerebellum;Cerebellum Gray Matter\nPons;Pons\n")
        array = np.zeros((20, 20, 20), dtype=np.int16)
        array[2:6, 2:6, 2:6] = 21
        array[2:6, 10:14, 2:6] = 72
        array[12:16, 2:6, 2:6] = 46
        array[12:16, 10:14, 2:6] = 97
        array[8:10, 8:10, 8:10] = 103
        array[8:10, 10:12, 8:10] = 104
        source = slicer.util.addVolumeFromArray(array, name="labels", nodeClassName="vtkMRMLLabelMapVolumeNode")
        slicer.util.saveNode(source, os.path.join(segmentationFolder, "labels.nrrd"))
        slicer.mrmlScene.RemoveNode(source)
        self.assertEqual([s["id"] for s in neuro.listSegmentations("TestSpace")], ["TestRegions"])
        logic = DynamicNeuroPETLogic()
        atlas = {}          # no space volumes: no brain mask, no white matter map
        # default reference: the same atlas with the sides merged, the cerebellar grey matter
        key = neuro.regionAtlasKey("TestRegions")
        segmentation, referenceId = logic.loadAtlasRegions(atlas, key)
        names = sorted(segmentation.GetSegmentation().GetSegment(i).GetName() for i in logic.segmentIds(segmentation))
        self.assertEqual(names, sorted(["Putamen R", "Putamen L", "Cerebellum Gray Matter R",
                                        "Cerebellum Gray Matter L", "Pons R", "Pons L",
                                        "Cerebellum Gray Matter" + REFERENCE_SUFFIX]))
        mask = slicer.util.arrayFromSegmentBinaryLabelmap(segmentation, referenceId)
        self.assertEqual(int(np.count_nonzero(mask)), 2 * 64)
        right = logic.segmentIdByName(segmentation, "Cerebellum Gray Matter R")
        self.assertEqual(int(np.count_nonzero(slicer.util.arrayFromSegmentBinaryLabelmap(segmentation, right))), 64)
        # the sides merged: made in the cache folder
        merged, mergedRef = logic.loadAtlasRegions(atlas, neuro.mergedRegionAtlas(key), neuro.mergedRegionAtlas(key),
                                                   "Pons")
        names = sorted(merged.GetSegmentation().GetSegment(i).GetName() for i in logic.segmentIds(merged))
        self.assertEqual(names, ["Cerebellum Gray Matter", "Pons", "Putamen"])
        self.assertEqual(int(np.count_nonzero(slicer.util.arrayFromSegmentBinaryLabelmap(merged, mergedRef))), 16)
        self.assertTrue(os.path.isfile(neuro.regionMapPaths(neuro.mergedRegionAtlas(key))[0]))
        # broad regions of the groups table; loading again into the same segmentation replaces its segments
        broad = neuro.regionAtlasKey("TestRegions", neuro.BROAD)
        grouped, _ = logic.loadAtlasRegions(atlas, broad)
        names = sorted(grouped.GetSegmentation().GetSegment(i).GetName() for i in logic.segmentIds(grouped))
        self.assertEqual(names, ["Cerebellum L", "Cerebellum R", "Pons L", "Pons R", "Striatum L", "Striatum R"])
        again, _ = logic.loadAtlasRegions(atlas, broad, target=grouped)
        self.assertIs(again, grouped)
        self.assertEqual(len(logic.segmentIds(again)), 6)
        # grown regions (no brain mask here) and an eroded reference of its own
        grown, grownRef = logic.loadAtlasRegions(atlas, neuro.mergedRegionAtlas(key), growMm=1.0,
                                                 erodeReferenceMm=0.5)
        putamen = logic.segmentIdByName(grown, "Putamen")
        self.assertGreater(int(np.count_nonzero(slicer.util.arrayFromSegmentBinaryLabelmap(grown, putamen))), 2 * 64)
        self.assertEqual(grown.GetSegmentation().GetSegment(grownRef).GetName(),
                         "Cerebellum Gray Matter" + REFERENCE_SUFFIX)
        self.assertLess(int(np.count_nonzero(slicer.util.arrayFromSegmentBinaryLabelmap(grown, grownRef))), 2 * 64)
        with self.assertRaises(ValueError):        # no white matter map
            logic.loadAtlasRegions(atlas, key, growMm=1.0, growInto=normaldb.GROW_ONLY_WM)
        # the extension's segmentations are there, on the extension's space
        self.assertIn(neuro.DEFAULT_SEGMENTATION, [s["id"] for s in neuro.listSegmentations(neuro.DEFAULT_SPACE)])
        self.assertIn("T1", neuro.atlasFiles())
        self.delayDisplay("Regions test passed")

    def test_Widget(self):
        widget = slicer.modules.dynamicneuropet.widgetRepresentation().self()
        for key, _, _, _ in STEPS:
            widget.showStep(key, enterStep=False)
            self.assertEqual(widget.current, key)
            self.assertEqual(widget.stack.count, 1)
        widget.showStep(STEP_DATA, enterStep=False)
        self.delayDisplay("Widget test passed")
