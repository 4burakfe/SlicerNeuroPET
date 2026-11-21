"""
Static Neuro PET/SPECT (3D Slicer module) - workflow app

A step-by-step workflow for a static brain PET or SPECT (FDG, amyloid, perfusion SPECT, ...), built on the Dynamic Neuro
Receptor PET module (same normalization and regions) and the Neuro PET normal databases:

    1. Data                 the static PET (PET4D DICOM plugin) or SPECT (DICOM SPECT reconstruction), or an image
                            file (BIDS PET: sub-<id>_ses-<session>_pet.nii.gz with its .json sidecar and the
                            participants.tsv; a file, or a subject / session folder with its CT / MR), the patient's
                            CT / MR (DICOM or a file), the age and sex (read from the tags, editable)
    2. CT/MR registration   optional: the image registered to the CT / MR (iterative rigid only) and hardened
    3. Normalization        the normal database (those of the image's modality, nuclide and tracer first; its
                            normalization settings are the defaults, others warn) and the normalization to a template
                            space (the database's; default MNI ICBM 152 2009c): the PET/SPECT to an MR template,
                            the CT/MR to the MR template with the same transform applied to the image, or the
                            PET/SPECT to the database's master mean map
    4. Regions              a segmentation of the space in a grouping (CerebrA, CerebrA PET, ...): with a database
                            its region settings (segmentation and grouping, growth, white matter,
                            reference region and its mask) are used, locked unless unlocked
    5. Review               the comparison with the database (4 x 4 layout: PET with region outlines, voxel z / t
                            map, 3D-SSP views, region table with SUVmean and regional scores; fusion with the CT /
                            MR / template); adding the patient to a user defined database (SUV and SUVR maps on the
                            database grid, a row in subjects.csv)
"""

import os
import sys

_MODULE_DIR = os.path.dirname(os.path.abspath(globals().get("__file__", "") or "."))
if globals().get("__file__") and _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from NeuroPETLib import requireBaseExtension  # noqa: E402

requireBaseExtension()

import json
import logging

import numpy as np
import qt
import slicer

from slicer.ScriptedLoadableModule import ScriptedLoadableModule, ScriptedLoadableModuleTest
from DICOMGatedDynamicPETLib import registration as reg
from NeuroWorkflowLib.base import (NeuroWorkflowLogic, NeuroWorkflowWidget, METHOD_PET, NEURO_PREFIX, REFERENCE_SUFFIX,
                                   REG_PREFIX, STATE_DONE, STATE_READY, STATE_WAITING, STEP_DATA, STEP_NORM, STEP_REG,
                                   STEP_REGIONS, _ijkToRas)
from NeuroPETLib import normaldb, neuroreview as nr, staticneuro as sn, ssp

from StaticNeuroPETLib.constants import (ICON_PATH, METHODS, NORM_HARDEN_ACTIONS, NORM_HARDEN_GRIDS, REGIONS_COLUMNS,
                                         REGIONS_ORIENTATIONS, STEPS, STEP_REVIEW)
from StaticNeuroPETLib.helpers import _sameRegion, _stepIcon

# The methods of the logic and the widget are in StaticNeuroPETLib, one file per step / topic (mixins): the classes
# below put them together; Reload (developer mode) reloads those files too (onReload).
from StaticNeuroPETLib.logic_image import ImageLogicMixin
from StaticNeuroPETLib.logic_database import DatabaseLogicMixin
from StaticNeuroPETLib.logic_review import ReviewLogicMixin
from StaticNeuroPETLib.logic_batch import BatchLogicMixin
from StaticNeuroPETLib.toolbar import ToolbarMixin
from StaticNeuroPETLib.page_data import DataPageMixin
from StaticNeuroPETLib.page_registration import RegistrationPageMixin
from StaticNeuroPETLib.page_normalization import NormalizationPageMixin
from StaticNeuroPETLib.page_regions import RegionsPageMixin
from StaticNeuroPETLib.review_page import ReviewPageMixin
from StaticNeuroPETLib.review_colors import ReviewColorsMixin
from StaticNeuroPETLib.review_table import ReviewTableMixin
from StaticNeuroPETLib.review_ssp import ReviewSspMixin
from StaticNeuroPETLib.review_cut import ReviewCutMixin
from StaticNeuroPETLib.review_peel import ReviewPeelMixin
from StaticNeuroPETLib.review_layout import ReviewLayoutMixin
from StaticNeuroPETLib.review_mouse import ReviewMouseMixin
from StaticNeuroPETLib.constants import BATCH_PET_FALLBACK, BATCH_SKIP  # noqa: F401 (NeuroPETDatabaseManager)


class StaticNeuroPET(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Mergen - Neuro PET - Static"
        parent.categories = ["Nuclear Medicine"]
        parent.dependencies = ["DICOMGatedDynamicPET", "DynamicPETMotionCorrection", "DynamicPETRegistration",
                               "DynamicNeuroPET"]     # the PET4D extension and Dynamic Neuro Receptor PET
        parent.contributors = ["Burak Demir"]
        parent.helpText = (
            "Workflow for a static brain PET or SPECT: <b>1. Data</b> (static PET / SPECT from DICOM or a file, the "
            "CT/MR, age and sex), <b>2. CT/MR registration</b> (optional), <b>3. Normalization</b> to the MNI ICBM "
            "152 2009c space (PET/SPECT to the T1 template, CT/MR to the MR template, or PET/SPECT to the mean map of "
            "a normal database; the normal database is picked here and gives its normalization settings), <b>4. "
            "Regions</b> (the CerebrA regions as in the database) and <b>5. Review</b> (and adding the patient to a "
            "user defined database). The extension brings the MNI 2009c space and the CerebrA segmentations; more "
            "go in your atlas folder (~/NeuroPET/Atlas, 'Open atlas folder' in step 3).")
        parent.acknowledgementText = (
            "Part of Mergen (Neuro PET/SPECT Analysis), built on the PET4D (DICOM Gated / Dynamic PET) extension. "
            "MNI ICBM 152 nonlinear symmetric 2009c template: Fonov "
            "VS et al., NeuroImage 2011;54:313-327. CerebrA: Manera AL et al., Scientific Data 2020;7:237 (CC0).")


# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------
class StaticNeuroPETLogic(ImageLogicMixin, DatabaseLogicMixin, ReviewLogicMixin, BatchLogicMixin, NeuroWorkflowLogic):
    """The Neuro logic on one static volume (the "series" of the Dynamic Neuro steps is the volume itself):

        logic = StaticNeuroPETLogic()
        volume = logic.loadFromFile("sub-00000016_ses-SUV_pet.nii.gz")
        result = logic.registerToAnatomy(volume, ct)           # optional
        undo, registered = logic.hardenRegistration(volume, ct, linearNode, reg.SAVE_SEPARATELY, reg.OUTPUT_PET_SPACING)
        result = logic.normalize(volume, template, METHOD_PET)
        undo, normalized, _, _ = logic.harden(volume, template, linearNode, reg.SAVE_SEPARATELY, ...)
    """


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------
class StaticNeuroPETWidget(ToolbarMixin,
                            DataPageMixin,
                            RegistrationPageMixin,
                            NormalizationPageMixin,
                            RegionsPageMixin,
                            ReviewPageMixin,
                            ReviewColorsMixin,
                            ReviewTableMixin,
                            ReviewSspMixin,
                            ReviewCutMixin,
                            ReviewPeelMixin,
                            ReviewLayoutMixin,
                            ReviewMouseMixin,
                            NeuroWorkflowWidget):
    """The Dynamic Neuro workflow on a static PET / SPECT: own Data, CT/MR registration, Regions (normal
    database settings) and Review pages; the Normalization page of the Dynamic Neuro module with the normal database
    and the PET template method."""

    MODULE_NAME = "StaticNeuroPET"
    STEPS = STEPS
    LOGIC_CLASS = StaticNeuroPETLogic
    TOOLBAR_NAME = "StaticNeuroPETWorkflowToolbar"
    TOOLBAR_TITLE = "Static Neuro PET/SPECT"
    TOOLBAR_TOOLTIP = "Static Neuro PET/SPECT: back to the first step"
    ICON = ICON_PATH
    STEP_AFTER_REGIONS = STEP_REVIEW
    BANNER_STEPS = (STEP_DATA, STEP_REVIEW)
    NORM_METHODS = METHODS
    LOAD_REGIONS_TEXT = "Load regions and go to review"
    NORM_HARDEN_GRIDS = NORM_HARDEN_GRIDS
    NORM_HARDEN_ACTIONS = NORM_HARDEN_ACTIONS
    def setup(self):
        self._patientTexts = {}         # volume ID: the patient's name (ID) shown on the toolbar
        self._contrastFor = None        # (CT/MR ID, (contrast, reason)): read once per CT/MR
        self.regSession = None
        self.regResult = None
        self.regUndo = None
        self.databaseRows = []          # [(folder, header, rank, problems)]
        self._databaseKey = None
        self._databaseChosen = False
        NeuroWorkflowWidget.setup(self)
        self.pages[STEP_REG] = self._buildRegPage()
        self.pages[STEP_REVIEW] = self._buildReviewPage()
        self.refresh()

    def volume(self):
        return self.sequence()

    # --- navigation: no CT/MR, no registration step; a step opens once the steps it needs are done ----------
    STEP_NEEDS = {STEP_REG: "Pick the PET / SPECT and its CT / MR in step 1 (Data) first.",
                  STEP_NORM: "Pick the PET / SPECT in step 1 (Data) first (the CT/MR registration is not needed).",
                  STEP_REGIONS: "Normalize the image first (Normalization step).",
                  STEP_REVIEW: "Load the regions first (Regions step)."}

    def _skipStep(self, key):
        """Back / Next pass over the CT/MR registration when there is no CT/MR."""
        return key == STEP_REG and self.anatomy() is None

    # --- workflow --------------------------------------------------------------------------------
    def _callStep(self, key, method):
        if key == STEP_REG:
            if method == "exit":
                self._closeRegSession()
            elif method == "enter" and self.regResult is not None:
                self._openRegSession()
            return
        if key == STEP_REGIONS:
            if method == "exit":
                self._exitRegionsViews()
            elif method == "enter":
                self._enterRegionsViews()
        if key == STEP_REVIEW:
            if method == "exit":
                self._exitReviewViews()
            elif method == "enter" and self.node("Segmentation") is not None:
                self._enterReviewViews()
                qt.QTimer.singleShot(200, self._autoCompare)
            return
        NeuroWorkflowWidget._callStep(self, key, method)

    def pendingHarden(self, key):
        if key == STEP_REG:
            return self.regResult is not None
        return NeuroWorkflowWidget.pendingHarden(self, key)

    def _hardenStep(self, key):
        if key == STEP_REG:
            self.onRegHarden()
        else:
            NeuroWorkflowWidget._hardenStep(self, key)

    def stepStates(self):
        volume, anatomy = self.volume(), self.anatomy()
        states = {STEP_DATA: STATE_DONE if volume is not None else STATE_READY}
        ready = STATE_READY if volume is not None else STATE_WAITING
        states[STEP_REG] = (STATE_DONE if volume is not None and volume.GetAttribute(REG_PREFIX + "Fixed") else
                            ready if anatomy is not None else STATE_WAITING)
        normalized = volume is not None and bool(volume.GetAttribute(NEURO_PREFIX + "Template"))
        states[STEP_NORM] = STATE_DONE if normalized else ready
        segmentation = self.node("Segmentation")
        states[STEP_REGIONS] = (STATE_DONE if segmentation is not None else
                                STATE_READY if normalized else STATE_WAITING)
        states[STEP_REVIEW] = STATE_READY if segmentation is not None else STATE_WAITING
        return states

    def refresh(self):
        NeuroWorkflowWidget.refresh(self)
        self._updateRegButtons()
        self._updateReviewPage()
        self._updatePatientLabel()
        self._updateReviewTools()

    def cleanup(self):
        self._closeRegSession()
        if getattr(self, "current", None) == STEP_REGIONS:
            self._exitRegionsViews()
        if getattr(self, "current", None) == STEP_REVIEW:
            self._exitReviewViews()
        NeuroWorkflowWidget.cleanup(self)

    def onSceneClose(self, caller, event):
        NeuroWorkflowWidget.onSceneClose(self, caller, event)
        self._closeRegSession()
        # the views' interactors and the slice and camera nodes of the layout outlive the scene: the review's
        # observers are taken off them before they are forgotten below (else every closed scene left one more
        # set on them and the next review handled each event once more)
        for stop in (self._watchCutSlices, self._watchMouse, self._syncCameras, self._watchPeelWheel):
            try:
                stop(False)
            except Exception:
                logging.exception("Static Neuro: could not remove the review's observers")
        self._regionsPreviousLayout = None
        self._regionsHelpers = None
        self.regResult = None
        self.regUndo = None
        self._databaseKey = None
        self._patientTexts = {}
        self._contrastFor = None
        self.reviewResult = self.scoreNode = self.reviewTable = self.greyPetNode = None
        self._brightnessCache = None
        self.sspModels = {}
        self.cutModels = {}
        self.peelNodes = {}
        self._peelDepth = None
        self._peelRays = None
        self._peelHighlighted = 0
        self._peelMaps = {}
        self._peelWheelObservers = []
        self._peelTexts = {}
        self._cutSliceObservers = []
        self._cutClipNodes = {}
        self._cutFacePlanes = {}
        self._sspMainDisplay = None
        self._sspLowDisplay = None
        self._sspPolys = {}
        self._mouseObservers = {}
        self._drag = None
        self._sspLabels = None
        self._sspOutlineModels = {}
        self._sspOutlineSurfaces = {}
        self._sspAllOutlineModels = {}
        self._sspBorderPolys = {}
        self._sspOutlineLabel = None
        self._sspColorsWritten = {}
        self._sspBounds = None
        self.scoreRgbNode = None
        self._cameraObservers = []
        self._segmentationDisplay = None
        self._anatomyWindowFor = None
        self._flash = self._autoCompared = None
        self._regionTableData = None
        if getattr(self, "regionTable", None) is not None:
            try:
                self.regionTable.setRowCount(0)
            except Exception:
                self.regionTable = None
        if hasattr(self, "compareLabel"):
            self.compareLabel.text = ""
        if hasattr(self, "regLabel"):
            self.regLabel.text = "Not registered."

    def onReload(self):
        """Developer reload: the NeuroWorkflowLib and StaticNeuroPETLib files reloaded first (Slicer reloads only this
        file)."""
        import importlib
        import NeuroWorkflowLib
        import StaticNeuroPETLib
        NeuroWorkflowLib.reloadModules()
        importlib.reload(StaticNeuroPETLib)
        for name in StaticNeuroPETLib.MODULES:
            importlib.reload(importlib.import_module("StaticNeuroPETLib." + name))
        NeuroWorkflowWidget.onReload(self)


def _testFolder(prefix="tmp"):
    """A temporary folder of a test, removed when Python (Slicer) exits."""
    import atexit
    import shutil
    import tempfile
    folder = tempfile.mkdtemp(prefix=prefix)
    atexit.register(shutil.rmtree, folder, True)
    return folder


# ---------------------------------------------------------------------------
# Test: import from a BIDS file, harden a volume, the workflow pages
# ---------------------------------------------------------------------------
class StaticNeuroPETTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_ImportAndHarden()
        self.setUp()
        self.test_Widget()
        self.setUp()
        self.test_Review()

    def test_Review(self):
        """The comparison with a (predefined, SUV) database: voxel z, regional values (approximate)."""
        self.delayDisplay("Static Neuro: review")
        logic = StaticNeuroPETLogic()
        patient = np.full((10, 10, 10), 5.0, dtype=np.float32)
        patient[:, :, :5] = 3.0                         # region A below the normals
        volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "patient")
        slicer.util.updateVolumeFromArray(volume, patient)
        labels = np.zeros((10, 10, 10), dtype=np.int16)
        labels[2:8, 2:8, 1:5] = 1
        labels[2:8, 2:8, 5:9] = 2
        labels[0, 0, 0] = 3
        labelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "labels")
        slicer.util.updateVolumeFromArray(labelNode, labels)
        segmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "regions")
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelNode, segmentation)
        ids = logic.segmentIds(segmentation)
        for segmentId, name in zip(ids, ("A", "B", "Optic Chiasm")):
            segmentation.GetSegmentation().GetSegment(segmentId).SetName(name)
        self.assertEqual(logic.dropSegments(segmentation), ["Optic Chiasm"])
        # without a database: the SUV of the regions
        plain = logic.reviewData(volume, segmentation)
        self.assertAlmostEqual(plain["suv"]["A"], 3.0, places=4)
        self.assertIsNone(plain["z"])
        # a predefined SUV database: mean 5, SD 1
        root = _testFolder()
        header = normaldb.newHeader("review test", kind=normaldb.PREDEFINED, nuclide="F-18", tracer="FDG",
                                    mapType=normaldb.MAP_SUV)
        folder = normaldb.createDatabase(root, header)
        mapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "map")
        for which, value in (("mean", 5.0), ("sd", 1.0)):
            slicer.util.updateVolumeFromArray(mapNode, np.full((10, 10, 10), value, dtype=np.float32))
            self.assertTrue(slicer.util.saveNode(mapNode, os.path.join(folder, normaldb.statsFileName(
                normaldb.MASTER, which, ".nrrd"))))
        header["statistics"] = {"computed": normaldb.now(), "stale": False, "groups": {normaldb.MASTER: {
            "name": normaldb.MASTER_TITLE, "n": 10, "mean": normaldb.statsFileName(normaldb.MASTER, "mean", ".nrrd"),
            "sd": normaldb.statsFileName(normaldb.MASTER, "sd", ".nrrd")}}}
        normaldb.writeHeader(folder, header)
        result = logic.reviewData(volume, segmentation, folder, normaldb.readHeader(folder))
        z = result["z"]
        self.assertAlmostEqual(float(z[5, 5, 2]), -2.0, places=3)       # (3 - 5) / 1
        self.assertAlmostEqual(float(z[5, 5, 7]), 0.0, places=3)
        self.assertTrue(nr.uptakeMask(result["mean"], result["mask"])[5, 5, 2])     # uniform normals: all shown
        self.assertAlmostEqual(result["reference"]["A"][0], 5.0, places=4)
        self.assertTrue(any("approximated" in note for note in result["notes"]))
        rows = nr.tableRows(result["values"], result["suv"], result["reference"], nr.SCORE_Z)
        self.assertAlmostEqual(dict((r[0], r[-1]) for r in rows)["A"], -2.0, places=3)
        # 3D-SSP: the surface of the regions, the voxel scores projected (no normals' projections: predefined)
        data = logic.reviewSsp(result, folder, normaldb.readHeader(folder))
        self.assertGreater(len(data["surface"]["points"]), 10)
        self.assertIsNone(data["mean"])
        self.assertLess(float(np.min(data["voxel"])), -1.5)
        self.assertAlmostEqual(float(np.max(data["patient"])), 5.0, places=2)
        poly = logic.surfacePolyData(data["surface"], data["voxel"])
        self.assertEqual(poly.GetNumberOfPoints(), len(data["surface"]["points"]))
        self.delayDisplay("Review passed")

    def test_ImportAndHarden(self):
        import vtk
        self.delayDisplay("Static Neuro: import and harden")
        folder = _testFolder()
        source = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "source")
        array = np.zeros((20, 24, 22), dtype=np.float32)
        array[6:14, 8:16, 7:15] = 5.0
        slicer.util.updateVolumeFromArray(source, array)
        path = os.path.join(folder, "sub-7_ses-SUV_pet.nii.gz")
        self.assertTrue(slicer.util.saveNode(source, path))
        with open(os.path.join(folder, "sub-7_ses-SUV_pet.json"), "w") as f:
            json.dump({"TracerName": "FDG", "TracerRadionuclide": "F18", "Units": "Bq/mL"}, f)
        with open(os.path.join(folder, "participants.tsv"), "w") as f:
            f.write("participant_id\tage\tsex\nsub-7\t64\tF\n")
        logic = StaticNeuroPETLogic()
        node = logic.loadFromFile(path)
        info = logic.imageInfo(node)
        self.assertEqual((info["modality"], info["unitsKind"], info["tracer"]), ("PET", sn.UNITS_SUV, "FDG"))
        self.assertEqual(logic.ageAndSex(node), (64, "F"))
        logic.setAgeAndSex(node, 65, "M")
        self.assertEqual(logic.ageAndSex(node), (65, "M"))
        self.assertTrue(logic.isStaticImage(node))

        fixed = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "fixed")
        slicer.util.updateVolumeFromArray(fixed, np.zeros((20, 24, 22), dtype=np.float32))
        linear = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "shift")
        matrix = vtk.vtkMatrix4x4()
        matrix.SetElement(0, 3, 2.0)
        linear.SetMatrixTransformToParent(matrix)
        undo, moved = logic.harden(node, fixed, linear, reg.SAVE_SEPARATELY, reg.OUTPUT_FIXED_GRID)[:2]
        self.assertIsNot(moved, node)
        self.assertEqual(moved.GetAttribute(NEURO_PREFIX + "Template"), "fixed")
        self.assertEqual(moved.GetAttribute(sn.STATIC_PREFIX + "Kind"), "static")
        self.assertAlmostEqual(float(slicer.util.arrayFromVolume(moved).sum()), float(array.sum()), delta=1.0)
        logic.revertHarden(undo)
        self.assertFalse(slicer.mrmlScene.IsNodePresent(moved))
        undo, same = logic.hardenRegistration(node, fixed, linear, reg.OVERWRITE, reg.OUTPUT_FIXED_GRID)
        self.assertIs(same, node)
        self.assertEqual(node.GetAttribute(REG_PREFIX + "Fixed"), "fixed")
        logic.revertHarden(undo)
        self.assertIsNone(node.GetAttribute(REG_PREFIX + "Fixed"))
        self.assertTrue(np.array_equal(slicer.util.arrayFromVolume(node), array))
        self.delayDisplay("Import and harden passed")

    def test_Widget(self):
        self.delayDisplay("Static Neuro: widget")
        widget = slicer.modules.staticneuropet.widgetRepresentation().self()
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "static")
        slicer.util.updateVolumeFromArray(node, np.ones((8, 8, 8), dtype=np.float32))
        node.SetAttribute("StaticPET.Kind", "static")
        node.SetAttribute("StaticPET.Units", "SUVbw")
        widget.setNode("Sequence", None)
        widget.selectInputs()
        self.assertIs(widget.volume(), node)
        for key, _, _, _ in STEPS:
            widget.showStep(key)
            self.assertEqual(widget.current, key)
        self.assertEqual(widget.stepStates()[STEP_DATA], STATE_DONE)
        self.assertEqual(widget.stepStates()[STEP_REVIEW], STATE_WAITING)
        self.assertTrue(hasattr(widget, "dbCombo") and hasattr(widget, "addReferenceCombo"))
        self.assertFalse(widget.addSubjectButton.enabled)        # not normalized
        settings = widget.normalizationSettings()
        self.assertIn(settings["method"], [m for m, _ in METHODS])
        self.assertTrue(_sameRegion("Cerebellum Gray Matter" + REFERENCE_SUFFIX, "cerebellum gray matter"))
        # no CT/MR: Next from the data goes to the normalization (and Back to the data)
        widget.setNode("Anatomy", None)
        widget.showStep(STEP_DATA)
        widget.moveStep(1)
        self.assertEqual(widget.current, STEP_NORM)
        widget.moveStep(-1)
        self.assertEqual(widget.current, STEP_DATA)
        # not normalized: the regions and the review cannot be opened yet, the normalization can (no registration)
        self.assertTrue(widget.stepAllowed(STEP_NORM))
        self.assertFalse(widget.stepAllowed(STEP_REG))
        self.assertFalse(widget.stepAllowed(STEP_REGIONS))
        self.assertFalse(widget.stepAllowed(STEP_REVIEW))
        widget.showStep(STEP_NORM)
        widget.refresh()
        self.assertFalse(widget.nextButton.enabled)
        self.assertFalse(widget.stepButtons[STEP_REGIONS].enabled)
        widget.showStep(STEP_DATA)
        import xml.etree.ElementTree as ElementTree
        views = ElementTree.fromstring(widget._regionsLayoutXml()).findall(".//view")
        self.assertEqual(len(views), len(REGIONS_COLUMNS) * len(REGIONS_ORIENTATIONS))
        # the region of a 3D-SSP vertex: the first label along its inward normal
        labels = np.zeros((20, 20, 20), np.int32)
        labels[5:15, 5:15, 5:10], labels[5:15, 5:15, 10:15] = 2, 3
        found = ssp.vertexLabels(labels, np.eye(4), [[4, 10, 10], [15, 10, 10], [0, 0, 0]],
                                 [[-1, 0, 0], [1, 0, 0], [1, 0, 0]])
        self.assertEqual(list(found), [2, 3, 0])
        # the patient's name and ID: the BIDS subject, then the values typed on the Data page
        node.SetAttribute(sn.STATIC_PREFIX + "SubjectID", "7")
        self.assertEqual(StaticNeuroPETLogic.patientNameAndId(node), ("", "sub-7"))
        StaticNeuroPETLogic.setPatientNameAndId(node, " DOE JOHN ", "123")
        self.assertEqual(StaticNeuroPETLogic.patientNameAndId(node), ("DOE JOHN", "123"))
        self.assertEqual(NORM_HARDEN_GRIDS[0][1], reg.OUTPUT_FIXED_GRID)
        self.assertEqual(NORM_HARDEN_ACTIONS[0][1], reg.OVERWRITE)
        self.assertFalse(widget._anatomyIsMr())
        for key, _, _, _ in STEPS:
            self.assertFalse(_stepIcon(key, STATE_READY, 1).isNull())
        # batch: the normalization plan of a database without settings, the QC picture
        plan, settings = StaticNeuroPETLogic().batchPlan(normaldb.newHeader("batch", nuclide="F-18", tracer="FDG"))
        self.assertEqual(plan["method"], METHOD_PET)
        self.assertTrue(settings["steps"])
        picture = os.path.join(_testFolder(), "qc.png")
        anatomy = sn.rasArray(slicer.util.arrayFromVolume(node), _ijkToRas(node))
        StaticNeuroPETLogic.writeQcPicture(picture, node, (anatomy, tuple(_ijkToRas(node).ravel())))
        self.assertTrue(os.path.getsize(picture) > 0)
        widget.showStep(STEP_DATA)
        self.delayDisplay("Widget passed")
