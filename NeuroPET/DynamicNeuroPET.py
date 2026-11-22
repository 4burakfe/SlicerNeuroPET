"""
Dynamic Neuro Receptor PET (3D Slicer module) - workflow app

A step-by-step workflow for dynamic receptor PET of the brain, built on the other PET4D modules:

    1. Data                 the patient's dynamic PET series (DICOM, or a BIDS 4D NIfTI: a file or a subject /
                            session folder, with its CT / MR) and its CT / MR (the contrast, T1 / T2 / PD / CT, read
                            from the DICOM tags, a BIDS name or the .json sidecar); name, ID, age and sex (editable);
                            optional Gaussian smoothing of the series (off by default) before the next steps
    2. Motion correction    the Dynamic PET Motion Correction panel, on the series of step 1 (iterative engine by
                            default, BRAINSFit selectable)
    3. CT/MR registration   the Dynamic PET to CT/MR Registration panel, the series to the CT / MR (iterative, rigid
                            by default, affine selectable, no deformable in this workflow)
    4. Normalization        to a template space (default: MNI ICBM 152 2009c; the iterative engine, one affine stage
                            by default; BRAINSFit and deformable passes selectable): the PET registered to the
                            template, or the CT / MR normalized and the same transform applied to the PET (default
                            when the study has a CT / MR). Checked in the registration layout (PET / fusion /
                            template, checkerboard, manual correction), then hardened (Overwrite by default). A
                            dynamic normal database may be picked here (or None): its normalization settings and
                            spaces, and its PET template (summed PET to PET template).
    5. Regions              a segmentation of the space (CerebrA with pons, CerebrA PET, or one of the user's) in a
                            grouping (its regions, broad regions, Broad (Compatible)), left and right separate; the
                            reference region a region of the same atlas with the sides merged (default: the
                            cerebellar grey matter) or the centrum semiovale of the space (white matter, for tracers
                            binding in all grey matter). The label maps of the groupings are made on first use.
                            With a database: locked to its region settings (Unlock), 3x2 regions views.
    6. Modeling             the Dynamic PET Modeling panel, ready with the normalized series, the regions, SRTM2 and
                            the reference region of step 5; with a database, its kinetic model and settings
    7. Review/Compare              the patient's parametric maps (BPnd, R1, k2 of SRTM2 by default) made with the Modeling
                            step's settings in the template's brain (or the Modeling step's own maps of the same
                            settings, asked first), one map picked; the Review step of the Static Neuro module on it
                            (slices, region table, 3D-SSP, cut, unfolded brain layouts; Slices by default), the
                            regional values the model fitted to the regions' curves (default) or the voxel map means

Steps 2, 3 and 6 show the panels of those modules inside this one; the series each step makes (a motion corrected or
registered series saved separately) becomes the series of the next step. The template spaces and segmentations come
from the atlas library (NeuroPETLib/neuro.py): the extension's Resources/Atlas and the user's ~/NeuroPET/Atlas.
"""


import logging
import json
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
from NeuroWorkflowLib import scenestate  # noqa: E402
from NeuroWorkflowLib.database import (NormalizationDatabaseMixin, RegionsDatabaseMixin,  # noqa: E402
                                       NORMALIZATION_ATTRIBUTE, registerRegionsLayout, seriesInfo)
from NeuroWorkflowLib.compare import ComparePageMixin, COMPARE_MAP_ROLE  # noqa: E402
from NeuroWorkflowLib.dynamicdb import DynamicDatabaseLogicMixin  # noqa: E402
from NeuroPETLib import neuroreview as nr  # noqa: E402
# the review of the Static Neuro module (its mixins) on the parametric maps: the Review/Compare step
from StaticNeuroPETLib.constants import STEP_REVIEW  # noqa: E402
from StaticNeuroPETLib.logic_database import DatabaseLogicMixin  # noqa: E402
from StaticNeuroPETLib.logic_review import ReviewLogicMixin  # noqa: E402
from StaticNeuroPETLib.toolbar import ToolbarMixin  # noqa: E402
from StaticNeuroPETLib.review_page import ReviewPageMixin  # noqa: E402
from StaticNeuroPETLib.review_colors import ReviewColorsMixin  # noqa: E402
from StaticNeuroPETLib.review_table import ReviewTableMixin  # noqa: E402
from StaticNeuroPETLib.review_ssp import ReviewSspMixin  # noqa: E402
from StaticNeuroPETLib.review_cut import ReviewCutMixin  # noqa: E402
from StaticNeuroPETLib.review_peel import ReviewPeelMixin  # noqa: E402
from StaticNeuroPETLib.review_layout import ReviewLayoutMixin, registerReviewLayouts  # noqa: E402
from StaticNeuroPETLib.review_mouse import ReviewMouseMixin  # noqa: E402

# the normalization methods: the summed PET to a PET template of a dynamic database too (its subjects' summed images)
DYNAMIC_METHODS = METHODS + ((METHOD_PET_TEMPLATE, "Register the summed PET to the database's PET template"),)
# step 7: the parametric maps compared (the Static Neuro Review step's key: its review mixins check it)
STEP_COMPARE = STEP_REVIEW
DYNAMIC_STEPS = STEPS + ((STEP_COMPARE, None, "Review/Compare", "The patient's parametric maps (BPnd, R1, ...) and their "
                                                       "regional values"),)


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
            "registration</b> (rigid by default, affine selectable), <b>4. Normalization</b> to a template space (default MNI ICBM 152 2009c), "
            "<b>5. Regions</b> (a segmentation of the space, default CerebrA PET), <b>6. Modeling</b> (SRTM2 with the "
            "cerebellar grey matter as the reference by default) and <b>7. Review/Compare</b> (the parametric maps in the "
            "review layouts with their regional values). Steps 2, 3 and 6 show the panels of the PET4D "
            "modules. The extension brings the MNI 2009c space and the CerebrA segmentations; more spaces and "
            "segmentations go in your atlas folder (~/NeuroPET/Atlas, 'Open atlas folder' in step 4). A dynamic "
            "normal database (Neuro PET Database Manager) can be picked in step 4: its normalization, region and "
            "kinetic settings are then used.")
        parent.acknowledgementText = (
            "Part of Mergen (Neuro PET/SPECT Analysis), built on the PET4D (DICOM Gated / Dynamic PET) extension. "
            "MNI ICBM 152 nonlinear symmetric 2009c template: Fonov "
            "VS et al., NeuroImage 2011;54:313-327 and NeuroImage 2009;47:S102 (Copyright (C) 1993-2004 Louis "
            "Collins, McConnell Brain Imaging Centre, Montreal Neurological Institute, McGill University). CerebrA: "
            "Manera AL et al., Scientific Data 2020;7:237 (CC0). Registration by the iterative multi-resolution "
            "engine (SimpleITK) or BRAINSFit.")
        # the layouts known before a scene saved in one of them is loaded; the workflow opened again on its step
        scenestate.installAtStartup("DynamicNeuroPET", DynamicNeuroPETWidget.registerLayouts,
                                    DynamicNeuroPETWidget.LAYOUT_STEPS)


class DynamicNeuroPETLogic(DynamicDatabaseLogicMixin, ReviewLogicMixin, DatabaseLogicMixin, NeuroWorkflowLogic):
    """The workflow's logic (NeuroWorkflowLib.base.NeuroWorkflowLogic): normalization and regions; usable without the
    widget (see there). The dynamic normal databases of the database folder (NeuroWorkflowLib.database); the
    comparison of a map (the Static Neuro review's logic: StaticNeuroPETLib.logic_review / logic_database); a patient
    added to a dynamic database (NeuroWorkflowLib.dynamicdb)."""

    DATABASE_MODE = normaldb.DYNAMIC
    REFERENCE_COLOR = (0.6, 0.6, 0.6)       # the reference region has no regional value of its own: grey

    @staticmethod
    def seriesInfo(sequence):
        """staticneuro.imageInfo() of a dynamic series (modality, units, nuclide, tracer)."""
        return seriesInfo(sequence)


class DynamicNeuroPETWidget(ComparePageMixin,
                            ToolbarMixin,
                            NormalizationDatabaseMixin,
                            RegionsDatabaseMixin,
                            ReviewPageMixin,
                            ReviewColorsMixin,
                            ReviewTableMixin,
                            ReviewSspMixin,
                            ReviewCutMixin,
                            ReviewPeelMixin,
                            ReviewLayoutMixin,
                            ReviewMouseMixin,
                            NeuroWorkflowWidget):
    """The workflow's widget (NeuroWorkflowLib.base.NeuroWorkflowWidget) with the Dynamic Neuro steps; the dynamic
    normal database picked on the Normalization page ('' for none: never blocking), its normalization settings the
    page's default, the Regions page locked to its region settings, the regions step views
    (NeuroWorkflowLib.database); the Review/Compare step: the Static Neuro review on the parametric maps
    (NeuroWorkflowLib.compare, StaticNeuroPETLib.review_*)."""

    MODULE_NAME = "DynamicNeuroPET"
    STEPS = DYNAMIC_STEPS
    LOGIC_CLASS = DynamicNeuroPETLogic
    DATABASE_MODE = normaldb.DYNAMIC
    NORM_METHODS = DYNAMIC_METHODS
    COMPARED_IN = "the Review/Compare step"
    STEP_NEEDS = dict(NeuroWorkflowWidget.STEP_NEEDS, **{STEP_COMPARE: "Load the regions first (Regions step)."})

    def setup(self):
        self.databaseRows = []          # [(folder, header, rank, problems)] of the dynamic databases
        self._databaseKey = None
        self._databaseChosen = False
        self._modelingPreset = None     # (database folder, series, regions) whose kinetics Modeling was set to
        NeuroWorkflowWidget.setup(self)
        self.pages[STEP_COMPARE] = self._buildComparePage()
        self.refresh()

    # the workflow toolbar of the base (its step icons) with the review's tools (shown on the Review/Compare step)
    def _stepButton(self, index, key, title, explanation):
        return NeuroWorkflowWidget._stepButton(self, index, key, title, explanation)

    def _stepIcon(self, key, state, number):
        return NeuroWorkflowWidget._stepIcon(self, key, state, number)

    def _buildToolbar(self):
        toolbar = NeuroWorkflowWidget._buildToolbar(self)
        if toolbar is not None:
            self._addReviewTools(toolbar)
        return toolbar

    def stepStates(self):
        states = NeuroWorkflowWidget.stepStates(self)
        sequence = self.sequence()
        normalized = sequence is not None and bool(sequence.GetAttribute(NEURO_PREFIX + "Template"))
        states[STEP_COMPARE] = (STATE_READY if normalized and self.node("Segmentation") is not None
                                else STATE_WAITING)
        return states

    def refresh(self):
        NeuroWorkflowWidget.refresh(self)
        if not hasattr(self, "kineticsLabel"):
            return
        self._updateCompareBox()
        self._updateAddDynBox()
        self._updateReviewPage()
        self._updateReviewTools()

    # --- the workflow saved with the scene: the review layouts too ------------------------------------
    LAYOUT_STEPS = {**NeuroWorkflowWidget.LAYOUT_STEPS, **{layoutId: STEP_COMPARE for layoutId in nr.LAYOUT_IDS}}

    @staticmethod
    def registerLayouts():
        registerRegionsLayout()
        registerReviewLayouts()

    def _adoptSceneNodes(self):
        """The comparison's map, score map and region table of the saved scene (Compare writes into them again)."""
        for attribute, role in (("scoreNode", "ReviewScore"), ("reviewTable", "ReviewTable")):
            node = self.node(role)
            if node is not None and slicer.mrmlScene.IsNodePresent(node):
                setattr(self, attribute, node)
        try:
            self._adoptCompareMap()
        except Exception:
            logging.exception("Neuro: could not take back the compared map")

    def _layoutOwned(self, layoutId):
        if layoutId in nr.LAYOUT_IDS:
            return self.current == STEP_COMPARE
        return NeuroWorkflowWidget._layoutOwned(self, layoutId)

    def volume(self):
        """The series' summed image (normalized with it): the image the databases are matched with and the regions
        views show."""
        sequence = self.sequence()
        return self.logic.displayImage(sequence) if sequence is not None else None

    def _databaseImageInfo(self):
        return self.logic.seriesInfo(self.sequence())

    def _callStep(self, key, method):
        if key == STEP_REGIONS:
            if method == "exit":
                self._exitRegionsViews()
            elif method == "enter":
                self._enterRegionsViews()
        if key == STEP_COMPARE:
            if method == "exit":
                self._exitReviewViews()
            elif method == "enter" and self.node("Segmentation") is not None:
                self._enterReviewViews()
                qt.QTimer.singleShot(200, self._autoCompare)
            return
        NeuroWorkflowWidget._callStep(self, key, method)

    def _prepareModeling(self, widget):
        """The workflow's inputs; with a dynamic database picked, once per database and series, its kinetic model
        and settings too (the maps are then made as the database's were)."""
        NeuroWorkflowWidget._prepareModeling(self, widget)
        folder = self._database()
        header = self._header(folder) if folder else None
        settings = normaldb.kinetics(header) if header else None
        # the base sets the default model with new regions: the database's again then
        key = (folder,) + tuple(node.GetID() if node is not None else None
                                for node in (self.sequence(), self.node("Segmentation")))
        if not settings or normaldb.kineticErrors(settings) or key == self._modelingPreset:
            return
        self._modelingPreset = key
        try:
            self._applyKinetics(widget, settings)
            logging.info(f"Neuro: Modeling set to the kinetics of '{header.get('name')}': "
                         f"{normaldb.kineticsText(settings)}")
        except Exception:
            logging.warning("Neuro: could not set the database's kinetic model in Modeling", exc_info=True)

    def _applyKinetics(self, widget, settings):
        """The Modeling step's model, weighting, time zero, t*, reference k2', SUVR window and clipping set to a
        dynamic database's kinetics (normaldb.kinetics); the workflow's reference region kept."""
        def pick(combo, data):
            index = combo.findData(data)
            if index >= 0:
                combo.setCurrentIndex(index)
            return index >= 0

        pick(widget.modelCombo, settings["model"])
        # a new model guesses the reference again: the workflow's (the database's) reference region back
        referenceId = self._parameter().GetParameter(NEURO_PREFIX + "ReferenceSegment")
        if referenceId:
            pick(widget.referenceCombo, referenceId)
        pick(widget.weightingCombo, settings["weighting"])
        index = widget.timeZeroCombo.findData(settings["timeZero"])
        try:
            enabled = widget.timeZeroCombo.model().item(index).isEnabled() if index >= 0 else False
        except Exception:
            enabled = index >= 0
        if enabled:
            widget.timeZeroCombo.setCurrentIndex(index)
        else:
            logging.warning(f"Neuro: the database's time zero '{settings['timeZero']}' is not available for this "
                            "series (no injection time); the start of the first frame is used")
        widget.autoTStarCheck.checked = False
        widget.tStarSpin.value = float(settings["tStarMin"])
        options = normaldb.kineticOptions(settings)
        widget.k2refSpin.value = options["k2ref"] or 0.0
        if options["window"]:
            widget.windowStartSpin.value, widget.windowEndSpin.value = options["window"]
        elif hasattr(widget, "_setDefaultWindow"):
            widget._setDefaultWindow()
        widget.clipCheck.checked = options["clipNegative"]

    def onNormHarden(self):
        """Harden; the settings of the normalization kept on the normalized series and its summed image (compared
        with a database's)."""
        settings = (self.normResult or {}).get("settings")
        done = NeuroWorkflowWidget.onNormHarden(self)
        sequence = self.sequence()
        if done and sequence is not None and settings:
            for node in (sequence, self.volume()):
                if node is not None:
                    node.SetAttribute(NORMALIZATION_ATTRIBUTE, json.dumps(settings))
        if done and ((self.normUndo or {}).get("undo") or {}).get("action") == reg.OVERWRITE:
            # Overwrite cannot be undone: the series in the patient space (a whole copy of it) is not kept
            self.normUndo = None
            self._updateNormButtons()
        self._updateDatabaseInfo()

    @staticmethod
    def _normHardenText(sequence, template, method):
        return (f"Resample the {sequence.GetNumberOfDataNodes()} frames of '{sequence.GetName()}' (and its summed "
                f"image) through the normalization onto the field of view of the template '{template.GetName()}'."
                + ("\nThe CT/MR is normalized too (on the template grid)." if method == METHOD_ANATOMY else "")
                + "\n\nOverwrite (default) replaces the series with the normalized one and cannot be undone: the "
                "series in the patient space is not kept in memory (load it again to start over). Save separately "
                "keeps it next to the normalized one (Revert removes the normalized one); a dynamic series takes a "
                "lot of memory.")

    def cleanup(self):
        if getattr(self, "current", None) == STEP_REGIONS:
            self._exitRegionsViews()
        if getattr(self, "current", None) == STEP_COMPARE:
            self._exitReviewViews()
        NeuroWorkflowWidget.cleanup(self)

    def onSceneClose(self, caller, event):
        NeuroWorkflowWidget.onSceneClose(self, caller, event)
        self._resetReviewState()
        self._resetCompareState()
        self._regionsPreviousLayout = None
        self._regionsHelpers = None
        self._databaseKey = None
        self._modelingPreset = None

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
        # the regions the reference is made of are left out; the reference is in the one labelmap layer
        self.assertEqual(names, sorted(["Putamen R", "Putamen L", "Pons R", "Pons L",
                                        "Cerebellum Gray Matter" + REFERENCE_SUFFIX]))
        mask = slicer.util.arrayFromSegmentBinaryLabelmap(segmentation, referenceId)
        self.assertEqual(int(np.count_nonzero(mask)), 2 * 64)
        self.assertIsNone(logic.segmentIdByName(segmentation, "Cerebellum Gray Matter R"))
        self.assertEqual(segmentation.GetSegmentation().GetNumberOfLayers(), 1)
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
        self.assertIsNone(logic.segmentIdByName(grown, "Cerebellum Gray Matter"))
        self.assertEqual(grown.GetSegmentation().GetNumberOfLayers(), 1)
        with self.assertRaises(ValueError):        # no white matter map
            logic.loadAtlasRegions(atlas, key, growMm=1.0, growInto=normaldb.GROW_ONLY_WM)
        # the extension's segmentations are there, on the extension's space
        self.assertIn(neuro.DEFAULT_SEGMENTATION, [s["id"] for s in neuro.listSegmentations(neuro.DEFAULT_SPACE)])
        self.assertIn("T1", neuro.atlasFiles())
        self.delayDisplay("Regions test passed")

    def test_Widget(self):
        widget = slicer.modules.dynamicneuropet.widgetRepresentation().self()
        self.assertEqual([k for k, *_ in widget.STEPS][-1], STEP_COMPARE)
        self.assertFalse(widget.REVIEW_MEAN_MASK)
        self.assertEqual(widget._defaultReviewLayout(), nr.LAYOUT_SLICES)
        self.assertIsNone(widget.reviewVolume())            # no maps yet
        for key, _, _, _ in DYNAMIC_STEPS:
            widget.showStep(key, enterStep=False)
            self.assertEqual(widget.current, key)
            self.assertEqual(widget.stack.count, 1)
        widget.showStep(STEP_DATA, enterStep=False)
        self.delayDisplay("Widget test passed")
