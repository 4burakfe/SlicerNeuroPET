"""
Dynamic Neuro Receptor PET (3D Slicer module) - workflow app

A step-by-step workflow for dynamic receptor PET of the brain, built on the other PET4D modules:

    1. Data                 the patient's dynamic PET series and its CT / MR (the MR contrast, T1 / T2 / PD, is read
                            from the DICOM tags)
    2. Motion correction    the Dynamic PET Motion Correction panel, on the series of step 1
    3. CT/MR registration   the Dynamic PET to CT/MR Registration panel, the series to the CT / MR
    4. Normalization        to a template space (default: MNI ICBM 152 2009c) (BRAINSFit, rigid + scale + skew then B-spline): the
                            PET registered to the template, or the CT / MR normalized and the same transform applied
                            to the PET (default when the study has a CT / MR). Checked in the registration layout
                            (PET / fusion / template, checkerboard, manual correction), then hardened into a new
                            series in the template space.
    5. Regions              a segmentation of the space (CerebrA with pons, CerebrA PET, or one of the user's) in a
                            grouping (its regions, broad regions, Broad (Compatible)), left and right separate; the
                            reference region a region of the same atlas with the sides merged (default: the
                            cerebellar grey matter). The label maps of the groupings are made on first use.
    6. Modeling             the Dynamic PET Modeling panel, ready with the normalized series, the regions, SRTM and
                            the cerebellar grey matter as the reference

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
from slicer.util import VTKObservationMixin

_MODULE_DIR = os.path.dirname(os.path.abspath(globals().get("__file__", "") or "."))
if globals().get("__file__") and _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from NeuroPETLib import requireBaseExtension  # noqa: E402
requireBaseExtension()     # the PET4D extension's modules (DICOMGatedDynamicPET folder) on sys.path
from DICOMGatedDynamicPETLib import dynamicviews  # noqa: E402
from DICOMGatedDynamicPETLib import registration as reg  # noqa: E402
from DICOMGatedDynamicPETLib import resampling  # noqa: E402
from DICOMGatedDynamicPETLib.hardendialog import hardenDialog  # noqa: E402
from NeuroPETLib import credits, neuro, normaldb  # noqa: E402
from NeuroPETLib.banner import MergenBanner  # noqa: E402

PREFIX = "DynamicPET."
REG_PREFIX = "DynamicPET.Registration."
MC_PREFIX = "DynamicPET.MotionCorrection."
NEURO_PREFIX = "DynamicPET.Neuro."
ATLAS_FILE_ATTRIBUTE = NEURO_PREFIX + "AtlasFile"
ATLAS_STAMP_ATTRIBUTE = NEURO_PREFIX + "AtlasFileStamp"      # neuro.fileStamp of the file when it was loaded
NORMALIZED_SUFFIX = " MNI"
REGIONS_NAME = "CerebrA regions"
REFERENCE_SUFFIX = " (reference)"
# Settings: the space and the region atlas picked last (the atlas library: neuro.listSpaces / listSegmentations)
SETTINGS_SPACE = "NeuroPET/Space"
SETTINGS_REGION_ATLAS = "NeuroPET/RegionAtlas"
GROW_MM, ERODE_REFERENCE_MM = 0.0, 0.0      # defaults: regions as in the atlas
WM_PROBABILITY_LIMIT = 0.5      # the regions grow only into voxels with a lower white matter probability
DEFAULT_TARGET = "Putamen"
DEFAULT_MODEL = "SRTM"

METHOD_PET, METHOD_ANATOMY, METHOD_PET_TEMPLATE = "pet", "anatomy", "petTemplate"   # the last: Static Neuro
METHODS = ((METHOD_PET, "Register the PET to the template"),
           (METHOD_ANATOMY, "Normalize the CT/MR, apply the same transform to the PET"))
PET_TEMPLATE_ATTRIBUTE = NEURO_PREFIX + "PetTemplate"     # on a template that is a PET (synthetic PET, a database's mean)
PET_TEMPLATE_COLORMAP = "Inverted grey"
# the normalization layout with a PET template: the template (synthetic / mean PET) in Blue, the patient's PET in Red,
# so the two PETs are told apart in the fusion
NORM_TEMPLATE_PET_COLOR_ID = "vtkMRMLColorTableNodeBlue"
NORM_PATIENT_PET_COLOR_ID = "vtkMRMLColorTableNodeRed"
NORMALIZATION_STEPS = (reg.SCALE_SKEW, "BSpline")      # rigid + scale + skew, then B-spline
# BRAINSFit registers a copy of the template and of the CT/MR / PET at this voxel size (as FSL's 2 mm MNI template;
# a 0.5 mm MR against the 1 mm template otherwise takes tens of GB); the result applies to the original images
NORMALIZATION_WORKING_MM = 2.0
# BRAINSFit B-spline to the template: 12 subdivisions per axis (control points about 16 mm apart over the template;
# BRAINSFit's own default is 14,10,12) and a stricter stopping factor than BRAINSFit's 2e13 (see registerStep)
NORMALIZATION_BSPLINE_GRID = 12
NORMALIZATION_SAMPLING = 0.005      # BRAINSFit samplingPercentage (its default 0.002; the registration module 0.05)
NORMALIZATION_CONVERGENCE_FACTOR = 1e9
# the other BRAINSFit options of the normalization (reg.BRAINSFIT_OPTIONS keys): BRAINSFit's defaults but the B-spline
# stopping factor (see NORMALIZATION_CONVERGENCE_FACTOR); numberOfThreads -1: all cores
NORMALIZATION_BRAINSFIT = {"numberOfIterations": 1500, "histogramBins": 50,
                           "bsplineConvergenceFactor": NORMALIZATION_CONVERGENCE_FACTOR,
                           "projectedGradientTolerance": 1e-5, "maximumNumberOfEvaluations": 900,
                           "maximumNumberOfCorrections": 25, "maxBSplineDisplacement": 0.0, "numberOfThreads": -1}
# Default engine (user 2026-11-07, tried by hand): the iterative engine, one affine stage on the 1 mm level with
# the "Thorough" sampling; BRAINSFit (the steps above) and the deformable passes can be picked
NORMALIZATION_ENGINE = reg.ENGINE_ITERATIVE
NORMALIZATION_ITERATIVE_LINEAR = "Affine"
NORMALIZATION_ITERATIVE_DEFORMABLE = False
NORMALIZATION_ITERATIVE_SPACING_MM = 1.0
NORMALIZATION_ITERATIVE_QUALITY = "Thorough"
NORMALIZATION_ITERATIVE_GRID_MM = 30.0     # iterative engine: last control-point spacing (brain to template)
# The PET is registered to the skull-stripped template by default (the scalp of the T1 has no PET counterpart);
# the CT/MR is registered to the whole template.
NORMALIZATION_BRAIN_MASK = {METHOD_PET: True, METHOD_ANATOMY: False, METHOD_PET_TEMPLATE: True}
NORMALIZATION_INITIALIZATION = {METHOD_PET: "useMomentsAlign", METHOD_ANATOMY: "useGeometryAlign",
                                METHOD_PET_TEMPLATE: "useMomentsAlign"}
# A CT (CT/MR method): its brain only (skull stripped, brain window) on a grid of about this voxel size, registered to
# the skull-stripped template from the centres of mass; fewer ml inside the skull is taken as no brain found
CT_BRAIN_WORKING_MM = 1.0
CT_SKULL_STRIP = False      # Burak 2026-11-09: the skull-stripped CT fit badly; try the whole CT clipped at CT_CLIP_HU
CT_CLIP_HU = 100.0
CT_BONE_HU = 100.0          # above this (bone) set to CT_BONE_VALUE: dark as the bone in the MR (Burak 2026-11-09)
CT_BONE_VALUE = -1000.0
CT_BRAIN_MIN_ML = 300.0
# Harden: the template's field of view with the PET voxel size (default), the template grid (1 mm), or custom
NORMALIZATION_GRIDS = (("Keep the PET voxel size (over the template field of view)", reg.OUTPUT_PET_SPACING),
                       ("Resample to the template grid (1 mm)", reg.OUTPUT_FIXED_GRID),
                       ("Resample to this voxel size:", reg.OUTPUT_CUSTOM))

STEP_DATA, STEP_MC, STEP_REG, STEP_NORM, STEP_REGIONS, STEP_MODEL = "data", "mc", "reg", "norm", "regions", "model"
# (key, embedded module or None, title, explanation)
STEPS = (
    (STEP_DATA, None, "Data", "Dynamic PET series and the patient's CT / MR"),
    (STEP_MC, "DynamicPETMotionCorrection", "Motion correction", "Frame-by-frame motion correction"),
    (STEP_REG, "DynamicPETRegistration", "CT/MR registration", "The series registered to the CT / MR"),
    (STEP_NORM, None, "Normalization", "To the MNI ICBM 152 2009c template"),
    (STEP_REGIONS, None, "Regions", "CerebrA atlas regions with their names"),
    (STEP_MODEL, "DynamicPETModeling", "Modeling", "Kinetic modelling of the regions (SRTM, Logan, ...)"),
)
TOOLBAR_OBJECT_NAME = "DynamicNeuroPETWorkflowToolbar"
ICON_PATH = os.path.join(_MODULE_DIR, "Resources", "Icons", "DynamicNeuroPET.png")
STATE_DONE, STATE_READY, STATE_WAITING = "done", "ready", "waiting"
STATE_COLORS = {STATE_DONE: "#2e9e5b", STATE_READY: "#2f7ed8", STATE_WAITING: "#8f969c"}


class DynamicNeuroPET(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Mergen - Neuro PET - Dynamic"
        parent.categories = ["Nuclear Medicine"]
        parent.dependencies = ["DICOMGatedDynamicPET", "DynamicPETEditor", "DynamicPETMotionCorrection",
                               "DynamicPETRegistration", "DynamicPETModeling"]     # the PET4D extension
        parent.contributors = ["Burak Demir"]
        parent.helpText = (
            "Workflow for dynamic receptor PET of the brain: <b>1. Data</b> (dynamic series and CT/MR), "
            "<b>2. Motion correction</b>, <b>3. CT/MR registration</b>, <b>4. Normalization</b> to the MNI ICBM 152 "
            "2009c template, <b>5. Regions</b> (CerebrA atlas) and <b>6. Modeling</b> (SRTM with the cerebellar grey "
            "matter as the reference by default). Steps 2, 3 and 6 show the panels of the PET4D modules. The atlases "
            "are not shipped: pick the folder that holds them in step 4.")
        parent.acknowledgementText = (
            "Part of Mergen (Neuro PET/SPECT Analysis), built on the PET4D (DICOM Gated / Dynamic PET) extension. "
            "MNI ICBM 152 nonlinear symmetric 2009c template: Fonov "
            "VS et al., NeuroImage 2011;54:313-327 and NeuroImage 2009;47:S102 (Copyright (C) 1993-2004 Louis "
            "Collins, McConnell Brain Imaging Centre, Montreal Neurological Institute, McGill University). CerebrA: "
            "Manera AL et al., Scientific Data 2020;7:237 (CC0). Registration by BRAINSFit.")


def _vtkMatrix(array):
    return resampling.vtkMatrix(array)


def _ijkToRas(node):
    matrix = vtk.vtkMatrix4x4()
    node.GetIJKToRASMatrix(matrix)
    return resampling.numpyMatrix(matrix)


def _percentiles(node, low=1.0, high=99.5):
    array = slicer.util.arrayFromVolume(node)
    if array is None or array.size == 0:
        return 0.0, 1.0
    sample = array[::2, ::2, ::2]
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return 0.0, 1.0
    lower, upper = (float(v) for v in np.percentile(sample, (low, high)))
    return (lower, upper) if upper > lower else (lower, lower + 1.0)


# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------
def defaultReferenceName(entries):
    """The cerebellar grey matter of an atlas version: the bilateral region, else the first side."""
    names = [n for _, n, _ in entries]
    for name in names:
        if name.lower() == neuro.REFERENCE_STRUCTURE.lower():
            return name
    for name in names:
        if name.lower().startswith(neuro.REFERENCE_STRUCTURE.lower()):
            return name
    return names[0] if names else None


class DynamicNeuroPETLogic(ScriptedLoadableModuleLogic):
    """Normalization and regions of the Neuro workflow; usable without the widget:

        logic = DynamicNeuroPETLogic()
        atlas = neuro.atlasFiles(spaceId)           # the volumes of a template space
        template = logic.loadAtlasVolume(atlas["T1"])
        result = logic.normalize(sequence, template, METHOD_PET)
        ... (a linear transform node with the chain: logic.chainNodes(result)) ...
        segmentation, referenceId = logic.loadAtlasRegions(atlas, "CerebrA:regions")
    """

    # --- atlas -------------------------------------------------------------------------------------
    @staticmethod
    def savedSpace():
        """The space picked last (settings), the default space when it is not there any more."""
        spaceId = str(qt.QSettings().value(SETTINGS_SPACE) or "")
        return spaceId if spaceId and neuro.spaceInfo(spaceId) else neuro.DEFAULT_SPACE

    @staticmethod
    def setSavedSpace(spaceId):
        qt.QSettings().setValue(SETTINGS_SPACE, spaceId or "")

    @staticmethod
    def savedRegionAtlas():
        key = str(qt.QSettings().value(SETTINGS_REGION_ATLAS) or "")
        return key if neuro.isRegionAtlasKey(key) else neuro.DEFAULT_REGION_ATLAS

    @staticmethod
    def setSavedRegionAtlas(key):
        qt.QSettings().setValue(SETTINGS_REGION_ATLAS, key or "")

    @staticmethod
    def openAtlasFolder():
        """Opens the user's atlas folder (~/NeuroPET/Atlas, made with its Space and Segmentation folders) in the
        file browser."""
        folder = neuro.ensureUserAtlasFolder() or neuro.BUNDLED_FOLDER
        qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(folder))
        return folder

    @staticmethod
    def loadAtlasVolume(path, labelmap=False, name=None):
        """The atlas file as a volume (label map with labelmap), reused when it is already in the scene and the
        file is still the one that was loaded (same modification time and size). A copy of a file written again
        since (a database's mean / SD maps or template made again, another database under the same name, a scene
        saved before) is removed and the file read again, so nothing is compared with out of date maps."""
        className = "vtkMRMLLabelMapVolumeNode" if labelmap else "vtkMRMLScalarVolumeNode"
        stamp = neuro.fileStamp(path)
        for node in list(slicer.util.getNodesByClass(className)):
            if node.GetAttribute(ATLAS_FILE_ATTRIBUTE) != path:
                continue
            if stamp and node.GetAttribute(ATLAS_STAMP_ATTRIBUTE) == stamp:
                return node
            logging.info(f"Neuro: '{node.GetName()}' is not the file on disk any more, reading {path} again")
            slicer.mrmlScene.RemoveNode(node)
        properties = {"show": False, "labelmap": bool(labelmap)}
        if name:
            properties["name"] = name
        node = slicer.util.loadVolume(path, properties)
        if node is None:
            raise RuntimeError(f"Could not load {path}")
        node.SetAttribute(ATLAS_FILE_ATTRIBUTE, path)
        node.SetAttribute(ATLAS_STAMP_ATTRIBUTE, stamp)
        return node

    # --- anatomy -----------------------------------------------------------------------------------
    @staticmethod
    def dicomValues(node, tags):
        """{tag: value} from the DICOM file of the volume's first instance ('' when not found)."""
        out = {t: "" for t in tags}
        try:
            uids = (node.GetAttribute("DICOM.instanceUIDs") or "").split()
            if not uids or slicer.dicomDatabase is None:
                return out
            path = slicer.dicomDatabase.fileForInstance(uids[0])
            if not path:
                return out
            for tag in tags:
                out[tag] = slicer.dicomDatabase.fileValue(path, tag) or ""
        except Exception:
            logging.debug("Neuro: could not read the DICOM tags", exc_info=True)
        return out

    @classmethod
    def anatomyContrast(cls, node):
        """(contrast "CT" / "T1" / "T2" / "PD" / None, reason) of the patient's CT / MR volume."""
        if node is None:
            return None, "no CT/MR"
        values = cls.dicomValues(node, ("0008,0060", "0008,103e", "0018,1030", "0018,0024", "0018,0080",
                                        "0018,0081", "0018,0082"))
        modality = values["0008,0060"]
        if not modality:
            array = slicer.util.arrayFromVolume(node)
            if array is not None and array.size and float(np.min(array[::4, ::4, ::4])) < -500.0:
                modality = "CT"
        return neuro.classifyAnatomy(modality, (values["0008,103e"], values["0018,1030"], values["0018,0024"],
                                                node.GetName()),
                                     values["0018,0080"], values["0018,0081"], values["0018,0082"])

    # --- normalization -----------------------------------------------------------------------------
    @staticmethod
    def firstImage(sequence):
        """A volume of the series with its geometry (the first frame)."""
        return sequence.GetNthDataNode(0)

    @staticmethod
    def displayImage(sequence):
        """The image shown for the series: its summed image, else the first frame."""
        from DynamicPETRegistration import DynamicPETRegistrationLogic
        return DynamicPETRegistrationLogic.summedVolumeOf(sequence) or sequence.GetNthDataNode(0)

    @staticmethod
    def revertHarden(undo):
        """Undo a Harden (the record harden() returned)."""
        from DynamicPETRegistration import DynamicPETRegistrationLogic
        DynamicPETRegistrationLogic().revert(undo)

    @staticmethod
    def referenceImage(sequence):
        """The PET image registered to the template: the summed image of the series, else the frame of the highest
        activity."""
        from DynamicPETRegistration import DynamicPETRegistrationLogic
        summed = DynamicPETRegistrationLogic.summedVolumeOf(sequence)
        if summed is not None:
            return summed
        from DynamicPETMotionCorrection import DynamicPETMotionCorrectionLogic
        from DICOMGatedDynamicPETLib import motioncorrection as mc
        activities = DynamicPETMotionCorrectionLogic.frameActivities(sequence)
        return sequence.GetNthDataNode(mc.maxActivityFrame(activities))

    @staticmethod
    def defaultSettings(method=METHOD_PET):
        from DynamicPETRegistration import DynamicPETRegistrationLogic
        settings = DynamicPETRegistrationLogic.defaultSettings()
        settings["steps"] = list(NORMALIZATION_STEPS)
        settings["resample"] = [True] * len(NORMALIZATION_STEPS)
        settings["workingVoxelMm"] = NORMALIZATION_WORKING_MM
        settings["bsplineGrid"] = NORMALIZATION_BSPLINE_GRID
        settings["samplingPercentage"] = NORMALIZATION_SAMPLING
        settings.update(NORMALIZATION_BRAINSFIT)
        settings["initialization"] = NORMALIZATION_INITIALIZATION[method]
        settings["iterativeLinear"] = NORMALIZATION_ITERATIVE_LINEAR
        settings["iterativeDeformable"] = NORMALIZATION_ITERATIVE_DEFORMABLE
        settings["iterativeGridMm"] = NORMALIZATION_ITERATIVE_GRID_MM
        settings["iterativeSpacingMm"] = NORMALIZATION_ITERATIVE_SPACING_MM
        settings["iterativeQuality"] = NORMALIZATION_ITERATIVE_QUALITY
        settings["engine"] = NORMALIZATION_ENGINE
        if NORMALIZATION_ENGINE == reg.ENGINE_ITERATIVE:
            settings["steps"] = reg.iterativeSteps(NORMALIZATION_ITERATIVE_LINEAR, NORMALIZATION_ITERATIVE_DEFORMABLE)
        return settings

    def normalize(self, sequence, template, method=METHOD_PET, anatomy=None, settings=None, mask=None,
                  progress=None, ctTemplate=None, ctCorrection=None):
        """Register the PET reference image (METHOD_PET) or the CT/MR (METHOD_ANATOMY) to the template; the result
        of DynamicPETRegistrationLogic.register() (linear matrix, B-spline node, ...) plus "moving". A CT with a CT
        template (in the MNI space, e.g. scct_unsmooth) is registered to that template instead (CT to CT, both
        stretched by neuro.ctStretch, no brain mask), then moved by ctCorrection (4x4, the CT template's space to the
        ICBM 152 2009c, see ctTemplateCorrection); the result is still on the template's grid."""
        from DynamicPETRegistration import DynamicPETRegistrationLogic
        settings = dict(settings or self.defaultSettings(method))
        ctBrain = None
        if method == METHOD_ANATOMY:
            if anatomy is None:
                raise ValueError("Pick the CT/MR in step 1 to normalize it.")
            moving = anatomy
            settings["movingFunctional"] = False
        else:
            moving = self.referenceImage(sequence)
        if moving.GetParentTransformNode() is not None:
            raise ValueError(f"'{moving.GetName()}' is under a transform. Harden or remove it first.")
        isCt = method == METHOD_ANATOMY and neuro.looksLikeCt(slicer.util.arrayFromVolume(anatomy))
        ctFixed = None
        if isCt and ctTemplate is not None:
            # a CT to the CT template (same modality, the skull fits the skull)
            ctBrain = self.ctStretched(anatomy)
            ctFixed = self.ctStretched(ctTemplate, workingMm=0.0) \
                if neuro.looksLikeCt(slicer.util.arrayFromVolume(ctTemplate)) else None
            mask = None
        elif isCt and not CT_SKULL_STRIP:
            # a CT: the whole head clipped at CT_CLIP_HU, its bone set to CT_BONE_VALUE (dark as in the MR)
            ctBrain = self.ctClipped(anatomy)
        elif isCt:
            # a CT: its brain (skull stripped, brain window) to the skull-stripped template, centres of mass first
            if mask is None:
                raise ValueError("A CT is normalized as its brain only: tick 'Skull-stripped template (brain mask)' "
                                 "(the brain mask must be in the atlas folder).")
            ctBrain = self.ctBrainOnly(anatomy)
            if settings.get("initialization") == "useGeometryAlign":
                settings["initialization"] = "useMomentsAlign"
        if ctTemplate is not None and isCt:
            fixed = ctFixed if ctFixed is not None else ctTemplate
        else:
            fixed = self.brainOnly(template, mask) if mask is not None else template
        try:
            registered = ctBrain if ctBrain is not None else moving
            result = DynamicPETRegistrationLogic().register(fixed, registered, settings, progress)
        finally:
            for helper in (fixed if fixed not in (template, ctTemplate) else None, ctBrain):
                if helper is not None:
                    slicer.mrmlScene.RemoveNode(helper)
        if ctTemplate is not None and isCt and ctCorrection is not None:
            if result.get("bspline") is not None:
                slicer.mrmlScene.RemoveNode(result["bspline"])
                result["bspline"], result["deformation"] = None, None
                result.setdefault("notes", []).append("CT to CT template: linear only (the deformable part dropped).")
            result["linear"] = np.asarray(ctCorrection, dtype=float) @ np.asarray(result["linear"], dtype=float)
        if ctTemplate is not None and isCt:
            logging.info(f"Neuro: the CT '{anatomy.GetName()}' registered to the CT template '{ctTemplate.GetName()}'")
        result["moving"] = moving
        result["masked"] = mask is not None
        result["ctBrain"] = ctBrain is not None
        result["ctTemplate"] = ctTemplate is not None and isCt
        return result

    def ctTemplateCorrection(self, templateGm, icbmGm, cachePath=None, ctTemplate=None, correctedPath=None):
        """4x4 (world, to parent) from the CT template's MNI space to the ICBM 152 2009c: its grey matter map
        registered (affine, iterative engine, centres of mass first) to the 2009c grey matter map. Read from cachePath
        when it is there (and written to it). The CT template moved by it is saved to correctedPath (when it is not
        there yet) to be looked at."""
        import json
        if cachePath and os.path.isfile(cachePath):
            try:
                with open(cachePath) as f:
                    matrix = np.asarray(json.load(f)["matrix"], dtype=float).reshape(4, 4)
                self.saveCorrectedCtTemplate(ctTemplate, matrix, correctedPath)
                return matrix
            except (OSError, ValueError, KeyError):
                logging.warning(f"Neuro: could not read {cachePath}, registering again")
        from DynamicPETRegistration import DynamicPETRegistrationLogic
        settings = self.defaultSettings(METHOD_PET)
        settings.update(engine=reg.ENGINE_ITERATIVE, iterativeLinear="Affine", iterativeDeformable=False,
                        initialization="useMomentsAlign", movingFunctional=False,
                        steps=reg.iterativeSteps("Affine", False))
        matrix = np.asarray(DynamicPETRegistrationLogic().register(icbmGm, templateGm, settings)["linear"], dtype=float)
        if cachePath:
            try:
                with open(cachePath, "w") as f:
                    json.dump({"from": os.path.basename(cachePath), "matrix": matrix.ravel().tolist()}, f)
            except OSError:
                logging.warning(f"Neuro: could not write {cachePath}")
        logging.info(f"Neuro: CT template space to ICBM 152 2009c: {matrix.round(4).tolist()}")
        self.saveCorrectedCtTemplate(ctTemplate, matrix, correctedPath)
        return matrix

    @staticmethod
    def saveCorrectedCtTemplate(ctTemplate, matrix, path):
        """The CT template with its geometry moved by matrix (lossless, no resampling) saved to path, unless it is
        already there."""
        if ctTemplate is None or not path or os.path.exists(path):
            return
        copy = slicer.modules.volumes.logic().CloneVolume(slicer.mrmlScene, ctTemplate, "CT template on 2009c")
        try:
            copy.SetIJKToRASMatrix(_vtkMatrix(np.asarray(matrix, dtype=float) @ _ijkToRas(ctTemplate)))
            if slicer.util.saveNode(copy, path):
                logging.info(f"Neuro: the CT template on the ICBM 152 2009c saved to {path}")
            else:
                logging.warning(f"Neuro: could not save {path}")
        finally:
            slicer.mrmlScene.RemoveNode(copy)

    @staticmethod
    def ctStretched(ct, workingMm=CT_BRAIN_WORKING_MM):
        """A hidden copy of a CT (Hounsfield units) with neuro.ctStretch intensities (soft tissue spread over the
        histogram), averaged down to about workingMm (0: its own grid); same place in space as the CT."""
        from DynamicPETRegistration import DynamicPETRegistrationLogic
        voxels, matrix = slicer.util.arrayFromVolume(ct), _ijkToRas(ct)
        if workingMm > 0:
            voxels, matrix = reg.poolToSpacing(voxels, matrix, workingMm)
        helper = DynamicPETRegistrationLogic.helperVolume(f"{ct.GetName()} stretched")
        helper.SetIJKToRASMatrix(_vtkMatrix(np.asarray(matrix, dtype=float)))
        slicer.util.updateVolumeFromArray(helper, neuro.ctStretch(voxels))
        return helper

    @staticmethod
    def ctClipped(ct, upperHu=None, workingMm=CT_BRAIN_WORKING_MM):
        """A hidden copy of a CT clipped at upperHu (CT_CLIP_HU) with its bone (above CT_BONE_HU) set to
        CT_BONE_VALUE (dark, as in the MR), averaged down to about workingMm; same place in space as the CT."""
        from DynamicPETRegistration import DynamicPETRegistrationLogic
        # the bone replaced on the CT's own grid first, so averaging down leaves no bright rim at its edges
        full = slicer.util.arrayFromVolume(ct)
        full = np.where(full > CT_BONE_HU, np.float32(CT_BONE_VALUE), full.astype(np.float32, copy=False))
        voxels, matrix = reg.poolToSpacing(full, _ijkToRas(ct), workingMm)
        image = np.minimum(np.asarray(voxels, dtype=np.float32), float(CT_CLIP_HU if upperHu is None else upperHu))
        helper = DynamicPETRegistrationLogic.helperVolume(f"{ct.GetName()} clipped")
        helper.SetIJKToRASMatrix(_vtkMatrix(np.asarray(matrix, dtype=float)))
        slicer.util.updateVolumeFromArray(helper, image)
        return helper

    @staticmethod
    def ctBrainOnly(ct, workingMm=CT_BRAIN_WORKING_MM):
        """A hidden copy of a head CT with its brain only (neuro.ctBrainImage: skull stripped, brain window, 0
        outside), averaged down to about workingMm: the MR template shows no skull, so the CT's bone must not pull the
        fit. Same place in space as the CT (its own grid)."""
        from DynamicPETRegistration import DynamicPETRegistrationLogic
        voxels, matrix = reg.poolToSpacing(slicer.util.arrayFromVolume(ct), _ijkToRas(ct), workingMm)
        spacing = np.linalg.norm(np.asarray(matrix, dtype=float)[:3, :3], axis=0)[::-1]    # numpy axes (k, j, i)
        image, count = neuro.ctBrainImage(voxels, spacing)
        if count * float(np.prod(spacing)) < CT_BRAIN_MIN_ML * 1000.0:
            raise ValueError(f"No brain found in the CT '{ct.GetName()}' ({count * float(np.prod(spacing)) / 1000:.0f}"
                             " ml inside the skull). Is it a head CT in Hounsfield units?")
        logging.info(f"Neuro: brain of the CT '{ct.GetName()}' for the normalization: "
                     f"{count * float(np.prod(spacing)) / 1000:.0f} ml")
        helper = DynamicPETRegistrationLogic.helperVolume(f"{ct.GetName()} brain")
        helper.SetIJKToRASMatrix(_vtkMatrix(np.asarray(matrix, dtype=float)))
        slicer.util.updateVolumeFromArray(helper, image)
        return helper

    @staticmethod
    def brainOnly(template, mask):
        """A hidden copy of the template with the voxels outside the brain mask set to 0 (skull stripped), on the
        template's grid: the registration then fits the brain, not the scalp and the neck."""
        from DynamicPETRegistration import DynamicPETRegistrationLogic
        array = slicer.util.arrayFromVolume(template).astype(np.float32)
        maskArray = slicer.util.arrayFromVolume(mask)
        if maskArray.shape != array.shape:
            raise ValueError(f"The brain mask '{mask.GetName()}' is not on the grid of '{template.GetName()}'.")
        helper = DynamicPETRegistrationLogic.helperVolume(f"{template.GetName()} brain")
        helper.SetIJKToRASMatrix(_vtkMatrix(_ijkToRas(template)))
        slicer.util.updateVolumeFromArray(helper, np.where(maskArray > 0.5, array, 0.0).astype(np.float32))
        return helper

    @staticmethod
    def petInAnatomySpace(sequence, anatomy):
        """Whether the series was registered (hardened) to this CT/MR (step 3)."""
        return bool(sequence and anatomy and sequence.GetAttribute(REG_PREFIX + "Fixed") == anatomy.GetName())

    @staticmethod
    def resampleKeepingAir(source, dims, fixedIjkToRas, worldToMoving=None, linear=True, interpolation="Linear"):
        """vtkImageData of the source resampled (resampling.Resampler) onto a grid. The voxels outside the source's
        field of view are 0; a CT is therefore resampled from a copy raised above 0 and those voxels set to air (its
        lowest value) after, so that they do not read as water (0 HU) in a mean CT."""
        resampler = resampling.Resampler(_ijkToRas(source), tuple(int(d) for d in dims), fixedIjkToRas,
                                         worldToMoving, linear, interpolation)
        array = slicer.util.arrayFromVolume(source)
        if array is None or not array.size or not neuro.looksLikeCt(array):
            return resampler.resample(source.GetImageData())
        low = float(np.min(array))
        scene = slicer.mrmlScene
        raised = scene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "CT raised (temporary)")
        out = scene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "CT resampled (temporary)")
        try:
            raised.SetIJKToRASMatrix(_vtkMatrix(_ijkToRas(source)))
            slicer.util.updateVolumeFromArray(raised, array.astype(np.float32) - low + 1.0)
            out.SetIJKToRASMatrix(_vtkMatrix(fixedIjkToRas))
            out.SetAndObserveImageData(resampler.resample(raised.GetImageData()))
            values = slicer.util.arrayFromVolume(out).astype(np.float32)
            slicer.util.updateVolumeFromArray(out, np.where(values < 0.5, low, values + low - 1.0).astype(np.float32))
            image = vtk.vtkImageData()
            image.DeepCopy(out.GetImageData())
            return image
        finally:
            scene.RemoveNode(raised)
            scene.RemoveNode(out)

    def normalizedAnatomy(self, anatomy, template, linearNode, interpolation="Linear"):
        """A new volume: the CT/MR resampled through the normalization chain onto the template grid (a CT's voxels
        outside its field of view as air, see resampleKeepingAir)."""
        from DynamicPETRegistration import DynamicPETRegistrationLogic
        worldToMoving, linear = DynamicPETRegistrationLogic.worldToPet(linearNode)
        dims = template.GetImageData().GetDimensions()
        scene = slicer.mrmlScene
        image = self.resampleKeepingAir(anatomy, dims, _ijkToRas(template), worldToMoving, linear, interpolation)
        node = scene.AddNewNodeByClass("vtkMRMLScalarVolumeNode",
                                       scene.GenerateUniqueName(anatomy.GetName() + NORMALIZED_SUFFIX))
        node.SetAndObserveImageData(image)
        node.SetIJKToRASMatrix(_vtkMatrix(_ijkToRas(template)))
        node.SetAttribute(NEURO_PREFIX + "Template", template.GetName())
        node.SetAttribute(NEURO_PREFIX + "Source", anatomy.GetName())
        node.CreateDefaultDisplayNodes()
        if anatomy.GetDisplayNode() is not None:
            node.GetDisplayNode().CopyContent(anatomy.GetDisplayNode())
        return node

    def harden(self, sequence, template, linearNode, action, grid, spacing=None, interpolation="Linear",
               description="", method=METHOD_PET, anatomy=None):
        """Resample the series (and its summed image) through the normalization chain onto the template field of
        view (DynamicPETRegistrationLogic.harden); with METHOD_ANATOMY the CT/MR is normalized too. Returns (undo,
        sequence, summed, normalized CT/MR or None)."""
        from DynamicPETRegistration import DynamicPETRegistrationLogic
        regLogic = DynamicPETRegistrationLogic()
        summed = regLogic.summedVolumeOf(sequence)
        undo, newSequence, newSummed = regLogic.harden(sequence, summed, template, linearNode, action, grid, spacing,
                                                       interpolation, description)
        for node in (newSequence, newSummed):
            if node is not None:
                node.SetAttribute(NEURO_PREFIX + "Template", template.GetName())
                node.SetAttribute(NEURO_PREFIX + "Method", method)
        if action == reg.SAVE_SEPARATELY:
            name = sequence.GetName() + NORMALIZED_SUFFIX
            newSequence.SetName(slicer.mrmlScene.GenerateUniqueName(name))
            if newSummed is not None and summed is not None:
                newSummed.SetName(slicer.mrmlScene.GenerateUniqueName(summed.GetName() + NORMALIZED_SUFFIX))
        normalized = None
        if method == METHOD_ANATOMY and anatomy is not None:
            try:
                normalized = self.normalizedAnatomy(anatomy, template, linearNode)
            except Exception:
                logging.exception("Neuro: could not normalize the CT/MR")
        return undo, newSequence, newSummed, normalized

    def ensurePseudoPet(self, atlas):
        """The synthetic PET template (neuro.pseudoPetArray of the grey and white matter maps): its path, made on
        first use and saved in the generated atlas folder (neuro.outputFolder; else in Slicer's temporary folder)."""
        path = neuro.pseudoPetPath(atlas)
        if path is None:
            raise ValueError("The grey and white matter maps (GM, WM) are not in the space's folder.")
        if os.path.isfile(path):
            return path
        scene = slicer.mrmlScene
        gm = slicer.util.loadVolume(atlas["GM"], {"show": False})
        wm = slicer.util.loadVolume(atlas["WM"], {"show": False})
        node = None
        try:
            spacing = tuple(float(x) for x in gm.GetSpacing())[::-1]
            array = neuro.pseudoPetArray(slicer.util.arrayFromVolume(gm), slicer.util.arrayFromVolume(wm), spacing)
            node = scene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "MNI synthetic PET")
            node.SetIJKToRASMatrix(_vtkMatrix(_ijkToRas(gm)))
            slicer.util.updateVolumeFromArray(node, array)
            for folder in (os.path.dirname(path), slicer.app.temporaryPath):
                target = os.path.join(folder, os.path.basename(path))
                if slicer.util.saveNode(node, target):
                    logging.info(f"Neuro: synthetic PET template saved as {target}")
                    return target
                logging.warning(f"Neuro: could not save the synthetic PET template in {folder}")
            raise RuntimeError("Could not save the synthetic PET template.")
        finally:
            for item in (gm, wm, node):
                if item is not None and scene.IsNodePresent(item):
                    scene.RemoveNode(item)

    # --- regions -----------------------------------------------------------------------------------
    @staticmethod
    def _loadLabels(path):
        """(label array (int), hidden label map node) of an atlas file; the caller removes the node."""
        node = slicer.util.loadVolume(path, {"show": False, "labelmap": True, "name": "Atlas labels (loading)"})
        if node is None:
            raise RuntimeError(f"Could not load {path}")
        return np.rint(slicer.util.arrayFromVolume(node)).astype(np.int32), node

    @staticmethod
    def _saveLabelMap(array, ijkToRas, path, csvPath=None, csvText=None):
        """Writes a label map (and its label table) at path, else in Slicer's temporary folder: (label map path,
        table path or None)."""
        scene = slicer.mrmlScene
        node = scene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "atlas labels (saving)")
        try:
            node.SetIJKToRASMatrix(_vtkMatrix(ijkToRas))
            slicer.util.updateVolumeFromArray(node, np.asarray(array).astype(np.int16))
            for folder in (os.path.dirname(path), slicer.app.temporaryPath):
                target = os.path.join(folder, os.path.basename(path))
                table = os.path.join(folder, os.path.basename(csvPath)) if csvPath else None
                try:
                    os.makedirs(folder, exist_ok=True)
                    if not slicer.util.saveNode(node, target):
                        raise OSError(f"could not write {target}")
                    if table:
                        with open(table, "w", newline="", encoding="utf-8") as f:
                            f.write(csvText or "")
                    logging.info(f"Neuro: region label map saved as {target}")
                    return target, table
                except OSError:
                    logging.warning(f"Neuro: could not save the region label map in {folder}", exc_info=True)
            raise RuntimeError(f"Could not save the region label map {os.path.basename(path)}.")
        finally:
            scene.RemoveNode(node)

    def segmentationLabelMap(self, info):
        """The label volume of a segmentation (neuro.segmentationInfo): its labels file when that is a label volume,
        else the label map made from labels.seg.nrrd (every segment by its name, numbered by labels.csv) on first use
        and whenever the segmentation or its table changed."""
        source, table = info["labels"], info["csv"]
        target = neuro.segmentationLabelMapPath(info)
        if target == source:
            return source
        try:
            if os.path.getmtime(target) >= max(os.path.getmtime(source), os.path.getmtime(table)):
                return target
        except OSError:
            pass
        structures = neuro.readCerebraLabels(table)
        layers, ijkToRas, segments = neuro.readSegmentationFile(source)
        labels, unknown = neuro.segmentationLabels(layers, segments, structures)
        if unknown:
            logging.warning(f"Neuro: segments of {source} not in its labels.csv: " + ", ".join(unknown))
        if not labels.any():
            raise ValueError(f"No region of labels.csv in {source}")
        return self._saveLabelMap(labels, ijkToRas, target)[0]

    @staticmethod
    def regionEntries(key):
        """[(label value, region name, (r, g, b))] of a region atlas key (neuro.regionAtlasKey: a segmentation and a
        grouping), read from the tables (no label map is made)."""
        return neuro.regionTable(key)[0]

    def regionAtlasPaths(self, key):
        """(label map path, entries) of a region atlas key: the segmentation's label map for its regions, else a
        version made from it (sides merged, broad regions of its groups tables) on first use and whenever the label
        map or the table changed (saved in neuro.generatedFolder(), else in Slicer's temporary folder)."""
        entries, mapping, groups, info = neuro.regionTable(key)
        basePath = self.segmentationLabelMap(info)
        labelsPath, csvPath = neuro.regionMapPaths(key, info)
        if labelsPath == basePath or labelsPath is None:
            return basePath, entries
        text = neuro.groupedCsvText(entries, groups or [])
        try:
            with open(csvPath, encoding="utf-8") as f:
                same = f.read() == text
            if same and os.path.getmtime(labelsPath) >= os.path.getmtime(basePath):
                return labelsPath, entries
        except OSError:
            pass
        array, source = self._loadLabels(basePath)
        try:
            ijkToRas = _ijkToRas(source)
        finally:
            slicer.mrmlScene.RemoveNode(source)
        path, _ = self._saveLabelMap(neuro.remapLabels(array, mapping), ijkToRas, labelsPath, csvPath, text)
        return path, entries

    def loadRegions(self, labelsPath, entries, referencePath=None, referenceName=None, referenceEntries=None,
                    name=REGIONS_NAME, overlayPath=None, referenceOverlayPath=None, growMm=0.0, erodeReferenceMm=0.0,
                    allowedMask=None, target=None, referenceMaskPath=None, referenceGroups=()):
        """A segmentation of the atlas labels (labelsPath, entries [(value, name, color)]) with their names. The
        reference region referenceName comes from referencePath (another version of the atlas on the same grid, e.g.
        the bilateral one, with its referenceEntries) as a segment of its own, or, when referencePath is labelsPath
        (or None), is the segment of that name. overlayPath / referenceOverlayPath: label maps of structures that
        overlap the others (the pons), added as segments of their own and taken out of the regions under them
        (Brainstem (except pons)). growMm: the regions grown into the unlabelled
        voxels (each to the nearest region, so they never overlap; only where allowedMask, when given on the same
        grid; ventricles fixed; the overlays not grown). erodeReferenceMm: the reference region eroded (then always a
        segment of its own, made from the atlas before growing). target: a segmentation node to fill (its segments are
        replaced) instead of a new one. referenceMaskPath: the reference region from a mask file on the atlas grid
        (a normal database's reference mask, used as it is) instead of referenceName / referencePath.
        referenceGroups: the reference-only regions of the atlas [(name, members)] (neuro.referenceGroups; 'Cerebellum
        Whole' of the broad regions), made from all their members. Returns
        (segmentation node, reference segment ID or None). The regions are made in a new segmentation and only on
        success copied into target: a failure (an empty reference region, another grid) leaves target as it was and
        no new segmentation behind."""
        scene = slicer.mrmlScene
        array, source = self._loadLabels(labelsPath)
        colorNode = labelNode = referenceNode = overlayNode = segmentation = None
        done = False
        try:
            values = {v for v, _, _ in entries}
            remapped = neuro.remapLabels(array, {v: v for v in values}).astype(np.int16)
            overlay = overlayMask = None
            if overlayPath:
                overlay, overlayNode = self._loadLabels(overlayPath)
                scene.RemoveNode(overlayNode)
                overlayNode = None
                if overlay.shape != array.shape:
                    raise ValueError("The pons overlay is not on the grid of the region atlas.")
                # the overlapping structures that are regions here (the pons) are taken out of the regions under
                # them (the brainstem), so that no voxel is in two regions
                overlayMask = np.isin(overlay, list(values))
                if overlayMask.any():
                    remapped = np.where(overlayMask, 0, remapped).astype(np.int16)
            original = remapped
            spacing = tuple(float(x) for x in source.GetSpacing())[::-1]       # of the numpy axes (k, j, i)
            if growMm > 0:
                allowed = allowedMask if allowedMask is not None and np.shape(allowedMask) == array.shape else None
                if allowedMask is not None and allowed is None:
                    logging.warning("Neuro: the brain mask is not on the atlas grid; regions grown without it")
                if overlayMask is not None and overlayMask.any():      # nothing grows into the pons
                    allowed = ~overlayMask if allowed is None else (np.asarray(allowed, dtype=bool) & ~overlayMask)
                remapped = neuro.growLabels(remapped, growMm, spacing, allowed,
                                            neuro.fixedValuesOf(entries)).astype(np.int16)
            colorNode = scene.AddNewNodeByClass("vtkMRMLColorTableNode", scene.GenerateUniqueName("CerebrA colors"))
            colorNode.SetTypeToUser()
            top = max(values) + 1
            colorNode.SetNumberOfColors(top)
            colorNode.SetColor(0, "Background", 0.0, 0.0, 0.0, 0.0)
            for value in range(1, top):
                colorNode.SetColor(value, "", 0.0, 0.0, 0.0, 0.0)
            for value, label, (r, g, b) in entries:
                colorNode.SetColor(value, label, r, g, b, 1.0)
            labelNode = scene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "CerebrA labels")
            labelNode.SetIJKToRASMatrix(_vtkMatrix(_ijkToRas(source)))
            slicer.util.updateVolumeFromArray(labelNode, remapped)
            labelNode.CreateDefaultDisplayNodes()
            labelNode.GetDisplayNode().SetAndObserveColorNodeID(colorNode.GetID())
            segmentation = scene.AddNewNodeByClass("vtkMRMLSegmentationNode", scene.GenerateUniqueName(name))
            segmentation.CreateDefaultDisplayNodes()
            segmentation.SetReferenceImageGeometryParameterFromVolumeNode(labelNode)
            if not slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelNode, segmentation):
                raise RuntimeError("Could not make the CerebrA segmentation")
            self._nameSegments(segmentation, entries)
            if overlay is not None:
                present = set(np.unique(overlay).tolist()) - {0}
                for value, label, color in entries:
                    if value in present and self.segmentIdByName(segmentation, label) is None:
                        self._addMaskSegment(segmentation, overlay == value, labelNode, label, color)
            referenceId = None
            if referenceMaskPath:
                referenceArray, referenceNode = self._loadLabels(referenceMaskPath)
                if referenceArray.shape != array.shape:
                    raise ValueError("The reference mask is not on the grid of the region atlas (MNI 2009c 1 mm).")
                if not (referenceArray > 0).any():
                    raise ValueError("The reference mask is empty.")
                referenceId = self._addMaskSegment(segmentation, referenceArray > 0, labelNode,
                                                   (referenceName or "Reference") + REFERENCE_SUFFIX,
                                                   (0.85, 0.85, 0.2))
            elif referenceName:
                sameAtlas = referencePath is None or os.path.normcase(os.path.abspath(referencePath)) == \
                    os.path.normcase(os.path.abspath(labelsPath))
                referenceValue = self.entryValue(entries if sameAtlas else (referenceEntries or ()), referenceName)
                # a reference-only region (no segment of its own): all the label values of its members
                unionValues = [] if referenceValue is not None else neuro.referenceRegionValues(
                    entries if sameAtlas else (referenceEntries or ()), referenceName, referenceGroups)
                if sameAtlas and erodeReferenceMm <= 0 and growMm <= 0 and not unionValues:
                    referenceId = self.segmentIdByName(segmentation, referenceName)
                elif sameAtlas:
                    # the reference of the atlas as drawn (not grown), eroded: a segment of its own
                    if referenceValue is None and not unionValues:
                        raise ValueError(f"No region '{referenceName}' in the atlas.")
                    referenceMask = np.isin(original, unionValues or [referenceValue])
                    if not referenceMask.any() and overlayPath:
                        overlay, overlayNode = self._loadLabels(overlayPath)
                        referenceMask = np.isin(overlay, unionValues or [referenceValue])
                    referenceId = self._addMaskSegment(
                        segmentation, neuro.erodeMask(referenceMask, erodeReferenceMm, spacing), labelNode,
                        referenceName + REFERENCE_SUFFIX, (0.85, 0.85, 0.2))
                else:
                    referenceArray, referenceNode = self._loadLabels(referencePath)
                    if referenceArray.shape != array.shape:
                        raise ValueError("The reference atlas is not on the grid of the region atlas.")
                    if referenceValue is None and not unionValues:
                        raise ValueError(f"No region '{referenceName}' in the reference atlas.")
                    referenceMask = np.isin(referenceArray, unionValues or [referenceValue])
                    if not referenceMask.any() and referenceOverlayPath:      # an overlapping structure (pons)
                        overlay, overlayNode = self._loadLabels(referenceOverlayPath)
                        if overlay.shape == array.shape:
                            referenceMask = np.isin(overlay, unionValues or [referenceValue])
                    referenceMask = neuro.erodeMask(referenceMask, erodeReferenceMm, spacing)
                    if not referenceMask.any():
                        raise ValueError(f"The reference region '{referenceName}' is empty (eroded too much?).")
                    referenceId = self._addMaskSegment(segmentation, referenceMask, labelNode,
                                                       referenceName + REFERENCE_SUFFIX, (0.85, 0.85, 0.2))
            if target is not None and scene.IsNodePresent(target):
                referenceId = self._replaceSegments(target, segmentation, labelNode, referenceId)
                scene.RemoveNode(segmentation)
                segmentation = target
            segmentation.SetAttribute(NEURO_PREFIX + "Atlas", os.path.basename(labelsPath))
            display = segmentation.GetDisplayNode()
            if display is not None:
                display.SetOpacity2DFill(0.3)
                display.SetVisibility3D(False)
            done = True
        finally:
            for node in (source, labelNode, colorNode, referenceNode, overlayNode):
                if node is not None and scene.IsNodePresent(node):
                    scene.RemoveNode(node)
            if not done and segmentation is not None and segmentation is not target and \
                    scene.IsNodePresent(segmentation):
                scene.RemoveNode(segmentation)
        logging.info(f"Neuro: '{segmentation.GetName()}' made from {os.path.basename(labelsPath)} "
                     f"({len(entries)} regions" + (f", reference '{referenceName}'" if referenceId else "") + ")")
        return segmentation, referenceId

    @staticmethod
    def _replaceSegments(target, source, referenceVolume, referenceId=None):
        """The segments of target replaced by copies of those of source (same IDs, names and colors); returns the
        ID the segment referenceId of source has in target."""
        referenceName = None
        if referenceId and source.GetSegmentation().GetSegment(referenceId) is not None:
            referenceName = source.GetSegmentation().GetSegment(referenceId).GetName()
        wasModifying = target.StartModify()
        try:
            segmentation = target.GetSegmentation()
            segmentation.RemoveAllSegments()
            target.CreateDefaultDisplayNodes()
            target.SetReferenceImageGeometryParameterFromVolumeNode(referenceVolume)
            fromSegmentation = source.GetSegmentation()
            for index in range(fromSegmentation.GetNumberOfSegments()):
                segmentId = fromSegmentation.GetNthSegmentID(index)
                if not segmentation.CopySegmentFromSegmentation(fromSegmentation, segmentId):
                    raise RuntimeError(f"Could not copy the segment {segmentId}")
        finally:
            target.EndModify(wasModifying)
        if referenceId and target.GetSegmentation().GetSegment(referenceId) is None and referenceName:
            referenceId = target.GetSegmentation().GetSegmentIdBySegmentName(referenceName) or None
        return referenceId

    @staticmethod
    def entryValue(entries, name):
        for value, entryName, _ in entries:
            if entryName.strip().lower() == name.strip().lower():
                return value
        return None

    def growthMask(self, atlas, growInto=normaldb.GROW_NO_WM):
        """Voxels the regions may grow into, by growInto (normaldb.GROW_*): in the MNI brain mask and outside the
        white matter (probability below WM_PROBABILITY_LIMIT; GROW_NO_WM; without the white matter map the brain
        mask), anywhere in the brain mask (GROW_WITH_WM), or only in the white matter (GROW_ONLY_WM; needs the white
        matter map, ValueError otherwise). None (no limit) without a brain mask, unless GROW_ONLY_WM."""
        onlyWm = growInto == normaldb.GROW_ONLY_WM
        if onlyWm and "WM" not in atlas:
            raise ValueError("Growing into the white matter only needs the white matter probability map (WM) in "
                             "the space's folder.")
        allowed = None
        if neuro.BRAIN_MASK in atlas:
            mask, node = self._loadLabels(atlas[neuro.BRAIN_MASK])
            slicer.mrmlScene.RemoveNode(node)
            allowed = mask > 0
        elif not onlyWm:
            return None
        if "WM" in atlas and growInto != normaldb.GROW_WITH_WM:
            wmNode = slicer.util.loadVolume(atlas["WM"], {"show": False})
            try:
                wm = slicer.util.arrayFromVolume(wmNode).astype(np.float32)
            finally:
                slicer.mrmlScene.RemoveNode(wmNode)
            if wm.max() > 1.5:      # stored 0-100 or 0-255
                wm = wm / wm.max()
            inWm = wm >= WM_PROBABILITY_LIMIT
            if allowed is None:
                allowed = inWm
            elif wm.shape == allowed.shape:
                allowed &= inWm if onlyWm else ~inWm
            elif onlyWm:
                raise ValueError("The white matter map is not on the grid of the brain mask.")
        return allowed

    def loadAtlasRegions(self, atlas, targetKind=None, referenceKind=None, referenceName=None, growMm=GROW_MM,
                         erodeReferenceMm=ERODE_REFERENCE_MM, growInto=normaldb.GROW_NO_WM, target=None,
                         referenceMaskPath=None):
        """Regions of a region atlas (targetKind: a key of neuro.regionAtlasKey, a segmentation and a grouping;
        default the default segmentation's regions), the reference region (referenceName, default the cerebellar grey
        matter) from referenceKind (default: the same atlas with the sides merged). atlas: the volumes of the space
        (neuro.atlasFiles; its brain mask and white matter map limit the growth). The label maps of the groupings
        are made on first use. growInto (normaldb.GROW_*): the grown regions stay out of the white matter, may take it
        too, or only take white matter voxels (growthMask). target: a segmentation to fill instead of a new one.
        Returns (segmentation, reference segment ID)."""
        targetKind = targetKind or neuro.DEFAULT_REGION_ATLAS
        referenceKind = referenceKind or neuro.mergedRegionAtlas(targetKind)
        labelsPath, entries = self.regionAtlasPaths(targetKind)
        referencePath, referenceEntries = self.regionAtlasPaths(referenceKind)
        if referenceName is None:
            referenceName = defaultReferenceName(referenceEntries)
        allowed = self.growthMask(atlas, growInto) if growMm > 0 else None
        return self.loadRegions(labelsPath, entries, referencePath, referenceName, referenceEntries, growMm=growMm,
                                erodeReferenceMm=erodeReferenceMm, allowedMask=allowed, target=target,
                                referenceMaskPath=referenceMaskPath,
                                referenceGroups=neuro.referenceGroups(referenceKind))

    @staticmethod
    def _addMaskSegment(segmentation, mask, geometryNode, name, color):
        """A segment of its own (own labelmap layer, so it may overlap the others) from a boolean array on the grid
        of geometryNode. Returns its ID."""
        scene = slicer.mrmlScene
        maskNode = scene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "CerebrA mask (temporary)")
        try:
            maskNode.SetIJKToRASMatrix(_vtkMatrix(_ijkToRas(geometryNode)))
            slicer.util.updateVolumeFromArray(maskNode, mask.astype(np.uint8))
            image = slicer.vtkSlicerSegmentationsModuleLogic.CreateOrientedImageDataFromVolumeNode(maskNode)
        finally:
            scene.RemoveNode(maskNode)
        segment = slicer.vtkSegment()
        segment.SetName(name)
        segment.SetColor(*color)
        segment.AddRepresentation(slicer.vtkSegmentationConverter.GetBinaryLabelmapRepresentationName(), image)
        seg = segmentation.GetSegmentation()
        before = set(DynamicNeuroPETLogic.segmentIds(segmentation))
        seg.AddSegment(segment)
        added = [i for i in DynamicNeuroPETLogic.segmentIds(segmentation) if i not in before]
        return added[0] if added else DynamicNeuroPETLogic.segmentIdByName(segmentation, name)

    @staticmethod
    def _nameSegments(segmentation, entries):
        """Names and colors from the table (older Slicer versions name imported segments 'Label_<n>')."""
        byValue = {v: (n, c) for v, n, c in entries}
        seg = segmentation.GetSegmentation()
        for segmentId in DynamicNeuroPETLogic.segmentIds(segmentation):
            segment = seg.GetSegment(segmentId)
            value = segment.GetLabelValue() if hasattr(segment, "GetLabelValue") else None
            if value not in byValue:
                digits = "".join(ch for ch in segmentId.split("_")[-1] if ch.isdigit())
                value = int(digits) if digits else None
            if value in byValue and segment.GetName() != byValue[value][0]:
                name, color = byValue[value]
                if segment.GetName().startswith("Label") or not segment.GetName():
                    segment.SetName(name)
                    segment.SetColor(*color)

    @staticmethod
    def segmentIds(segmentation):
        ids = vtk.vtkStringArray()
        segmentation.GetSegmentation().GetSegmentIDs(ids)
        return [ids.GetValue(i) for i in range(ids.GetNumberOfValues())]

    @classmethod
    def segmentIdByName(cls, segmentation, name, startswith=False):
        name = name.strip().lower()
        for segmentId in cls.segmentIds(segmentation):
            segmentName = segmentation.GetSegmentation().GetSegment(segmentId).GetName().strip().lower()
            if segmentName == name or (startswith and segmentName.startswith(name)):
                return segmentId
        return None


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------
def _badge(state, number, size=22):
    return qt.QIcon(_badgePixmap(state, number, size))


def _badgePixmap(state, number, size=22):
    """The state badge of a step (a ring with its number, a filled disc with a tick when done), drawn at twice the
    size (sharp on high DPI screens)."""
    scale = 2
    pixmap = qt.QPixmap(size * scale, size * scale)
    pixmap.fill(qt.QColor(0, 0, 0, 0))
    painter = qt.QPainter(pixmap)
    try:
        painter.setRenderHint(qt.QPainter.Antialiasing)
        color = qt.QColor(STATE_COLORS[state])
        margin = 2 * scale
        full = size * scale
        if state == STATE_DONE:
            painter.setPen(qt.Qt.NoPen)
            painter.setBrush(qt.QBrush(color))
        else:
            pen = qt.QPen(color)
            pen.setWidth(2 * scale)
            painter.setPen(pen)
            painter.setBrush(qt.QBrush(qt.QColor(0, 0, 0, 0)))
        painter.drawEllipse(margin, margin, full - 2 * margin, full - 2 * margin)
        font = qt.QFont()
        font.setBold(True)
        font.setPixelSize(int(full * 0.55))
        painter.setFont(font)
        painter.setPen(qt.QColor("white") if state == STATE_DONE else color)
        painter.drawText(qt.QRect(0, 0, full, full), qt.Qt.AlignCenter, "✓" if state == STATE_DONE
                         else str(number))
    finally:
        painter.end()
    return pixmap


def _button(text, role="action"):
    return dynamicviews.colorButton(qt.QPushButton(text), role)


def _growIntoCombo():
    """Where grown regions may go (normaldb.GROW_INTO); the Neuro workflows and the Database Manager."""
    combo = qt.QComboBox()
    for key, title in normaldb.GROW_INTO:
        combo.addItem(title, key)
    combo.toolTip = ("Where the grown regions may go (all inside the MNI brain mask, each voxel to the nearest "
                     "region): CSF and the cortex edge only (white matter probability < 0.5), white matter too, or "
                     "white matter only (they do not grow outward into CSF).")
    return combo


class DynamicNeuroPETWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Step bar on top, the page of the current step below: the module's own pages (Data, Normalization, Regions)
    and the panels of the PET4D modules (Motion Correction, Registration, Modeling). The Static Neuro PET/SPECT
    module derives from it (its own steps, logic, toolbar and pages)."""

    MODULE_NAME = "DynamicNeuroPET"
    STEPS = STEPS
    LOGIC_CLASS = None              # None: DynamicNeuroPETLogic
    TOOLBAR_NAME = TOOLBAR_OBJECT_NAME
    TOOLBAR_TITLE = "Neuro Receptor PET"
    TOOLBAR_TOOLTIP = "Dynamic Neuro Receptor PET: back to the first step"
    ICON = ICON_PATH
    STEP_AFTER_REGIONS = STEP_MODEL
    BANNER_STEPS = (STEP_DATA,)     # the Mergen banner at the top of the panel on these steps
    NORM_METHODS = METHODS
    LOAD_REGIONS_TEXT = "Load regions and go to modeling"
    NORM_HARDEN_GRIDS = NORMALIZATION_GRIDS         # the first one the default of the Harden dialog
    NORM_HARDEN_ACTIONS = (("Save separately", reg.SAVE_SEPARATELY), ("Overwrite", reg.OVERWRITE))

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = (self.LOGIC_CLASS or DynamicNeuroPETLogic)()
        self.current = STEP_DATA
        self.pages = {}            # step key -> page widget (own pages)
        self.embedded = {}         # step key -> widget representation of the embedded module
        self.stepButtons = {}
        self.normSession = None    # RegistrationSession of the normalization check
        self.normResult = None     # pending normalization (until Harden)
        self.normUndo = None
        self.atlas = {}
        self._updating = False
        self._entered = False

        # the Mergen banner at the top of the panel (shown on BANNER_STEPS)
        self.banner = MergenBanner()
        self.layout.addWidget(self.banner)

        # --- step bar: a toolbar at the top of the main window (shown while this module is open), in the panel
        # when there is no main window
        self.toolbar = self._buildToolbar()
        if self.toolbar is None:
            bar = qt.QFrame()
            bar.setFrameShape(qt.QFrame.StyledPanel)
            barLayout = qt.QGridLayout(bar)
            barLayout.setContentsMargins(4, 4, 4, 4)
            barLayout.setSpacing(4)
            for index, (key, _, title, explanation) in enumerate(self.STEPS):
                button = self._stepButton(index, key, title, explanation)
                button.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)
                barLayout.addWidget(button, index // 3, index % 3)
            self.layout.addWidget(bar)
        self.titleLabel = qt.QLabel()
        self.titleLabel.wordWrap = True
        self.layout.addWidget(self.titleLabel)

        # the page, then Back / Next right under its last element (not at the bottom of the panel), then the
        # free space
        self.stack = qt.QStackedWidget()
        self.layout.addWidget(self.stack)

        nav = qt.QHBoxLayout()
        self.backButton = qt.QPushButton("‹ Back")
        self.nextButton = _button("Next ›")
        nav.addWidget(self.backButton)
        nav.addStretch(1)
        nav.addWidget(self.nextButton)
        self.layout.addLayout(nav)
        self.layout.addStretch(1)
        self.backButton.connect("clicked()", lambda: self.moveStep(-1))
        self.nextButton.connect("clicked()", lambda: self.moveStep(1))

        self.pages[STEP_DATA] = self._buildDataPage()
        self.pages[STEP_NORM] = self._buildNormPage()
        self.pages[STEP_REGIONS] = self._buildRegionsPage()

        self.refreshTimer = qt.QTimer()
        self.refreshTimer.setSingleShot(True)
        self.refreshTimer.setInterval(300)
        self.refreshTimer.connect("timeout()", self.refresh)
        for event in (slicer.mrmlScene.NodeAddedEvent, slicer.mrmlScene.NodeRemovedEvent,
                      slicer.mrmlScene.EndImportEvent):
            self.addObserver(slicer.mrmlScene, event, lambda caller, event: self._refreshLater())
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneClosed)
        self.refreshAtlasLibrary(save=False)
        self.selectInputs()
        self.showStep(STEP_DATA, enterStep=False)

    # --- workflow toolbar ------------------------------------------------------------------------------
    def _stepButton(self, index, key, title, explanation):
        button = qt.QToolButton()
        button.setText(title)
        button.setToolButtonStyle(qt.Qt.ToolButtonTextBesideIcon)
        button.setAutoRaise(True)
        button.setCheckable(True)
        button.setFocusPolicy(qt.Qt.NoFocus)
        button.toolTip = f"{index + 1}. {title}: {explanation}"
        button.setProperty("stepTip", button.toolTip)
        button.connect("clicked(bool)", lambda checked, k=key: self.showStep(k))
        self.stepButtons[key] = button
        return button

    def _buildToolbar(self):
        """The workflow toolbar: the module's name and the steps with their state, like the Tarvos toolbar. One per
        widget; a toolbar left by a developer reload is hidden and renamed, never deleted (Python may still wrap
        it)."""
        try:
            if slicer.app.commandOptions().noMainWindow:
                return None
            mainWindow = slicer.util.mainWindow()
        except Exception:
            mainWindow = None
        if mainWindow is None:
            return None
        for old in [w for w in mainWindow.findChildren(qt.QToolBar) if w.objectName == self.TOOLBAR_NAME]:
            old.setVisible(False)
            old.objectName = self.TOOLBAR_NAME + "Retired"
            try:
                old.toggleViewAction().setVisible(False)
            except Exception:
                pass
        toolbar = qt.QToolBar(self.TOOLBAR_TITLE + " workflow", mainWindow)
        toolbar.objectName = self.TOOLBAR_NAME
        mainWindow.addToolBarBreak()
        mainWindow.addToolBar(toolbar)
        toolbar.setMovable(True)
        toolbar.setVisible(False)
        title = qt.QToolButton()
        title.setText(self.TOOLBAR_TITLE)
        title.setToolButtonStyle(qt.Qt.ToolButtonTextBesideIcon)
        title.setAutoRaise(True)
        title.setFocusPolicy(qt.Qt.NoFocus)
        title.setStyleSheet("QToolButton { font-weight: bold; }")
        if os.path.exists(self.ICON):
            title.setIcon(qt.QIcon(self.ICON))
        title.toolTip = self.TOOLBAR_TOOLTIP
        title.connect("clicked(bool)", lambda checked: self.showStep(STEP_DATA))
        toolbar.addWidget(title)
        toolbar.addSeparator()
        for index, (key, _, stepTitle, explanation) in enumerate(self.STEPS):
            if index > 0:
                arrow = qt.QLabel("\u203a")
                arrow.setStyleSheet("color: #8f969c; font-weight: bold; padding: 0 2px;")
                toolbar.addWidget(arrow)
            toolbar.addWidget(self._stepButton(index, key, stepTitle, explanation))
        return toolbar

    def _refreshLater(self):
        """A scene change: refresh soon while the module is shown (entering it refreshes anyway)."""
        if getattr(self, "_entered", False):
            self.refreshTimer.start()

    def _showToolbar(self, show):
        if getattr(self, "toolbar", None) is not None:
            try:
                self.toolbar.setVisible(bool(show))
            except Exception:
                logging.debug("Neuro: the toolbar is gone", exc_info=True)

    # --- parameter node: the series and images shared by the steps ---------------------------------
    def _parameter(self):
        return self.logic.getParameterNode()

    def node(self, role):
        return self._parameter().GetNodeReference(NEURO_PREFIX + role)

    def setNode(self, role, node):
        self._parameter().SetNodeReferenceID(NEURO_PREFIX + role, node.GetID() if node is not None else None)

    def sequence(self):
        return self.node("Sequence")

    def anatomy(self):
        return self.node("Anatomy")

    # --- page 1: data --------------------------------------------------------------------------------
    @staticmethod
    def _nodeSelector(nodeTypes, toolTip):
        selector = slicer.qMRMLNodeComboBox()
        selector.nodeTypes = nodeTypes
        selector.noneEnabled = True
        selector.addEnabled = False
        selector.removeEnabled = False
        selector.renameEnabled = False
        selector.setMRMLScene(slicer.mrmlScene)
        selector.setCurrentNode(None)
        selector.toolTip = toolTip
        return selector

    def _buildDataPage(self):
        page = qt.QWidget()
        layout = qt.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        box = qt.QGroupBox("Patient data")
        form = qt.QFormLayout(box)
        self.sequenceSelector = self._nodeSelector(["vtkMRMLSequenceNode"],
                                                   "Dynamic PET series of the patient (as loaded from DICOM: "
                                                   "'<series> dynamic Bq/ml')")
        form.addRow("Dynamic PET:", self.sequenceSelector)
        self.sequenceInfo = qt.QLabel()
        self.sequenceInfo.wordWrap = True
        form.addRow(self.sequenceInfo)
        self.anatomySelector = self._nodeSelector(["vtkMRMLScalarVolumeNode"],
                                                  "CT or MR of the same patient (registration in step 3, and the "
                                                  "normalization through the CT/MR in step 4)")
        form.addRow("CT / MR:", self.anatomySelector)
        self.anatomyInfo = qt.QLabel()
        self.anatomyInfo.wordWrap = True
        form.addRow(self.anatomyInfo)
        self.contrastCombo = qt.QComboBox()
        for key, title in (("", "Automatic"), ("T1", "T1"), ("T2", "T2"), ("PD", "PD"), ("CT", "CT")):
            self.contrastCombo.addItem(title, key)
        self.contrastCombo.toolTip = ("Contrast of the CT/MR, read from the DICOM tags (series description, "
                                      "protocol, TR / TE). It picks the template of step 4 when the CT/MR is "
                                      "normalized.")
        form.addRow("Contrast:", self.contrastCombo)
        layout.addWidget(box)
        dicomButton = qt.QPushButton("Open the DICOM browser")
        dicomButton.toolTip = "Load the dynamic PET ('dynamic Bq/ml') and the CT/MR from DICOM"
        dicomButton.connect("clicked()", lambda: slicer.util.selectModule("DICOM"))
        layout.addWidget(dicomButton)
        note = qt.QLabel("This workflow is for dynamic receptor imaging of the brain: the series is motion corrected, "
                         "registered to the CT/MR, normalized to the MNI ICBM 152 2009c template, and the CerebrA "
                         "regions are modelled (SRTM with the cerebellar grey matter as the reference by default). "
                         "Steps can be skipped.")
        note.wordWrap = True
        note.setStyleSheet("color: #7d868d")
        layout.addWidget(note)
        layout.addStretch(1)
        self.sequenceSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onDataChanged)
        self.anatomySelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onDataChanged)
        self.contrastCombo.connect("currentIndexChanged(int)", lambda i: self.onDataChanged())
        return page

    def selectInputs(self):
        if self.sequence() is None:
            from DynamicPETEditor import DynamicPETEditorLogic
            nodes = [n for n in slicer.util.getNodesByClass("vtkMRMLSequenceNode")
                     if (n.GetAttribute(PREFIX + "Kind") == "dynamic" or n.GetAttribute(PREFIX + "FrameDurationS"))
                     and DynamicPETEditorLogic.isDynamicSequence(n)]
            if nodes:
                original = [n for n in nodes if not n.GetAttribute(MC_PREFIX + "Method")
                            and not n.GetAttribute(REG_PREFIX + "Fixed")]
                self.setNode("Sequence", (original or nodes)[-1])
        if self.anatomy() is None:
            from DynamicPETRegistration import fixedCandidates
            candidates = [n for n in fixedCandidates() if not n.GetAttribute(ATLAS_FILE_ATTRIBUTE)
                          and not n.GetAttribute(NEURO_PREFIX + "Template")]
            if candidates:
                self.setNode("Anatomy", candidates[0])
        self._syncDataPage()

    def _syncDataPage(self):
        self._updating = True
        try:
            self.sequenceSelector.setCurrentNode(self.sequence())
            self.anatomySelector.setCurrentNode(self.anatomy())
        finally:
            self._updating = False
        self._updateDataInfo()

    def onDataChanged(self, *args):
        if self._updating:
            return
        sequence, anatomy = self.sequenceSelector.currentNode(), self.anatomySelector.currentNode()
        if sequence is not self.sequence():
            self.setNode("Sequence", sequence)
            self.setNode("OriginalSequence", sequence)
            self._clearNormalization()
        if anatomy is not self.anatomy():
            self.setNode("Anatomy", anatomy)
            self._clearNormalization()
            self._methodChosen = False          # a new CT/MR: the method follows the data again
        self._updateDataInfo()
        self._updateTemplateChoice()
        self.refresh()

    def contrast(self):
        """(contrast, reason) of the CT/MR: the user's choice, else read from the DICOM tags."""
        chosen = self.contrastCombo.currentData
        if chosen:
            return chosen, "chosen"
        return self.logic.anatomyContrast(self.anatomy())

    def _updateDataInfo(self):
        sequence = self.sequence()
        if sequence is None:
            self.sequenceInfo.text = "Pick the dynamic PET series."
        else:
            from DynamicPETRegistration import DynamicPETRegistrationLogic
            summed = DynamicPETRegistrationLogic.summedVolumeOf(sequence)
            done = []
            if sequence.GetAttribute(MC_PREFIX + "Method"):
                done.append("motion corrected")
            if sequence.GetAttribute(REG_PREFIX + "Fixed"):
                done.append(f"registered to '{sequence.GetAttribute(REG_PREFIX + 'Fixed')}'")
            if sequence.GetAttribute(NEURO_PREFIX + "Template"):
                done.append("normalized")
            self.sequenceInfo.text = (f"{sequence.GetNumberOfDataNodes()} frames"
                                      + (f", summed image '{summed.GetName()}'" if summed else ", no summed image")
                                      + (f"; {', '.join(done)}" if done else "") + ".")
        anatomy = self.anatomy()
        if anatomy is None:
            self.anatomyInfo.text = "No CT/MR: steps 3 and the CT/MR normalization are skipped."
        else:
            contrast, reason = self.logic.anatomyContrast(anatomy)
            self.anatomyInfo.text = (f"Recognised as <b>{contrast}</b> ({reason})." if contrast else
                                     f"Contrast not recognised ({reason}); the T1 template is used unless you pick "
                                     "one.")

    # --- page 4: normalization ---------------------------------------------------------------------
    def _buildNormPage(self):
        page = qt.QWidget()
        layout = qt.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        box = qt.QGroupBox("Normalization")
        form = qt.QFormLayout(box)
        # the settings (method, template, registration) folded away: set by default (or by the database)
        self.normSettingsBox = ctk.ctkCollapsibleButton()
        self.normSettingsBox.text = "Normalization settings"
        self.normSettingsBox.collapsed = True
        settingsForm = qt.QFormLayout(self.normSettingsBox)
        self.methodCombo = qt.QComboBox()
        for key, title in self.NORM_METHODS:
            self.methodCombo.addItem(title, key)
        self.methodCombo.toolTip = (
            "PET: the summed image (or the frame of the highest activity) is registered to the template. CT/MR: the "
            "CT/MR is registered to the template and the same transform moves the PET, which must already lie on "
            "the CT/MR (step 3).")
        settingsForm.addRow("Method:", self.methodCombo)
        # the template space (atlas library: Space/<id> of the extension and of ~/NeuroPET/Atlas)
        spaceRow = qt.QHBoxLayout()
        self.spaceCombo = qt.QComboBox()
        self.spaceCombo.toolTip = ("Template space normalized to: the spaces of the extension and of your atlas folder "
                                   "(~/NeuroPET/Atlas/Space/<name>, with space.json and T1 / T2 / PD / GM / WM / CSF / "
                                   "mask / CT volumes). Picked by itself when there is one.")
        spaceRow.addWidget(self.spaceCombo, 1)
        self.openAtlasButton = qt.QPushButton("Open atlas folder")
        self.openAtlasButton.toolTip = ("Open your atlas folder (~/NeuroPET/Atlas) in the file browser: add a space "
                                        "(Space/<name>) or a segmentation (Segmentation/<name>) there, then Refresh.")
        spaceRow.addWidget(self.openAtlasButton)
        self.refreshAtlasButton = qt.QPushButton("Refresh")
        self.refreshAtlasButton.toolTip = "Read the spaces and segmentations of the atlas folders again"
        spaceRow.addWidget(self.refreshAtlasButton)
        settingsForm.addRow("Space:", spaceRow)
        self.atlasInfo = qt.QLabel()
        self.atlasInfo.wordWrap = True
        settingsForm.addRow(self.atlasInfo)
        self.templateCombo = qt.QComboBox()      # filled with the space's volumes that fit the method
        self.templateCombo.toolTip = ("Volume of the space registered to. PET: an MR template (T1 by default) or the "
                                      "grey matter / synthetic PET (GM + 0.3 WM, smoothed 6 mm; made on first use). "
                                      "CT/MR: the template of the MR contrast (T1 / T2 / PD), a CT the space's CT "
                                      "template. A volume picked by hand is kept.")
        settingsForm.addRow("Volume:", self.templateCombo)
        self.templateNote = qt.QLabel()
        self.templateNote.wordWrap = True
        settingsForm.addRow(self.templateNote)
        self._addTemplateRows(settingsForm)
        self.maskCheck = qt.QCheckBox("Skull-stripped template (brain mask)")
        self.maskCheck.checked = NORMALIZATION_BRAIN_MASK[METHOD_PET]
        self.maskCheck.toolTip = ("Register to the template with the voxels outside its brain mask "
                                  "(mask.nrrd of the space) set to 0, so the scalp and the neck do not "
                                  "pull the fit. On by default when the PET or a CT is registered (a CT as its brain "
                                  "only), off for an MR.")
        settingsForm.addRow(self.maskCheck)
        from DynamicPETRegistration import iterativeOptions
        self.normEngineCombo = qt.QComboBox()
        for title, key in reg.ENGINES:
            self.normEngineCombo.addItem(title, key)
        self.normEngineCombo.setCurrentIndex(self.normEngineCombo.findData(NORMALIZATION_ENGINE))
        self.normEngineCombo.toolTip = ("BRAINSFit: the steps below. Iterative multi-resolution (as in EasyReg): a "
                                        "rigid (or affine) stage, then B-spline passes coarse to fine on a working "
                                        "grid of a few mm (little memory).")
        settingsForm.addRow("Registration:", self.normEngineCombo)
        self.normBrainsBox = qt.QWidget()
        brainsForm = qt.QFormLayout(self.normBrainsBox)
        brainsForm.setContentsMargins(0, 0, 0, 0)
        self.normIterativeBox = qt.QWidget()
        iterativeForm = qt.QFormLayout(self.normIterativeBox)
        iterativeForm.setContentsMargins(0, 0, 0, 0)
        self.normIterative = iterativeOptions(iterativeForm, gridMm=NORMALIZATION_ITERATIVE_GRID_MM,
                                              linear=NORMALIZATION_ITERATIVE_LINEAR,
                                              deformable=NORMALIZATION_ITERATIVE_DEFORMABLE,
                                              spacingMm=NORMALIZATION_ITERATIVE_SPACING_MM,
                                              quality=NORMALIZATION_ITERATIVE_QUALITY)
        self.normStepCombos = []
        for index, default in enumerate(NORMALIZATION_STEPS):
            combo = qt.QComboBox()
            if index > 0:
                combo.addItem("(none)", "")
            for name in reg.STEP_TYPES:
                combo.addItem(reg.stepTitle(name), name)
            combo.setCurrentIndex(combo.findData(default))
            combo.toolTip = ("Rigid + scale + skew then B-spline by default (a B-spline only as the last step)")
            self.normStepCombos.append(combo)
            brainsForm.addRow(f"Step {index + 1}:", combo)
        advanced = ctk.ctkCollapsibleButton()
        advanced.text = "Advanced (BRAINSFit)"
        advanced.collapsed = True
        advancedForm = qt.QFormLayout(advanced)
        self.normWorkingSpin = qt.QDoubleSpinBox()
        self.normWorkingSpin.setRange(0.0, 8.0)
        self.normWorkingSpin.setDecimals(1)
        self.normWorkingSpin.setSingleStep(0.5)
        self.normWorkingSpin.suffix = " mm"
        self.normWorkingSpin.specialValueText = "Full resolution"
        self.normWorkingSpin.toolTip = (
            "BRAINSFit registers copies of the template and of the CT/MR (or PET) smoothed and resampled to this voxel "
            "size (never finer than the image itself); the transform applies to the originals. 2 mm (as the 2 mm "
            "MNI template of FSL) keeps the memory low: a 0.5 mm MR against the 1 mm template otherwise takes tens "
            "of GB. 0: the template and the images as they are.")
        advancedForm.addRow("Register at:", self.normWorkingSpin)
        self.normGridEdit = qt.QLineEdit()
        self.normGridEdit.toolTip = (
            "B-spline control grid: subdivisions along each axis of the template, one number or three (i, j, k), "
            "e.g. 12 (control points about 16 mm apart) or 14,10,12 (BRAINSFit's default). More subdivisions deform "
            "more locally, take longer and more memory.")
        advancedForm.addRow("B-spline grid:", self.normGridEdit)
        self.normSamplingSpin = qt.QDoubleSpinBox()
        self.normSamplingSpin.setRange(0.0001, 1.0)
        self.normSamplingSpin.setDecimals(4)
        self.normSamplingSpin.setSingleStep(0.001)
        self.normSamplingSpin.toolTip = ("Fraction of the template voxels sampled for the mutual information "
                                         "(samplingPercentage; BRAINSFit's default 0.002). Lower is faster and "
                                         "lighter.")
        advancedForm.addRow("Sampling:", self.normSamplingSpin)

        def intSpin(low, high, tip, special=None):
            spin = qt.QSpinBox()
            spin.setRange(low, high)
            spin.toolTip = tip
            if special:
                spin.specialValueText = special
            return spin

        self.normIterationsSpin = intSpin(1, 100000, "numberOfIterations: most optimizer iterations per step "
                                          "(BRAINSFit's default 1500)")
        advancedForm.addRow("Iterations:", self.normIterationsSpin)
        self.normBinsSpin = intSpin(8, 256, "numberOfHistogramBins of the mutual information (default 50); fewer "
                                    "bins need less memory per thread for the B-spline")
        advancedForm.addRow("Histogram bins:", self.normBinsSpin)
        self.normFactorEdit = qt.QLineEdit()
        self.normFactorEdit.toolTip = ("costFunctionConvergenceFactor of the B-spline (L-BFGS-B): when it stops. "
                                       "1e+12 low accuracy, 1e+7 moderate, 1e+1 very high (BRAINSFit's doc); "
                                       "BRAINSFit's default 2e+13 stops early; 1e+9 by default here.")
        advancedForm.addRow("Convergence factor:", self.normFactorEdit)
        self.normToleranceEdit = qt.QLineEdit()
        self.normToleranceEdit.toolTip = ("projectedGradientTolerance of the B-spline (L-BFGS-B), default 1e-5 "
                                          "(BRAINSFit's doc: 1e-4 seems to work well)")
        advancedForm.addRow("Gradient tolerance:", self.normToleranceEdit)
        self.normEvaluationsSpin = intSpin(1, 100000, "maximumNumberOfEvaluations of the B-spline line search "
                                           "(default 900)")
        advancedForm.addRow("Max evaluations:", self.normEvaluationsSpin)
        self.normCorrectionsSpin = intSpin(1, 1000, "maximumNumberOfCorrections of L-BFGS-B (default 25)")
        advancedForm.addRow("Max corrections:", self.normCorrectionsSpin)
        self.normDisplacementSpin = qt.QDoubleSpinBox()
        self.normDisplacementSpin.setRange(0.0, 200.0)
        self.normDisplacementSpin.setDecimals(1)
        self.normDisplacementSpin.suffix = " mm"
        self.normDisplacementSpin.specialValueText = "Unbounded"
        self.normDisplacementSpin.toolTip = "maxBSplineDisplacement: largest control point shift (0: unbounded)"
        advancedForm.addRow("Max B-spline shift:", self.normDisplacementSpin)
        self.normThreadsSpin = intSpin(-1, 256, "numberOfThreads of BRAINSFit (Auto: all cores). Memory of the "
                                       "B-spline metric grows with the threads, so fewer threads use less memory.",
                                       special="Auto")
        advancedForm.addRow("Threads:", self.normThreadsSpin)
        resetButton = qt.QPushButton("Reset to defaults")
        resetButton.connect("clicked()", self._resetNormAdvanced)
        advancedForm.addRow(resetButton)
        brainsForm.addRow(advanced)
        self._resetNormAdvanced()
        settingsForm.addRow(self.normBrainsBox)
        settingsForm.addRow(self.normIterativeBox)
        self.normInitCombo = qt.QComboBox()
        for title, mode in reg.INITIALIZATIONS:
            self.normInitCombo.addItem(title, mode)
        self.normInitCombo.toolTip = ("How the first step starts. Centers of mass (default for the PET) puts the "
                                      "brain activity on the template's head; geometric centers (default for the "
                                      "CT/MR) aligns the volume centres.")
        settingsForm.addRow("Initialization:", self.normInitCombo)
        form.addRow(self.normSettingsBox)
        # what the template is, the references to cite and its license (read only)
        self.templateCredits = credits.infoBox()
        form.addRow(self.templateCredits)
        self.normalizeButton = _button("Normalize")
        self.normalizeButton.toolTip = "Register to the template and show the result in the registration layout"
        form.addRow(self.normalizeButton)
        row = qt.QHBoxLayout()
        self.normManualButton = qt.QPushButton("Correct manually")
        self.normManualButton.checkable = True
        self.normManualButton.toolTip = "Move / rotate handles on the PET, on top of the normalization"
        self.normRevertButton = _button("Revert", "revert")
        self.normRevertButton.toolTip = ("Before Harden: drop the normalization. After Harden: remove the normalized "
                                         "series (or put an overwritten one back).")
        self.normHardenButton = _button("Harden", "harden")
        self.normHardenButton.toolTip = "Resample every frame (and the summed image) into the template space"
        row.addWidget(self.normManualButton)
        row.addWidget(self.normRevertButton)
        row.addWidget(self.normHardenButton)
        form.addRow(row)
        self.normLabel = qt.QLabel("Not normalized yet.")
        self.normLabel.wordWrap = True
        self.normLabel.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        form.addRow(self.normLabel)
        layout.addWidget(box)

        viewBox = ctk.ctkCollapsibleButton()
        viewBox.text = "View"
        viewBox.collapsed = True
        view = qt.QFormLayout(viewBox)
        self.normOpacity = ctk.ctkSliderWidget()
        self.normOpacity.minimum, self.normOpacity.maximum = 0.0, 1.0
        self.normOpacity.singleStep, self.normOpacity.decimals = 0.05, 2
        self.normOpacity.value = 0.5
        view.addRow("Fusion opacity:", self.normOpacity)
        self.normCheckerCheck = qt.QCheckBox("Checkerboard column")
        view.addRow(self.normCheckerCheck)
        layout.addWidget(viewBox)
        layout.addStretch(1)

        self.spaceCombo.connect("activated(int)", lambda i: self.onSpaceChanged(self.spaceCombo.currentData))
        self.openAtlasButton.connect("clicked()", self.logic.openAtlasFolder)
        self.refreshAtlasButton.connect("clicked()", self.refreshAtlasLibrary)
        self.methodCombo.connect("currentIndexChanged(int)", self.onMethodChanged)
        self.methodCombo.connect("activated(int)", lambda i: setattr(self, "_methodChosen", True))
        self.templateCombo.connect("currentIndexChanged(int)", lambda i: self._clearNormalization())
        self.templateCombo.connect("activated(int)", lambda i: self._onTemplatePicked())
        self.templateCombo.connect("currentIndexChanged(int)", lambda i: self._updateTemplateCredits())
        self.methodCombo.connect("currentIndexChanged(int)", lambda i: self._updateTemplateCredits())
        self.normalizeButton.connect("clicked()", self.onNormalize)
        self.normEngineCombo.connect("currentIndexChanged(int)", self.onNormEngineChanged)
        self.normManualButton.connect("toggled(bool)", lambda on: self.normSession and
                                      self.normSession.setManualHandles(on))
        self.normRevertButton.connect("clicked()", self.onNormRevert)
        self.normHardenButton.connect("clicked()", self.onNormHarden)
        self.normOpacity.connect("valueChanged(double)", lambda v: self.normSession and
                                 self.normSession.setFusionOpacity(v))
        self.normCheckerCheck.connect("toggled(bool)", self._onNormCheckerToggled)
        self.onMethodChanged()
        self.onNormEngineChanged()
        return page

    def _addTemplateRows(self, form):
        """More rows under the template choice (the Static Neuro module's normal database template)."""

    def spaceId(self):
        """The space picked (a database's own, 'From DB', is its space id)."""
        data = (self.spaceCombo.currentData if hasattr(self, "spaceCombo") else None) or neuro.DEFAULT_SPACE
        return data[len(normaldb.DB_TEMPLATE_PREFIX):] if normaldb.isDatabaseTemplate(data) else data

    def _isDatabaseSpace(self):
        """Whether the space picked is a database's own ('From DB': its space is not in the atlas folders)."""
        return hasattr(self, "spaceCombo") and normaldb.isDatabaseTemplate(self.spaceCombo.currentData)

    def _databaseTemplates(self):
        """({template key: path}, database name) of the normal database's own volumes ('From DB'); none here (the
        Static Neuro module's database)."""
        return {}, ""

    def _extraSpaces(self):
        """[(title, data)] of spaces offered besides the atlas folders' ('From DB (name)'); none here."""
        return []

    def _templateTitle(self, key):
        if normaldb.isDatabaseTemplate(key):
            return normaldb.templateTitle(key, self._databaseTemplates()[1])
        return neuro.templateTitle(key)

    def _templateAvailable(self, key):
        if normaldb.isDatabaseTemplate(key):
            return key in self._databaseTemplates()[0]
        return neuro.canMakeTemplate(self.atlas, key)

    def spaceName(self):
        return neuro.spaceTitle(self.spaceId())

    def refreshAtlasLibrary(self, save=True):
        """Reads the spaces and segmentations of the atlas folders again (the space picked kept)."""
        current = self.spaceCombo.currentData if self.spaceCombo.count else self.logic.savedSpace()
        self.spaceCombo.blockSignals(True)
        self.spaceCombo.clear()
        for space in neuro.listSpaces():
            self.spaceCombo.addItem(space["name"], space["id"])
        for title, data in self._extraSpaces():
            self.spaceCombo.addItem(title, data)
        index = self.spaceCombo.findData(current)
        self.spaceCombo.setCurrentIndex(index if index >= 0 else 0)
        self.spaceCombo.blockSignals(False)
        self.onSpaceChanged(self.spaceCombo.currentData, save=save)

    def setSpace(self, spaceId):
        """Picks a space (a database's: 'From DB' when it is not in the atlas folders); False when it is not
        offered."""
        index = self.spaceCombo.findData(spaceId or neuro.DEFAULT_SPACE)
        if index < 0:
            index = self.spaceCombo.findData(normaldb.DB_TEMPLATE_PREFIX + (spaceId or ""))
        if index < 0:
            return False
        if index != self.spaceCombo.currentIndex:
            self.spaceCombo.setCurrentIndex(index)
            self.onSpaceChanged(self.spaceCombo.currentData)
        return True

    def onSpaceChanged(self, spaceId, save=True):
        spaceId = spaceId or ""
        fromDatabase = normaldb.isDatabaseTemplate(spaceId)
        if save and spaceId and not fromDatabase:
            self.logic.setSavedSpace(spaceId)
        info = neuro.spaceInfo(spaceId) if spaceId and not fromDatabase else None
        self.atlas = neuro.atlasFiles(info) if info else {}
        self.spaceCombo.enabled = self.spaceCombo.count > 1
        if fromDatabase:
            volumes, name = self._databaseTemplates()
            self.atlasInfo.text = (f"The space of '{name}' ({spaceId[len(normaldb.DB_TEMPLATE_PREFIX):]}) is not in "
                                   "the atlas folders: its own volumes are offered ("
                                   + (", ".join(normaldb.templateTitle(k)[len("From DB: "):] for k in volumes)
                                      or "none yet") + "); no segmentation.")
        elif not info:
            self.atlasInfo.text = ("<span style='color:#c0392b'>No template space found. The extension's atlas "
                                   f"folder is {neuro.BUNDLED_FOLDER}; your own spaces go in "
                                   f"{neuro.userAtlasFolder()}/Space.</span>")
        else:
            volumes = [k for k, _ in neuro.SPACE_VOLUMES if k in info["files"]]
            names = {neuro.BRAIN_MASK: "mask", neuro.CT_TEMPLATE: "CT", neuro.CT_TEMPLATE_GM: "CT_GM"}
            text = f"{len(volumes)} volume(s): " + ", ".join(names.get(k, k) for k in volumes) + "."
            segmentations = neuro.listSegmentations(info["id"])
            text += (f" Segmentations: {', '.join(s['name'] for s in segmentations)}." if segmentations else
                     " No segmentation of this space.")
            if any(k not in self.atlas for k in neuro.REQUIRED):
                text = "<span style='color:#c0392b'>" + text + " No T1: the normalization cannot run.</span>"
            self.atlasInfo.text = text
        self.maskCheck.enabled = neuro.BRAIN_MASK in self.atlas
        self._updateTemplateChoice()
        if hasattr(self, "segmentationCombo"):
            self._fillSegmentations()
        self.refreshTimer.start()

    def _fillTemplateCombo(self):
        """The space's volumes that fit the method (neuro.normalizationTemplates), then the normal database's that fit
        ('From DB: <name> - CT mean', normaldb.databaseTemplateKeys); the current one kept."""
        method = self.method()
        contrast = self.contrast()[0] if method == METHOD_ANATOMY else None
        keys = [] if method == METHOD_PET_TEMPLATE else neuro.normalizationTemplates(
            self.atlas, METHOD_ANATOMY if method == METHOD_ANATOMY else METHOD_PET, contrast)
        volumes, name = self._databaseTemplates()
        keys = keys + normaldb.databaseTemplateKeys(volumes, method, contrast)
        titles = [self._templateTitle(k) for k in keys]
        if [(self.templateCombo.itemData(i), self.templateCombo.itemText(i))
                for i in range(self.templateCombo.count)] == list(zip(keys, titles)):
            return
        current = self.templateCombo.currentData
        self.templateCombo.blockSignals(True)
        self.templateCombo.clear()
        for key, title in zip(keys, titles):
            self.templateCombo.addItem(title, key)
        index = self.templateCombo.findData(current)
        self.templateCombo.setCurrentIndex(index if index >= 0 else 0)
        self.templateCombo.blockSignals(False)

    def method(self):
        return self.methodCombo.currentData or METHOD_PET

    def _anatomyIsCt(self):
        return self.anatomy() is not None and self.contrast()[0] == "CT"

    def _normalizationDefaults(self, method):
        """(initialization, brain mask) of a method; a CT (CT/MR method) is registered as its brain to the
        skull-stripped template from the centres of mass."""
        if method == METHOD_ANATOMY and self._anatomyIsCt() and CT_SKULL_STRIP:
            return "useMomentsAlign", True
        return NORMALIZATION_INITIALIZATION[method], NORMALIZATION_BRAIN_MASK[method]

    def _syncCtDefaults(self):
        """The CT/MR method's initialization and brain mask set again when the CT/MR changes between a CT and an MR
        (not while a normalization is pending)."""
        if not hasattr(self, "maskCheck") or self.normResult is not None:
            return
        isCt = self.method() == METHOD_ANATOMY and self._anatomyIsCt()
        if getattr(self, "_defaultsForCt", None) == isCt:
            return
        self._defaultsForCt = isCt
        initialization, masked = self._normalizationDefaults(self.method())
        self.normInitCombo.setCurrentIndex(self.normInitCombo.findData(initialization))
        self.maskCheck.checked = masked

    def onMethodChanged(self, *args):
        method = self.method()
        self._defaultsForCt = method == METHOD_ANATOMY and self._anatomyIsCt()
        initialization, masked = self._normalizationDefaults(method)
        self.normInitCombo.setCurrentIndex(self.normInitCombo.findData(initialization))
        self.maskCheck.checked = masked
        self._templateChosen = False        # another method: its default template again
        self._updateTemplateChoice()
        self._clearNormalization()

    def _onTemplatePicked(self):
        self._templateChosen = True
        self._updateTemplateChoice()

    def _updateTemplateChoice(self):
        """T1 for the PET; the template of the MR contrast for the CT/MR method (the space's volumes that fit)."""
        if not hasattr(self, "templateCombo"):
            return
        self._fillTemplateCombo()
        method = self.method()
        if method == METHOD_ANATOMY:
            contrast, reason = self.contrast()
            key = neuro.templateForContrast(contrast)
            note = (f"CT/MR recognised as {contrast} ({reason})." if contrast else f"CT/MR contrast: {reason}.")
        else:
            key, note = neuro.DEFAULT_TEMPLATE, "The PET is registered to the T1 template by default."
        chosen = self.templateCombo.currentData
        if getattr(self, "_templateChosen", False) and chosen and self._templateAvailable(chosen):
            # picked by hand: kept (the registration always uses the template shown)
            self.templateNote.text = f"{self._templateTitle(chosen)}, picked by hand."
            self._updateTemplateCredits()
            return
        if self.templateCombo.findData(key) < 0 and self.templateCombo.count:
            fallback = self.templateCombo.itemData(0)
            note += (f" The {neuro.templateTitle(key)} template is not in the space, "
                     f"{self._templateTitle(fallback)} is used.")
            key = fallback
        index = self.templateCombo.findData(key)
        if index >= 0 and index != self.templateCombo.currentIndex:
            self.templateCombo.blockSignals(True)
            self.templateCombo.setCurrentIndex(index)
            self.templateCombo.blockSignals(False)
        self.templateNote.text = note
        self._updateTemplateCredits()

    def _updateTemplateCredits(self):
        """Description, references and license of the template registered to."""
        if not hasattr(self, "templateCredits"):
            return
        key = self.templateCombo.currentData or ""
        if normaldb.isDatabaseTemplate(key):
            self.templateCredits.setHtml(f"<p>Template: {self._templateTitle(key)}, made from the normal database's "
                                         f"subjects in the {self.spaceName()} space.</p>")
            return
        self.templateCredits.setHtml(credits.templateHtml(key, neuro.templateTitle(key) if key else "",
                                                          self.spaceId()))

    def normEngine(self):
        return self.normEngineCombo.currentData or NORMALIZATION_ENGINE

    def onNormEngineChanged(self, *args):
        iterative = self.normEngine() == reg.ENGINE_ITERATIVE
        self.normBrainsBox.visible = not iterative
        self.normIterativeBox.visible = iterative
        self._clearNormalization()

    def normSettings(self):
        from DynamicPETRegistration import iterativeSettings
        settings = self.logic.defaultSettings(self.method())
        settings["initialization"] = self.normInitCombo.currentData
        if self.normEngine() == reg.ENGINE_ITERATIVE:
            settings.update(iterativeSettings(self.normIterative), engine=reg.ENGINE_ITERATIVE)
            settings["steps"] = reg.iterativeSteps(settings["iterativeLinear"], settings["iterativeDeformable"])
            return settings
        steps = [c.currentData for c in self.normStepCombos if c.currentData]
        settings["engine"] = reg.ENGINE_BRAINSFIT
        settings["steps"] = reg.validateSteps(steps)
        settings["bsplineGrid"] = reg.parseGrid(self.normGridEdit.text)
        settings["samplingPercentage"] = float(self.normSamplingSpin.value)
        try:
            factor, tolerance = float(self.normFactorEdit.text), float(self.normToleranceEdit.text)
        except ValueError:
            raise ValueError("The convergence factor and the gradient tolerance are numbers (e.g. 1e9, 1e-5).")
        settings.update(numberOfIterations=int(self.normIterationsSpin.value),
                        histogramBins=int(self.normBinsSpin.value),
                        bsplineConvergenceFactor=factor, projectedGradientTolerance=tolerance,
                        maximumNumberOfEvaluations=int(self.normEvaluationsSpin.value),
                        maximumNumberOfCorrections=int(self.normCorrectionsSpin.value),
                        maxBSplineDisplacement=float(self.normDisplacementSpin.value),
                        numberOfThreads=int(self.normThreadsSpin.value))
        working = float(self.normWorkingSpin.value)
        settings["resample"] = [working > 0] * len(settings["steps"])
        settings["workingVoxelMm"] = working
        return settings

    def _resetNormAdvanced(self):
        """The BRAINSFit options of the normalization back to their defaults."""
        options = NORMALIZATION_BRAINSFIT
        self.normWorkingSpin.value = NORMALIZATION_WORKING_MM
        self.normGridEdit.text = str(NORMALIZATION_BSPLINE_GRID)
        self.normSamplingSpin.value = NORMALIZATION_SAMPLING
        self.normIterationsSpin.value = options["numberOfIterations"]
        self.normBinsSpin.value = options["histogramBins"]
        self.normFactorEdit.text = f"{options['bsplineConvergenceFactor']:g}"
        self.normToleranceEdit.text = f"{options['projectedGradientTolerance']:g}"
        self.normEvaluationsSpin.value = options["maximumNumberOfEvaluations"]
        self.normCorrectionsSpin.value = options["maximumNumberOfCorrections"]
        self.normDisplacementSpin.value = options["maxBSplineDisplacement"]
        self.normThreadsSpin.value = options["numberOfThreads"]

    def _template(self):
        """The template the result lies on (its grid); with the CT template the 2009c T1 (the CT is registered to
        the CT template, see _ctTemplate)."""
        key = self.templateCombo.currentData
        if normaldb.isDatabaseTemplate(key):
            volumes, name = self._databaseTemplates()
            if key not in volumes:
                raise ValueError(f"'{name}' has no {self._templateTitle(key)[len('From DB: '):]}.")
            template = self.logic.loadAtlasVolume(volumes[key], name=f"{name} {normaldb.databaseTemplateVolume(key)}")
            if key == normaldb.DB_PET:
                template.SetAttribute(PET_TEMPLATE_ATTRIBUTE, "1")      # a PET: shown in inverted grey
            self.setNode("Template", template)
            return template
        if key == neuro.CT_TEMPLATE:
            key = neuro.DEFAULT_TEMPLATE
        path = self.atlas.get(key)
        if not path and key == neuro.PSEUDO_PET and neuro.canMakeTemplate(self.atlas, key):
            path = self.logic.ensurePseudoPet(self.atlas)
            self.atlas = neuro.atlasFiles(self.spaceId()) or self.atlas
        if not path:
            raise ValueError(f"The space {self.spaceName()} has no {neuro.templateTitle(key)} volume "
                             f"({key}.nrrd in its folder).")
        template = self.logic.loadAtlasVolume(path, name=f"{self.spaceName()} {key}")
        if key == neuro.PSEUDO_PET:
            template.SetAttribute(PET_TEMPLATE_ATTRIBUTE, "1")      # a PET: shown in inverted grey
        self.setNode("Template", template)
        return template

    @staticmethod
    def _isPetTemplate(node):
        """Whether a template is a PET (the synthetic PET, a database's mean PET): shown in inverted grey."""
        return node is not None and (bool(node.GetAttribute(PET_TEMPLATE_ATTRIBUTE))
                                     or (node.GetName() or "").endswith(" " + neuro.PSEUDO_PET))

    @staticmethod
    def _invertedGrey(node):
        """A volume's display in the inverted grey of the PET (dark = high)."""
        from DICOMGatedDynamicPETLib import colormaps
        display = node.GetDisplayNode() if node is not None else None
        if display is None and node is not None:
            node.CreateDefaultDisplayNodes()
            display = node.GetDisplayNode()
        if display is not None:
            display.SetAndObserveColorNodeID(colormaps.colorNodeForLabel(PET_TEMPLATE_COLORMAP).GetID())

    @staticmethod
    def _setColor(node, colorId):
        """A volume's display in one of Slicer's color tables (by node ID); nothing when the node or table is missing."""
        display = node.GetDisplayNode() if node is not None else None
        if display is None and node is not None:
            node.CreateDefaultDisplayNodes()
            display = node.GetDisplayNode()
        if display is not None and slicer.mrmlScene.GetNodeByID(colorId) is not None:
            display.SetAndObserveColorNodeID(colorId)

    def _mask(self):
        if not self.maskCheck.checked or neuro.BRAIN_MASK not in self.atlas:
            return None
        return self.logic.loadAtlasVolume(self.atlas[neuro.BRAIN_MASK], name=f"{self.spaceName()} brain mask")

    def _ctTemplate(self):
        """The CT template of the atlas folder (scct_unsmooth / scct) when a CT is normalized (CT/MR method), else
        None."""
        key = self.templateCombo.currentData
        if key == normaldb.databaseTemplateKey("CT") and self.method() == METHOD_ANATOMY and self._anatomyIsCt():
            return self._template()         # the database's mean CT: CT to CT on its own grid (no correction)
        if key != neuro.CT_TEMPLATE or neuro.CT_TEMPLATE not in self.atlas:
            return None
        if self.method() != METHOD_ANATOMY or not self._anatomyIsCt():
            raise ValueError("The CT template is for a CT with the 'CT/MR to MR template' method; pick another "
                             "template.")
        return self.logic.loadAtlasVolume(self.atlas[neuro.CT_TEMPLATE], name="MNI CT template")

    def _ctCorrection(self, ctTemplate=None):
        """The CT template's space to the ICBM 152 2009c (4x4): its grey matter map (scgrey) registered to the 2009c
        grey matter map once (cached next to the CT template); identity without the two maps."""
        if neuro.CT_TEMPLATE_GM not in self.atlas or "GM" not in self.atlas:
            logging.warning("Neuro: scgrey (or the 2009c grey matter map) not in the atlas folder: the CT template's "
                            "MNI space is taken as the ICBM 152 2009c")
            return None
        templateGm = self.logic.loadAtlasVolume(self.atlas[neuro.CT_TEMPLATE_GM], name="MNI CT template grey matter")
        icbmGm = self.logic.loadAtlasVolume(self.atlas["GM"], name=f"{self.spaceName()} GM")
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            return self.logic.ctTemplateCorrection(templateGm, icbmGm, neuro.ctCorrectionPath(self.atlas), ctTemplate,
                                                   neuro.ctCorrectedPath(self.atlas))
        finally:
            qt.QApplication.restoreOverrideCursor()

    def onNormalize(self):
        sequence, anatomy, method = self.sequence(), self.anatomy(), self.method()
        if sequence is None:
            slicer.util.warningDisplay("Pick the dynamic PET series in step 1.", windowTitle="Normalization")
            return
        if method == METHOD_ANATOMY:
            if anatomy is None:
                slicer.util.warningDisplay("Pick the CT/MR in step 1, or register the PET to the template.",
                                           windowTitle="Normalization")
                return
            if not self.logic.petInAnatomySpace(sequence, anatomy) and not slicer.util.confirmOkCancelDisplay(
                    f"'{sequence.GetName()}' was not registered to '{anatomy.GetName()}' in step 3. The CT/MR "
                    "transform moves the PET correctly only when the PET already lies on the CT/MR (e.g. PET/MR, or "
                    "registered and hardened). Go on?", windowTitle="Normalization"):
                return
        try:
            settings = self.normSettings()
            template = self._template()
            mask = self._mask()
            ctTemplate = self._ctTemplate()
            ctCorrection = (self._ctCorrection(ctTemplate) if ctTemplate is not None
                            and not normaldb.isDatabaseTemplate(self.templateCombo.currentData) else None)
        except Exception as error:
            slicer.util.errorDisplay(str(error), windowTitle="Normalization")
            return
        self._clearNormalization()
        progressDialog = slicer.util.createProgressDialog(windowTitle="Normalization", labelText="Registering...",
                                                          maximum=len(settings["steps"]))

        def progress(index, total, text):
            progressDialog.setValue(index)
            progressDialog.setLabelText(f"Normalization: {text}")
            slicer.app.processEvents()
            return not progressDialog.wasCanceled

        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            self.normResult = self.logic.normalize(sequence, template, method, anatomy, settings, mask, progress,
                                                   ctTemplate=ctTemplate, ctCorrection=ctCorrection)
        except Exception as error:
            logging.exception("Neuro: normalization failed")
            slicer.util.errorDisplay(f"Normalization failed:\n{error}")
            self.normResult = None
            return
        finally:
            qt.QApplication.restoreOverrideCursor()
            progressDialog.close()
            progressDialog.deleteLater()
        if not self.normResult["steps"]:
            bspline = self.normResult.get("bspline")
            if bspline is not None and slicer.mrmlScene.IsNodePresent(bspline):
                slicer.mrmlScene.RemoveNode(bspline)
            self.normResult = None
            self.normLabel.text = "Stopped before the first step."
            self._updateNormButtons()
            return
        self.normResult["template"] = template
        self.normResult["method"] = method
        self._openNormSession()
        lines = reg.summaryLines(self.normResult["steps"], self.normResult["linear"], self.normResult["center"],
                                 self.normResult["deformation"])
        moving = self.normResult["moving"]
        notes = self.normResult.get("notes") or []
        self.normLabel.text = (f"'{moving.GetName()}' registered to '{template.GetName()}'"
                               + (" (stopped)" if self.normResult["stopped"] else "") + ":\n" + "\n".join(lines)
                               + ("\n" + " ".join(notes) if notes else "")
                               + "\nCheck it in the views, correct it by hand if needed, then Harden.")
        self._updateNormButtons()

    def _openNormSession(self):
        """The registration layout (PET / fusion / template, checkerboard) with the normalization chain on the PET.
        With the CT/MR method the PET is shown too: it lies on the CT/MR, so the same chain moves it."""
        from DynamicPETRegistration import RegistrationSession, DynamicPETRegistrationWidget
        self._closeNormSession()
        sequence = self.sequence()
        if self.normResult is None or sequence is None:
            return
        if DynamicPETRegistrationWidget._closeOtherModules() is False:
            return              # Review/Edit has unsaved edits and the user keeps it open
        registrationWidget = getattr(slicer.modules, "DynamicPETRegistrationWidget", None)
        if registrationWidget is not None and getattr(registrationWidget, "session", None) is not None:
            registrationWidget.onClose()
        session = None
        try:
            session = RegistrationSession(self.normResult["template"])
            session.fusionOpacity = self.normOpacity.value
            session.showCheckerboard = self.normCheckerCheck.checked
            session.open(self.logic.referenceImage(sequence))
            session.setRegistration(self.normResult["linear"], self.normResult["bspline"])
            session.setFixedWindow(*_percentiles(self.normResult["template"]))
            if self._isPetTemplate(self.normResult["template"]):
                # PET to PET: the template in Blue, the patient's PET in Red (alone and in the fusion)
                self._setColor(session.fixedHelper, NORM_TEMPLATE_PET_COLOR_ID)
                self._setColor(session.petView, NORM_PATIENT_PET_COLOR_ID)
                self._setColor(session.petFusion, NORM_PATIENT_PET_COLOR_ID)
            if self.normResult.get("method") == METHOD_ANATOMY and self.normResult.get("moving") is not None:
                # the normalized CT/MR alone on the right, over the template in the fusion column
                moving = self.normResult["moving"]
                session.setAnatomy(moving, _percentiles(moving))
            self.normSession = session
            self._matchNormZoom()
            qt.QTimer.singleShot(200, self._matchNormZoom)      # again once the new layout has its final size
        except Exception as error:
            logging.exception("Neuro: could not open the normalization layout")
            if session is not None:
                self._closeHalfOpen(session)
            slicer.util.errorDisplay(f"Could not open the normalization layout:\n{error}")


    def _onNormCheckerToggled(self, on):
        if self.normSession is not None:
            self.normSession.setShowCheckerboard(on)
            self._matchNormZoom()
            qt.QTimer.singleShot(200, self._matchNormZoom)

    def _matchNormZoom(self):
        """Every view of the normalization layout at the zoom of the template view of its row (the same mm per pixel)
        and on the centre of the template, so the patient's PET and the template start at the same size."""
        session = self.normSession
        if session is None or session.fixedHelper is None:
            return
        try:
            views = session._views()
            anchors = {}
            for orientation, column, widget in views:
                if column == "Anat":
                    widget.sliceController().fitSliceToBackground()
                    anchors[orientation] = widget.mrmlSliceNode()
            bounds = [0.0] * 6
            session.fixedHelper.GetRASBounds(bounds)
            center = [(bounds[2 * i] + bounds[2 * i + 1]) / 2.0 for i in range(3)]
            for orientation, column, widget in views:
                anchor = anchors.get(orientation)
                sliceNode = widget.mrmlSliceNode()
                if anchor is not None and sliceNode is not anchor:
                    fov, dims = anchor.GetFieldOfView(), anchor.GetDimensions()
                    own = sliceNode.GetDimensions()
                    if dims[0] > 0 and dims[1] > 0 and own[0] > 0 and own[1] > 0:
                        mmPerPixel = max(fov[0] / dims[0], fov[1] / dims[1])
                        sliceNode.SetFieldOfView(mmPerPixel * own[0], mmPerPixel * own[1], fov[2])
                sliceNode.JumpSliceByCentering(*center)
        except Exception:
            logging.debug("Neuro: could not match the zoom of the normalization views", exc_info=True)

    @staticmethod
    def _closeHalfOpen(session):
        """A registration layout that failed while opening: its helper volumes, transforms and layout taken back."""
        try:
            session.close()
        except Exception:
            logging.exception("Neuro: could not close the half open registration layout")

    def _closeNormSession(self):
        if self.normSession is not None:
            if self.normManualButton.checked:
                self.normManualButton.blockSignals(True)
                self.normManualButton.checked = False
                self.normManualButton.blockSignals(False)
            self.normSession.close()
            self.normSession = None

    def _clearNormalization(self):
        """Drop a pending (not hardened) normalization."""
        if not hasattr(self, "normLabel"):
            return
        self._closeNormSession()
        if self.normResult is not None:
            bspline = self.normResult.get("bspline")
            if bspline is not None and slicer.mrmlScene.IsNodePresent(bspline):
                slicer.mrmlScene.RemoveNode(bspline)
            self.normResult = None
            self.normLabel.text = "Not normalized yet."
        self._updateNormButtons()

    def _updateNormButtons(self):
        if not hasattr(self, "normHardenButton"):
            return
        pending = self.normResult is not None and self.normSession is not None
        self.normHardenButton.enabled = pending
        self.normManualButton.enabled = pending
        self.normRevertButton.enabled = pending or self.normUndo is not None
        self.normalizeButton.enabled = self.sequence() is not None and neuro.DEFAULT_TEMPLATE in self.atlas

    def onNormHarden(self):
        sequence = self.sequence()
        if self.normResult is None or self.normSession is None or sequence is None:
            return
        template, method = self.normResult["template"], self.normResult["method"]
        petSpacing = reg.gridSpacing(_ijkToRas(self.logic.firstImage(sequence)))
        text = self._normHardenText(sequence, template, method)
        choice = hardenDialog("Harden normalization", text, self.NORM_HARDEN_GRIDS, petSpacing,
                              list(self.NORM_HARDEN_ACTIONS), slicer.util.mainWindow())
        if choice is None:
            return
        steps = reg.stepsText(self.normResult["steps"])
        description = (f"Normalized to {template.GetName()} ({steps}"
                       + (", through the CT/MR" if method == METHOD_ANATOMY else "") + ")")
        self.normManualButton.checked = False
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            undo, newSequence, newSummed, normalizedAnatomy = self._hardenNormalization(
                sequence, template, choice, description, method)
        except Exception as error:
            logging.exception("Neuro: could not harden the normalization")
            slicer.util.errorDisplay(f"Could not harden the normalization:\n{error}")
            return
        finally:
            qt.QApplication.restoreOverrideCursor()
        self._clearNormalization()
        self.normUndo = {"undo": undo, "previous": sequence, "anatomy": normalizedAnatomy}
        self.setNode("Sequence", newSequence)
        self.setNode("NormalizedAnatomy", normalizedAnatomy)
        self.normLabel.text = (description + ".\n"
                               + (f"Overwrote '{sequence.GetName()}'" if choice["action"] == reg.OVERWRITE else
                                  f"Saved as '{newSequence.GetName()}'"
                                  + (f" and '{newSummed.GetName()}'" if newSummed is not None else ""))
                               + (f"; CT/MR as '{normalizedAnatomy.GetName()}'" if normalizedAnatomy else "")
                               + ". Next: load the regions.")
        self._showNormalized(newSequence, template)
        self._updateNormButtons()
        self.refresh()

    def _hardenNormalization(self, sequence, template, choice, description, method):
        """logic.harden() of the pending normalization with the Harden dialog's choice."""
        return self.logic.harden(sequence, template, self.normSession.linear, choice["action"], choice["grid"],
                                 choice["spacing"], choice["interpolation"], description, method, self.anatomy())

    @staticmethod
    def _normHardenText(sequence, template, method):
        return (f"Resample the {sequence.GetNumberOfDataNodes()} frames of '{sequence.GetName()}' (and its summed "
                f"image) through the normalization onto the field of view of the template '{template.GetName()}'."
                + ("\nThe CT/MR is normalized too (on the template grid)." if method == METHOD_ANATOMY else "")
                + "\n\nSave separately (default) keeps the series in the patient space next to the normalized one.")

    def _showNormalized(self, sequence, template):
        """The normalized summed image over the template in the standard views."""
        summed = self.logic.displayImage(sequence)
        try:
            slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
            slicer.util.setSliceViewerLayers(background=template, foreground=summed, foregroundOpacity=0.5,
                                             label=None, fit=True)
        except Exception:
            logging.debug("Neuro: could not show the normalized series", exc_info=True)

    def onNormRevert(self):
        if self.normResult is not None:
            self._clearNormalization()
            return
        if self.normUndo is None:
            return
        record = self.normUndo
        try:
            self.logic.revertHarden(record["undo"])
        except Exception as error:
            logging.exception("Neuro: could not revert the normalization")
            slicer.util.errorDisplay(f"Could not revert the normalization:\n{error}")
            return
        anatomy = record.get("anatomy")
        if anatomy is not None and slicer.mrmlScene.IsNodePresent(anatomy):
            slicer.mrmlScene.RemoveNode(anatomy)
        if slicer.mrmlScene.IsNodePresent(record["previous"]):
            self.setNode("Sequence", record["previous"])
        self.setNode("NormalizedAnatomy", None)
        self.normUndo = None
        self.normLabel.text = "Reverted: the series is in the patient space."
        self._updateNormButtons()
        self.refresh()

    # --- page 5: regions ---------------------------------------------------------------------------
    def _buildRegionsPage(self):
        page = qt.QWidget()
        layout = qt.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        box = qt.QGroupBox("Regions")
        form = qt.QFormLayout(box)
        self.regionsInfo = qt.QLabel()
        self.regionsInfo.wordWrap = True
        form.addRow(self.regionsInfo)
        self.segmentationCombo = qt.QComboBox()
        self.segmentationCombo.toolTip = ("Segmentation the regions come from: those of the space normalized to, in "
                                          "the extension and in your atlas folder (~/NeuroPET/Atlas/Segmentation/"
                                          "<name>: segmentation.json, labels.seg.nrrd or labels.nrrd, labels.csv).")
        form.addRow("Segmentation:", self.segmentationCombo)
        self.groupingCombo = qt.QComboBox()
        self.groupingCombo.toolTip = ("Regions as in the segmentation (left and right separate), its broad regions "
                                      "(groups.csv: Prefrontal, Lateral Occipital, ...) or Broad (Compatible), GE "
                                      "CortexID-like regions (groups_compatible.csv). The label maps are made on first "
                                      "use.")
        form.addRow("Grouping:", self.groupingCombo)
        self.referenceRegionCombo = qt.QComboBox()
        self.referenceRegionCombo.toolTip = ("Reference region of the reference tissue models (no specific binding; "
                                             "the cerebellar grey matter by default): a region of the regions above, "
                                             "both sides together")
        form.addRow("Reference region:", self.referenceRegionCombo)
        # the atlas the regions come from: references to cite and license (read only)
        self.regionsCredits = credits.infoBox()
        form.addRow(self.regionsCredits)
        self.growSpin = qt.QDoubleSpinBox()
        self.growSpin.setRange(0.0, 10.0)
        self.growSpin.setDecimals(1)
        self.growSpin.singleStep = 0.5
        self.growSpin.suffix = " mm"
        self.growSpin.specialValueText = "Off"
        self.growSpin.value = GROW_MM
        self.growSpin.toolTip = (
            "Grow the regions into the unlabelled voxels around them (sulcal CSF, the edge of the cortex), for the "
            "PET resolution. Each voxel goes to the nearest region, so regions never overlap. Only inside the MNI "
            "brain mask and outside the white matter (probability < 0.5, when the space has a WM map); the "
            "ventricles do not grow.")
        form.addRow("Grow regions:", self.growSpin)
        self.growIntoCombo = _growIntoCombo()
        form.addRow("Grow into:", self.growIntoCombo)
        self.growSpin.connect("valueChanged(double)", lambda v: setattr(self.growIntoCombo, "enabled", v > 0))
        self.growIntoCombo.enabled = GROW_MM > 0
        self.erodeSpin = qt.QDoubleSpinBox()
        self.erodeSpin.setRange(0.0, 10.0)
        self.erodeSpin.setDecimals(1)
        self.erodeSpin.singleStep = 0.5
        self.erodeSpin.suffix = " mm"
        self.erodeSpin.specialValueText = "Off"
        self.erodeSpin.value = ERODE_REFERENCE_MM
        self.erodeSpin.toolTip = ("Erode the reference region (taken from the atlas before growing), so spill-over "
                                  "from the neighbouring tissue does not reach it.")
        form.addRow("Erode reference:", self.erodeSpin)
        self.regionsTargetSelector = self._nodeSelector(
            ["vtkMRMLSegmentationNode"],
            "Segmentation the regions are written into: its segments are replaced, so loading again (e.g. after "
            "changing the atlas or the growth) does not pile up segmentations. 'Create new' makes a new one.")
        self.regionsTargetSelector.noneDisplay = "Create new segmentation"
        form.addRow("Output segmentation:", self.regionsTargetSelector)
        self.loadRegionsButton = _button(self.LOAD_REGIONS_TEXT)
        form.addRow(self.loadRegionsButton)
        self.regionsLabel = qt.QLabel()
        self.regionsLabel.wordWrap = True
        form.addRow(self.regionsLabel)
        layout.addWidget(box)
        layout.addStretch(1)
        self.segmentationCombo.connect("currentIndexChanged(int)", lambda i: self._fillGroupings())
        self.groupingCombo.connect("currentIndexChanged(int)", lambda i: self._onRegionAtlasChanged())
        self.referenceRegionCombo.connect("currentIndexChanged(int)", lambda i: self._updateRegionsCredits())
        self._updateRegionsCredits()
        self.loadRegionsButton.connect("clicked()", self.onLoadRegions)
        return page

    def regionAtlasKey(self):
        """The region atlas picked (neuro.regionAtlasKey: segmentation and grouping)."""
        return neuro.regionAtlasKey(self.segmentationCombo.currentData or neuro.DEFAULT_SEGMENTATION,
                                    self.groupingCombo.currentData or neuro.REGIONS)

    def setRegionAtlas(self, key):
        """Picks a region atlas key (a database's); False when its segmentation or grouping is not there."""
        segmentation, grouping, _ = neuro.splitRegionAtlas(key)
        index = self.segmentationCombo.findData(segmentation)
        if index < 0:
            return False
        self.segmentationCombo.setCurrentIndex(index)
        self._fillGroupings()
        index = self.groupingCombo.findData(grouping)
        if index < 0:
            return False
        self.groupingCombo.setCurrentIndex(index)
        return True

    def _fillSegmentations(self):
        """The segmentations of the space (the one picked kept, else the one picked last)."""
        current = self.segmentationCombo.currentData or neuro.splitRegionAtlas(self.logic.savedRegionAtlas())[0]
        self.segmentationCombo.blockSignals(True)
        self.segmentationCombo.clear()
        for info in neuro.listSegmentations(self.spaceId()):
            self.segmentationCombo.addItem(info["name"], info["id"])
        index = self.segmentationCombo.findData(current)
        self.segmentationCombo.setCurrentIndex(index if index >= 0 else 0)
        self.segmentationCombo.blockSignals(False)
        self._fillGroupings()

    def _fillGroupings(self):
        info = neuro.segmentationInfo(self.segmentationCombo.currentData) if self.segmentationCombo.count else None
        current = self.groupingCombo.currentData or neuro.splitRegionAtlas(self.logic.savedRegionAtlas())[1]
        self.groupingCombo.blockSignals(True)
        self.groupingCombo.clear()
        titles = dict(neuro.GROUPINGS)
        for grouping in neuro.segmentationGroupings(info):
            self.groupingCombo.addItem(titles[grouping], grouping)
        index = self.groupingCombo.findData(current)
        self.groupingCombo.setCurrentIndex(index if index >= 0 else 0)
        self.groupingCombo.blockSignals(False)
        self._onRegionAtlasChanged()

    def _onRegionAtlasChanged(self):
        if self.segmentationCombo.count and self.groupingCombo.count:
            self.logic.setSavedRegionAtlas(self.regionAtlasKey())
        self._fillReferenceRegions()
        self._updateRegionsCredits()

    def _fillReferenceRegions(self):
        """The regions of the region atlas with the sides merged (from the tables, so they are listed before the
        label maps exist), the cerebellar grey matter picked."""
        current = self.referenceRegionCombo.currentText
        kind = neuro.mergedRegionAtlas(self.regionAtlasKey()) if self.segmentationCombo.count else None
        try:
            entries = self.logic.regionEntries(kind) if kind else []
        except Exception:
            logging.warning("Neuro: could not read the region labels", exc_info=True)
            entries = []
        try:
            references = neuro.referenceGroups(kind) if kind else []
        except Exception:
            logging.warning("Neuro: could not read the reference regions of the groups table", exc_info=True)
            references = []
        # the regions, then the reference-only regions of the groups table (Cerebellum Whole)
        names = sorted(n for _, n, _ in entries)
        names += [n for n in neuro.referenceRegionNames(entries, references) if n not in names]
        key = (kind, tuple(names))
        if getattr(self, "_referenceKey", None) == key:
            return
        self._referenceKey = key
        self.referenceRegionCombo.blockSignals(True)
        self.referenceRegionCombo.clear()
        for name in names:
            self.referenceRegionCombo.addItem(name)
        index = self.referenceRegionCombo.findText(current) if current else -1
        if index < 0:
            index = self.referenceRegionCombo.findText(defaultReferenceName(entries) or "")
        self.referenceRegionCombo.setCurrentIndex(max(0, index))
        self.referenceRegionCombo.blockSignals(False)

    def _updateRegionsPage(self):
        self._fillReferenceRegions()
        sequence = self.sequence()
        normalized = sequence is not None and bool(sequence.GetAttribute(NEURO_PREFIX + "Template"))
        files = self.segmentationCombo.count > 0
        text = []
        if not normalized:
            text.append("<span style='color:#c0392b'>The image is not normalized yet (Normalization step): the "
                        "regions lie in the template space.</span>")
        if not files:
            text.append(f"<span style='color:#c0392b'>The space {self.spaceName()} has no segmentation (a folder "
                        "of Segmentation/ in the atlas folders whose segmentation.json names this space).</span>")
        segmentation = self.node("Segmentation")
        if segmentation is not None:
            text.append(f"Loaded: '{segmentation.GetName()}'.")
            if not getattr(self, "_targetInitialized", False):     # once: loading again replaces it by default
                self._targetInitialized = True
                self.regionsTargetSelector.setCurrentNode(segmentation)
        if not text:
            text.append(f"'{sequence.GetName()}' is in the template space. Load the regions.")
        self.regionsInfo.text = " ".join(text)
        self.loadRegionsButton.enabled = files and sequence is not None
        self._updateRegionsCredits()

    def _updateRegionsCredits(self):
        """Description, references and license of the region atlas (and of the pons when it is the reference)."""
        if not hasattr(self, "regionsCredits"):
            return
        key = self.regionAtlasKey() if self.segmentationCombo.count else ""
        self.regionsCredits.setHtml(credits.regionsHtml(key, self.referenceRegionCombo.currentText,
                                                        neuro.regionAtlasTitle(key) if key else ""))

    def onLoadRegions(self):
        sequence = self.sequence()
        if sequence is None:
            return
        if not sequence.GetAttribute(NEURO_PREFIX + "Template") and not slicer.util.confirmOkCancelDisplay(
                "The image is not normalized (Normalization step), so the regions will not lie on the brain. "
                "Load them anyway?", windowTitle="Regions"):
            return
        try:
            arguments = self._regionsArguments()
        except ValueError as error:
            slicer.util.errorDisplay(str(error), windowTitle="Regions")
            return
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            segmentation, referenceId = self.logic.loadAtlasRegions(self.atlas, **arguments)
        except Exception as error:
            logging.exception("Neuro: could not load the regions")
            slicer.util.errorDisplay(f"Could not load the regions:\n{error}")
            return
        finally:
            qt.QApplication.restoreOverrideCursor()
        old = self.node("Segmentation")
        if old is segmentation:
            old = None
        self.setNode("Segmentation", segmentation)
        self.regionsTargetSelector.setCurrentNode(segmentation)
        self._parameter().SetParameter(NEURO_PREFIX + "ReferenceSegment", referenceId or "")
        count = len(self.logic.segmentIds(segmentation))
        reference = segmentation.GetSegmentation().GetSegment(referenceId).GetName() if referenceId else None
        self.regionsLabel.text = (f"'{segmentation.GetName()}': {count} segments"
                                  + (f", reference '{reference}'" if reference else ", no reference region") + "."
                                  + (f" (The earlier '{old.GetName()}' is still in the scene.)" if old else ""))
        self.showStep(self.STEP_AFTER_REGIONS)

    def _regionsArguments(self):
        """Keyword arguments of logic.loadAtlasRegions() from the page."""
        if not self.segmentationCombo.count:
            raise ValueError(f"The space {self.spaceName()} has no segmentation.")
        key = self.regionAtlasKey()
        return {"targetKind": key, "referenceKind": neuro.mergedRegionAtlas(key),
                "referenceName": self.referenceRegionCombo.currentText or None,
                "growMm": float(self.growSpin.value), "erodeReferenceMm": float(self.erodeSpin.value),
                "growInto": self.growIntoCombo.currentData or normaldb.GROW_NO_WM,
                "target": self.regionsTargetSelector.currentNode()}

    # --- embedded modules --------------------------------------------------------------------------
    def _moduleName(self, key):
        return next(m for k, m, _, _ in self.STEPS if k == key)

    def stepPanel(self, key):
        """The page of a step: an own page, or the widget representation of the embedded module (taken back into
        the stack when Slicer showed it in its own panel meanwhile)."""
        if key in self.pages:
            panel = self.pages[key]
        else:
            moduleName = self._moduleName(key)
            module = getattr(slicer.modules, moduleName.lower(), None)
            if module is None:
                raise RuntimeError(f"Module {moduleName} is not loaded")
            panel = module.widgetRepresentation()
            self.embedded[key] = panel
        if self.stack.indexOf(panel) < 0:
            self.stack.addWidget(panel)
        return panel

    def stepWidget(self, key):
        panel = self.embedded.get(key)
        if panel is None:
            return None
        try:
            return panel.self()
        except Exception:
            return None

    def _callStep(self, key, method):
        if key in self.embedded:
            widget = self.stepWidget(key)
            function = getattr(widget, method, None)
            if callable(function):
                try:
                    function()
                except Exception:
                    logging.warning(f"Neuro: {method}() of step {key} failed", exc_info=True)
        elif key == STEP_NORM and method == "exit":
            self._closeNormSession()
        elif key == STEP_NORM and method == "enter" and self.normResult is not None:
            self._openNormSession()

    def _takeSeriesFrom(self, key):
        """The series the embedded module ended with (e.g. a new series saved separately) is the workflow's."""
        widget = self.stepWidget(key)
        selector = getattr(widget, "sequenceSelector", None) if widget is not None else None
        node = selector.currentNode() if selector is not None else None
        if node is not None and node is not self.sequence():
            self.setNode("Sequence", node)
            self._clearNormalization()

    def _prepareStep(self, key):
        """Give the embedded module the workflow's series (and CT/MR, regions) before it is shown."""
        widget = self.stepWidget(key)
        sequence = self.sequence()
        if widget is None:
            return
        try:
            if sequence is not None and widget.sequenceSelector.currentNode() is not sequence:
                widget.sequenceSelector.setCurrentNode(sequence)
            if key == STEP_REG and self.anatomy() is not None and widget.fixedSelector.currentNode() is not \
                    self.anatomy():
                widget.fixedSelector.setCurrentNode(self.anatomy())
            if key == STEP_MODEL:
                self._prepareModeling(widget)
        except Exception:
            logging.warning(f"Neuro: could not set the inputs of step {key}", exc_info=True)

    def _prepareModeling(self, widget):
        self._prepareBrainMask(widget)
        segmentation = self.node("Segmentation")
        if segmentation is None:
            return
        fresh = widget.segmentationSelector.currentNode() is not segmentation
        if fresh:
            index = widget.modelCombo.findData(DEFAULT_MODEL)
            if index >= 0:
                widget.modelCombo.setCurrentIndex(index)
            widget.segmentationSelector.setCurrentNode(segmentation)
            referenceId = self._parameter().GetParameter(NEURO_PREFIX + "ReferenceSegment")
            index = widget.referenceCombo.findData(referenceId) if referenceId else -1
            if index >= 0:
                widget.referenceCombo.setCurrentIndex(index)
            target = (self.logic.segmentIdByName(segmentation, DEFAULT_TARGET)
                      or self.logic.segmentIdByName(segmentation, DEFAULT_TARGET + " R")
                      or self.logic.segmentIdByName(segmentation, DEFAULT_TARGET, startswith=True))
            index = widget.targetCombo.findData(target) if target else -1
            if index >= 0:
                widget.targetCombo.setCurrentIndex(index)

    def _prepareBrainMask(self, widget):
        """The MNI template's brain mask for 'Mask outside the brain' of the voxel-wise maps (the series is in MNI
        space after the normalization), unless another mask was picked. A series that is not normalized gets no
        mask (the template's mask is taken off), so 'Mask outside the brain' is off."""
        selector = getattr(widget, "brainMaskSelector", None)
        if selector is None:
            return
        sequence = self.sequence()
        normalized = sequence is not None and bool(sequence.GetAttribute(NEURO_PREFIX + "Template"))
        current = selector.currentNode()
        if not normalized:
            if current is not None and current.GetAttribute(ATLAS_FILE_ATTRIBUTE):
                selector.setCurrentNode(None)
            return
        if current is None and neuro.BRAIN_MASK in self.atlas:
            selector.setCurrentNode(self.logic.loadAtlasVolume(self.atlas[neuro.BRAIN_MASK],
                                                               name=f"{self.spaceName()} brain mask"))

    def pendingHarden(self, key):
        """Whether step key (motion correction, CT/MR registration, normalization) has a result that is not hardened
        yet."""
        if key == STEP_NORM:
            return self.normResult is not None
        if key in (STEP_MC, STEP_REG) and key in self.embedded:
            button = getattr(self.stepWidget(key), "hardenButton", None)
            return bool(button is not None and button.enabled)
        return False

    def _hardenStep(self, key):
        if key == STEP_NORM:
            self.onNormHarden()
        else:
            widget = self.stepWidget(key)
            if widget is not None:
                widget.onHarden()

    def _confirmLeaving(self, key):
        """Leaving a step with a result that is not hardened: offer Harden. False to stay on the step."""
        if not self.pendingHarden(key):
            return True
        title = dict((k, t) for k, _, t, _ in self.STEPS)[key]
        box = qt.QMessageBox(slicer.util.mainWindow())
        box.setIcon(qt.QMessageBox.Question)
        box.setWindowTitle(title)
        box.setText(f"The {title.lower()} result is not hardened yet. Harden it now?")
        box.setInformativeText("Without Harden the series stays as it was; the result is kept and you can come back "
                               "to harden it.")
        hardenButton = box.addButton("Harden", qt.QMessageBox.AcceptRole)
        continueButton = box.addButton("Continue without", qt.QMessageBox.DestructiveRole)
        box.addButton(qt.QMessageBox.Cancel)
        box.setDefaultButton(hardenButton)
        box.exec_()
        clicked = box.clickedButton()
        box.deleteLater()
        if clicked == continueButton:
            return True
        if clicked == hardenButton:
            self._hardenStep(key)
            return not self.pendingHarden(key)          # the Harden dialog cancelled: stay
        return False

    def showStep(self, key, enterStep=True):
        previous = self.current
        if previous != key and not self._confirmLeaving(previous):
            self.refresh()
            return
        if previous != key:
            if previous in self.embedded:
                self._takeSeriesFrom(previous)
            self._callStep(previous, "exit")
        try:
            panel = self.stepPanel(key)
        except Exception as error:
            logging.warning(f"Neuro: cannot show step {key}: {error}")
            key, panel = STEP_DATA, self.stepPanel(STEP_DATA)
        self.current = key
        self.stack.setCurrentWidget(panel)
        self.banner.setVisible(key in self.BANNER_STEPS and self.banner.hasPicture())
        # only the shown page stays in the stack (the stack takes the height of its tallest page otherwise)
        for index in reversed(range(self.stack.count)):
            page = self.stack.widget(index)
            if page is not panel:
                self.stack.removeWidget(page)
        if key == STEP_DATA:
            self._syncDataPage()
        if key in self.embedded:
            self._prepareStep(key)
        if enterStep and slicer.util.selectedModule() == self.MODULE_NAME:
            self._callStep(key, "enter")
        self.refresh()
        if previous != key:
            self._scrollToTop()

    def moveStep(self, delta):
        key = self._neighbourStep(delta)
        if key is not None:
            self.showStep(key)

    # --- which steps can be opened: the steps they need done first ---------------------------------
    STEP_NEEDS = {STEP_MC: "Pick the image in step 1 (Data) first.",
                  STEP_REG: "Pick the image and its CT / MR in step 1 (Data) first.",
                  STEP_NORM: "Pick the image in step 1 (Data) first.",
                  STEP_REGIONS: "Normalize the image first (Normalization step).",
                  STEP_MODEL: "Load the regions first (Regions step)."}

    def stepAllowed(self, key, states=None):
        """Whether step key can be opened: the data step and the shown step always, another step once the steps it
        needs are done (its state is not 'waiting'): an image for the CT/MR registration (with a CT / MR) and the
        normalization (the registration is never needed), the normalization for the regions, the regions for the
        steps after them."""
        if key == STEP_DATA or key == getattr(self, "current", None):
            return True
        states = states if states is not None else self.stepStates()
        return states.get(key) != STATE_WAITING

    def _skipStep(self, key):
        """Whether Back / Next pass over step key (none here)."""
        return False

    def _neighbourStep(self, delta, states=None):
        """The step Back (delta -1) / Next (+1) goes to (passing over the skipped steps); None when there is none or
        it cannot be opened yet."""
        keys = [k for k, _, _, _ in self.STEPS]
        index = keys.index(self.current) + delta
        while 0 <= index < len(keys) and self._skipStep(keys[index]):
            index += 1 if delta > 0 else -1
        if not 0 <= index < len(keys):
            return None
        return keys[index] if self.stepAllowed(keys[index], states) else None

    # --- state ---------------------------------------------------------------------------------------
    def stepStates(self):
        sequence, anatomy = self.sequence(), self.anatomy()
        states = {}
        states[STEP_DATA] = STATE_DONE if sequence is not None else STATE_READY
        ready = STATE_READY if sequence is not None else STATE_WAITING
        states[STEP_MC] = STATE_DONE if sequence is not None and sequence.GetAttribute(MC_PREFIX + "Method") else ready
        states[STEP_REG] = (STATE_DONE if sequence is not None and sequence.GetAttribute(REG_PREFIX + "Fixed") else
                            ready if anatomy is not None else STATE_WAITING)
        normalized = sequence is not None and bool(sequence.GetAttribute(NEURO_PREFIX + "Template"))
        states[STEP_NORM] = STATE_DONE if normalized else ready
        segmentation = self.node("Segmentation")
        states[STEP_REGIONS] = (STATE_DONE if segmentation is not None else
                                STATE_READY if normalized else STATE_WAITING)
        states[STEP_MODEL] = STATE_READY if segmentation is not None else STATE_WAITING
        return states

    def _stepIcon(self, key, state, number):
        """The icon of a step button: its state and number."""
        return _badge(state, number)

    def _scrollToTop(self):
        """The module panel scrolled back to its top (a step shown after a long one opened where that one ended)."""
        widget = self.parent
        while widget is not None:
            if widget.inherits("QScrollArea"):
                widget.verticalScrollBar().setValue(0)
                return
            widget = widget.parentWidget()

    def _defaultMethod(self):
        """Normalize with the patient's CT/MR when the study has one (else the PET), until the method is picked by
        hand; a pending normalization is not touched."""
        if getattr(self, "_methodChosen", False) or not hasattr(self, "methodCombo") or self.normResult is not None:
            return
        wanted = METHOD_ANATOMY if self.anatomy() is not None else METHOD_PET
        if self.methodCombo.currentData != wanted:
            self.methodCombo.setCurrentIndex(self.methodCombo.findData(wanted))

    def refresh(self):
        if not hasattr(self, "stepButtons"):
            return
        self._defaultMethod()
        self._syncCtDefaults()
        states = self.stepStates()
        for number, (key, _, title, explanation) in enumerate(self.STEPS, start=1):
            button = self.stepButtons[key]
            button.setIcon(self._stepIcon(key, states[key], number))
            button.checked = key == self.current
            allowed = self.stepAllowed(key, states)
            button.enabled = allowed
            tip = button.property("stepTip") or f"{number}. {title}: {explanation}"
            need = self.STEP_NEEDS.get(key)
            button.toolTip = tip if allowed or not need else f"{tip}\n{need}"
            if key == self.current:
                self.titleLabel.text = f"<b>{number}. {title}</b> - {explanation}"
        self.backButton.enabled = self._neighbourStep(-1, states) is not None
        following = self._neighbourStep(1, states)
        self.nextButton.enabled = following is not None
        keys = [k for k, _, _, _ in self.STEPS]
        later = [k for k in keys[keys.index(self.current) + 1:] if not self._skipStep(k)]
        need = self.STEP_NEEDS.get(later[0]) if later and following is None else None
        self.nextButton.toolTip = need or ""
        if self.current == STEP_DATA:
            self._updateDataInfo()
        self._updateNormButtons()
        self._updateRegionsPage()

    # --- module ------------------------------------------------------------------------------------
    def enter(self):
        self._entered = True
        self._showToolbar(True)
        self.selectInputs()
        self.showStep(self.current)

    def exit(self):
        self._entered = False
        self.refreshTimer.stop()
        self._showToolbar(False)
        if self.current in self.embedded:
            self._takeSeriesFrom(self.current)
        self._callStep(self.current, "exit")

    def cleanup(self):
        if getattr(self, "_entered", False) and self.current in self.embedded:
            DynamicNeuroPETWidget._callStep(self, self.current, "exit")     # the embedded module shown in the step
        self._entered = False
        self._showToolbar(False)
        if getattr(self, "toolbar", None) is not None:
            try:        # out of the main window's toolbar area (not deleted: Python may still wrap it)
                slicer.util.mainWindow().removeToolBar(self.toolbar)
            except Exception:
                logging.debug("Neuro: the toolbar is gone", exc_info=True)
        self._closeNormSession()
        self.refreshTimer.stop()
        self.removeObservers()
        for panel in self.embedded.values():   # the panels belong to their modules
            try:
                self.stack.removeWidget(panel)
                panel.setParent(None)
            except Exception:
                pass
        self.embedded = {}
        ScriptedLoadableModuleWidget.cleanup(self)

    def onSceneClose(self, caller, event):
        # a closed scene ends the patient: the shown step is left now (its views and observers still need the nodes
        # of the scene) and the workflow starts again on the Data step once the scene is empty (onSceneClosed), so
        # the next patient does not open on an empty Review / Modeling step
        if getattr(self, "current", STEP_DATA) != STEP_DATA:
            if getattr(self, "_entered", False):    # not shown: exit() has left the step already
                try:
                    self._callStep(self.current, "exit")
                except Exception:
                    logging.exception("Neuro: could not leave the step on scene close")
            self.current = STEP_DATA
        self._closeNormSession()
        self.normResult = None
        self.normUndo = None
        if hasattr(self, "normLabel"):
            self.normLabel.text = "Not normalized yet."

    def onSceneClosed(self, caller, event):
        """The scene is empty: show the Data step (onSceneClose has left the previous step)."""
        try:
            self.showStep(STEP_DATA)
        except Exception:
            logging.exception("Neuro: could not show the Data step after scene close")
        self._refreshLater()


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
