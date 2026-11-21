"""
Neuro PET Database Manager (3D Slicer module)

Normal databases of brain PET and SPECT in a template space (default MNI ICBM 152 2009c), kept in a database folder
picked by the user (one folder per database, see NeuroPETLib/normaldb.py for the layout):

    Static tab      the static databases of the folder in a table (name, kind, nuclide, tracer, subjects, reference
                    region, age groups, statistics), with
                        Create new database...      an empty user defined database (the subjects are added later
                                                    from the Neuro workflows, its mean / SD maps computed from them)
                        Add predefined database...  a packaged database (.zip) or mean / SD maps picked one by one
                                                    (master and each age group, with the number of subjects)
                        Export as predefined...     a user defined database as a .zip without its subject images
                        Download predefined...      the predefined databases published on GitHub Releases
                                                    (NeuroPETLib/dbdownload.py), installed or updated
                        Delete database             the folder removed from disk, after a warning
    Dynamic tab     to come

The reference region mask is copied into every database: made from the CerebrA atlas or the MNI brain mask of the
Neuro atlas folder (the one of the Dynamic Neuro Receptor PET module), or a mask file of the user.
"""

import logging
import os
import sys

import ctk
import numpy as np
import qt
import slicer
import vtk
from slicer.ScriptedLoadableModule import (ScriptedLoadableModule, ScriptedLoadableModuleWidget,
                                           ScriptedLoadableModuleLogic, ScriptedLoadableModuleTest)

_MODULE_DIR = os.path.dirname(os.path.abspath(globals().get("__file__", "") or "."))
if globals().get("__file__") and _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from NeuroPETLib import requireBaseExtension  # noqa: E402
requireBaseExtension()     # the PET4D extension's modules (DICOMGatedDynamicPET folder) on sys.path
from DICOMGatedDynamicPETLib import dynamicviews  # noqa: E402
from NeuroPETLib import credits, dbdownload, neuro, normaldb  # noqa: E402
from NeuroPETLib import neuroreview as nr  # noqa: E402
from NeuroPETLib import ssp  # noqa: E402
from NeuroPETLib import staticneuro as sn  # noqa: E402
from NeuroPETLib.banner import MergenBanner  # noqa: E402

# application settings keys: kept from the PET4D extension (the neuro modules came from it), so the folders stay picked
SETTINGS_ROOT = "DICOMGatedDynamicPET/NeuroDatabase/RootFolder"
SETTINGS_BATCH_FOLDER = "DICOMGatedDynamicPET/NeuroDatabase/BatchFolder"
COLUMNS = (("name", "Name"), ("kind", "Type"), ("modality", "Modality"), ("nuclide", "Nuclide"), ("tracer", "Tracer"),
           ("subjects", "Subjects"), ("reference", "Reference region"), ("ageGroups", "Age groups"),
           ("statistics", "Mean / SD"), ("version", "Version"), ("created", "Created"))
DEFAULT_SMOOTHING_MM = 8.0


class NeuroPETDatabaseManager(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Mergen - Neuro PET - Database Manager"     # PET and SPECT
        parent.categories = ["Nuclear Medicine"]
        parent.dependencies = ["DynamicNeuroPET", "StaticNeuroPET"]
        parent.contributors = ["Burak Demir"]
        parent.helpText = (
            "Normal databases of brain PET and SPECT in a template space (default MNI ICBM 152 2009c). Pick the "
            "database folder; each "
            "database is a folder in it with its header (database.json), the reference region mask, the subjects (user "
            "defined databases) and the mean / SD maps (master and per age group). <b>Create new database</b> makes "
            "an empty user defined database, <b>Add predefined database</b> one from a package (.zip) or from mean / "
            "SD maps. A user defined database can be exported as a predefined one.")
        parent.acknowledgementText = (
            "Part of Mergen (Neuro PET/SPECT Analysis), built on the PET4D (DICOM Gated / Dynamic PET) extension. "
            "MNI ICBM 152 2009c: Fonov VS et al., NeuroImage 2011;"
            "54:313-327. CerebrA: Manera AL et al., Scientific Data 2020;7:237 (CC0).")


def _ijkToRasOf(node):
    matrix = vtk.vtkMatrix4x4()
    node.GetIJKToRASMatrix(matrix)
    return np.array([[matrix.GetElement(r, c) for c in range(4)] for r in range(4)])


def _button(text, role="action"):
    return dynamicviews.colorButton(qt.QPushButton(text), role)


def _number(text):
    text = str(text or "").strip().replace(",", ".")
    if not text:
        return None
    return float(text)


def _pathEdit(dirs=False, filters="Images (*.nii *.nii.gz *.nrrd *.nhdr *.mha *.mhd)"):
    edit = ctk.ctkPathLineEdit()
    edit.filters = ctk.ctkPathLineEdit.Dirs if dirs else ctk.ctkPathLineEdit.Files
    if not dirs:
        edit.nameFilters = [filters, "All files (*)"]
    edit.settingKey = ""
    return edit


# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------
class NeuroPETDatabaseManagerLogic(ScriptedLoadableModuleLogic):
    """The database folder and the reference masks; the files themselves through normaldb (no Slicer)."""

    @staticmethod
    def rootFolder():
        """The picked database folder, else the default one (~/NeuroPET/Databases)."""
        return str(qt.QSettings().value(SETTINGS_ROOT) or "") or normaldb.defaultRootFolder()

    @staticmethod
    def ensureRootFolder():
        """With no folder picked yet: the default folder made and kept in the settings. Returns the folder."""
        if not str(qt.QSettings().value(SETTINGS_ROOT) or ""):
            folder = normaldb.defaultRootFolder()
            try:
                os.makedirs(folder, exist_ok=True)
            except OSError as e:
                logging.warning(f"Could not make the default database folder {folder}: {e}")
                return folder
            qt.QSettings().setValue(SETTINGS_ROOT, folder)
        return NeuroPETDatabaseManagerLogic.rootFolder()

    @staticmethod
    def setRootFolder(folder):
        qt.QSettings().setValue(SETTINGS_ROOT, folder or "")

    @staticmethod
    def maskSourceProblem(regionKey, header=None):
        """Why the mask of a reference region cannot be made from the atlas library, or None: the whole brain needs
        the brain mask of the database's space (the region atlas regions: checked when the mask is made)."""
        if regionKey != normaldb.REF_GLOBAL:
            return None
        space = (header or {}).get("space")
        if neuro.BRAIN_MASK not in neuro.atlasFiles(space):
            return f"The space {neuro.spaceTitle(space)} has no brain mask (mask.nrrd in its folder)"
        return None

    def writeReferenceMask(self, folder, header, customPath=None):
        """Write the reference region mask of a database into its folder (reference_mask.nii.gz, or the custom
        file's format) and set header["referenceMask"]. Global: the brain mask of its space; a region of its region
        atlas: that region, both sides; custom: a copy of the file."""
        import shutil
        key = header.get("referenceRegion")
        if key == normaldb.REF_CUSTOM:
            if not customPath or not os.path.isfile(customPath):
                raise ValueError("Pick the custom reference mask file")
            relative = "reference_mask" + (normaldb.imageExtension(customPath) or ".nii.gz")
            shutil.copyfile(customPath, os.path.join(folder, relative))
            header["referenceMask"] = relative
            return relative
        problem = self.maskSourceProblem(key, header)
        if problem:
            raise ValueError(problem)
        if key == normaldb.REF_ATLAS:
            return self._writeAtlasRegionMask(folder, header)
        if key != normaldb.REF_GLOBAL:
            raise ValueError(f"Unknown reference region '{key}'")
        source = slicer.util.loadLabelVolume(neuro.atlasFiles(header.get("space"))[neuro.BRAIN_MASK], {"show": False})
        return self._saveReferenceMask(folder, header, source, slicer.util.arrayFromVolume(source) > 0)

    def _saveReferenceMask(self, folder, header, source, mask):
        """The reference mask (eroded by the header's erosion) saved on the grid of source; source removed."""
        target = None
        try:
            mask = neuro.erodeMask(mask, float(header.get("referenceErosionMm") or 0),
                                   tuple(float(x) for x in source.GetSpacing())[::-1])
            if not mask.any():
                raise ValueError("The reference mask is empty (eroded too much?)")
            target = slicer.modules.volumes.logic().CreateAndAddLabelVolume(slicer.mrmlScene, source,
                                                                           "reference mask")
            slicer.util.updateVolumeFromArray(target, mask.astype(np.uint8))
            relative = normaldb.REFERENCE_MASK_NAME
            if not slicer.util.saveNode(target, os.path.join(folder, relative)):
                raise ValueError("The reference mask could not be saved")
        finally:
            for node in (source, target):
                if node is not None and slicer.mrmlScene.IsNodePresent(node):
                    slicer.mrmlScene.RemoveNode(node)
        header["referenceMask"] = relative
        return relative

    def _writeAtlasRegionMask(self, folder, header):
        """Reference mask of a region of the database's own region atlas (header regionAtlas), both sides
        (header referenceStructure: 'Cerebellum Gray Matter' takes 'Cerebellum Gray Matter R' and '... L')."""
        from DynamicNeuroPET import DynamicNeuroPETLogic
        name = str(header.get("referenceStructure") or "").strip()
        if neuro.isWhiteMatterReference(name):      # of the space, not of the region atlas
            mask, labelsPath = DynamicNeuroPETLogic().whiteMatterReferenceMask(neuro.atlasFiles(header.get("space")))
            source = slicer.util.loadLabelVolume(labelsPath, {"show": False})
            return self._saveReferenceMask(folder, header, source, mask)
        kind = neuro.mergedRegionAtlas(header.get("regionAtlas") or normaldb.DEFAULT_REGION_ATLAS)
        labelsPath, entries = DynamicNeuroPETLogic().regionAtlasPaths(kind)
        values = neuro.referenceRegionValues(entries, name, neuro.referenceGroups(kind))
        if not values:
            raise ValueError(f"The region atlas ({normaldb.regionAtlasTitle(kind)}) has no region '{name}'")
        source = slicer.util.loadLabelVolume(labelsPath, {"show": False})
        array = np.rint(slicer.util.arrayFromVolume(source)).astype(np.int64)
        return self._saveReferenceMask(folder, header, source, np.isin(array, values))

    def editDatabase(self, folder, root, header, changes, customMask=None, progress=None):
        """Save edited settings (normaldb.editedHeader): a new reference region makes the reference mask (user
        defined: and the subjects' SUVR maps) again, a new name renames the folder. Returns the folder."""
        self.editedFolder = folder         # where the database is, also when this fails half way
        if changes.get("name"):            # first: a failure below leaves the folder named as its header
            normaldb.writeHeader(folder, header)
            folder = self.editedFolder = normaldb.renameDatabase(folder, root, header["name"])
        if changes.get("reference") and header.get("kind") == normaldb.USER_DEFINED:
            sameCustom = (header.get("referenceRegion") == normaldb.REF_CUSTOM and not customMask
                          and header.get("referenceMask"))
            if not sameCustom:
                self.writeReferenceMask(folder, header, customMask)
            normaldb.writeHeader(folder, header)
            self.recomputeRatios(folder, header, progress)
        else:
            normaldb.writeHeader(folder, header)
        return folder

    @staticmethod
    def recomputeRatios(folder, header, progress=None):
        """The SUVR maps of the subjects of a user defined database again from their SUV maps and its reference
        mask (after the reference region changed); their reference mean and region in subjects.csv too."""
        from NeuroPETLib import staticneuro as sn
        subjects = normaldb.readSubjects(folder)
        if not subjects:
            return
        maskPath = sn.referenceMaskPath(folder, header)
        if not maskPath:
            raise ValueError("The database has no reference mask.")
        maskNode = slicer.util.loadLabelVolume(maskPath, {"show": False})
        try:
            mask = slicer.util.arrayFromVolume(maskNode) > 0
        finally:
            slicer.mrmlScene.RemoveNode(maskNode)
        title = normaldb.referenceTitle(header)
        problems = []
        for index, subject in enumerate(subjects):
            if progress is not None:
                progress(index, len(subjects), f"SUVR map of {subject.get('id')}")
            suvPath = os.path.join(folder, subject.get("suvFile") or "")
            if not subject.get("suvFile") or not os.path.isfile(suvPath):
                problems.append(f"{subject.get('id')}: no SUV map")
                continue
            node = slicer.util.loadVolume(suvPath, {"show": False})
            try:
                ratio, mean = sn.ratioMap(slicer.util.arrayFromVolume(node), mask)
                slicer.util.updateVolumeFromArray(node, ratio)
                suvrFile = subject.get("suvrFile") or normaldb.subjectFileNames(subject.get("id"))[1]
                if not slicer.util.saveNode(node, os.path.join(folder, suvrFile)):
                    raise ValueError("could not write the SUVR map")
                subject.update(suvrFile=suvrFile, referenceMean=f"{mean:.6g}", reference=title)
            except Exception as error:
                problems.append(f"{subject.get('id')}: {error}")
            finally:
                slicer.mrmlScene.RemoveNode(node)
        normaldb.writeSubjects(folder, subjects)
        if problems:
            raise ValueError("Some SUVR maps were not made again:\n" + "\n".join(problems[:20]))

    @staticmethod
    def forgetLoaded(folder):
        """Remove the scene's cached copies of a database's files (template, mean maps) loaded by the Neuro workflows,
        so they are read again after they changed."""
        from DynamicNeuroPET import ATLAS_FILE_ATTRIBUTE
        root = os.path.normcase(os.path.realpath(folder)) + os.sep
        for node in list(slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")):
            path = node.GetAttribute(ATLAS_FILE_ATTRIBUTE)
            if path and os.path.normcase(os.path.realpath(path)).startswith(root):
                slicer.mrmlScene.RemoveNode(node)

    def computeStatistics(self, folder, progress=None):
        """The mean / SD maps of a user defined database from its subjects' maps (SUVR or SUV by the map type),
        each smoothed with the database's FWHM: the master (all subjects) and, when age stratified, each age group
        with at least two subjects. Written to stats/; the header's statistics, version and age summary updated.
        progress(index, total, text) returns False to stop (nothing written). Returns (header, notes)."""
        header = normaldb.readHeader(folder)
        if header.get("kind") != normaldb.USER_DEFINED:
            raise ValueError("The mean / SD maps are computed for user defined databases only.")
        subjects = normaldb.readSubjects(folder)
        if len(subjects) < 2:
            raise ValueError("The mean / SD maps need at least two subjects.")
        mapKey = normaldb.subjectMapKey(header)
        groups = normaldb.statisticsGroups(header, subjects)
        notes = [f"Age group '{g['name']}': {len(g['ids'])} subject(s), no mean / SD maps (2 needed)."
                 for k, g in groups.items() if k != normaldb.MASTER and len(g["ids"]) < 2]
        groups = {k: g for k, g in groups.items() if len(g["ids"]) >= 2}
        accumulators = {k: normaldb.RunningStatistics() for k in groups}
        membership = {}
        for key, group in groups.items():
            for subjectId in group["ids"]:
                membership.setdefault(subjectId, []).append(key)
        fwhm = float(header.get("smoothingFwhmMm") or 0)
        masks, surface, regionNote = self._regionMasks(folder, header)
        if regionNote:
            notes.append(regionNote)
        regionValues = {}
        sspAccumulators = {k: normaldb.RunningStatistics() for k in groups} if surface is not None else None
        # the brain the 3D-SSP rays stop at: the union of the regions, on the grid of the subjects' maps
        brain = nr.unionMask(masks, next(iter(masks.values())).shape) if masks and surface is not None else None
        geometry = None
        for index, subject in enumerate(subjects):
            if progress is not None and not progress(index, len(subjects), f"{subject['id']} ({index + 1} / "
                                                                           f"{len(subjects)})"):
                self._dropPendingRegions(folder)
                return None, ["Stopped: nothing written."]
            relative = subject.get(mapKey)
            path = os.path.join(folder, relative) if relative else ""
            if not path or not os.path.isfile(path):
                raise ValueError(f"Subject '{subject['id']}' has no {mapKey[:-4].upper()} map ({relative}).")
            node = slicer.util.loadVolume(path, {"show": False})
            if node is None:
                raise ValueError(f"Could not load {path}")
            try:
                matrix = vtk.vtkMatrix4x4()
                node.GetIJKToRASMatrix(matrix)
                if geometry is None:
                    geometry = matrix
                if masks:
                    try:
                        regionValues[subject["id"]] = nr.regionMeans(slicer.util.arrayFromVolume(node), masks)
                    except ValueError as error:
                        notes.append(f"No regional statistics: {error}")
                        masks = None
                smoothed = normaldb.smoothMap(slicer.util.arrayFromVolume(node), fwhm,
                                              tuple(float(x) for x in node.GetSpacing())[::-1])
                for key in membership.get(subject["id"], []):
                    accumulators[key].add(smoothed)
                if sspAccumulators is not None:      # 3D-SSP: the smoothed map projected on the regions' surface
                    if brain is not None and brain.shape != smoothed.shape:
                        notes.append(f"No 3D-SSP projections: subject '{subject['id']}' is not on the grid of the "
                                     f"regions.")
                        sspAccumulators = None
                    else:
                        projected = ssp.project(smoothed, _ijkToRasOf(node), surface["points"], surface["normals"],
                                                mask=brain)
                        for key in membership.get(subject["id"], []):
                            sspAccumulators[key].add(projected)
            finally:
                slicer.mrmlScene.RemoveNode(node)
        # the maps are written beside the old ones first (a failure leaves the database as it was), then the
        # header is marked out of date while they replace the old ones, then written with the new statistics: a
        # database stopped half way says its maps are out of date instead of mixing old and new maps silently
        output = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "normal database map")
        written, pendingFiles = {}, []
        try:
            output.SetIJKToRASMatrix(geometry)
            for key, group in groups.items():
                files = {}
                for which, array in (("mean", accumulators[key].mean()), ("sd", accumulators[key].sd())):
                    relative = normaldb.statsFileName(key, which)
                    pending = normaldb.statsFileName(key, which, ".pending.nii.gz")
                    slicer.util.updateVolumeFromArray(output, array)
                    if not slicer.util.saveNode(output, os.path.join(folder, pending)):
                        raise ValueError(f"Could not write {relative}")
                    pendingFiles.append((pending, relative))
                    files[which] = relative
                ages = [s.get("age") for s in subjects if s["id"] in set(group["ids"])]
                written[key] = {"name": group["name"], "n": accumulators[key].n, **files,
                                "ages": normaldb.ageSummary(ages)}
        except Exception:
            for pending, _ in pendingFiles:
                if os.path.isfile(os.path.join(folder, pending)):
                    os.remove(os.path.join(folder, pending))
            raise
        finally:
            slicer.mrmlScene.RemoveNode(output)
        old = header.get("statistics") or {}
        if old.get("computed") and not old.get("stale"):
            normaldb.writeHeader(folder, dict(header, statistics=dict(old, stale=True)), touch=False)
        for pending, relative in pendingFiles:
            os.replace(os.path.join(folder, pending), os.path.join(folder, relative))
        for key, entry in (old.get("groups") or {}).items():
            if key not in written:
                for which in ("mean", "sd"):
                    path = os.path.join(folder, entry.get(which) or "")
                    if entry.get(which) and os.path.isfile(path):
                        os.remove(path)
        if old.get("computed"):
            header["version"] = int(header.get("version") or 1) + 1
        header["statistics"] = {"computed": normaldb.now(), "stale": False, "groups": written,
                                "smoothingFwhmMm": fwhm, "maps": mapKey}
        regionsFile = os.path.join(folder, nr.REGIONS_FILE)
        if masks and regionValues:
            statistics = nr.regionStatistics(regionValues, groups)
            settings = {k: v for k, v in sn.regionSettings(folder, header).items() if k != "referenceMaskPath"}
            nr.writeJson(regionsFile, nr.regionsDocument(statistics, mapKey, header.get("version", 1), settings,
                                                         nr.masksSignature(masks)))
            header["statistics"]["regions"] = nr.REGIONS_FILE
        elif os.path.isfile(regionsFile):
            os.remove(regionsFile)
        # the regions themselves (a segmentation), for the Regions step of everyone using the database
        pending = os.path.join(folder, nr.REGIONS_SEGMENTATION_PENDING)
        stored = os.path.join(folder, nr.REGIONS_SEGMENTATION_FILE)
        if header["statistics"].get("regions") and os.path.isfile(pending):
            os.replace(pending, stored)
            header["statistics"]["regionsSegmentation"] = nr.REGIONS_SEGMENTATION_FILE
            header["statistics"]["regionsReference"] = getattr(self, "_regionsReference", None) or ""
        else:
            self._dropPendingRegions(folder)
            if os.path.isfile(stored):
                os.remove(stored)
            if masks:
                notes.append("The regions were not stored with the statistics (the segmentation could not be saved): "
                             "they are drawn from the atlas folder.")
        sspFile = os.path.join(folder, ssp.SSP_FILE)
        if sspAccumulators is not None:
            sspGroups = {k: {"mean": a.mean(), "sd": a.sd(), "n": a.n, "name": groups[k]["name"]}
                         for k, a in sspAccumulators.items() if a.n >= 2}
            ssp.write(sspFile, surface, sspGroups, {"depthMm": ssp.DEPTH_MM, "stepMm": ssp.STEP_MM,
                                                    "clipToBrain": True, "surface": ssp.SURFACE_VERSION,
                                                    "fwhmMm": fwhm,
                                                    "maps": mapKey, "version": header.get("version", 1)})
            header["statistics"]["ssp"] = ssp.SSP_FILE
        elif os.path.isfile(sspFile):
            os.remove(sspFile)
        header["statistics"]["anatomy"] = self._writeAnatomyMeans(folder, header, subjects, old.get("anatomy"))
        header["subjectAges"] = normaldb.ageSummary(s.get("age") for s in subjects)
        normaldb.writeHeader(folder, header)
        self.forgetLoaded(folder)
        return header, notes

    @staticmethod
    def _writeAnatomyMeans(folder, header, subjects, old=None):
        """The mean CT / MR of each contrast (the subjects that have one, at least two; unsmoothed, an MR scaled to its
        brain median) written to stats/anatomy_<contrast>_mean.nii.gz; those of contrasts gone removed. Returns
        {contrast: {"mean", "n"}}."""
        from StaticNeuroPET import StaticNeuroPETLogic
        atlas = neuro.atlasFiles(header.get("space"))
        mask = None
        if neuro.BRAIN_MASK in atlas:
            mask = StaticNeuroPETLogic.loadAtlasVolume(atlas[neuro.BRAIN_MASK],
                                                       name=f"{neuro.spaceTitle(header.get('space'))} brain mask")
        written = {}
        for contrast, (array, geometry, n) in StaticNeuroPETLogic._meanAnatomyMaps(
                folder, header, [s["id"] for s in subjects], mask).items():
            relative = normaldb.anatomyStatsFileName(contrast)
            StaticNeuroPETLogic._saveArray(array, geometry, os.path.join(folder, relative))
            written[contrast] = {"mean": relative, "n": n}
        for contrast, entry in (old or {}).items():
            relative = (entry or {}).get("mean")
            if relative and contrast not in written and os.path.isfile(os.path.join(folder, relative)):
                os.remove(os.path.join(folder, relative))
        return written

    @staticmethod
    def _dropPendingRegions(folder):
        pending = os.path.join(folder, nr.REGIONS_SEGMENTATION_PENDING)
        if os.path.isfile(pending):
            try:
                os.remove(pending)
            except OSError:
                logging.warning(f"Neuro DB: could not remove {pending}", exc_info=True)

    def _regionMasks(self, folder, header):
        """(masks {region: boolean array} on the subjects' grid, 3D-SSP surface of their union, note) of the
        database's regions (its region settings: region atlas of the atlas library) for the regional statistics and
        the normals' surface projections; (None, None, why) when they cannot be made."""
        from StaticNeuroPET import StaticNeuroPETLogic
        atlas = neuro.atlasFiles(header.get("space"))
        key = header.get("regionAtlas") or normaldb.DEFAULT_REGION_ATLAS
        if neuro.segmentationInfo(neuro.splitRegionAtlas(key)[0]) is None:
            return None, None, (f"No regional statistics: the segmentation of the region atlas "
                                f"({neuro.splitRegionAtlas(key)[0]}) is not in the atlas folders.")
        gridPath = sn.referenceMaskPath(folder, header) or atlas.get(neuro.BRAIN_MASK)
        if not gridPath:
            return None, None, "No regional statistics: no reference mask or brain mask of the space for the grid."
        logic = StaticNeuroPETLogic()
        gridNode = slicer.util.loadLabelVolume(gridPath, {"show": False})
        try:
            try:
                self._regionsReference = None
                self._dropPendingRegions(folder)
                masks = logic.databaseRegionMasks(atlas, folder, header, gridNode, savePath=os.path.join(
                    folder, nr.REGIONS_SEGMENTATION_PENDING))
                self._regionsReference = logic.savedRegionsReference
            except Exception as error:
                logging.exception("Neuro DB: could not make the regions")
                return None, None, f"No regional statistics: the regions could not be made ({error})."
            if not masks:
                return None, None, "No regional statistics: the regions are empty."
            try:
                shape = next(iter(masks.values())).shape
                # the regions with every CerebrA structure (not hollow when the region atlas leaves some out)
                hull = logic.brainHull(nr.unionMask(masks, shape), atlas, gridNode)
                surface = logic.reviewSurface(hull, gridNode, hemispheres=ssp.hemisphereMasks(masks, shape))
            except Exception as error:
                logging.exception("Neuro DB: could not make the 3D-SSP surface")
                return masks, None, f"No 3D-SSP projections: the surface could not be made ({error})."
            return masks, surface, None
        finally:
            slicer.mrmlScene.RemoveNode(gridNode)

    def createUserDefined(self, root, header, customMask=None):
        """New empty user defined database with its reference mask. Returns the folder."""
        import shutil
        folder = normaldb.createDatabase(root, header)
        try:
            self.writeReferenceMask(folder, header, customMask)
            normaldb.writeHeader(folder, header)
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        return folder

    @staticmethod
    def installPackage(root, asset, tag, folder=None, progress=None):
        """Download a package of the release (dbdownload) into root and install it as a predefined database, in place
        of folder when given (an update). progress(done, total) may return False to cancel. Returns the folder."""
        os.makedirs(root, exist_ok=True)
        path = os.path.join(root, ".download-" + normaldb.safeName(asset["name"]))
        try:
            dbdownload.download(asset, path, progress)
            source = dbdownload.sourceInfo(asset, tag)
            if folder:
                return normaldb.replacePredefined(root, path, folder, source)
            return normaldb.importPackage(root, path, origin=source)
        finally:
            for leftover in (path, path + ".part"):
                if os.path.exists(leftover):
                    os.remove(leftover)

    def createPredefined(self, root, header, maps, maskPath=None):
        """Predefined database from mean / SD maps; the reference mask is the file given, else made from the atlas
        library when it can be (else none)."""
        folder = normaldb.createPredefined(root, header, maps, maskPath)
        if not maskPath and header.get("referenceRegion") != normaldb.REF_CUSTOM \
                and self.maskSourceProblem(header.get("referenceRegion"), header) is None:
            saved = normaldb.readHeader(folder)
            try:
                self.writeReferenceMask(folder, saved)
                normaldb.writeHeader(folder, saved)
            except Exception as e:
                logging.warning(f"Reference mask of {folder} not made: {e}")
        return folder


# ---------------------------------------------------------------------------
# Header form (shared by the dialogs)
# ---------------------------------------------------------------------------
class HeaderForm:
    """Fields of a database header: name, nuclide, tracer, acquisition, template space, region atlas (segmentation and
    grouping), reference region, smoothing, map type (predefined), scanner and description, age groups (with maps and
    subjects for a predefined one)."""

    def __init__(self, layout, predefined=False):
        self.predefined = predefined
        self.widget = qt.QWidget()
        layout.addWidget(self.widget)
        outer = qt.QVBoxLayout(self.widget)
        outer.setContentsMargins(0, 0, 0, 0)

        box = qt.QGroupBox("Database")
        form = qt.QFormLayout(box)
        self.nameEdit = qt.QLineEdit()
        self.nameEdit.placeholderText = "e.g. FDG normals 2026"
        self.nameEdit.toolTip = "The database folder is named after it."
        form.addRow("Name:", self.nameEdit)
        self.modalityCombo = qt.QComboBox()
        self.modalityCombo.addItems(list(normaldb.MODALITIES))
        self.modalityCombo.toolTip = "PET (SUV / SUVR maps) or SPECT (counts / ratio maps)."
        form.addRow("Modality:", self.modalityCombo)
        self.nuclideCombo = qt.QComboBox()
        self.nuclideCombo.editable = True
        self.nuclideCombo.toolTip = "Nuclides of the modality; another one can be typed."
        form.addRow("Nuclide:", self.nuclideCombo)
        self.tracerCombo = qt.QComboBox()
        self.tracerCombo.editable = True
        self.tracerCombo.toolTip = ("Tracers of the nuclide. The nuclide itself means the tracer is not given; "
                                    "another tracer can be typed.")
        form.addRow("Tracer:", self.tracerCombo)
        acquisition = qt.QHBoxLayout()
        self.startSpin = qt.QDoubleSpinBox()
        self.startSpin.setRange(0, 100000)
        self.startSpin.setDecimals(1)
        self.startSpin.suffix = " min p.i."
        self.startSpin.value = 60
        self.durationSpin = qt.QDoubleSpinBox()
        self.durationSpin.setRange(0, 100000)
        self.durationSpin.setDecimals(1)
        self.durationSpin.suffix = " min"
        self.durationSpin.value = 10
        acquisition.addWidget(qt.QLabel("start"))
        acquisition.addWidget(self.startSpin)
        acquisition.addWidget(qt.QLabel("duration"))
        acquisition.addWidget(self.durationSpin)
        form.addRow("Acquisition:", acquisition)
        spaceRow = qt.QHBoxLayout()
        self.spaceCombo = qt.QComboBox()
        for space in neuro.listSpaces():
            self.spaceCombo.addItem(space["name"], space["id"])
        self.spaceCombo.toolTip = ("Template space of the database (the extension's and those of your atlas folder, "
                                   "~/NeuroPET/Atlas/Space): the subjects are normalized to it and the region atlas is "
                                   "one of its segmentations. Fixed once the database is made.")
        spaceRow.addWidget(self.spaceCombo, 1)
        openAtlas = qt.QPushButton("Open atlas folder")
        openAtlas.toolTip = ("Open your atlas folder (~/NeuroPET/Atlas) in the file browser: add a space "
                             "(Space/<name>) or a segmentation (Segmentation/<name>) there.")
        openAtlas.connect("clicked()", self._openAtlasFolder)
        spaceRow.addWidget(openAtlas)
        form.addRow("Space:", spaceRow)
        self.spaceInfo = qt.QLabel()
        self.spaceInfo.wordWrap = True
        form.addRow(self.spaceInfo)
        outer.addWidget(box)

        refBox = qt.QGroupBox("Normalization")
        refForm = qt.QFormLayout(refBox)
        self.segmentationCombo = qt.QComboBox()
        self.segmentationCombo.toolTip = ("Segmentation the regions come from (those of the space: the extension's "
                                          "and those of your atlas folder, ~/NeuroPET/Atlas/Segmentation).")
        refForm.addRow("Segmentation:", self.segmentationCombo)
        self.groupingCombo = qt.QComboBox()
        self.groupingCombo.toolTip = ("Regions the patients are reported in (the Neuro workflows use the same), "
                                      "left and right separate: the segmentation's regions, its broad regions or "
                                      "Broad (Compatible). The reference region is one of its regions.")
        refForm.addRow("Grouping:", self.groupingCombo)
        self.referenceCombo = qt.QComboBox()
        self.referenceCombo.toolTip = ("Region the images are divided by (SUVR): the whole brain, a region of the "
                                       "segmentation and grouping above (both sides together), the centrum semiovale "
                                       "(white matter above the lateral ventricles, from the space's white matter "
                                       "map: for tracers binding in all grey matter, e.g. [11C]UCB-J) or a custom mask "
                                       "file. Its mask is copied into the database.")
        refForm.addRow("Reference region:", self.referenceCombo)
        self._fillSegmentations()
        self.maskPath = _pathEdit()
        self.maskPath.toolTip = ("Reference mask in the template space. Custom region: required. Otherwise made from "
                                 "the region atlas (or the space's brain mask)"
                                 + (", or this file when given." if predefined else "."))
        self.maskLabel = qt.QLabel("Mask file:")
        refForm.addRow(self.maskLabel, self.maskPath)
        self.smoothingSpin = qt.QDoubleSpinBox()
        self.smoothingSpin.setRange(0, 30)
        self.smoothingSpin.setDecimals(1)
        self.smoothingSpin.suffix = " mm FWHM"
        self.smoothingSpin.specialValueText = "Off"
        self.smoothingSpin.value = DEFAULT_SMOOTHING_MM
        self.smoothingSpin.toolTip = ("Gaussian smoothing of the maps before the mean / SD" if not predefined
                                      else "Smoothing applied to the maps of the database (for the record).")
        refForm.addRow("Smoothing:", self.smoothingSpin)
        self.erosionSpin = qt.QDoubleSpinBox()
        self.erosionSpin.setRange(0, 10)
        self.erosionSpin.setDecimals(1)
        self.erosionSpin.singleStep = 0.5
        self.erosionSpin.suffix = " mm"
        self.erosionSpin.specialValueText = "Off"
        self.erosionSpin.toolTip = ("Erode the reference mask made from the atlas (not a custom mask file), so "
                                    "spill-over from the neighbouring tissue does not reach it.")
        refForm.addRow("Erode reference:", self.erosionSpin)
        # the atlas of the regions and of the reference: references to cite and license (read only)
        self.atlasCredits = credits.infoBox()
        refForm.addRow(self.atlasCredits)
        self.growSpin = qt.QDoubleSpinBox()
        self.growSpin.setRange(0, 10)
        self.growSpin.setDecimals(1)
        self.growSpin.singleStep = 0.5
        self.growSpin.suffix = " mm"
        self.growSpin.specialValueText = "Off"
        self.growSpin.toolTip = ("Grow the regions into the unlabelled voxels around them (each voxel to the nearest "
                                 "region, inside the brain mask), for the PET / SPECT resolution.")
        refForm.addRow("Grow regions:", self.growSpin)
        self.growIntoCombo = qt.QComboBox()
        for key, title in normaldb.GROW_INTO:
            self.growIntoCombo.addItem(title, key)
        self.growIntoCombo.toolTip = ("Where the grown regions may go (inside the brain mask): CSF and the cortex edge "
                                      "only, white matter too, or white matter only (no outward growth into CSF).")
        self.growIntoCombo.enabled = False
        self.growSpin.connect("valueChanged(double)", lambda v: setattr(self.growIntoCombo, "enabled", v > 0))
        refForm.addRow("Grow into:", self.growIntoCombo)
        if predefined:
            self.mapTypeCombo = qt.QComboBox()
            self.mapTypeCombo.toolTip = ("What the mean / SD maps hold: PET SUVR (divided by the reference) or SUV, "
                                         "SPECT ratio (counts divided by the reference) or counts.")
            refForm.addRow("Map type:", self.mapTypeCombo)
        outer.addWidget(refBox)

        normBox = qt.QGroupBox("Spatial normalization (to the space)")
        normBox.toolTip = ("How the subjects are normalized to the template space. The Neuro workflows take it as the "
                           "default when this database is picked and warn when a patient is normalized otherwise "
                           "(never blocked). Left 'Not specified', the first subject added sets it.")
        normForm = qt.QFormLayout(normBox)

        def choiceCombo(items):
            combo = qt.QComboBox()
            for key, title in items:
                combo.addItem(title, key)
            return combo

        self.normMethodCombo = choiceCombo(normaldb.NORM_METHODS)
        normForm.addRow("Method:", self.normMethodCombo)
        self.databaseVolumes, self.databaseName = {}, ""        # the database edited: its own ('From DB') volumes
        self.normTemplateCombo = qt.QComboBox()
        self.normTemplateCombo.toolTip = ("The volume registered to, by the method: the space's templates, then the "
                                          "database's own (From DB: its PET template, its CT / MR means). For the "
                                          "CT/MR method a subject of another contrast takes the space's template of "
                                          "its contrast.")
        normForm.addRow("Volume:", self.normTemplateCombo)
        self.templateCredits = credits.infoBox()
        normForm.addRow(self.templateCredits)
        self.normEngineCombo = choiceCombo(normaldb.NORM_ENGINES)
        normForm.addRow("Registration:", self.normEngineCombo)
        self.normLinearCombo = choiceCombo(normaldb.NORM_LINEAR)
        normForm.addRow("Linear stage:", self.normLinearCombo)
        yesNo = (("", "Not specified"), ("yes", "Yes"), ("no", "No"))
        self.normDeformableCombo = choiceCombo(yesNo)
        normForm.addRow("Deformable stage:", self.normDeformableCombo)
        self.normMaskCombo = choiceCombo(yesNo)
        normForm.addRow("Skull-stripped template:", self.normMaskCombo)
        self.normMethodCombo.connect("currentIndexChanged(int)", lambda i: self._updateNormTemplate())
        self._updateNormTemplate()
        outer.addWidget(normBox)

        ageBox = qt.QGroupBox("Age groups" if not predefined else "Mean / SD maps")
        ageLayout = qt.QVBoxLayout(ageBox)
        self.ageCheck = qt.QCheckBox("Stratify by age")
        self.ageCheck.toolTip = ("Separate mean / SD maps per age group (ages in years, both ends inclusive, "
                                 "empty max: open). The master mean / SD of all subjects is always made.")
        ageLayout.addWidget(self.ageCheck)
        headers = ["Group", "Min age", "Max age"]
        if predefined:
            headers += ["Subjects", "Mean map", "SD map"]
        self.ageTable = qt.QTableWidget(0, len(headers))
        self.ageTable.setHorizontalHeaderLabels(headers)
        self.ageTable.verticalHeader().visible = False
        self.ageTable.horizontalHeader().setStretchLastSection(True)
        self.ageTable.minimumHeight = 140
        ageLayout.addWidget(self.ageTable)
        rowButtons = qt.QHBoxLayout()
        self.addGroupButton = qt.QPushButton("Add group")
        self.removeGroupButton = qt.QPushButton("Remove group")
        rowButtons.addWidget(self.addGroupButton)
        rowButtons.addWidget(self.removeGroupButton)
        rowButtons.addStretch(1)
        ageLayout.addLayout(rowButtons)
        outer.addWidget(ageBox)

        textBox = qt.QGroupBox("Notes")
        textForm = qt.QFormLayout(textBox)
        self.scannerEdit = qt.QLineEdit()
        self.scannerEdit.placeholderText = "scanner, reconstruction, ..."
        textForm.addRow("Scanner:", self.scannerEdit)
        self.descriptionEdit = qt.QPlainTextEdit()
        self.descriptionEdit.placeholderText = ("Description, source of the subjects, how the database was made, "
                                                "references to cite, license of the data, ...")
        self.descriptionEdit.toolTip = ("Shown (with the references and licenses of the atlases) wherever the "
                                        "database is picked, and kept in its exported package.")
        self.descriptionEdit.maximumHeight = 160
        textForm.addRow("Description:", self.descriptionEdit)
        outer.addWidget(textBox)

        if predefined:
            self._addRow(normaldb.MASTER_TITLE, master=True)
        self.addGroupButton.connect("clicked()", lambda: self._addRow())
        self.removeGroupButton.connect("clicked()", self._removeRow)
        self.ageCheck.connect("toggled(bool)", lambda on: self._updateAge())
        self.referenceCombo.connect("currentIndexChanged(int)", lambda i: self._updateReference())
        self.spaceCombo.connect("currentIndexChanged(int)", lambda i: self._fillSegmentations())
        self.segmentationCombo.connect("currentIndexChanged(int)", lambda i: self._fillGroupings())
        self.groupingCombo.connect("currentIndexChanged(int)", lambda i: self._fillReferences())
        for combo in (self.referenceCombo, self.groupingCombo):
            combo.connect("currentIndexChanged(int)", lambda i: self._updateCredits())
        self.normTemplateCombo.connect("currentIndexChanged(int)", lambda i: self._updateCredits())
        self.normMethodCombo.connect("currentIndexChanged(int)", lambda i: self._updateCredits())
        self._updateCredits()
        self._updateAge()
        self._updateReference()
        self.modalityCombo.connect("currentIndexChanged(int)", lambda i: self._updateModality())
        self.nuclideCombo.connect("currentTextChanged(QString)", self._updateTracers)
        self._updateModality()

    # --- modality, nuclide, tracer --------------------------------------------------------------------
    def _updateModality(self):
        modality = self.modalityCombo.currentText
        self.nuclideCombo.blockSignals(True)
        self.nuclideCombo.clear()
        self.nuclideCombo.addItems(normaldb.nuclides(modality))
        self.nuclideCombo.currentIndex = 0
        self.nuclideCombo.blockSignals(False)
        self._updateTracers(self.nuclideCombo.currentText)
        if self.predefined:
            self.mapTypeCombo.clear()
            self.mapTypeCombo.addItems(list(normaldb.mapTypes(modality)))

    def _updateTracers(self, nuclide):
        """The tracers of the nuclide: the nuclide itself (tracer not given) first, then the second one (the usual
        tracer) selected."""
        options = normaldb.tracers(nuclide, self.modalityCombo.currentText)
        self.tracerCombo.clear()
        self.tracerCombo.addItems(options)
        self.tracerCombo.currentIndex = 1 if len(options) > 1 else 0

    # --- age table -----------------------------------------------------------------------------------
    def _firstGroupRow(self):
        return 1 if self.predefined else 0

    def _addRow(self, name="", low="", high="", master=False):
        row = self.ageTable.rowCount
        self.ageTable.insertRow(row)
        item = qt.QTableWidgetItem(name)
        if master:
            item.setFlags(qt.Qt.ItemIsEnabled)
        self.ageTable.setItem(row, 0, item)
        for column, text in ((1, low), (2, high)):
            cell = qt.QTableWidgetItem("" if master else str(text))
            if master:
                cell.setFlags(qt.Qt.ItemIsEnabled)
            self.ageTable.setItem(row, column, cell)
        if self.predefined:
            spin = qt.QSpinBox()
            spin.setRange(0, 1000000)
            self.ageTable.setCellWidget(row, 3, spin)
            for column in (4, 5):
                self.ageTable.setCellWidget(row, column, _pathEdit())
        self.ageTable.resizeColumnsToContents()

    def _removeRow(self):
        row = self.ageTable.currentRow()
        if row >= self._firstGroupRow():
            self.ageTable.removeRow(row)

    def _updateAge(self):
        on = self.ageCheck.checked
        self.addGroupButton.enabled = on
        self.removeGroupButton.enabled = on
        if not self.predefined:
            self.ageTable.enabled = on
        else:
            for row in range(self._firstGroupRow(), self.ageTable.rowCount):
                self.ageTable.setRowHidden(row, not on)
        if on and self.ageTable.rowCount == self._firstGroupRow():
            for name, low, high in (("20-39", 20, 39), ("40-59", 40, 59), ("60+", 60, "")):
                self._addRow(name, low, high)

    def _updateReference(self):
        custom = self.referenceKey() == normaldb.REF_CUSTOM
        self.maskPath.visible = custom or self.predefined
        self.maskLabel.visible = custom or self.predefined
        self.maskLabel.text = "Mask file:" if custom else "Mask file (optional):"

    def referenceKey(self):
        return str(self.referenceCombo.itemData(self.referenceCombo.currentIndex) or "").split(":", 1)[0]

    def referenceStructure(self):
        data = str(self.referenceCombo.itemData(self.referenceCombo.currentIndex) or "")
        return data.split(":", 1)[1] if data.startswith(normaldb.REF_ATLAS + ":") else ""

    def spaceId(self):
        return self.spaceCombo.currentData or normaldb.SPACE

    def regionAtlasKey(self):
        return neuro.regionAtlasKey(self.segmentationCombo.currentData or neuro.DEFAULT_SEGMENTATION,
                                    self.groupingCombo.currentData or neuro.REGIONS)

    def _fillSegmentations(self, keep=None):
        """The segmentations of the space picked (keep: a segmentation id to pick)."""
        info = neuro.spaceInfo(self.spaceId())
        if info:
            volumes = [k for k, _ in neuro.SPACE_VOLUMES if k in info["files"]]
            size = info["manifest"].get("voxelSizeMm")
            size = size[0] if isinstance(size, (list, tuple)) and size else size
            self.spaceInfo.text = ((f"{float(size):g} mm voxels; " if size else "") + "volumes: "
                                   + ", ".join({neuro.BRAIN_MASK: "mask", neuro.CT_TEMPLATE: "CT",
                                                neuro.CT_TEMPLATE_GM: "CT_GM"}.get(k, k) for k in volumes))
        else:
            self.spaceInfo.text = "<span style='color:#c0392b'>No template space in the atlas folders.</span>"
        current = keep or self.segmentationCombo.currentData or neuro.DEFAULT_SEGMENTATION
        combo = self.segmentationCombo
        combo.blockSignals(True)
        combo.clear()
        for segmentation in neuro.listSegmentations(self.spaceId()):
            combo.addItem(segmentation["name"], segmentation["id"])
        index = combo.findData(current)
        combo.setCurrentIndex(index if index >= 0 else 0)
        combo.blockSignals(False)
        self._fillGroupings()

    def _fillGroupings(self, keep=None):
        current = keep or self.groupingCombo.currentData or neuro.REGIONS
        info = neuro.segmentationInfo(self.segmentationCombo.currentData) if self.segmentationCombo.count else None
        combo = self.groupingCombo
        combo.blockSignals(True)
        combo.clear()
        titles = dict(neuro.GROUPINGS)
        for grouping in neuro.segmentationGroupings(info):
            combo.addItem(titles[grouping], grouping)
        index = combo.findData(current)
        combo.setCurrentIndex(index if index >= 0 else 0)
        combo.blockSignals(False)
        self._fillReferences()

    def _atlasRegionNames(self):
        """The region names of the region atlas picked, sides merged ('Putamen')."""
        if not self.segmentationCombo.count:
            return []
        try:
            kind = neuro.mergedRegionAtlas(self.regionAtlasKey())
            names = neuro.referenceRegionNames(neuro.regionTable(kind)[0], neuro.referenceGroups(kind))
            if names and "WM" in neuro.atlasFiles(self.spaceId()):     # of the space: the centrum semiovale
                names.append(neuro.WHITE_MATTER_REFERENCE)
            return names
        except Exception:       # noqa: BLE001 - shown as a list without atlas regions
            logging.warning("Neuro DB: could not read the regions of the region atlas", exc_info=True)
            return []

    def _fillReferences(self, keep=None):
        """The reference regions: whole brain, every region of the region atlas (sides merged), custom mask file;
        keep (item data) or the one picked stays picked when it is still there, else a usual reference region."""
        combo = self.referenceCombo
        if not combo.enabled and keep is None:          # fixed (a predefined database): left as it is
            return
        current = keep if keep is not None else combo.itemData(combo.currentIndex)
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(normaldb.referenceTitle(normaldb.REF_GLOBAL), normaldb.REF_GLOBAL)
        names = self._atlasRegionNames()
        for name in names:
            combo.addItem(name, f"{normaldb.REF_ATLAS}:{name}")
        combo.addItem(normaldb.referenceTitle(normaldb.REF_CUSTOM), normaldb.REF_CUSTOM)
        index = combo.findData(current) if current else -1
        if index < 0:
            preferred = [n for n in normaldb.PREFERRED_REFERENCES if n in names]
            index = combo.findData(f"{normaldb.REF_ATLAS}:{preferred[0]}") if preferred else 0
        combo.setCurrentIndex(max(0, index))
        combo.blockSignals(False)
        if hasattr(self, "maskPath"):           # not while the form is being built
            self._updateReference()
        if hasattr(self, "templateCredits"):
            self._updateCredits()

    @staticmethod
    def _openAtlasFolder():
        from DynamicNeuroPET import DynamicNeuroPETLogic
        DynamicNeuroPETLogic.openAtlasFolder()

    def ageGroups(self):
        groups = []
        for row in range(self._firstGroupRow(), self.ageTable.rowCount):
            def text(column):
                item = self.ageTable.item(row, column)
                return item.text().strip() if item else ""
            try:
                low, high = _number(text(1)), _number(text(2))
            except ValueError:
                low, high = text(1), text(2)        # reported by the validation
            groups.append({"name": text(0), "min": low, "max": high})
        return groups

    def maps(self):
        """{MASTER or age group name: {"mean", "sd", "n"}} of a predefined database."""
        maps = {}
        groups = self.ageGroups() if self.ageCheck.checked else []
        rows = [(0, normaldb.MASTER)] + [(self._firstGroupRow() + i, g["name"]) for i, g in enumerate(groups)]
        for row, name in rows:
            maps[name] = {"n": self.ageTable.cellWidget(row, 3).value,
                          "mean": self.ageTable.cellWidget(row, 4).currentPath,
                          "sd": self.ageTable.cellWidget(row, 5).currentPath}
        return maps

    def setHeader(self, header, fixed=()):
        """The fields from an existing database's header (Edit settings); the keys in fixed are shown but locked."""
        def pick(combo, value, editable=False):
            index = combo.findData(value) if combo.findData(value) >= 0 else combo.findText(str(value or ""))
            if index >= 0:
                combo.setCurrentIndex(index)
            elif editable:
                combo.setEditText(str(value or ""))

        self.nameEdit.text = header.get("name", "")
        pick(self.modalityCombo, header.get("modality"))
        self._updateModality()
        pick(self.nuclideCombo, header.get("nuclide"), editable=True)
        self._updateTracers(self.nuclideCombo.currentText)
        pick(self.tracerCombo, header.get("tracer"), editable=True)
        acquisition = header.get("acquisition") or {}
        self.startSpin.value = float(acquisition.get("startMinutes") or 0)
        self.durationSpin.value = float(acquisition.get("durationMinutes") or 0)

        self.smoothingSpin.value = float(header.get("smoothingFwhmMm") or 0)
        self.erosionSpin.value = float(header.get("referenceErosionMm") or 0)
        space = header.get("space") or normaldb.SPACE
        if self.spaceCombo.findData(space) < 0:      # not in the atlas folders here: shown, its regions not listed
            self.spaceCombo.addItem(f"{space} (not in the atlas folders)", space)
        self.spaceCombo.blockSignals(True)
        pick(self.spaceCombo, space)
        self.spaceCombo.blockSignals(False)
        self.spaceCombo.enabled = False             # fixed once the database is made
        segmentation, grouping, _ = neuro.splitRegionAtlas(header.get("regionAtlas") or normaldb.DEFAULT_REGION_ATLAS)
        self._fillSegmentations(keep=segmentation)
        if self.segmentationCombo.currentData != segmentation:
            self.segmentationCombo.addItem(f"{segmentation} (not in the atlas folders)", segmentation)
            self.segmentationCombo.setCurrentIndex(self.segmentationCombo.count - 1)
        self._fillGroupings(keep=grouping)
        if self.groupingCombo.currentData != grouping:
            self.groupingCombo.addItem(grouping, grouping)
            self.groupingCombo.setCurrentIndex(self.groupingCombo.count - 1)
        reference = header.get("referenceRegion") or ""
        self._fillReferences(keep=f"{reference}:{header.get('referenceStructure') or ''}"
                             if reference == normaldb.REF_ATLAS else reference)
        self.growSpin.value = float(header.get("regionGrowMm") or 0)
        pick(self.growIntoCombo, normaldb.growInto(header))
        norm = dict(normaldb.emptyNormalization(), **(header.get("normalization") or {}))
        for combo, key in ((self.normMethodCombo, "method"), (self.normEngineCombo, "engine"),
                           (self.normLinearCombo, "linear")):
            pick(combo, norm.get(key) or "")
        for combo, key in ((self.normDeformableCombo, "deformable"), (self.normMaskCombo, "brainMask")):
            pick(combo, "" if norm.get(key) is None else ("yes" if norm[key] else "no"))
        self._updateNormTemplate(keep=norm.get("template") or "")
        self.ageTable.setRowCount(self._firstGroupRow())
        for group in header.get("ageGroups") or []:
            self._addRow(group.get("name", ""), "" if group.get("min") is None else f"{group['min']:g}",
                         "" if group.get("max") is None else f"{group['max']:g}")
        self.ageCheck.checked = bool(header.get("ageStratified"))
        self._updateAge()
        self.scannerEdit.text = header.get("scanner", "")
        self.descriptionEdit.setPlainText(header.get("description", ""))
        self._updateReference()
        widgets = {"referenceRegion": [self.referenceCombo], "referenceErosionMm": [self.erosionSpin],
                   "smoothingFwhmMm": [self.smoothingSpin],
                   "regionAtlas": [self.segmentationCombo, self.groupingCombo], "regionGrowMm": [self.growSpin],
                   "growInto": [self.growIntoCombo],
                   "ageStratified": [self.ageCheck, self.ageTable, self.addGroupButton, self.removeGroupButton]}
        for key in fixed:
            for widget in widgets.get(key, []):
                widget.enabled = False

    def _updateCredits(self):
        """References and licenses of the region atlas, the reference region and the template picked."""
        reference = self.referenceKey()
        key = self.regionAtlasKey() if self.segmentationCombo.count else ""
        keys = credits.regionAtlasSources(key) + credits.referenceSources(reference, self.referenceStructure())
        self.atlasCredits.setHtml(credits.sourcesHtml(
            keys, f"Regions: {normaldb.regionAtlasTitle(key)}; reference region: {self.referenceCombo.currentText}."))
        template = self.normTemplateCombo.currentData
        spaceName = neuro.spaceTitle(self.spaceId())
        if normaldb.isDatabaseTemplate(template) and self.normMethodCombo.currentData != normaldb.NORM_PET_TEMPLATE:
            self.templateCredits.setHtml(f"<p>Template: {self.normTemplateCombo.currentText}, made from the "
                                         f"database's subjects in the {spaceName} space.</p>"
                                         + credits.sourcesHtml(credits.spaceSources(self.spaceId())))
        elif self.normMethodCombo.currentData == normaldb.NORM_PET_TEMPLATE:
            self.templateCredits.setHtml("<p>Template: the database's own PET template (built from its subjects; "
                                         f"Build template), in the {spaceName} space.</p>"
                                         + credits.sourcesHtml(credits.spaceSources(self.spaceId())))
        else:
            self.templateCredits.setHtml(credits.templateHtml(template, self.normTemplateCombo.currentText,
                                                              self.spaceId())
                                         if template else credits.sourcesHtml(
                                             credits.spaceSources(self.spaceId()),
                                             f"Template not specified: the templates of {spaceName}."))

    def _updateNormTemplate(self, keep=None):
        """The volumes the method registers to: PET to MR template the space's MR / grey matter / synthetic PET and
        the database's MR means; CT/MR the CT and MR templates and the database's CT / MR means; PET to PET the
        database's PET template (fixed); not specified: any of the space's."""
        method = self.normMethodCombo.itemData(self.normMethodCombo.currentIndex) or ""
        current = keep if keep is not None else self.normTemplateCombo.currentData
        if method == normaldb.NORM_PET:
            keys = [k for k in neuro.TEMPLATES_PET]
        elif method == normaldb.NORM_ANATOMY:
            keys = list(neuro.TEMPLATES_MR) + [neuro.CT_TEMPLATE]
        elif method == normaldb.NORM_PET_TEMPLATE:
            keys = []
        else:
            keys = [k for k, _ in normaldb.NORM_TEMPLATES if k]
        keys = [""] + keys + normaldb.databaseTemplateKeys(self.databaseVolumes, method or normaldb.NORM_PET)
        if method == normaldb.NORM_PET_TEMPLATE:
            keys = [normaldb.DB_PET]
        self.normTemplateCombo.blockSignals(True)
        self.normTemplateCombo.clear()
        for key in keys:
            title = "Not specified" if not key else normaldb.templateTitle(key, self.databaseName or None)
            self.normTemplateCombo.addItem(title, key)
        if current and self.normTemplateCombo.findData(current) < 0:
            self.normTemplateCombo.addItem(normaldb.templateTitle(current, self.databaseName or None)
                                           + " (not available)", current)
        index = self.normTemplateCombo.findData(current or "")
        self.normTemplateCombo.setCurrentIndex(index if index >= 0 else 0)
        self.normTemplateCombo.blockSignals(False)
        self.normTemplateCombo.enabled = method != normaldb.NORM_PET_TEMPLATE
        self._updateCredits()

    def normalization(self):
        def data(combo):
            return combo.itemData(combo.currentIndex) or ""

        def yesNo(combo):
            value = data(combo)
            return None if not value else value == "yes"

        method = data(self.normMethodCombo)
        return {"method": method, "template": "" if method == normaldb.NORM_PET_TEMPLATE else
                data(self.normTemplateCombo),
                "engine": data(self.normEngineCombo), "linear": data(self.normLinearCombo),
                "deformable": yesNo(self.normDeformableCombo), "brainMask": yesNo(self.normMaskCombo)}

    def _voxelSize(self):
        info = neuro.spaceInfo(self.spaceId())
        size = (info["manifest"].get("voxelSizeMm") if info else None) or normaldb.VOXEL_MM
        return float(size[0] if isinstance(size, (list, tuple)) else size)

    def header(self):
        stratified = bool(self.ageCheck.checked)
        fields = dict(
            modality=self.modalityCombo.currentText,
            nuclide=self.nuclideCombo.currentText.strip(),
            tracer=self.tracerCombo.currentText.strip(),
            acquisition={"startMinutes": float(self.startSpin.value),
                         "durationMinutes": float(self.durationSpin.value)},
            space=self.spaceId(),
            voxelSizeMm=self._voxelSize(),
            referenceRegion=self.referenceKey(),
            referenceStructure=self.referenceStructure(),
            smoothingFwhmMm=float(self.smoothingSpin.value),
            referenceErosionMm=float(self.erosionSpin.value),
            regionAtlas=self.regionAtlasKey(),
            regionGrowMm=float(self.growSpin.value),
            growInto=(self.growIntoCombo.itemData(self.growIntoCombo.currentIndex) if self.growSpin.value > 0
                      else normaldb.GROW_NO_WM),
            normalization=self.normalization(),
            ageStratified=stratified,
            ageGroups=self.ageGroups() if stratified else [],
            scanner=self.scannerEdit.text.strip(),
            description=self.descriptionEdit.toPlainText().strip(),
        )
        if self.predefined:
            fields["mapType"] = self.mapTypeCombo.currentText
        return normaldb.newHeader(self.nameEdit.text.strip(), normaldb.STATIC,
                                  normaldb.PREDEFINED if self.predefined else normaldb.USER_DEFINED, **fields)


def _scrollDialog(title, parent):
    dialog = qt.QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.resize(720, 760)
    outer = qt.QVBoxLayout(dialog)
    scroll = qt.QScrollArea()
    scroll.widgetResizable = True
    content = qt.QWidget()
    layout = qt.QVBoxLayout(content)
    scroll.setWidget(content)
    outer.addWidget(scroll, 1)
    buttons = qt.QDialogButtonBox()
    outer.addWidget(buttons)
    return dialog, layout, buttons


def _execDialog(dialog):
    """Run a dialog made by _scrollDialog and free it afterwards (its answers are kept by the accept handlers)."""
    try:
        return dialog.exec_()
    finally:
        dialog.deleteLater()


def _closeProgress(progressDialog, free=True):
    """Close a progress dialog (None is fine; closing twice too). Before a message is shown (free False: only hidden,
    freed later): a message box opened while the modal progress dialog is up can end up under it, and Slicer then
    waits on a box nobody sees."""
    if progressDialog is None:
        return
    progressDialog.close()
    if free:
        progressDialog.deleteLater()
    else:
        slicer.app.processEvents()


def _error(parent, text):
    slicer.util.errorDisplay(text, parent=parent, windowTitle="Neuro PET Database Manager")


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------
class NeuroPETDatabaseManagerWidget(ScriptedLoadableModuleWidget):
    """Database folder on top, the Static / Dynamic tabs below: the table of the databases, details of the selected
    one and the database buttons."""

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = NeuroPETDatabaseManagerLogic()
        self.databases = []         # [(folder, header)] of the static table
        self.layout.addWidget(MergenBanner())       # the Mergen banner at the top of the panel

        rootBox = qt.QGroupBox("Database folder")
        rootLayout = qt.QVBoxLayout(rootBox)
        self.rootPath = _pathEdit(dirs=True)
        self.rootPath.toolTip = ("Folder that holds the databases (one folder per database). Kept in the settings; "
                                 "~/NeuroPET/Databases until another one is picked.")
        self.rootPath.currentPath = self.logic.ensureRootFolder()
        rootLayout.addWidget(self.rootPath)
        self.rootInfo = qt.QLabel()
        self.rootInfo.wordWrap = True
        rootLayout.addWidget(self.rootInfo)
        self.layout.addWidget(rootBox)

        self.tabs = qt.QTabWidget()
        self.layout.addWidget(self.tabs, 1)
        self.tabs.addTab(self._buildStaticTab(), "Static")
        dynamic = qt.QLabel("Dynamic databases (BPnd and other parametric maps) come next.")
        dynamic.wordWrap = True
        dynamic.alignment = qt.Qt.AlignTop
        self.tabs.addTab(dynamic, "Dynamic")

        self.rootPath.connect("currentPathChanged(QString)", self._onRootChanged)
        self.refresh()

    def _buildStaticTab(self):
        page = qt.QWidget()
        layout = qt.QVBoxLayout(page)
        self.table = qt.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels([title for _, title in COLUMNS])
        self.table.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().visible = False
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.minimumHeight = 220
        layout.addWidget(self.table, 1)

        self.details = qt.QTextBrowser()
        self.details.maximumHeight = 170
        layout.addWidget(self.details)
        # the selected database's description (data source, citation, license) and its atlases (read only)
        creditsLabel = qt.QLabel("<b>Description, references and licenses</b>")
        layout.addWidget(creditsLabel)
        self.credits = credits.infoBox(170)
        layout.addWidget(self.credits)

        grid = qt.QGridLayout()
        self.createButton = _button("Create new database...")
        self.createButton.toolTip = "An empty user defined database; subjects are added from the Neuro workflows."
        self.predefinedButton = _button("Add predefined database...")
        self.predefinedButton.toolTip = "A packaged database (.zip), or mean / SD maps picked one by one."
        self.downloadButton = _button("Download predefined databases...")
        self.downloadButton.toolTip = ("The predefined normal databases published with the extension (GitHub "
                                       "Releases of SlicerNeuroPET whose tag starts with static-): pick a release, "
                                       "then download, install or update its databases.")
        self.exportButton = qt.QPushButton("Export as predefined...")
        self.exportButton.toolTip = "The selected user defined database as a .zip without its subject images."
        self.deleteButton = _button("Delete database", "revert")
        self.deleteButton.toolTip = "Remove the selected database folder from disk (asks first)."
        self.addSubjectsButton = _button("1. Add subjects...")
        self.addSubjectsButton.toolTip = (
            "Add normal subjects to the selected user defined database: a BIDS folder, or image files with a subject "
            "table (CSV / TSV: id, age, sex). Each native image is stored in the database, normalized with the "
            "database's settings and resampled to its grid, with a QC picture. Patients are added from the Neuro "
            "workflow (Review: Add this patient to DB).")
        self.normalizeAgainButton = _button("2. Normalize again...")
        self.normalizeAgainButton.toolTip = ("Normalize all subjects again from their stored native images with the "
                                             "database's current settings (e.g. after a new template), replacing "
                                             "their maps.")
        self.buildTemplateButton = _button("3. Build template (iterative)...")
        self.buildTemplateButton.toolTip = (
            "A study specific template from the subjects' stored native images: pass 0 to the normalization target "
            "(e.g. the synthetic PET), then passes to the group's own mean, each subject resampled once from its "
            "native image, the group's drift taken out, until the subjects move less than the stop value. The last "
            "template becomes the database's normalization template (PET to PET) and the subjects' maps are "
            "replaced.")
        self.statsButton = _button("4. Compute mean / SD")
        self.statsButton.toolTip = ("The mean / SD maps of the selected user defined database from its subjects' "
                                    "maps, smoothed with its FWHM (master and age groups).")
        self.removeSubjectsButton = qt.QPushButton("Remove subjects...")
        self.removeSubjectsButton.toolTip = ("Remove subjects (e.g. badly normalized ones, after the QC pictures) "
                                             "from the selected user defined database, with their files.")
        self.editButton = qt.QPushButton("Edit settings...")
        self.editButton.toolTip = ("Change the settings of the selected database (name, tracer, reference region, "
                                   "region and normalization settings, age groups, notes).")
        self.openButton = qt.QPushButton("Open folder")
        self.refreshButton = qt.QPushButton("Refresh")
        grid.addWidget(self.createButton, 0, 0)
        grid.addWidget(self.predefinedButton, 0, 1)
        grid.addWidget(self.addSubjectsButton, 1, 0)
        grid.addWidget(self.normalizeAgainButton, 1, 1)
        grid.addWidget(self.buildTemplateButton, 2, 0)
        grid.addWidget(self.statsButton, 2, 1)
        grid.addWidget(self.removeSubjectsButton, 3, 0)
        grid.addWidget(self.editButton, 3, 1)
        grid.addWidget(self.exportButton, 4, 0)
        grid.addWidget(self.deleteButton, 4, 1)
        grid.addWidget(self.openButton, 5, 0)
        grid.addWidget(self.refreshButton, 5, 1)
        grid.addWidget(self.downloadButton, 6, 0, 1, 2)
        layout.addLayout(grid)

        self.table.connect("itemSelectionChanged()", self._updateSelection)
        self.createButton.connect("clicked()", self.onCreate)
        self.predefinedButton.connect("clicked()", self.onAddPredefined)
        self.downloadButton.connect("clicked()", self.onDownload)
        self.exportButton.connect("clicked()", self.onExport)
        self.addSubjectsButton.connect("clicked()", self.onAddSubjects)
        self.normalizeAgainButton.connect("clicked()", self.onNormalizeAgain)
        self.buildTemplateButton.connect("clicked()", self.onBuildTemplate)
        self.statsButton.connect("clicked()", self.onComputeStatistics)
        self.removeSubjectsButton.connect("clicked()", self.onRemoveSubjects)
        self.editButton.connect("clicked()", self.onEdit)
        self.deleteButton.connect("clicked()", self.onDelete)
        self.openButton.connect("clicked()", self.onOpenFolder)
        self.refreshButton.connect("clicked()", self.refresh)
        return page

    # --- table ---------------------------------------------------------------------------------------
    def root(self):
        return self.rootPath.currentPath

    def _onRootChanged(self, path):
        self.logic.setRootFolder(path)
        self.refresh()

    def refresh(self, select=None):
        root = self.root()
        self.databases, errors = normaldb.listDatabases(root, normaldb.STATIC)
        if not root:
            self.rootInfo.text = "Pick the folder that holds (or will hold) the databases."
        elif not os.path.isdir(root):
            self.rootInfo.text = "The folder does not exist."
        else:
            text = f"{len(self.databases)} static database(s)."
            if errors:
                text += " Unreadable: " + ", ".join(os.path.basename(f) for f, _ in errors)
                for folder, error in errors:
                    logging.warning(f"Database {folder}: {error}")
            self.rootInfo.text = text
        self.table.setRowCount(len(self.databases))
        selectRow = -1
        for row, (folder, header) in enumerate(self.databases):
            summary = normaldb.summary(folder, header)
            for column, (key, _) in enumerate(COLUMNS):
                item = qt.QTableWidgetItem(str(summary[key]))
                if key == "statistics" and summary[key] == "Out of date":
                    item.setForeground(qt.QBrush(qt.QColor("#c8553d")))
                item.setToolTip(folder)
                self.table.setItem(row, column, item)
            if select and os.path.normcase(folder) == os.path.normcase(select):
                selectRow = row
        self.table.resizeColumnsToContents()
        if selectRow >= 0:
            self.table.selectRow(selectRow)
        self._updateSelection()

    def selected(self):
        rows = {index.row() for index in self.table.selectedIndexes()}
        if len(rows) != 1:
            return None, None
        row = rows.pop()
        return self.databases[row] if 0 <= row < len(self.databases) else (None, None)

    def _updateSelection(self):
        folder, header = self.selected()
        rootOk = bool(self.root()) and os.path.isdir(self.root())
        self.createButton.enabled = rootOk
        self.predefinedButton.enabled = rootOk
        self.downloadButton.enabled = bool(self.root())
        self.deleteButton.enabled = folder is not None
        self.editButton.enabled = folder is not None
        self.openButton.enabled = rootOk
        self.exportButton.enabled = folder is not None and header.get("kind") == normaldb.USER_DEFINED
        userDefined = self.exportButton.enabled
        if userDefined:
            header = self._migrated(folder, header)
        have, count = normaldb.nativeCount(folder) if userDefined else (0, 0)
        self.addSubjectsButton.enabled = userDefined
        self.removeSubjectsButton.enabled = userDefined and count > 0
        self.statsButton.enabled = userDefined and count > 0
        self.normalizeAgainButton.enabled = userDefined and have > 0
        self.buildTemplateButton.enabled = userDefined and have > 1
        self.details.setHtml(self._detailsHtml(folder, header) if folder else "")
        self.credits.setHtml(credits.databaseHtml(header, "Select a database.") if folder else "")

    def _migrated(self, folder, header):
        """A user defined database of the old layout (format 1) moved to the folder per subject, once."""
        if int(header.get("formatVersion") or 1) >= normaldb.FORMAT_VERSION:
            return header
        try:
            moved = normaldb.migrateLayout(folder)
        except Exception as e:
            logging.error(f"Could not move the database {folder} to the folder per subject layout: {e}")
            slicer.util.showStatusMessage(f"Could not update the layout of the database '{header.get('name')}': "
                                          f"{e}", 10000)
            return header
        logging.info(f"Database {folder} moved to the folder per subject layout (format {normaldb.FORMAT_VERSION}): "
                     f"{moved} file(s) moved.")
        slicer.util.showStatusMessage(f"The database '{header.get('name')}' was moved to the folder per subject "
                                      f"layout ({moved} file(s) moved).", 10000)
        header = normaldb.readHeader(folder)
        self.databases = [(f, header if f == folder else h) for f, h in self.databases]
        return header

    @staticmethod
    def _detailsHtml(folder, header):
        def esc(text):
            return (str(text if text is not None else "").replace("&", "&amp;").replace("<", "&lt;")
                    .replace(">", "&gt;"))

        def mm(value):          # a number of the header (a hand edited header may hold text)
            try:
                return f"{float(value or 0):g}"
            except (TypeError, ValueError):
                return esc(value)
        acquisition = header.get("acquisition") or {}
        rows = [
            ("Folder", folder),
            ("Space", f"{neuro.spaceTitle(header.get('space'))}, {header.get('voxelSizeMm', '')} mm"),
            ("Acquisition", f"start {acquisition.get('startMinutes')} min p.i., duration "
                            f"{acquisition.get('durationMinutes')} min"),
            ("Reference", normaldb.referenceTitle(header)
             + (f" (mask {header.get('referenceMask')})" if header.get("referenceMask") else " (no mask)")),
            ("Reference erosion", f"{mm(header.get('referenceErosionMm'))} mm"),
            ("Regions", f"{normaldb.regionAtlasTitle(header.get('regionAtlas'))}, grown "
                        f"{mm(header.get('regionGrowMm'))} mm"
                        + (f" ({normaldb.GROW_INTO_TEXT[normaldb.growInto(header)]})"
                           if normaldb.GROW_INTO_TEXT[normaldb.growInto(header)] else "")),
            ("Spatial normalization", normaldb.normalizationText(header.get("normalization"))),
            ("Maps", f"{header.get('mapType')}, smoothing {mm(header.get('smoothingFwhmMm'))} mm FWHM"),
            ("Scanner", header.get("scanner")),
            ("Description", "below" if str(header.get("description") or "").strip() else ""),
        ]
        groups = ((header.get("statistics") or {}).get("groups") or {})
        if groups:
            rows.append(("Mean / SD", "; ".join(f"{g.get('name', k)}: n={g.get('n')}" for k, g in groups.items())))
            rows.append(("Regional statistics", "yes (stats/regions.json)" if nr.regionsPath(folder, header) else
                         "no (computed with the mean / SD when the CerebrA is in the atlas folder)"))
            rows.append(("3D-SSP projections", "yes (stats/ssp.npz)" if ssp.sspPath(folder, header) else "no"))
        ages = header.get("subjectAges")
        if ages and ages.get("n"):
            rows.append(("Ages", f"{ages.get('mean')} ± {ages.get('sd')} ({ages.get('min')}-{ages.get('max')})"))
        rows += normaldb.pipelineRows(folder, header)
        rows.append(("Version", f"{header.get('version', 1)}, format {header.get('formatVersion') or 1}, modified "
                                f"{str(header.get('modified', ''))[:16]}"))
        return "<table>" + "".join(f"<tr><td><b>{esc(k)}</b>&nbsp;</td><td>{esc(v)}</td></tr>" for k, v in rows
                                   if v not in (None, "")) + "</table>"

    # --- buttons -------------------------------------------------------------------------------------
    def onCreate(self):
        dialog, layout, buttons = _scrollDialog("Create new static database", slicer.util.mainWindow())
        info = qt.QLabel("An empty user defined database. Subjects are added later from the Neuro workflow ("
                         "Add patient to database); their maps are copied into the database and its mean / SD maps "
                         "computed.")
        info.wordWrap = True
        layout.addWidget(info)
        form = HeaderForm(layout, predefined=False)
        layout.addStretch(1)
        create = buttons.addButton("Create", qt.QDialogButtonBox.AcceptRole)
        buttons.addButton(qt.QDialogButtonBox.Cancel)
        buttons.connect("rejected()", dialog.reject)

        def accept():
            header = form.header()
            errors = normaldb.validateHeader(header)
            problem = self.logic.maskSourceProblem(header["referenceRegion"], header)
            if problem:
                errors.append(problem + " - pick a region or a custom mask.")
            if errors:
                _error(dialog, "\n".join(errors))
                return
            try:
                with slicer.util.tryWithErrorDisplay("Could not create the database", waitCursor=True):
                    folder = self.logic.createUserDefined(self.root(), header, form.maskPath.currentPath)
            except Exception:
                return
            dialog.accept()
            self.refresh(select=folder)

        create.connect("clicked()", accept)
        _execDialog(dialog)

    def onAddPredefined(self):
        dialog, layout, buttons = _scrollDialog("Add predefined static database", slicer.util.mainWindow())
        sourceBox = qt.QGroupBox("Source")
        sourceLayout = qt.QVBoxLayout(sourceBox)
        packageRadio = qt.QRadioButton("Packaged database (.zip, e.g. exported from another computer)")
        mapsRadio = qt.QRadioButton("Mean and SD maps, picked one by one")
        packageRadio.checked = True
        sourceLayout.addWidget(packageRadio)
        packagePath = _pathEdit(filters="Database package (*.zip)")
        sourceLayout.addWidget(packagePath)
        packageInfo = qt.QLabel()
        packageInfo.wordWrap = True
        sourceLayout.addWidget(packageInfo)
        packageName = qt.QLineEdit()
        packageName.placeholderText = "name (default: the name in the package)"
        sourceLayout.addWidget(packageName)
        sourceLayout.addWidget(mapsRadio)
        layout.addWidget(sourceBox)
        form = HeaderForm(layout, predefined=True)
        layout.addStretch(1)
        add = buttons.addButton("Add", qt.QDialogButtonBox.AcceptRole)
        buttons.addButton(qt.QDialogButtonBox.Cancel)
        buttons.connect("rejected()", dialog.reject)

        def updateSource():
            package = packageRadio.checked
            packagePath.enabled = package
            packageName.enabled = package
            form.widget.visible = not package

        def readPackage(path):
            if not path or not os.path.isfile(path):
                packageInfo.text = ""
                return
            try:
                header = normaldb.readPackage(path)
            except Exception as e:
                packageInfo.text = f"Not a database package: {e}"
                return
            groups = ((header.get("statistics") or {}).get("groups") or {})
            n = (groups.get(normaldb.MASTER) or {}).get("n")
            packageInfo.text = (f"{header.get('name')}: {header.get('tracer', '')} ({header.get('nuclide', '')}), "
                                f"{n} subjects, reference {normaldb.referenceTitle(header)}, "
                                f"{len(groups)} mean / SD map pair(s).")

        packageRadio.connect("toggled(bool)", lambda on: updateSource())
        packagePath.connect("currentPathChanged(QString)", readPackage)
        updateSource()

        def accept():
            try:
                with slicer.util.tryWithErrorDisplay("Could not add the database", waitCursor=True):
                    if packageRadio.checked:
                        path = packagePath.currentPath
                        if not path or not os.path.isfile(path):
                            raise ValueError("Pick the database package (.zip)")
                        folder = normaldb.importPackage(self.root(), path, packageName.text.strip() or None)
                    else:
                        header = form.header()
                        if header["referenceRegion"] == normaldb.REF_CUSTOM and not form.maskPath.currentPath:
                            raise ValueError("Pick the custom reference mask file")
                        folder = self.logic.createPredefined(self.root(), header, form.maps(),
                                                             form.maskPath.currentPath or None)
            except Exception:
                return
            dialog.accept()
            self.refresh(select=folder)

        add.connect("clicked()", accept)
        _execDialog(dialog)

    def onDownload(self):
        """The releases of static databases (tag static-...): pick one, see its description and its packages
        (installed, update, not installed); the checked ones are downloaded and installed into the database folder."""
        root = self.root()
        try:
            os.makedirs(root, exist_ok=True)
        except OSError as e:
            _error(slicer.util.mainWindow(), f"Could not make the database folder {root}: {e}")
            return
        try:
            with slicer.util.tryWithErrorDisplay("Could not read the releases of databases", waitCursor=True):
                releases = dbdownload.fetchReleases(normaldb.STATIC)
        except Exception:
            return
        if not releases:
            _error(slicer.util.mainWindow(), "No release of static databases yet:\n" + dbdownload.releasePage(tag=""))
            return

        dialog, layout, buttons = _scrollDialog("Download predefined databases", slicer.util.mainWindow())
        dialog.resize(860, 760)
        releaseForm = qt.QFormLayout()
        releaseCombo = qt.QComboBox()
        for release in releases:
            date = release["published"][:10]
            releaseCombo.addItem(f"{release['name']}  ({release['tag']}, {date})"
                                 + ("  pre-release" if release["prerelease"] else ""))
        releaseForm.addRow("Release:", releaseCombo)
        layout.addLayout(releaseForm)
        pageLabel = qt.QLabel()
        pageLabel.openExternalLinks = True
        layout.addWidget(pageLabel)
        description = qt.QTextBrowser()
        description.openExternalLinks = True
        description.minimumHeight = 240
        layout.addWidget(description, 1)
        target = qt.QLabel(f"Installed into <b>{root}</b>. Each database also carries its description, the references "
                           "to cite and the licenses of its sources (shown under the table once installed).")
        target.wordWrap = True
        layout.addWidget(target)
        table = qt.QTableWidget(0, 3)
        table.setHorizontalHeaderLabels(["Database", "Size", "Status"])
        table.verticalHeader().visible = False
        table.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(qt.QAbstractItemView.NoSelection)
        table.horizontalHeader().setStretchLastSection(True)
        table.minimumHeight = 220
        layout.addWidget(table, 1)
        rows = []

        def showRelease(number):
            release = releases[number]
            pageLabel.text = f"<a href=\"{release['page']}\">{release['page']}</a>"
            if hasattr(description, "setMarkdown"):
                description.setMarkdown(release["body"] or "(no description)")
            else:
                description.setPlainText(release["body"] or "(no description)")
            index = dbdownload.installedIndex(normaldb.listDatabases(root)[0])
            del rows[:]
            table.setRowCount(len(release["assets"]))
            for row, asset in enumerate(release["assets"]):
                state, folder = dbdownload.status(asset, index, normaldb.PREDEFINED)
                item = qt.QTableWidgetItem(asset["title"])
                flags = qt.Qt.ItemIsEnabled
                if state in (dbdownload.NEW, dbdownload.UPDATE):
                    flags |= qt.Qt.ItemIsUserCheckable
                    item.setCheckState(qt.Qt.Checked if state == dbdownload.UPDATE else qt.Qt.Unchecked)
                item.setFlags(flags)
                item.setToolTip(asset["name"] + (f"\n{folder}" if folder else ""))
                table.setItem(row, 0, item)
                table.setItem(row, 1, qt.QTableWidgetItem(dbdownload.sizeText(asset["size"])))
                stateItem = qt.QTableWidgetItem(dbdownload.STATUS_TEXT[state])
                if state == dbdownload.UPDATE:
                    stateItem.setForeground(qt.QBrush(qt.QColor("#c8553d")))
                table.setItem(row, 2, stateItem)
                rows.append((asset, state, folder, item))
            table.resizeColumnsToContents()
            get.enabled = any(r[1] in (dbdownload.NEW, dbdownload.UPDATE) for r in rows)

        def checked():
            return [r for r in rows if r[1] in (dbdownload.NEW, dbdownload.UPDATE)
                    and r[3].checkState() == qt.Qt.Checked]

        allNew = buttons.addButton("Check all", qt.QDialogButtonBox.ActionRole)
        get = buttons.addButton("Download and install", qt.QDialogButtonBox.AcceptRole)
        buttons.addButton(qt.QDialogButtonBox.Close)
        buttons.connect("rejected()", dialog.reject)

        def checkAll():
            for _, state, _, item in rows:
                if state in (dbdownload.NEW, dbdownload.UPDATE):
                    item.setCheckState(qt.Qt.Checked)

        allNew.connect("clicked()", checkAll)
        releaseCombo.connect("currentIndexChanged(int)", showRelease)
        showRelease(0)
        done = {"folders": [], "errors": []}

        def accept():
            picked = checked()
            if not picked:
                return
            tag = releases[releaseCombo.currentIndex]["tag"]
            total = sum(a["size"] for a, _, _, _ in picked) or 1
            progressDialog = qt.QProgressDialog("Downloading...", "Cancel", 0, 1000, dialog)
            progressDialog.setWindowTitle("Download predefined databases")
            progressDialog.windowModality = qt.Qt.WindowModal
            progressDialog.minimumDuration = 0
            before = 0
            for number, (asset, state, folder, _) in enumerate(picked, 1):
                progressDialog.labelText = f"{number} / {len(picked)}: {asset['title']}"

                def progress(got, size, before=before):
                    progressDialog.value = int(1000 * (before + got) / total)
                    slicer.app.processEvents()
                    return not progressDialog.wasCanceled

                try:
                    if folder:
                        self.logic.forgetLoaded(folder)     # the scene's copies of the maps being replaced
                    done["folders"].append(self.logic.installPackage(
                        root, asset, tag, folder if state == dbdownload.UPDATE else None, progress))
                except dbdownload.Cancelled:
                    break
                except Exception as e:
                    logging.error(f"Download of {asset['name']}: {e}")
                    done["errors"].append(f"{asset['title']}: {e}")
                before += asset["size"]
            progressDialog.close()
            dialog.accept()

        get.connect("clicked()", accept)
        _execDialog(dialog)
        if done["folders"] or done["errors"]:
            self.refresh(select=done["folders"][-1] if done["folders"] else None)
            text = f"{len(done['folders'])} database(s) installed."
            if done["errors"]:
                _error(slicer.util.mainWindow(), text + "\n\nNot installed:\n" + "\n".join(done["errors"]))
            else:
                slicer.util.infoDisplay(text + " Please also cite the works listed in each database's description.",
                                        windowTitle="Download predefined databases")

    def onExport(self):
        folder, header = self.selected()
        if not folder:
            return
        errors, warnings, contents = normaldb.exportCheck(folder, header)
        if errors:
            slicer.util.errorDisplay("The database cannot be exported yet:\n\n" + "\n\n".join(errors),
                                     windowTitle="Export as predefined database")
            return
        if warnings and not slicer.util.confirmOkCancelDisplay(
                "The package will lack:\n\n" + "\n\n".join(warnings) + "\n\nExport anyway?",
                windowTitle="Export as predefined database"):
            return
        path = qt.QFileDialog.getSaveFileName(slicer.util.mainWindow(), "Export as predefined database",
                                              os.path.join(os.path.expanduser("~"),
                                                           normaldb.safeName(header["name"]) + ".zip"),
                                              "Database package (*.zip)")
        if not path:
            return
        if not path.lower().endswith(".zip"):
            path += ".zip"
        try:
            with slicer.util.tryWithErrorDisplay("Could not export the database", waitCursor=True):
                normaldb.exportPredefined(folder, path)
        except Exception:
            return
        slicer.util.infoDisplay(f"Saved {path}\n\n" + "\n".join(contents)
                                + ("\n\nNot in the package:\n" + "\n".join(warnings) if warnings else ""),
                                windowTitle="Neuro PET Database Manager")

    def onComputeStatistics(self):
        folder, header = self.selected()
        if not folder or header.get("kind") != normaldb.USER_DEFINED:
            return
        count = normaldb.subjectCount(folder, header)
        fwhm = float(header.get("smoothingFwhmMm") or 0)
        if not slicer.util.confirmOkCancelDisplay(
                f"Compute the mean / SD maps of '{header.get('name')}' from its {count} subject(s) ("
                f"{normaldb.subjectMapKey(header)[:-4].upper()} maps, smoothing "
                + (f"{fwhm:g} mm FWHM" if fwhm > 0 else "off") + ")?", windowTitle="Compute mean / SD"):
            return
        progressDialog = slicer.util.createProgressDialog(windowTitle="Compute mean / SD", labelText="Starting...",
                                                          maximum=count)

        def progress(index, total, text):
            progressDialog.setValue(index)
            progressDialog.setLabelText(f"Smoothing and adding: {text}")
            slicer.app.processEvents()
            return not progressDialog.wasCanceled

        result = None
        try:
            result = self.logic.computeStatistics(folder, progress)
        except Exception as error:
            # the progress dialog closed before the message: a message under it looks like a frozen Slicer
            logging.exception("Compute mean / SD failed")
            _closeProgress(progressDialog, free=False)
            _error(slicer.util.mainWindow(), f"Could not compute the mean / SD maps:\n{error}")
        finally:
            _closeProgress(progressDialog)
            self.refresh(select=folder)
        if result is None:
            return
        saved, notes = result
        if saved is None:
            slicer.util.infoDisplay("\n".join(notes), windowTitle="Compute mean / SD")
            return
        groups = saved["statistics"]["groups"]
        text = "Done: " + "; ".join(f"{g['name']} n = {g['n']}" for g in groups.values()) + "."
        if notes:
            text += "\n\n" + "\n".join(notes)
        slicer.util.infoDisplay(text, windowTitle="Compute mean / SD")

    def onEdit(self):
        folder, old = self.selected()
        if not folder:
            return
        predefined = old.get("kind") == normaldb.PREDEFINED
        dialog, layout, buttons = _scrollDialog(f"Edit '{old.get('name')}'", slicer.util.mainWindow())
        count = normaldb.subjectCount(folder, old)
        info = qt.QLabel(
            ("A predefined database: the settings of its mean / SD maps (reference region, smoothing, age groups) "
             "stay as they are" + (", and so do its regions (stored with its statistics)."
                                   if "regionAtlas" in normaldb.fixedKeys(old) else ".") if predefined else
             f"{count} subject(s). A new reference region (or erosion) makes the reference mask again and the SUVR "
             "maps of the subjects from their SUV maps; reference, smoothing or age group changes put the mean / SD "
             "maps out of date. A new name renames the folder."))
        info.wordWrap = True
        layout.addWidget(info)
        form = HeaderForm(layout, predefined=False)
        form.databaseVolumes, form.databaseName = normaldb.databaseVolumes(folder, old), old.get("name", "")
        form.setHeader(old, normaldb.fixedKeys(old))
        layout.addStretch(1)
        save = buttons.addButton("Save", qt.QDialogButtonBox.AcceptRole)
        buttons.addButton(qt.QDialogButtonBox.Cancel)
        buttons.connect("rejected()", dialog.reject)
        done = {}

        def accept():
            try:
                header, changes = normaldb.editedHeader(old, form.header())
            except ValueError as error:
                _error(dialog, str(error))
                return
            if not changes["keys"]:
                dialog.reject()
                return
            customPath = form.maskPath.currentPath
            notes = []
            if changes["reference"] and not predefined:
                if header["referenceRegion"] == normaldb.REF_CUSTOM:
                    if not customPath and old.get("referenceRegion") != normaldb.REF_CUSTOM:
                        _error(dialog, "Pick the custom reference mask file.")
                        return
                else:
                    problem = self.logic.maskSourceProblem(header["referenceRegion"], header)
                    if problem:
                        _error(dialog, problem + ".")
                        return
                notes.append("The reference mask is made again" + (f" and the SUVR maps of the {count} subject(s) "
                                                                   "computed again from their SUV maps"
                                                                   if count else "") + ".")
            if changes["statistics"] and (old.get("statistics") or {}).get("computed"):
                notes.append("The mean / SD maps become out of date.")
            if changes["normalization"] and count and not predefined:
                notes.append(f"The {count} subject(s) already in were normalized with the earlier settings (they "
                             "stay; remove and add them again to normalize them anew).")
            if changes["name"]:
                notes.append(f"The folder is renamed to '{normaldb.safeName(header['name'])}'.")
            if notes and not slicer.util.confirmOkCancelDisplay("\n\n".join(notes), windowTitle="Edit database",
                                                                parent=dialog):
                return
            done.update(header=header, changes=changes, customPath=customPath)
            dialog.accept()

        save.connect("clicked()", accept)
        if not _execDialog(dialog) or not done:
            return
        progressDialog = None
        if done["changes"]["reference"] and count and not predefined:
            progressDialog = slicer.util.createProgressDialog(windowTitle="Edit database", maximum=count,
                                                              labelText="SUVR maps...")

        def progress(index, total, text):
            if progressDialog is not None:
                progressDialog.setValue(index)
                progressDialog.setLabelText(text)
                slicer.app.processEvents()
            return True

        editedFolder = folder
        self.logic.editedFolder = folder
        try:
            if progressDialog is None:
                slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
            try:
                folder = self.logic.editDatabase(folder, self.root(), done["header"], done["changes"],
                                                 done["customPath"], progress)
            finally:
                if progressDialog is None:
                    slicer.app.restoreOverrideCursor()
        except Exception as error:
            folder = self.logic.editedFolder       # renamed before the failure: shown under its new name
            # the progress dialog closed before the message: a message under it looks like a frozen Slicer
            logging.exception("Edit database failed")
            _closeProgress(progressDialog, free=False)
            _error(slicer.util.mainWindow(), f"Could not save the database settings:\n{error}")
        finally:
            _closeProgress(progressDialog)
            for changed in {editedFolder, folder}:      # the scene's copies of its files (renamed, made again)
                self.logic.forgetLoaded(changed)
            self.refresh(select=folder)

    # --- subjects: add, normalize again, template, natives ---------------------------------------------
    def _bidsPicker(self, form, label="BIDS folder:"):
        """Folder and session rows of a form (state["folderEdit"], state["sessionCombo"]): the BIDS PET files of a
        folder and the session picked (state["files"]); state["changed"] is called after each change."""
        from NeuroPETLib import staticneuro as sn
        folderEdit = _pathEdit(dirs=True)
        folderEdit.currentPath = str(qt.QSettings().value(SETTINGS_BATCH_FOLDER) or "")
        form.addRow(label, folderEdit)
        sessionCombo = qt.QComboBox()
        sessionCombo.toolTip = "Session of the images (BIDS ses-), e.g. SUV"
        form.addRow("Session:", sessionCombo)
        state = {"files": [], "all": [], "changed": lambda: None}

        def scan(*args):
            root = folderEdit.currentPath
            state["all"] = sn.findPetFiles(root) if root and os.path.isdir(root) else []
            current = sessionCombo.currentData
            sessions = sorted({sn.bidsSession(f) for f in state["all"]})
            sessionCombo.blockSignals(True)
            sessionCombo.clear()
            for session in sessions:
                count = sum(1 for f in state["all"] if sn.bidsSession(f) == session)
                sessionCombo.addItem(f"{session or '(none)'} ({count})", session)
            index = sessionCombo.findData(current) if current is not None else -1
            sessionCombo.setCurrentIndex(index if index >= 0 else max(0, sessionCombo.findData("SUV")))
            sessionCombo.blockSignals(False)
            pick()

        def pick(*args):
            session = sessionCombo.currentData
            state["files"] = [f for f in state["all"] if sn.bidsSession(f) == session]
            state["changed"]()

        folderEdit.connect("currentPathChanged(QString)", scan)
        sessionCombo.connect("currentIndexChanged(int)", pick)
        state["scan"] = scan
        state["folderEdit"] = folderEdit
        state["sessionCombo"] = sessionCombo
        return state

    def _atlasOrAsk(self, folder, parent=None):
        """The volumes of the database's template space (neuro.atlasFiles); {} (after an error) when the space is
        not in the atlas folders."""
        space = normaldb.readHeader(folder).get("space")
        atlas = neuro.atlasFiles(space)
        if not atlas:
            _error(parent or slicer.util.mainWindow(), f"The database's space ({space}) is not in the atlas folders "
                                                        f"(the extension's or {neuro.userAtlasFolder()}).")
        return atlas

    def _progressDialog(self, title, maximum):
        progressDialog = slicer.util.createProgressDialog(windowTitle=title, labelText="Starting...",
                                                          maximum=max(1, maximum))

        def progress(index, total, text):
            progressDialog.maximum = max(1, total)
            progressDialog.setValue(index)
            progressDialog.setLabelText(text)
            slicer.app.processEvents()
            return not progressDialog.wasCanceled
        return progressDialog, progress

    @staticmethod
    def _withoutAnatomyChoice(header, items, parent=None):
        """With the CT/MR method: what to do with the subjects of a batch without a CT / MR, asked when there are
        some (BATCH_PET_FALLBACK / BATCH_SKIP); None when cancelled. BATCH_PET_FALLBACK when nothing to ask."""
        from StaticNeuroPET import BATCH_PET_FALLBACK, BATCH_SKIP
        method = (header.get("normalization") or {}).get("method")
        missing = [i[1] for i in items if not ((i[2] if len(i) > 2 else None) or {}).get("anatomy")]
        if method != normaldb.NORM_ANATOMY or not missing:
            return BATCH_PET_FALLBACK
        box = qt.QMessageBox(parent or slicer.util.mainWindow())
        box.setWindowTitle("Subjects without a CT/MR")
        box.setIcon(qt.QMessageBox.Question)
        target = normaldb.templateTitle(normaldb.normalizationTarget(header))
        box.setText(f"The database is normalized CT/MR to template, but {len(missing)} of the {len(items)} subject(s) "
                    f"have no CT/MR: " + ", ".join(missing[:8]) + (", ..." if len(missing) > 8 else "") + ".")
        box.setInformativeText(f"Normalize their PET to the template ({target}) instead (noted in the log), or "
                               "skip them?")
        fallback = box.addButton("Normalize the PET", qt.QMessageBox.AcceptRole)
        skip = box.addButton("Skip them", qt.QMessageBox.DestructiveRole)
        box.addButton(qt.QMessageBox.Cancel)
        box.setDefaultButton(fallback)
        box.exec_()
        clicked = box.clickedButton()
        return BATCH_PET_FALLBACK if clicked == fallback else BATCH_SKIP if clicked == skip else None

    def _runBatch(self, folder, items, atlas, qc, replace, title, align=True, withoutAnatomy=None):
        """Run StaticNeuroPETLogic.batchAdd on batch items and report it."""
        from StaticNeuroPET import StaticNeuroPETLogic, BATCH_PET_FALLBACK
        progressDialog, progress = self._progressDialog(title, len(items))
        try:
            rows, logPath, stopped = StaticNeuroPETLogic().batchAdd(folder, items, atlas, qc, progress, replace,
                                                                    align, withoutAnatomy or BATCH_PET_FALLBACK)
        except Exception as error:
            logging.exception(f"{title} failed")
            _error(slicer.util.mainWindow(), f"{title} could not start:\n{error}")
            return
        finally:
            progressDialog.close()
            progressDialog.deleteLater()
            self.logic.forgetLoaded(folder)
            self.refresh(select=folder)
        counts = {status: sum(1 for r in rows if r["status"] == status)
                  for status in ("added", "skipped", "failed", "stopped")}
        withCt = sum(1 for i in items if ((i[2] if len(i) > 2 else None) or {}).get("anatomy"))
        text = (f"Done {counts['added']}, skipped {counts['skipped']}, failed {counts['failed']}"
                + (" - stopped" if stopped else "") + (f"; {withCt} with a CT/MR" if withCt else "")
                + f".\n\nLog: {logPath}"
                + ("\nQC pictures: " + os.path.join(folder, "qc") if qc else "")
                + "\n\nCheck the QC pictures, remove the badly normalized subjects, then build the template or "
                  "compute the mean / SD maps.")
        failed = [r for r in rows if r["status"] == "failed"]
        if failed:
            text += "\n\nFailed:\n" + "\n".join(f"{r['id']}: {r['message']}" for r in failed[:10])
            if len(failed) > 10:
                text += f"\n... ({len(failed) - 10} more in the log)"
        slicer.util.infoDisplay(text, windowTitle=title)

    def onAddSubjects(self):
        from StaticNeuroPET import StaticNeuroPETLogic
        folder, header = self.selected()
        if not folder or header.get("kind") != normaldb.USER_DEFINED:
            return
        logic = StaticNeuroPETLogic()
        try:
            plan, _ = logic.batchPlan(header, folder)
        except ValueError as error:
            _error(slicer.util.mainWindow(), str(error))
            return
        dialog, layout, buttons = _scrollDialog(f"Add subjects to '{header.get('name')}'", slicer.util.mainWindow())
        dialog.resize(720, 560)
        info = qt.QLabel(
            "Adds normal subjects from a BIDS folder (sub-<id>[_ses-<session>]_pet.nii[.gz] with their .json and the "
            "participants.tsv for age and sex) or from image files (a folder of .nii / .nii.gz / .nrrd; age and sex "
            "from a CSV / TSV with the columns id, age, sex - the ID is matched to the file name). Each image is "
            "stored as it is (native, in subjects/<id>/), normalized to the MNI space with the database's "
            "normalization settings (no manual check), resampled onto the database grid (its space, linear) and "
            "saved with its SUVR map (the database's reference mask). The subject's CT / MR (optional, 'CT/MR'): in "
            "a BIDS folder from anat/ of the same session, with image files from the subject table's column "
            "anatomy (or ct / mr); the PET is registered to it (rigid) unless they are already aligned, both are "
            "stored and normalized together (with the CT/MR method the CT/MR is registered to the template). Subject "
            "IDs already in the database are skipped unless 'replace' is ticked; a subject that fails is logged and "
            "the batch goes on.")
        info.wordWrap = True
        info.textFormat = qt.Qt.PlainText
        layout.addWidget(info)
        form = qt.QFormLayout()
        layout.addLayout(form)
        sourceCombo = qt.QComboBox()
        sourceCombo.addItem("BIDS folder", "bids")
        sourceCombo.addItem("Image files (a folder) + subject table", "files")
        form.addRow("Source:", sourceCombo)
        bids = self._bidsPicker(form)
        filesEdit = _pathEdit(dirs=True)
        filesEdit.toolTip = "A folder with one image per subject (not searched below)."
        form.addRow("Image folder:", filesEdit)
        tableEdit = _pathEdit(filters="Subject table (*.csv *.tsv *.txt)")
        tableEdit.toolTip = ("Optional: columns id (or subject / participant_id), age, sex, and the CT / MR file "
                             "(column anatomy, ct or mr; relative to the table's folder or absolute).")
        form.addRow("Subject table:", tableEdit)
        anatomyCombo = qt.QComboBox()
        anatomyCombo.toolTip = ("The subjects' CT / MR (optional): stored with them, normalized with them, averaged "
                                "into the database's CT / MR means; required by the CT/MR normalization method.")
        form.addRow("CT/MR:", anatomyCombo)
        alignCheck = qt.QCheckBox("CT/MR already aligned with the PET (PET/CT, PET/MR): no registration")
        alignCheck.toolTip = "Off: the PET is registered to its CT / MR (rigid) before the normalization."
        form.addRow(alignCheck)
        spaceLabel = qt.QLabel(neuro.spaceTitle(header.get("space")))
        spaceLabel.toolTip = "The database's template space (atlas library): the subjects are normalized to it."
        form.addRow("Space:", spaceLabel)
        qcCheck = qt.QCheckBox("Save a QC picture of each subject (PET over the MNI T1, 3 slices) in qc/")
        qcCheck.checked = True
        form.addRow(qcCheck)
        replaceCheck = qt.QCheckBox("Subjects already in the database: add them again (replace)")
        form.addRow(replaceCheck)
        found = qt.QLabel()
        found.wordWrap = True
        form.addRow(found)
        normLabel = qt.QLabel("Normalization: " + normaldb.normalizationText(plan)
                              + ("" if normaldb.isSpecified(header.get("normalization")) else
                                 " (the module defaults: the database has none; they are stored with the first "
                                 "subject)"))
        normLabel.wordWrap = True
        form.addRow(normLabel)
        layout.addStretch(1)
        run = buttons.addButton("Run", qt.QDialogButtonBox.AcceptRole)
        buttons.addButton(qt.QDialogButtonBox.Cancel)
        buttons.connect("rejected()", dialog.reject)
        state = {"items": []}

        def rowVisible(widget, visible):
            form.labelForField(widget).visible = visible
            widget.visible = visible

        def fillAnatomyRoles(isBids):
            choices = sn.ANATOMY_ROLES if isBids else (("", "None"), ("table", "From the subject table's column "
                                                                               "(anatomy / ct / mr)"))
            if [anatomyCombo.itemData(i) for i in range(anatomyCombo.count)] == [k for k, _ in choices]:
                return
            anatomyCombo.blockSignals(True)
            anatomyCombo.clear()
            for key, title in choices:
                anatomyCombo.addItem(title, key)
            anatomyCombo.blockSignals(False)

        def update(*args):
            isBids = sourceCombo.currentData == "bids"
            for widget in (bids["folderEdit"], bids["sessionCombo"]):
                rowVisible(widget, isBids)
            rowVisible(filesEdit, not isBids)
            rowVisible(tableEdit, not isBids)
            fillAnatomyRoles(isBids)
            role = anatomyCombo.currentData or ""
            alignCheck.enabled = bool(role)
            if isBids:
                files, table = bids["files"], None
            else:
                root = filesEdit.currentPath
                files = sorted(os.path.join(root, f) for f in (os.listdir(root) if root and os.path.isdir(root)
                                                               else [])
                               if normaldb.imageExtension(f) and os.path.isfile(os.path.join(root, f)))
            tableProblem = ""
            if not isBids:
                try:
                    table = normaldb.readSubjectTable(tableEdit.currentPath)
                except (OSError, ValueError) as error:
                    table = {}
                    tableProblem = f"Could not read the subject table: {error}. "
            items = logic.batchItems(folder, files, table)
            withAnatomy = []
            for path, subjectId, extra in items:
                anatomy, contrast = (sn.findAnatomy(path, role) if isBids else
                                     ((extra or {}).get("anatomy") or None, "") if role else (None, ""))
                if anatomy:
                    extra = dict(extra or {}, anatomy=anatomy, contrast=contrast or "")
                elif extra:
                    extra = {k: v for k, v in extra.items() if k not in ("anatomy", "contrast")}
                withAnatomy.append((path, subjectId, extra))
            items = withAnatomy
            state["items"] = items
            anatomyCount = sum(1 for i in items if (i[2] or {}).get("anatomy"))
            ids = [i[1] for i in items]
            existing = sum(1 for i in ids if normaldb.hasSubject(folder, i))
            duplicates = len(ids) - len(set(i.lower() for i in ids))
            noTable = sum(1 for i in items if not isBids and table and i[2] is None)
            replace = bool(replaceCheck.checked)
            found.text = (tableProblem + f"{len(items)} image(s)" + (f", {existing} already in the database ("
                                                      + ("added again" if replace else "skipped") + ")"
                                                      if existing else "")
                          + (f"; {duplicates} subject(s) appear twice: pick one session" if duplicates else "")
                          + (f"; {noTable} not in the subject table (no age / sex)" if noTable else "")
                          + (f"; CT/MR found for {anatomyCount}" if role else "") + ".")
            run.enabled = duplicates == 0 and (len(items) > existing or (replace and len(items) > 0))

        bids["changed"] = update
        sourceCombo.connect("currentIndexChanged(int)", update)
        filesEdit.connect("currentPathChanged(QString)", update)
        tableEdit.connect("currentPathChanged(QString)", update)
        replaceCheck.connect("toggled(bool)", update)
        anatomyCombo.connect("currentIndexChanged(int)", update)
        bids["scan"]()
        result = {}

        def accept():
            atlas = self._atlasOrAsk(folder, dialog)
            if not atlas:
                return
            withoutAnatomy = self._withoutAnatomyChoice(header, state["items"], dialog)
            if withoutAnatomy is None:
                return
            result.update(items=list(state["items"]), atlas=atlas, qc=bool(qcCheck.checked),
                          replace=bool(replaceCheck.checked), align=not alignCheck.checked,
                          withoutAnatomy=withoutAnatomy)
            if sourceCombo.currentData == "bids":
                qt.QSettings().setValue(SETTINGS_BATCH_FOLDER, bids["folderEdit"].currentPath)
            dialog.accept()

        run.connect("clicked()", accept)
        if not _execDialog(dialog) or not result.get("items"):
            return
        self._runBatch(folder, result["items"], result["atlas"], result["qc"], result["replace"], "Add subjects",
                       result["align"], result["withoutAnatomy"])

    def onNormalizeAgain(self):
        from StaticNeuroPET import StaticNeuroPETLogic
        folder, header = self.selected()
        if not folder or header.get("kind") != normaldb.USER_DEFINED:
            return
        have, count = normaldb.nativeCount(folder)
        if not have:
            _error(slicer.util.mainWindow(), "No subject has its native image in the database.")
            return
        try:
            plan, _ = StaticNeuroPETLogic().batchPlan(header, folder)
        except ValueError as error:
            _error(slicer.util.mainWindow(), str(error))
            return
        text = (f"Normalize again the {have} subject(s) of '{header.get('name')}' from their stored native images, "
                f"with the database's normalization:\n\n{normaldb.normalizationText(plan)}"
                + (f" to its template ({(header.get('normalizationTemplate') or {}).get('file')})"
                   if plan.get("method") == normaldb.NORM_PET_TEMPLATE else "")
                + ".\n\nTheir maps are replaced (one resampling from the native image)."
                + (f"\n\n{count - have} subject(s) have no native image and stay as they are." if have < count else ""))
        if not slicer.util.confirmOkCancelDisplay(text, windowTitle="Normalize again"):
            return
        atlas = self._atlasOrAsk(folder)
        if not atlas:
            return
        items = StaticNeuroPETLogic.batchItems(folder)
        withoutAnatomy = self._withoutAnatomyChoice(header, items)
        if withoutAnatomy is None:
            return
        self._runBatch(folder, items, atlas, True, True, "Normalize again", True, withoutAnatomy)

    def onBuildTemplate(self):
        from StaticNeuroPET import StaticNeuroPETLogic
        from NeuroPETLib import templatebuild as tb
        folder, header = self.selected()
        if not folder or header.get("kind") != normaldb.USER_DEFINED:
            return
        have, count = normaldb.nativeCount(folder)
        if have < count:
            _error(slicer.util.mainWindow(), f"{count - have} of the {count} subjects have no native image in the "
                                             "database: add them again or remove them first.")
            return
        if count < 2:
            _error(slicer.util.mainWindow(), "The template needs at least two subjects.")
            return
        dialog, layout, buttons = _scrollDialog(f"Build the template of '{header.get('name')}'",
                                                slicer.util.mainWindow())
        dialog.resize(680, 460)
        target = normaldb.normalizationTarget(header)
        info = qt.QLabel(
            f"Builds a study specific template from the {count} subjects' native images. Pass 0 registers every "
            f"subject to the MNI target ({neuro.templateTitle(target)}) with the database's engine, linear type and "
            "brain mask, and averages their maps (smoothed with the database's FWHM). Each next pass registers every "
            "subject to the previous mean, starting from its last matrix. After each pass the group's average drift "
            "since pass 0 is taken out (the template cannot grow or move), the matrices are saved "
            "(subjects/<id>/transform.json), every subject is resampled once from its native image and replaced, "
            "and the template is saved (templates/pass_<k>.nii.gz). It stops when the subjects move less than the "
            "stop value on average, or after the maximum passes; the last template becomes the database's "
            "normalization template (PET to PET). Linear registration only. Log: qc/" + tb.ITERATIONS_LOG
            + ". Then compute the mean / SD maps.")
        info.wordWrap = True
        info.textFormat = qt.Qt.PlainText
        layout.addWidget(info)
        form = qt.QFormLayout()
        layout.addLayout(form)
        passesSpin = qt.QSpinBox()
        passesSpin.setRange(2, 10)
        passesSpin.value = tb.MAX_PASSES
        passesSpin.toolTip = "Pass 0 (to the MNI target) and the passes to the group's own mean, at most."
        form.addRow("Maximum passes:", passesSpin)
        stopSpin = qt.QDoubleSpinBox()
        stopSpin.setRange(0.05, 5.0)
        stopSpin.decimals = 2
        stopSpin.singleStep = 0.1
        stopSpin.suffix = " mm"
        stopSpin.value = tb.STOP_MM
        stopSpin.toolTip = ("Stop when the subjects' matrices move less than this on average between two passes (RMS "
                            "over the corners of a brain-sized box).")
        form.addRow("Stop below:", stopSpin)
        qcCheck = qt.QCheckBox("QC picture of each subject (qc/, the last pass)")
        qcCheck.checked = True
        form.addRow(qcCheck)
        layout.addStretch(1)
        buttons.addButton("Build", qt.QDialogButtonBox.AcceptRole)
        buttons.addButton(qt.QDialogButtonBox.Cancel)
        buttons.connect("accepted()", dialog.accept)
        buttons.connect("rejected()", dialog.reject)
        accepted = dialog.exec_()
        passes, stopMm, qc = int(passesSpin.value), float(stopSpin.value), bool(qcCheck.checked)
        dialog.deleteLater()
        if not accepted:
            return
        atlas = self._atlasOrAsk(folder)
        if not atlas:
            return
        progressDialog, progress = self._progressDialog("Build template", count)
        try:
            rows, logPath, stopped = StaticNeuroPETLogic().buildTemplate(folder, atlas, passes, stopMm, qc, progress)
        except Exception as error:
            logging.exception("Build template failed")
            _error(slicer.util.mainWindow(), f"The template could not be built:\n{error}")
            return
        finally:
            progressDialog.close()
            progressDialog.deleteLater()
            self.logic.forgetLoaded(folder)
            self.refresh(select=folder)
        lines = [f"Pass {r['pass']}: {r['subjects']} subjects, moved {r['meanMoveMm'] or '-'} mm (max "
                 f"{r['maxMoveMm'] or '-'}), drift scale {r['driftScale']}, "
                 f"template r {r['templateCorrelation'] or '-'}"
                 + (f", {r['failed']} failed" if r['failed'] else "") for r in rows]
        used = normaldb.readHeader(folder).get("normalizationTemplate") or {}
        text = ("\n".join(lines) + ("\n\nStopped." if stopped else "")
                + (f"\n\nTemplate: pass {used.get('iteration')} ("
                   + ("converged" if used.get("converged") else "not converged: the maximum passes") + ")."
                   if used.get("built") == "iterative" and rows else "")
                + f"\n\nLog: {logPath}\n\nCheck the QC pictures, then compute the mean / SD maps.")
        slicer.util.infoDisplay(text, windowTitle="Build template")

    def onRemoveSubjects(self):
        folder, header = self.selected()
        if not folder or header.get("kind") != normaldb.USER_DEFINED:
            return
        subjects = normaldb.readSubjects(folder)
        if not subjects:
            slicer.util.infoDisplay("The database has no subjects.", windowTitle="Remove subjects")
            return
        dialog, layout, buttons = _scrollDialog(f"Remove subjects from '{header.get('name')}'",
                                                slicer.util.mainWindow())
        dialog.resize(520, 600)
        label = qt.QLabel("Select the subjects to remove (Ctrl / Shift for several). Their rows and SUV / SUVR maps "
                          "are deleted; the mean / SD maps become out of date.")
        label.wordWrap = True
        layout.addWidget(label)
        filterEdit = qt.QLineEdit()
        filterEdit.placeholderText = "Filter by ID"
        layout.addWidget(filterEdit)
        listWidget = qt.QListWidget()
        listWidget.setSelectionMode(qt.QAbstractItemView.ExtendedSelection)
        for subject in subjects:
            text = (f"{subject.get('id')}  (age {subject.get('age') or '?'}, {subject.get('sex') or '?'}, "
                    f"reference mean {subject.get('referenceMean') or '?'})")
            item = qt.QListWidgetItem(text)
            item.setData(qt.Qt.UserRole, subject.get("id"))
            listWidget.addItem(item)
        layout.addWidget(listWidget, 1)
        filterEdit.connect("textChanged(QString)", lambda text: [
            listWidget.item(i).setHidden(text.strip().lower() not in listWidget.item(i).text().lower())
            for i in range(listWidget.count)])
        remove = buttons.addButton("Remove", qt.QDialogButtonBox.AcceptRole)
        buttons.addButton(qt.QDialogButtonBox.Cancel)
        buttons.connect("rejected()", dialog.reject)
        chosen = []

        def accept():
            ids = [item.data(qt.Qt.UserRole) for item in listWidget.selectedItems()]
            if not ids:
                return
            if not slicer.util.confirmYesNoDisplay(f"Remove {len(ids)} subject(s) and their maps from disk?\n\n"
                                                   + ", ".join(ids[:20]) + (" ..." if len(ids) > 20 else ""),
                                                   windowTitle="Remove subjects", parent=dialog):
                return
            chosen.extend(ids)
            dialog.accept()

        remove.connect("clicked()", accept)
        if not _execDialog(dialog) or not chosen:
            return
        try:
            with slicer.util.tryWithErrorDisplay("Could not remove the subjects", waitCursor=True):
                normaldb.removeSubjects(folder, chosen)
        except Exception:
            pass                # shown by tryWithErrorDisplay
        finally:
            self.logic.forgetLoaded(folder)
            self.refresh(select=folder)

    def onDelete(self):
        folder, header = self.selected()
        if not folder:
            return
        count = normaldb.subjectCount(folder, header)
        what = (f"{count} subject(s) with their maps" if header.get("kind") == normaldb.USER_DEFINED
                else "the mean / SD maps")
        if not slicer.util.confirmYesNoDisplay(
                f"Delete the database '{header.get('name')}' from disk?\n\n{folder}\n\nThe folder is removed with "
                f"{what}. This cannot be undone.", windowTitle="Delete database"):
            return
        try:
            with slicer.util.tryWithErrorDisplay("Could not delete the database", waitCursor=True):
                normaldb.deleteDatabase(folder, self.root())
        except Exception:
            pass                # shown by tryWithErrorDisplay
        finally:
            self.logic.forgetLoaded(folder)       # the scene's copies of the deleted maps and template
            self.refresh()

    def onOpenFolder(self):
        folder, _ = self.selected()
        qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(folder or self.root()))


def _testFolder(prefix="tmp"):
    """A temporary folder of a test, removed when Python (Slicer) exits."""
    import atexit
    import shutil
    import tempfile
    folder = tempfile.mkdtemp(prefix=prefix)
    atexit.register(shutil.rmtree, folder, True)
    return folder


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
class NeuroPETDatabaseManagerTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_ReferenceMask()
        self.setUp()
        self.test_Statistics()

    def test_Statistics(self):
        root = _testFolder()
        folder = normaldb.createDatabase(root, normaldb.newHeader("stats", nuclide="F-18", tracer="FDG",
                                                                  smoothingFwhmMm=0.0))
        values = (1.0, 2.0, 4.0)
        for index, value in enumerate(values):
            node = slicer.util.addVolumeFromArray(np.full((5, 6, 7), value, dtype=np.float32))
            suv, suvr = normaldb.subjectFileNames(f"S{index}")
            for relative in (suv, suvr):
                slicer.util.saveNode(node, os.path.join(folder, relative))
            slicer.mrmlScene.RemoveNode(node)
            normaldb.addSubject(folder, {"id": f"S{index}", "age": 50, "suvFile": suv, "suvrFile": suvr})
        logic = NeuroPETDatabaseManagerLogic()
        header, notes = logic.computeStatistics(folder)
        master = header["statistics"]["groups"][normaldb.MASTER]
        self.assertEqual(master["n"], 3)
        mean = slicer.util.loadVolume(os.path.join(folder, master["mean"]), {"show": False})
        sd = slicer.util.loadVolume(os.path.join(folder, master["sd"]), {"show": False})
        self.assertAlmostEqual(float(slicer.util.arrayFromVolume(mean)[2, 3, 3]), np.mean(values), places=4)
        self.assertAlmostEqual(float(slicer.util.arrayFromVolume(sd)[2, 3, 3]), np.std(values, ddof=1), places=4)
        frozen = normaldb.freezeTemplate(folder)
        self.assertEqual(frozen["normalization"]["method"], normaldb.NORM_PET_TEMPLATE)
        self.assertTrue(normaldb.templatePath(folder, frozen))
        self.delayDisplay("Statistics test passed")

    def test_ReferenceMask(self):
        atlasFolder, root, cache = _testFolder(), _testFolder(), _testFolder()
        previous = {v: os.environ.get(v) for v in (neuro.USER_FOLDER_VARIABLE, neuro.GENERATED_FOLDER_VARIABLE)}
        os.environ[neuro.USER_FOLDER_VARIABLE] = atlasFolder
        os.environ[neuro.GENERATED_FOLDER_VARIABLE] = cache
        try:
            self._testReferenceMask(atlasFolder, root)
        finally:
            for variable, value in previous.items():
                if value is None:
                    os.environ.pop(variable, None)
                else:
                    os.environ[variable] = value

    def _testReferenceMask(self, atlasFolder, root):
        # a segmentation of the user's atlas folder (a label volume and its table)
        segmentationFolder = os.path.join(atlasFolder, neuro.SEGMENTATION_FOLDER, "TestRegions")
        os.makedirs(segmentationFolder)
        with open(os.path.join(segmentationFolder, neuro.SEGMENTATION_MANIFEST), "w") as f:
            f.write('{"name": "Test regions", "space": "%s"}' % normaldb.SPACE)
        with open(os.path.join(segmentationFolder, neuro.SEGMENTATION_CSV_NAME), "w") as f:
            f.write("Mindboggle ID;Label Name;RH Label;LH Labels;Notes;Dice Kappa\n"
                    "45;Cerebellum Gray Matter;46;97;;0,83\n444;Pons;103;104;;0\n")
        array = np.zeros((10, 10, 10), dtype=np.int16)
        array[1:3, 1:3, 1:3] = 46
        array[5:7, 5:7, 5:7] = 97
        array[8, 8, 8] = 103
        source = slicer.util.addVolumeFromArray(array, name="labels", nodeClassName="vtkMRMLLabelMapVolumeNode")
        slicer.util.saveNode(source, os.path.join(segmentationFolder, "labels.nrrd"))
        slicer.mrmlScene.RemoveNode(source)
        logic = NeuroPETDatabaseManagerLogic()
        header = normaldb.newHeader("test", nuclide="F-18", tracer="FDG", regionAtlas="TestRegions:regions")
        self.assertEqual((header["referenceRegion"], header["referenceStructure"]),
                         (normaldb.REF_ATLAS, "Cerebellum Gray Matter"))
        folder = logic.createUserDefined(root, header)
        saved = normaldb.readHeader(folder)
        self.assertEqual(saved["referenceMask"], normaldb.REFERENCE_MASK_NAME)
        mask = slicer.util.loadLabelVolume(os.path.join(folder, normaldb.REFERENCE_MASK_NAME), {"show": False})
        self.assertEqual(int(np.count_nonzero(slicer.util.arrayFromVolume(mask))), 16)
        # edit: another reference region makes the mask again; a new name renames the folder
        edited, changes = normaldb.editedHeader(saved, dict(saved, referenceStructure="Pons", name="test 2"))
        folder = logic.editDatabase(folder, root, edited, changes)
        self.assertTrue(folder.endswith("test 2") or folder.endswith("test_2"))
        mask = slicer.util.loadLabelVolume(os.path.join(folder, normaldb.REFERENCE_MASK_NAME), {"show": False})
        self.assertEqual(int(np.count_nonzero(slicer.util.arrayFromVolume(mask))), 1)
        self.assertEqual(normaldb.readHeader(folder)["referenceStructure"], "Pons")
        normaldb.deleteDatabase(folder, root)
        self.assertFalse(os.path.exists(folder))
        # the whole brain: the brain mask of the extension's space
        self.assertIsNone(logic.maskSourceProblem(normaldb.REF_GLOBAL, {"space": normaldb.SPACE}))
        self.assertIsNotNone(logic.maskSourceProblem(normaldb.REF_GLOBAL, {"space": "NoSuchSpace"}))
        self.delayDisplay("Reference mask test passed")
