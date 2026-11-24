"""
Dynamic normal databases: a patient's parametric maps, summed image and regional TAC fits added to a user defined
dynamic database (the Review/Compare step of the Dynamic Neuro workflow, DynamicNeuroPET.py), or many normal subjects
added in a batch from their BIDS series (the Neuro PET Database Manager), and the database's template made of its
subjects' summed images.

A dynamic subject's folder (subjects/<id>/, normaldb):
    <id>_<map>.nii.gz       each map of the database's model (normaldb.kinetics maps), on the database grid
    <id>_summed.nii.gz      the summed image normalized, on the grid (the PET to PET template's source)
    <id>_native.nii.gz      (batch) the summed image as it was normalized: motion corrected, in the patient space,
                            with transform.json (its normalization matrix)
    <id>_fits.json          the regions' TAC fit values {"model", "k2ref", "reference", "settings", "regions":
                            {region: {map: value}}}
    <id>_regions.csv        the region fits as the Modeling export writes them, the TACs beside it
                            (<id>_regions_TACs.csv)
subjects.csv: suvFile = the summed image, mapFiles = {map: file} (JSON), fitsFile, tacsFile, nativeFile (the native
summed image), seriesSmoothingMm (the Data step's smoothing of the frames), sourceFile (the series file of a batch).

A batch subject goes through the workflow's steps without a manual check, with the database's settings: the BIDS
series loaded; its frames smoothed (the database's series smoothing); motion corrected (PET4D defaults: iterative
rigid, every frame to the summed image; Overwrite); the summed image normalized to the database's template (PET to
MR template / synthetic PET, or PET to PET with the database's template; the CT/MR method is not used in a batch)
and the series hardened onto the template's field of view (PET voxel size, Overwrite); the database's regions and
reference region (locked, as the Regions step loads them); the regions fitted and the voxel maps made with the
database's kinetic settings (PET4D Dynamic PET Modeling logic, the template's brain mask); then added as above.
"""

import json
import logging
import os
import time

import numpy as np
import slicer
import vtk

from DICOMGatedDynamicPETLib import registration as reg, resampling
from NeuroWorkflowLib.base import (METHOD_PET, METHOD_PET_TEMPLATE, NORMALIZATION_ENGINE,
                                   NORMALIZATION_ITERATIVE_DEFORMABLE, NORMALIZATION_ITERATIVE_LINEAR,
                                   NORMALIZATION_STEPS, NORMALIZATION_WORKING_MM, PET_TEMPLATE_ATTRIBUTE,
                                   SMOOTHING_ATTRIBUTE, ATLAS_FILE_ATTRIBUTE, _ijkToRas, _vtkMatrix)
from NeuroPETLib import (dynamicneuro as dn, neuro, neuroreview as nr, normaldb, staticneuro as sn,
                         templatebuild as tb)

NORMALIZATION_ATTRIBUTE = "DynamicPET.Neuro.Normalization"     # NeuroWorkflowLib.database (no widget import here)
BATCH_COLUMNS = ("file", "id", "status", "message", "k2ref", "seconds")
# the classes of the nodes a batch subject leaves behind (removed after it, the sequence browsers first)
BATCH_NODE_CLASSES = ("vtkMRMLSequenceBrowserNode", "vtkMRMLSequenceNode", "vtkMRMLScalarVolumeNode",
                      "vtkMRMLSegmentationNode", "vtkMRMLTableNode", "vtkMRMLTransformNode")
TEMPLATE_SCALE = tb.SUMMED_SCALE


def seriesSmoothingOf(sequence):
    """The Data step's smoothing of a series (FWHM mm; 0 when it was not smoothed)."""
    text = sequence.GetAttribute(SMOOTHING_ATTRIBUTE) if sequence is not None else None
    try:
        return float(text) if text else 0.0
    except ValueError:
        return 0.0


class DynamicDatabaseLogicMixin:
    """The logic of adding a patient (or a batch of normal subjects) to a dynamic database (mixed into
    DynamicNeuroPETLogic)."""

    @staticmethod
    def attributes(node):
        """The attributes of a node (storeNative keeps them with the native image)."""
        return {name: node.GetAttribute(name) for name in (node.GetAttributeNames() or ())} if node else {}

    def _resampledNode(self, volume, gridNode, name):
        """A temporary volume of volume resampled (linear) onto the grid of gridNode."""
        dims = tuple(int(x) for x in gridNode.GetImageData().GetDimensions())
        image = resampling.Resampler(_ijkToRas(volume), dims, _ijkToRas(gridNode), None, True,
                                     "Linear").resample(volume.GetImageData())
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", name)
        node.SetIJKToRASMatrix(_vtkMatrix(_ijkToRas(gridNode)))
        node.SetAndObserveImageData(image)
        return node

    def addDynamicSubject(self, folder, header, maps, summed, subjectId, gridPath, age, sex, normalization, fits,
                          k2ref=None, referenceName="", settings=None, writeRegionsCsv=None, source="",
                          scanDate="", replace=False, native="", seriesSmoothing=None, sourceFile=""):
        """Add a patient to a user defined dynamic database: maps {map name: volume} (every map of the database's
        model; normalized, made with its kinetic settings) and the summed image (normalized) resampled onto the
        database grid (gridPath: its reference mask, or the space's brain mask) and saved in subjects/<id>/; fits
        {region: {map: value}} the regions' TAC fit values (k2ref: the reference's k2' used, settings: the maps'
        kinetic settings, both kept with them); writeRegionsCsv(path) writes the region fits CSV (and its TACs).
        native: the relative file of the native summed image already stored (a batch); seriesSmoothing: the Data
        step's smoothing of the frames (FWHM mm, 0: none; the first subject sets the database's); sourceFile: the
        series file (a batch: the subject can be made again from it). Returns the row written into subjects.csv."""
        subjectId = normaldb.checkSubjectId(folder, subjectId, replace)
        wanted = list(normaldb.kinetics(header)["maps"])
        missing = [m for m in wanted if maps.get(m) is None]
        if missing:
            raise ValueError("The maps " + ", ".join(missing) + " of the database's model are missing.")
        scene = slicer.mrmlScene
        old = next((s for s in normaldb.readSubjects(folder) if s.get("id", "").lower() == subjectId.lower()), None)
        os.makedirs(os.path.join(folder, normaldb.subjectDir(subjectId)), exist_ok=True)
        _, gridNode = self._loadLabels(gridPath)
        nodes = [gridNode]
        files = {}
        try:
            for name in wanted:
                node = self._resampledNode(maps[name], gridNode, f"Neuro subject {name} (temporary)")
                nodes.append(node)
                values = np.nan_to_num(slicer.util.arrayFromVolume(node).astype(np.float32))
                slicer.util.updateVolumeFromArray(node, values)
                relative = normaldb.mapFileName(subjectId, name)
                if not slicer.util.saveNode(node, os.path.join(folder, relative)):
                    raise RuntimeError(f"Could not write {relative}")
                files[name] = relative
            summedFile = ""
            if summed is not None:
                node = self._resampledNode(summed, gridNode, "Neuro subject summed (temporary)")
                nodes.append(node)
                summedFile = normaldb.summedFileName(subjectId)
                if not slicer.util.saveNode(node, os.path.join(folder, summedFile)):
                    raise RuntimeError(f"Could not write {summedFile}")
        finally:
            for node in nodes:
                if node is not None and scene.IsNodePresent(node):
                    scene.RemoveNode(node)
        fitsFile = normaldb.fitsFileName(subjectId)
        document = {"model": normaldb.kinetics(header)["model"], "k2ref": k2ref, "reference": referenceName,
                    "settings": settings or {}, "regions": {
                        region: {k: (None if v is None or not np.isfinite(float(v)) else float(v))
                                 for k, v in values.items()} for region, values in (fits or {}).items()}}
        with open(os.path.join(folder, fitsFile), "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=1)
        tacsFile = ""
        if writeRegionsCsv is not None:
            relative = normaldb.regionsCsvName(subjectId)
            try:
                written = writeRegionsCsv(os.path.join(folder, relative)) or []
                tacs = [p for p in written if p.endswith("_TACs.csv")]
                tacsFile = os.path.relpath(tacs[0], folder).replace(os.sep, "/") if tacs else relative
            except Exception:
                logging.warning("Dynamic Neuro: could not write the region fits CSV", exc_info=True)
        if scanDate and len(scanDate) == 8 and scanDate.isdigit():
            scanDate = f"{scanDate[:4]}-{scanDate[4:6]}-{scanDate[6:]}"
        row = {"id": subjectId, "age": "" if age is None else f"{age:g}", "sex": sex or "", "scanDate": scanDate,
               "suvFile": summedFile, "suvrFile": "",
               "referenceMean": "" if k2ref is None else f"{float(k2ref):.6g}", "reference": referenceName or "",
               "normalization": normaldb.normalizationText(normalization), "source": source,
               "mapFiles": json.dumps(files, sort_keys=True), "fitsFile": fitsFile, "tacsFile": tacsFile,
               "nativeFile": native or "",
               "seriesSmoothingMm": "" if seriesSmoothing is None else f"{float(seriesSmoothing):g}",
               "sourceFile": sourceFile or ""}
        if old and replace:         # the old subject's files not written again go; its native image is kept
            for key in ("nativeFile", "scanDate", "source", "sourceFile"):
                if not row.get(key) and old.get(key):
                    row[key] = old[key]
            stale = set(normaldb.subjectMapFiles(old).values()) | {old.get("suvFile"), old.get("suvrFile")}
            stale -= set(files.values()) | {summedFile, row.get("nativeFile")}
            for relative in stale:
                path = os.path.join(folder, relative) if relative else ""
                if path and os.path.isfile(path):
                    os.remove(path)
        normaldb.addSubject(folder, row, normalization, replace)
        logging.info(f"Dynamic Neuro: '{subjectId}' added to {folder} ({', '.join(files)})")
        return row

    # --- batch: normal subjects from their BIDS series --------------------------------------------------
    def dynamicBatchPlan(self, folder, header):
        """(plan, registration settings, notes) of a batch into a dynamic database: its normalization settings
        (staticneuro.batchNormalization, the module defaults where it has none; the target its normalizationTarget);
        PET to PET needs the database's template (of its summed images). The CT/MR method is not used in a batch: the
        summed image is registered to the database's MNI target instead (a note says so)."""
        norm = dict(normaldb.emptyNormalization(), **(header.get("normalization") or {}))
        notes = []
        if norm.get("method") == normaldb.NORM_ANATOMY:
            norm.update(method=normaldb.NORM_PET, template=normaldb.normalizationTarget(header))
            notes.append("the database normalizes the CT/MR; a batch registers the summed image to "
                         + normaldb.templateTitle(norm["template"]))
        hasTemplate = bool(normaldb.templatePath(folder, header))
        plan = sn.batchNormalization(norm, normaldb.normalizationTarget(header), NORMALIZATION_ENGINE,
                                     {reg.ENGINE_ITERATIVE: (NORMALIZATION_ITERATIVE_LINEAR,
                                                             NORMALIZATION_ITERATIVE_DEFORMABLE),
                                      reg.ENGINE_BRAINSFIT: (NORMALIZATION_STEPS[0], "BSpline" in NORMALIZATION_STEPS)},
                                     hasTemplate)
        method = METHOD_PET_TEMPLATE if plan["method"] == normaldb.NORM_PET_TEMPLATE else METHOD_PET
        settings = self.defaultSettings(method)
        if plan["engine"] == reg.ENGINE_ITERATIVE:
            settings.update(engine=reg.ENGINE_ITERATIVE, iterativeLinear=plan["linear"],
                            iterativeDeformable=plan["deformable"],
                            steps=reg.iterativeSteps(plan["linear"], plan["deformable"]))
        else:
            steps = reg.validateSteps([plan["linear"]] + (["BSpline"] if plan["deformable"] else []))
            settings.update(engine=reg.ENGINE_BRAINSFIT, steps=steps,
                            resample=[NORMALIZATION_WORKING_MM > 0] * len(steps))
        return plan, settings, notes

    def _batchTemplate(self, folder, header, atlas, plan):
        """The template volume of a batch plan: the database's template (PET to PET) or the space's template (the
        synthetic PET made when needed)."""
        if plan["method"] == normaldb.NORM_PET_TEMPLATE:
            path = normaldb.templatePath(folder, header)
            if not path:
                raise ValueError(f"'{header.get('name')}' has no template of its summed images: make it first.")
            node = self.loadAtlasVolume(path, name=f"{header.get('name')} template")
            node.SetAttribute(PET_TEMPLATE_ATTRIBUTE, "1")
            return node
        key = plan["template"]
        path = atlas.get(key)
        if not path and key in neuro.GENERATED and neuro.canMakeTemplate(atlas, key):
            path = self.ensureSyntheticPet(atlas, key)
        if not path:
            raise ValueError(f"The template '{neuro.templateTitle(key)}' is not in the database's space "
                             f"({neuro.spaceTitle(header.get('space'))}).")
        node = self.loadAtlasVolume(path, name=f"{neuro.spaceTitle(header.get('space'))} {key}")
        if key in neuro.GENERATED:
            node.SetAttribute(PET_TEMPLATE_ATTRIBUTE, "1")
        return node

    def batchRegionArguments(self, folder, header):
        """loadAtlasRegions() keyword arguments of a database's regions and reference region (as the Regions step
        loads them locked to the database)."""
        settings = sn.regionSettings(folder, header)
        key = settings["regionAtlas"]
        arguments = {"targetKind": key, "referenceKind": neuro.mergedRegionAtlas(key),
                     "growMm": settings["growMm"], "erodeReferenceMm": settings["erodeReferenceMm"],
                     "growInto": settings["growInto"]}
        stored = sn.storedRegions(folder, header)
        if stored:
            arguments["storedRegions"] = stored
        elif settings["referenceMaskPath"]:
            arguments["referenceMaskPath"] = settings["referenceMaskPath"]
            arguments["referenceName"] = settings["referenceTitle"] or "Reference"
            arguments.update(sn.referenceMaskDrop(settings))
        elif settings["referenceAtlasRegion"]:
            arguments["referenceName"] = settings["referenceAtlasRegion"]
        else:
            raise ValueError(f"The database '{header.get('name')}' has no copy of its reference mask "
                             f"({settings['referenceTitle']}): edit its reference region first.")
        return arguments

    @staticmethod
    def batchItemsOfSources(folder):
        """Batch items (path, subject ID, None) of the subjects of a dynamic database made from a series file that is
        still there (sourceFile): to make them again with the database's current settings. Returns (items, the IDs
        without such a file)."""
        items, missing = [], []
        for subject in normaldb.readSubjects(folder):
            path = subject.get("sourceFile") or ""
            if path and os.path.isfile(path):
                items.append((path, subject["id"], None))
            else:
                missing.append(subject["id"])
        return items, missing

    def batchAddDynamic(self, folder, items, atlas=None, qc=True, progress=None, replace=False,
                        seriesSmoothing=None):
        """Add normal subjects to a user defined dynamic database from their dynamic series (items: (BIDS 4D NIfTI
        path, subject ID, extra); see the module docstring for the steps). seriesSmoothing: the Data step's
        smoothing of the frames (FWHM mm, 0: none; None: the database's, none when it has none). IDs already in the
        database are skipped (replace: made again and replaced); a subject that fails is logged and the batch goes
        on. qc: a PNG of the normalized summed image over the space's T1 in <database>/qc/. progress(index, total,
        text) returns False to stop. The log is <database>/qc/batch_<time>.csv. Returns (rows, log path, stopped)."""
        header = normaldb.readHeader(folder)
        if header.get("kind") != normaldb.USER_DEFINED or not normaldb.isDynamic(header):
            raise ValueError("Series can be added to user defined dynamic databases only.")
        kinetics = normaldb.kinetics(header)
        errors = normaldb.kineticErrors(kinetics)
        if errors:
            raise ValueError("The database's kinetic settings: " + "; ".join(errors))
        atlas = atlas or neuro.atlasFiles(header.get("space"))
        gridPath = sn.referenceMaskPath(folder, header) or atlas.get(neuro.BRAIN_MASK)
        if not gridPath:
            raise ValueError(f"'{header['name']}' has no reference mask and its space no brain mask: no grid.")
        if seriesSmoothing is None:
            seriesSmoothing = normaldb.seriesSmoothing(header) or 0.0
        plan, settings, planNotes = self.dynamicBatchPlan(folder, header)
        regionArguments = self.batchRegionArguments(folder, header)
        template = self._batchTemplate(folder, header, atlas, plan)
        mask = None
        if plan["brainMask"] and neuro.BRAIN_MASK in atlas:
            candidate = self.loadAtlasVolume(atlas[neuro.BRAIN_MASK],
                                             name=f"{neuro.spaceTitle(header.get('space'))} brain mask")
            if candidate.GetImageData().GetDimensions() == template.GetImageData().GetDimensions():
                mask = candidate
        brainMask = (self.loadAtlasVolume(atlas[neuro.BRAIN_MASK],
                                          name=f"{neuro.spaceTitle(header.get('space'))} brain mask")
                     if neuro.BRAIN_MASK in atlas else None)
        anatomy = None
        if qc and "T1" in atlas:
            t1 = self.loadAtlasVolume(atlas["T1"], name=f"{neuro.spaceTitle(header.get('space'))} T1")
            anatomy = (sn.rasArray(slicer.util.arrayFromVolume(t1), _ijkToRas(t1)), tuple(_ijkToRas(t1).ravel()))
        qcFolder = os.path.join(folder, "qc")
        os.makedirs(qcFolder, exist_ok=True)
        logPath = os.path.join(qcFolder, "batch_" + time.strftime("%Y%m%d_%H%M%S") + ".csv")
        context = {"header": header, "kinetics": kinetics, "plan": plan, "settings": settings,
                   "planNotes": planNotes, "regions": regionArguments, "template": template, "mask": mask,
                   "brainMask": brainMask, "gridPath": gridPath, "smoothing": float(seriesSmoothing or 0.0),
                   "anatomy": anatomy, "qcFolder": qcFolder, "atlas": atlas}
        rows, stopped, total = [], False, len(items)
        for index, item in enumerate(items):
            path, subjectId = item[0], item[1]
            if progress is not None and not progress(index, total, f"{subjectId} ({index + 1} / {total})"):
                stopped = True
                break
            row = {"file": path, "id": subjectId, "status": "", "message": "", "k2ref": "", "seconds": ""}
            start = time.time()
            if normaldb.hasSubject(folder, subjectId) and not replace:
                row.update(status="skipped", message="already in the database")
            else:
                def stepProgress(text, index=index, subjectId=subjectId):
                    return progress is None or progress(index, total, f"{subjectId} ({index + 1} / {total}): {text}")
                backup = normaldb.backupSubjectFolder(folder, subjectId)
                try:
                    added = self._batchDynamicSubject(folder, path, subjectId, context, stepProgress, replace)
                    if added is None:
                        normaldb.restoreSubjectFolder(folder, subjectId, backup)
                        row.update(status="stopped", message="stopped")
                        stopped = True
                    else:
                        row.update(status="added", k2ref=added.get("referenceMean", ""),
                                   message=added.get("notes", ""))
                        normaldb.dropSubjectBackup(folder, backup)
                except Exception as error:
                    logging.exception(f"Dynamic Neuro batch: {path} failed")
                    message = str(error).replace("\n", " ")
                    try:
                        normaldb.restoreSubjectFolder(folder, subjectId, backup)
                        message += " (its files left as they were)" if backup["existed"] else ""
                    except Exception as restoreError:       # noqa: BLE001 - said in the log row
                        logging.exception(f"Dynamic Neuro batch: could not restore the files of {subjectId}")
                        message += f" (its old files could not be put back: {restoreError}; copy in {backup['backup']})"
                    row.update(status="failed", message=message)
                context["header"] = normaldb.readHeader(folder)
            row["seconds"] = f"{time.time() - start:.0f}"
            rows.append(row)
            self._writeDynamicBatchLog(logPath, rows)
            if stopped:
                break
        if not rows:
            self._writeDynamicBatchLog(logPath, rows)
        return rows, logPath, stopped

    @staticmethod
    def _sceneNodeIds():
        return {slicer.mrmlScene.GetNthNode(i).GetID() for i in range(slicer.mrmlScene.GetNumberOfNodes())}

    @staticmethod
    def _removeNewNodes(before):
        """Remove the nodes of BATCH_NODE_CLASSES added since before (a set of node IDs), the browsers first; the
        space's volumes loaded meanwhile (loadAtlasVolume, kept for the next subject) stay."""
        scene = slicer.mrmlScene
        for className in BATCH_NODE_CLASSES:
            for node in list(slicer.util.getNodesByClass(className)):
                if node.GetID() not in before and scene.IsNodePresent(node) \
                        and not node.GetAttribute(ATLAS_FILE_ATTRIBUTE):
                    scene.RemoveNode(node)

    def _batchDynamicSubject(self, folder, path, subjectId, context, progress, replace=False):
        """One subject of batchAddDynamic(): the row added (with "notes"), None when stopped."""
        before = self._sceneNodeIds()
        try:
            return self._batchDynamicSteps(folder, path, subjectId, context, progress, replace)
        finally:
            self._removeNewNodes(before)

    def _batchDynamicSteps(self, folder, path, subjectId, context, progress, replace):
        from DynamicPETMotionCorrection import DynamicPETMotionCorrectionLogic
        from DynamicPETModeling import DynamicPETModelingLogic
        from DynamicPETRegistration import DynamicPETRegistrationLogic
        from DICOMGatedDynamicPETLib import kinetics as kin, motioncorrection as mc
        header, kinetics = context["header"], context["kinetics"]
        notes = list(context["planNotes"])

        if not progress("loading the series"):
            return None
        sequence, summed = self.importDynamic(path)
        if summed is None:
            raise ValueError("the series has no summed image")
        age, sex = self.ageAndSex(sequence)
        if age is None:
            notes.append("age not known")

        if context["smoothing"] > 0:
            if not progress(f"smoothing ({context['smoothing']:g} mm FWHM)"):
                return None
            self.smoothSeries(sequence, context["smoothing"])

        # motion correction: every frame to the summed image (the PET4D defaults), hardened into the series
        mcLogic = DynamicPETMotionCorrectionLogic()
        mcSettings = mcLogic.defaultSettings()
        results = mcLogic.registerFrames(sequence, summed, None, mcSettings,
                                         lambda i, n, text: progress(f"motion correction {i + 1} / {n}"))
        if any(r["status"] == mc.NOT_RUN for r in results):
            return None
        _, _, resummed = mcLogic.overwrite(sequence, results, mcSettings, summed.GetName(), True, summed, None)
        for result in results:
            result["image"] = None
        summed = resummed or DynamicPETRegistrationLogic.summedVolumeOf(sequence) or summed
        notes.append("motion: " + mc.summaryText([r["status"] for r in results]))

        # the native summed image (motion corrected, patient space) and its normalization
        native = self.storeNative(folder, subjectId, summed, path)
        plan, template = context["plan"], context["template"]
        if not progress("normalization"):
            return None
        method = METHOD_PET_TEMPLATE if plan["method"] == normaldb.NORM_PET_TEMPLATE else METHOD_PET
        result = self.normalize(sequence, template, method, None, context["settings"], context["mask"],
                                lambda i, n, text: progress(f"normalization: {text}"), petImage=summed)
        if result.get("stopped") or not result.get("steps"):
            if result.get("bspline") is not None:
                slicer.mrmlScene.RemoveNode(result["bspline"])
            return None
        notes.extend(result.get("notes") or [])
        chain = DynamicPETRegistrationLogic.transformNode("vtkMRMLLinearTransformNode", "Batch normalization")
        chain.SetMatrixTransformToParent(_vtkMatrix(result["linear"]))
        if result.get("bspline") is not None:
            chain.SetAndObserveTransformNodeID(result["bspline"].GetID())
        description = f"Normalized to {template.GetName()} ({reg.stepsText(result['steps'])}, batch)"
        _, sequence, summed, _ = self.harden(sequence, template, chain, reg.OVERWRITE, reg.OUTPUT_PET_SPACING, None,
                                             "Linear", description, method)
        used = {k: plan.get(k) for k in normaldb.NORM_KEYS}
        for node in (sequence, summed):
            if node is not None:
                node.SetAttribute(NORMALIZATION_ATTRIBUTE, json.dumps(used))
        tb.writeCurrent(folder, subjectId, result["linear"], template.GetName(), result.get("bspline") is not None)

        # the database's regions, the region fits and the voxel maps with its kinetic settings
        if not progress("regions"):
            return None
        segmentation, referenceId = self.loadAtlasRegions(context["atlas"], **context["regions"])
        if not referenceId:
            raise ValueError("the database's reference region was not found in the regions")
        modeling = DynamicPETModelingLogic()
        timeZero = kinetics["timeZero"]
        offsetS = 60.0 * float(kinetics.get("injectionMin") or 0.0)
        try:
            analysis = modeling.prepare(sequence, segmentation, timeZero, offsetS)
        except ValueError as error:
            if timeZero == normaldb.TIME_FIRST_FRAME:
                raise
            timeZero = normaldb.TIME_FIRST_FRAME
            analysis = modeling.prepare(sequence, segmentation, timeZero)
            notes.append(f"time zero: the first frame ({error})")
        model = kinetics["model"]
        options = kin.defaultOptions(model)
        options.update(normaldb.kineticOptions(kinetics))
        if not progress("fitting the regions"):
            return None
        modeling.lastK2ref = None
        results = modeling.fitSegments(analysis, model, referenceId, options)
        k2ref = None
        if model == "SRTM2":
            k2ref = options.get("k2ref") or (modeling.lastK2ref[0] if modeling.lastK2ref else None)
            if k2ref:
                options["k2ref"] = k2ref
        referenceName = analysis.name(referenceId)
        names = {segmentId: analysis.name(segmentId) for segmentId in results}
        fits = {names[s]: dict(r.get("values") or {}) for s, r in results.items()
                if not r.get("error") and not nr.isReferenceRegion(names.get(s, ""))}
        failed = [names.get(s, s) for s, r in results.items() if r.get("error")]
        if failed:
            notes.append("regions not fitted: " + ", ".join(failed[:6]) + (" ..." if len(failed) > 6 else ""))
        window = options.get("window")
        state = {"kind": "all", "model": model, "timeZero": timeZero,
                 "weighting": options["weighting"], "tStar": float(options.get("tStar") or 0.0), "autoTStar": False,
                 "k2ref": float(kinetics.get("k2ref") or 0.0), "window": list(window) if window else [0.0, 0.0],
                 "clip": bool(options.get("clipNegative", True)), "referenceName": referenceName, "batch": True}
        if not progress(f"the {model} maps"):
            return None
        maps = modeling.parametricMaps(analysis, model, referenceId, options, None, 0.0, False,
                                       lambda i, n: progress(f"the {model} maps {i} / {n}"),
                                       brainMask=context["brainMask"], settings=state)
        if not maps:
            return None

        if not progress("saving"):
            return None
        if timeZero != kinetics["timeZero"]:
            notes.append("the maps were not made with the database's time zero")
        settingsUsed = dn.mapSettings(state, referenceName)
        row = self.addDynamicSubject(
            folder, header, maps, summed, subjectId, context["gridPath"], age, sex, used, fits, k2ref=k2ref,
            referenceName=referenceName, settings=settingsUsed,
            writeRegionsCsv=lambda csvPath: modeling.exportCsv(csvPath, analysis, results, model),
            source=os.path.basename(path), scanDate="", replace=replace, native=native,
            seriesSmoothing=context["smoothing"], sourceFile=os.path.abspath(path))
        if context["anatomy"] is not None and row.get("suvFile"):
            try:
                self._writeSummedQc(folder, row["suvFile"], subjectId, context)
            except Exception as error:      # noqa: BLE001 - said in the log
                logging.exception(f"Dynamic Neuro batch: no QC picture of {subjectId}")
                notes.append(f"no QC picture ({error})")
        row["notes"] = "; ".join(n for n in notes if n)
        return row

    def _writeSummedQc(self, folder, relative, subjectId, context):
        """The QC picture (the normalized summed image over the space's T1) of a subject's stored summed image."""
        from StaticNeuroPETLib.logic_batch import BatchLogicMixin
        node = slicer.util.loadVolume(os.path.join(folder, relative), {"show": False})
        try:
            BatchLogicMixin.writeQcPicture(os.path.join(context["qcFolder"], normaldb.safeName(subjectId) + "_qc.png"),
                                           node, context["anatomy"])
        finally:
            slicer.mrmlScene.RemoveNode(node)

    @staticmethod
    def _writeDynamicBatchLog(path, rows):
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=BATCH_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    # --- the database's template of its subjects' summed images ----------------------------------------
    def buildSummedTemplate(self, folder, atlas=None):
        """The database's normalization template (PET to PET): the mean of its subjects' normalized summed images
        (suvFile), each scaled to a mean of TEMPLATE_SCALE in the space's brain mask; saved as template.nii.gz and
        made the database's normalization (normaldb.useTemplate). The subjects keep their maps: make them again
        with the template (batch, replace) for a database normalized to it. Returns (header, n)."""
        header = normaldb.readHeader(folder)
        if not normaldb.isDynamic(header):
            raise ValueError("The template of summed images is for dynamic databases.")
        subjects = [s for s in normaldb.readSubjects(folder) if s.get("suvFile")]
        if len(subjects) < 2:
            raise ValueError("The template needs the summed images of two subjects at least.")
        atlas = atlas or neuro.atlasFiles(header.get("space"))
        mask, geometry = None, None
        if neuro.BRAIN_MASK in atlas:
            maskNode = self.loadAtlasVolume(atlas[neuro.BRAIN_MASK],
                                            name=f"{neuro.spaceTitle(header.get('space'))} brain mask")
            mask = slicer.util.arrayFromVolume(maskNode) > 0
        arrays = []
        for subject in subjects:
            path = os.path.join(folder, subject["suvFile"])
            if not os.path.isfile(path):
                continue
            node = slicer.util.loadVolume(path, {"show": False})
            try:
                if geometry is None:
                    geometry = vtk.vtkMatrix4x4()
                    node.GetIJKToRASMatrix(geometry)
                arrays.append(slicer.util.arrayFromVolume(node).copy())
            finally:
                slicer.mrmlScene.RemoveNode(node)
        if mask is not None and arrays and mask.shape != arrays[0].shape:
            mask = None
        mean, count = tb.scaledMean(arrays, mask)
        output = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "summed template (temporary)")
        pending = os.path.join(folder, "template_summed.pending.nii.gz")
        try:
            output.SetIJKToRASMatrix(geometry)
            slicer.util.updateVolumeFromArray(output, mean)
            if not slicer.util.saveNode(output, pending):
                raise RuntimeError(f"Could not write {pending}")
        finally:
            slicer.mrmlScene.RemoveNode(output)
        try:
            header = normaldb.useTemplate(folder, pending, n=count, built="summedMean",
                                          scaling=f"each summed image to a mean of {TEMPLATE_SCALE:g} in the brain")
        finally:
            if os.path.isfile(pending):
                os.remove(pending)
        return header, count
