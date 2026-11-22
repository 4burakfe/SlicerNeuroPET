"""
Batch: many normal subjects into a database, the iterative template of a database, QC pictures and logs.

Part of the Static Neuro PET/SPECT module (StaticNeuroPET.py): the methods of BatchLogicMixin, mixed into
StaticNeuroPETLogic there.
"""

import json
import logging
import os
import time

import numpy as np
import slicer
import vtk

from DICOMGatedDynamicPETLib import registration as reg
from NeuroWorkflowLib.base import (METHOD_ANATOMY, METHOD_PET, METHOD_PET_TEMPLATE, NORMALIZATION_ENGINE,
                                   NORMALIZATION_ITERATIVE_DEFORMABLE, NORMALIZATION_ITERATIVE_LINEAR,
                                   NORMALIZATION_STEPS, NORMALIZATION_WORKING_MM, PET_TEMPLATE_ATTRIBUTE, _ijkToRas,
                                   _vtkMatrix)
from NeuroPETLib import neuro, normaldb, staticneuro as sn, templatebuild as tb

from StaticNeuroPETLib.constants import BATCH_PET_FALLBACK, BATCH_SKIP, NORMALIZATION_ATTRIBUTE, TEMPLATE_KEY_ATTRIBUTE
from StaticNeuroPETLib.helpers import _fmt


# ---------------------------------------------------------------------------
class BatchTemplates:
    """The templates of a batch into a database, each loaded once: the volumes of its space (atlasFiles keys; the
    synthetic PET made when needed), its own ("From DB": 'db:PET', 'db:CT', ...), the brain mask and the CT
    template's correction to the space."""

    def __init__(self, logic, folder, header, atlas):
        self.logic, self.folder, self.header, self.atlas = logic, folder, header, atlas
        self.database = normaldb.databaseVolumes(folder, header)
        self.spaceName = neuro.spaceTitle(header.get("space"))
        self._nodes, self._correction = {}, False

    def path(self, key):
        if normaldb.isDatabaseTemplate(key):
            path = self.database.get(key)
            if not path:
                raise ValueError(f"'{self.header['name']}' has no {normaldb.templateTitle(key)[len('From DB: '):]} "
                                 "(compute its mean / SD maps or build its template).")
            return path
        path = self.atlas.get(key)
        if not path and key in neuro.GENERATED and neuro.canMakeTemplate(self.atlas, key):
            path = self.logic.ensureSyntheticPet(self.atlas, key)
        if not path:
            raise ValueError(f"The template '{neuro.templateTitle(key)}' is not in the database's space "
                             f"({self.spaceName}).")
        return path

    def get(self, key):
        """The template volume of a key (loaded once, tagged with its key)."""
        if key not in self._nodes:
            volume = normaldb.databaseTemplateVolume(key)
            name = f"{self.header['name']} {volume}" if volume else f"{self.spaceName} {key}"
            node = self.logic.loadAtlasVolume(self.path(key), name=name)
            node.SetAttribute(TEMPLATE_KEY_ATTRIBUTE, key)
            if key in neuro.GENERATED or key == normaldb.DB_PET:
                node.SetAttribute(PET_TEMPLATE_ATTRIBUTE, "1")
            self._nodes[key] = node
        return self._nodes[key]

    def forAnatomy(self, key, contrast):
        """(template whose grid the result lies on, CT template or None, its correction or None) of a CT / MR
        registered to key: a CT to the space's CT template (on the T1's grid, corrected to the space), to the
        database's mean CT, or (no CT template in the space) to the T1; an MR to its contrast's template, the T1 when
        the space has none of that contrast."""
        if contrast == "CT" and key == neuro.CT_TEMPLATE and neuro.CT_TEMPLATE not in self.atlas:
            key = neuro.DEFAULT_TEMPLATE
        if contrast == "CT" and key == neuro.CT_TEMPLATE:
            ct = self.get(key)
            return self.get(neuro.DEFAULT_TEMPLATE), ct, self.ctCorrection(ct)
        if contrast == "CT" and key == normaldb.databaseTemplateKey("CT"):
            ct = self.get(key)
            return ct, ct, None
        if contrast != "CT" and not normaldb.isDatabaseTemplate(key) and not neuro.canMakeTemplate(self.atlas, key):
            key = neuro.DEFAULT_TEMPLATE        # a T2 / PD without its template in the space: the T1 (as the workflow)
        return self.get(key), None, None

    def ctCorrection(self, ctTemplate):
        """The space's CT template to the space (NeuroWorkflowLogic.ctTemplateCorrection), once; None without
        the grey matter maps."""
        if self._correction is False:
            self._correction = None
            if neuro.CT_TEMPLATE_GM in self.atlas and "GM" in self.atlas:
                self._correction = self.logic.ctTemplateCorrection(
                    self.logic.loadAtlasVolume(self.atlas[neuro.CT_TEMPLATE_GM], name="CT template grey matter"),
                    self.logic.loadAtlasVolume(self.atlas["GM"], name=f"{self.spaceName} GM"),
                    neuro.ctCorrectionPath(self.atlas), ctTemplate, neuro.ctCorrectedPath(self.atlas))
        return self._correction

    def mask(self, template):
        """The space's brain mask when it lies on the template's grid, else None."""
        if neuro.BRAIN_MASK not in self.atlas:
            return None
        mask = self.logic.loadAtlasVolume(self.atlas[neuro.BRAIN_MASK], name=f"{self.spaceName} brain mask")
        same = mask.GetImageData().GetDimensions() == template.GetImageData().GetDimensions()
        return mask if same else None


class BatchLogicMixin:
    """The logic batch methods of StaticNeuroPETLogic."""

    # --- batch: many normal subjects into a database ----------------------------------------------
    BATCH_COLUMNS = ("file", "id", "status", "message", "referenceMean", "seconds")

    def batchPlan(self, header, folder=None):
        """(plan, registration settings) of a batch into the database: its normalization settings
        (staticneuro.batchNormalization; PET to PET when the database in folder has its template) and the registration
        settings they give."""
        hasTemplate = bool(folder) and bool(sn.petTemplatePath(folder, header))
        plan = sn.batchNormalization(header.get("normalization"), neuro.DEFAULT_TEMPLATE, NORMALIZATION_ENGINE,
                                     {reg.ENGINE_ITERATIVE: (NORMALIZATION_ITERATIVE_LINEAR,
                                                             NORMALIZATION_ITERATIVE_DEFORMABLE),
                                      reg.ENGINE_BRAINSFIT: (NORMALIZATION_STEPS[0], "BSpline" in NORMALIZATION_STEPS)},
                                     hasTemplate)
        settings = self.defaultSettings(plan["method"])
        if plan["engine"] == reg.ENGINE_ITERATIVE:
            settings.update(engine=reg.ENGINE_ITERATIVE, iterativeLinear=plan["linear"],
                            iterativeDeformable=plan["deformable"],
                            steps=reg.iterativeSteps(plan["linear"], plan["deformable"]))
        else:
            steps = reg.validateSteps([plan["linear"]] + (["BSpline"] if plan["deformable"] else []))
            settings.update(engine=reg.ENGINE_BRAINSFIT, steps=steps,
                            resample=[NORMALIZATION_WORKING_MM > 0] * len(steps))
        return plan, settings

    def batchAdd(self, folder, files, atlas=None, qc=True, progress=None, replace=False, align=True,
                 withoutAnatomy=BATCH_PET_FALLBACK):
        """Add images of normal subjects to a user defined database, one after the other. files: batch items
        (batchItems: path, subject ID, extra) - BIDS PET files, plain image files with a subject table, or the
        database's stored native images (normalized again) - or plain paths (BIDS). extra {"anatomy": file,
        "contrast"}: the subject's CT / MR (optional), the PET put on it first (registered rigidly with align, else
        taken as lying on it), stored with it and normalized with it. Each loaded (and its native images stored),
        normalized with the database's normalization settings (no manual check): the PET to the space's template or
        to the database's own (PET to PET), or the CT / MR to the template of its contrast (CT/MR method; a subject
        without one: withoutAnatomy BATCH_PET_FALLBACK registers its PET to the database's MNI target, noted in the
        log, BATCH_SKIP skips it), resampled onto the template grid and added (logic.addSubject, the database's
        reference mask). IDs already in the database are skipped (replace: normalized again and replaced); a subject
        that fails is logged and the batch goes on. qc: a PNG of the normalized PET over the space's T1 (3 slices)
        in <database>/qc/. progress(index, total, text) returns False to stop. The log is
        <database>/qc/batch_<time>.csv. Returns (rows, log path, stopped)."""
        header = normaldb.readHeader(folder)
        if header.get("kind") != normaldb.USER_DEFINED:
            raise ValueError("Subjects can be added to user defined databases only.")
        atlas = atlas or neuro.atlasFiles(header.get("space"))      # the volumes of the database's space
        gridPath = sn.referenceMaskPath(folder, header)
        if not gridPath:
            raise ValueError(f"'{header['name']}' has no reference mask (reference_mask in its folder).")
        plan, settings = self.batchPlan(header, folder)
        fallback = None
        if plan["method"] == METHOD_ANATOMY:
            norm = dict(normaldb.emptyNormalization(), **(header.get("normalization") or {}))
            norm.update(method=normaldb.NORM_PET, template=normaldb.normalizationTarget(header), brainMask=None)
            fallback = self.batchPlan(dict(header, normalization=norm), None)
        templates = BatchTemplates(self, folder, header, atlas)
        anatomy = None
        if qc and "T1" in atlas:
            t1 = self.loadAtlasVolume(atlas["T1"], name=f"{neuro.spaceTitle(header.get('space'))} T1")
            anatomy = (sn.rasArray(slicer.util.arrayFromVolume(t1), _ijkToRas(t1)), tuple(_ijkToRas(t1).ravel()))
        qcFolder = os.path.join(folder, "qc")
        os.makedirs(qcFolder, exist_ok=True)
        logPath = os.path.join(qcFolder, "batch_" + time.strftime("%Y%m%d_%H%M%S") + ".csv")
        items = [f if isinstance(f, (tuple, list)) else self.batchItems(folder, [f])[0] for f in files]
        rows, stopped, total = [], False, len(items)
        for index, item in enumerate(items):
            path, subjectId = item[0], item[1]
            if progress is not None and not progress(index, total, f"{subjectId} ({index + 1} / {total})"):
                stopped = True
                break
            row = {"file": path, "id": subjectId, "status": "", "message": "", "referenceMean": "", "seconds": ""}
            start = time.time()
            hasAnatomy = bool(((item[2] if len(item) > 2 else None) or {}).get("anatomy"))
            if normaldb.hasSubject(folder, subjectId) and not replace:
                row.update(status="skipped", message="already in the database")
            elif fallback is not None and not hasAnatomy and withoutAnatomy == BATCH_SKIP:
                row.update(status="skipped", message="no CT/MR (the database's method is CT/MR)")
            else:
                def stepProgress(step, steps, text, index=index, subjectId=subjectId):
                    return progress is None or progress(index, total, f"{subjectId} ({index + 1} / {total}): {text}")
                # the subject's files as they were: a failure (or a stop) half way leaves them so - the old native
                # image with its own maps (replace), no folder of a half added new subject
                backup = normaldb.backupSubjectFolder(folder, subjectId)
                try:
                    subjectPlan = (plan, settings) if fallback is None or hasAnatomy else fallback
                    added = self._batchSubject(folder, header, item, subjectId, templates, subjectPlan, gridPath,
                                               anatomy, qcFolder, stepProgress, replace, align)
                    if added is None:
                        normaldb.restoreSubjectFolder(folder, subjectId, backup)
                        row.update(status="stopped", message="stopped during the registration")
                        stopped = True
                    else:
                        notes = added.get("notes", "")
                        if subjectPlan is fallback:
                            notes = "; ".join(n for n in ("no CT/MR: the PET registered to "
                                                          + normaldb.templateTitle(fallback[0]["template"]), notes)
                                              if n)
                        row.update(status="added", referenceMean=added["referenceMean"], message=notes)
                        normaldb.dropSubjectBackup(folder, backup)
                except Exception as error:
                    logging.exception(f"Static Neuro batch: {path} failed")
                    message = str(error).replace("\n", " ")
                    try:
                        normaldb.restoreSubjectFolder(folder, subjectId, backup)
                        message += " (its files left as they were)" if backup["existed"] else ""
                    except Exception as restoreError:       # noqa: BLE001 - said in the log row
                        logging.exception(f"Static Neuro batch: could not restore the files of {subjectId}")
                        message += f" (its old files could not be put back: {restoreError}; copy in {backup['backup']})"
                    row.update(status="failed", message=message)
                header = normaldb.readHeader(folder)
            row["seconds"] = f"{time.time() - start:.0f}"
            rows.append(row)
            self._writeBatchLog(logPath, rows)
            if stopped:
                break
        if not rows:
            self._writeBatchLog(logPath, rows)
        return rows, logPath, stopped

    def _batchSubject(self, folder, header, item, subjectId, templates, subjectPlan, gridPath, qcAnatomy, qcFolder,
                      progress, replace=False, align=True):
        """One subject of batchAdd(): the row added (with "notes"), None when the registration was stopped."""
        from DynamicPETRegistration import DynamicPETRegistrationLogic
        scene = slicer.mrmlScene
        nodes = []
        plan, settings = subjectPlan
        try:
            volume, isNative = self._loadSource(folder, item)
            nodes.append(volume)
            anatomy, contrast, anatomyStored, placement = self._loadAnatomy(folder, item, volume, align, progress)
            nodes.append(anatomy)
            # the images as added (the PET put on its CT / MR), before anything else moves them; stored ones as they are
            native = None if isNative else self.storeNative(folder, subjectId, volume, item[0])
            anatomyNative = None
            if anatomy is not None and anatomyStored:
                anatomyNative = os.path.relpath(item[2]["anatomy"], folder).replace(os.sep, "/")
            elif anatomy is not None:
                anatomyNative = self.storeAnatomy(folder, subjectId, anatomy, contrast,
                                                  (placement or {}).get("file", ""), placement)
            notes = sn.unitWarnings(self.imageInfo(volume), header)
            method = plan["method"]
            if method == METHOD_ANATOMY:
                key = sn.anatomyTemplateKey(plan["template"], contrast)
                template, ctTemplate, ctCorrection = templates.forAnatomy(key, contrast)
            else:
                template, ctTemplate, ctCorrection = templates.get(plan["template"] if method == METHOD_PET
                                                                   else normaldb.DB_PET), None, None
            mask = templates.mask(template) if plan["brainMask"] else None
            result = self.normalize(volume, template, method, anatomy, settings, mask, progress,
                                    ctTemplate=ctTemplate, ctCorrection=ctCorrection)
            nodes.append(result.get("bspline"))
            if result.get("stopped") or not result.get("steps"):
                return None
            notes.extend(result.get("notes") or [])
            chain = DynamicPETRegistrationLogic.transformNode("vtkMRMLLinearTransformNode", "Batch normalization")
            nodes.append(chain)
            chain.SetMatrixTransformToParent(_vtkMatrix(result["linear"]))
            if result.get("bspline") is not None:
                chain.SetAndObserveTransformNodeID(result["bspline"].GetID())
            description = f"Normalized to {template.GetName()} ({reg.stepsText(result['steps'])}, batch)"
            _, normalized, _, normalizedAnatomy = self.harden(volume, template, chain, reg.SAVE_SEPARATELY,
                                                              reg.OUTPUT_FIXED_GRID, None, "Linear", description,
                                                              method, anatomy)
            nodes.extend((normalized, normalizedAnatomy))
            used = dict(plan, template=template.GetAttribute(TEMPLATE_KEY_ATTRIBUTE) or plan["template"])
            normalized.SetAttribute(NORMALIZATION_ATTRIBUTE, json.dumps(used))
            age, sex = self.ageAndSex(normalized)
            if age is None:
                notes.append("age not known")
            # the transform first: addSubject writing the subject's row is the last step (a failure before it is
            # undone by restoring the subject's folder)
            tb.writeCurrent(folder, subjectId, result["linear"], template.GetName(), result.get("bspline") is not None)
            row = self.addSubject(folder, header, normalized, subjectId, gridPath, age, sex, used, replace,
                                  native=native, anatomy=normalizedAnatomy, contrast=contrast,
                                  anatomyNative=anatomyNative, placement=placement)
            if qcAnatomy is not None:
                try:
                    self.writeQcPicture(os.path.join(qcFolder, normaldb.safeName(subjectId) + "_qc.png"), normalized,
                                        qcAnatomy)
                except Exception as error:
                    logging.exception(f"Static Neuro batch: no QC picture of {subjectId}")
                    notes.append(f"no QC picture ({error})")
            row["notes"] = "; ".join(notes)
            return row
        finally:
            for node in nodes:
                if node is not None and scene.IsNodePresent(node):
                    scene.RemoveNode(node)

    # --- iterative template of a normal database ---------------------------------------------------
    def buildTemplate(self, folder, atlas=None, maxPasses=tb.MAX_PASSES, stopMm=tb.STOP_MM, qc=True, progress=None,
                      items=None):
        """An iterative (group-wise) template of a user defined database from the stored native images of its
        subjects (items: batch items instead, see batchItems; see templatebuild): pass 0 registers each subject to
        the database's MNI target (normaldb.normalizationTarget, e.g. the synthetic PET), the mean of their maps
        (smoothed with the database's FWHM) is template 0; pass k registers each subject to template k-1, starting
        from its previous matrix. After each pass the group's drift since pass 0 is taken out, the matrices are
        saved (subjects/<id>/transform.json), every subject is resampled
        once from its native image and replaced in the database, and the new template is saved
        (templates/pass_<k>.nii.gz). Stops when the subjects move less than stopMm between two passes, after
        maxPasses, or when progress(index, total, text) returns False. The last template becomes the database's
        normalization template (PET to PET), also when a later pass fails. A stop while the subjects of a pass are
        resampled finishes that pass first (all subjects on one pass); a failure there puts the subjects already
        resampled back to the pass before. Linear registration only. Returns (log rows, log path, stopped)."""
        header = normaldb.readHeader(folder)
        if header.get("kind") != normaldb.USER_DEFINED:
            raise ValueError("A template is built for user defined databases only.")
        atlas = atlas or neuro.atlasFiles(header.get("space"))      # the volumes of the database's space
        gridPath = sn.referenceMaskPath(folder, header)
        if not gridPath:
            raise ValueError(f"'{header['name']}' has no reference mask (reference_mask in its folder).")
        if items is None:
            items = self.batchItems(folder)
            have, count = normaldb.nativeCount(folder)
            if have < count:
                raise ValueError(f"{count - have} of the {count} subjects have no native image in the database: "
                                 "add them again (replace) or remove them.")
        files = [i[0] for i in items]
        # pass 0: the database's PET to MR template normalization, to its MNI target (kept when it became PET to PET)
        norm0 = dict(normaldb.emptyNormalization(), **(header.get("normalization") or {}))
        norm0.update(method=normaldb.NORM_PET, template=normaldb.normalizationTarget(header))
        plan, settings = self.batchPlan(dict(header, normalization=norm0), None)
        plan = dict(plan, deformable=False)
        if settings.get("engine") == reg.ENGINE_ITERATIVE:
            settings.update(iterativeDeformable=False, steps=reg.iterativeSteps(plan["linear"], False))
        else:
            steps = [s for s in settings["steps"] if s != "BSpline"] or ["Affine"]
            settings.update(steps=steps, resample=list(settings.get("resample") or [True] * len(steps))[:len(steps)])
        key = plan["template"]
        templatePath = atlas.get(key)
        if not templatePath and key in neuro.GENERATED and neuro.canMakeTemplate(atlas, key):
            templatePath = self.ensureSyntheticPet(atlas, key)
        if not templatePath:
            raise ValueError(f"The template '{key}' (pass 0) is not in the database's space "
                             f"({neuro.spaceTitle(header.get('space'))}).")
        mniTemplate = self.loadAtlasVolume(templatePath, name=f"{neuro.spaceTitle(header.get('space'))} {key}")
        spaceName = neuro.spaceTitle(header.get("space"))
        scaleMask = brainMask = None        # the brain mask: the MR templates' scale, the skull strip of the target
        if neuro.BRAIN_MASK in atlas:
            scaleMask = self.loadAtlasVolume(atlas[neuro.BRAIN_MASK], name=f"{spaceName} brain mask")
            brainMask = scaleMask if plan["brainMask"] else None
        anatomy = None
        if qc and "T1" in atlas:
            t1 = self.loadAtlasVolume(atlas["T1"], name=f"{spaceName} T1")
            anatomy = (sn.rasArray(slicer.util.arrayFromVolume(t1), _ijkToRas(t1)), tuple(_ijkToRas(t1).ravel()))
        qcFolder = os.path.join(folder, "qc")
        os.makedirs(qcFolder, exist_ok=True)
        logPath = os.path.join(qcFolder, tb.ITERATIONS_LOG)
        ids = {i[0]: i[1] for i in items}
        itemOf = {i[0]: i for i in items}
        rows, initial, previous, previousTemplate, isConverged, lastAnatomy = [], {}, {}, None, False, {}
        target, method, stepSettings, stopped, lastPass = mniTemplate, METHOD_PET, dict(settings), False, None
        targetName = mniTemplate.GetName()
        try:
            for passIndex in range(int(maxPasses)):
                start = time.time()
                todo = [p for p in files if passIndex == 0 or ids[p] in previous]
                total = len(todo)
                # the brain mask strips the target (the MNI template, or a template on its grid)
                mask = brainMask
                if mask is not None and target.GetImageData().GetDimensions() != mask.GetImageData().GetDimensions():
                    mask = None
                current, failed = {}, []
                for index, path in enumerate(todo):
                    subjectId = ids[path]
                    label = f"Pass {passIndex}: registering {subjectId} ({index + 1} / {total})"
                    if progress is not None and not progress(index, total, label):
                        stopped = True
                        break
                    try:
                        matrix = self._templateRegister(folder, itemOf[path], target, method, stepSettings, mask,
                                                        previous.get(subjectId),
                                                        lambda s, n, t, i=index, text=label: progress is None
                                                        or progress(i, total, f"{text}: {t}"))
                    except Exception as error:
                        logging.exception(f"Static Neuro template: {path} failed in pass {passIndex}")
                        failed.append(f"{subjectId}: {error}")
                        matrix = previous.get(subjectId)
                    if matrix is False:
                        stopped = True
                        break
                    if matrix is not None:
                        current[subjectId] = matrix
                if stopped or len(current) < 2:
                    if not stopped:
                        raise ValueError(f"Pass {passIndex}: fewer than two subjects registered.")
                    break
                if passIndex == 0:
                    initial, drift = dict(current), np.eye(4)
                else:
                    current, drift = tb.driftCorrected(current, initial)
                byId = {ids[p]: p for p in todo}
                passPlan = dict(plan, method=method if passIndex else normaldb.NORM_PET, templatePass=passIndex)
                resampled = []
                try:
                    for index, (subjectId, matrix) in enumerate(current.items()):
                        tb.writeTransform(folder, subjectId, passIndex, matrix, byId.get(subjectId, ""))
                        text = (f"Pass {passIndex}: resampling {subjectId} ({index + 1} / {len(current)})"
                                + ("; stopping after this pass" if stopped else ""))
                        if progress is not None and not progress(index, len(current), text):
                            stopped = True      # the pass is finished first: every subject on the same pass
                        resampled.append(subjectId)
                        self._templateResample(folder, header, itemOf[byId[subjectId]], subjectId, matrix,
                                               mniTemplate, gridPath, passPlan, anatomy, qcFolder)
                        header = normaldb.readHeader(folder)
                except Exception:
                    if passIndex:
                        self._templateTakeBack(folder, header, itemOf, byId, resampled, previous, passIndex,
                                               dict(plan, method=method if passIndex > 1 else normaldb.NORM_PET,
                                                    templatePass=passIndex - 1),
                                               mniTemplate, gridPath, anatomy, qcFolder)
                    raise
                templateArray, geometry = self._meanSubjectMap(folder, header, list(current))
                relative = tb.templateFile(passIndex)
                self._saveArray(templateArray, geometry, os.path.join(folder, relative))
                # the subjects' CT / MR through the same matrices: a CT / MR template that goes with the PET's
                anatomyTemplates = {}
                for contrast, (array, anatomyGeometry, n) in self._meanAnatomyMaps(
                        folder, header, list(current), scaleMask).items():
                    anatomyRelative = tb.templateFile(passIndex, contrast)
                    self._saveArray(array, anatomyGeometry, os.path.join(folder, anatomyRelative))
                    anatomyTemplates[contrast] = {"file": os.path.join(folder, anatomyRelative), "n": n}
                moved = tb.passStatistics(current, previous) if passIndex else {"n": 0, "meanMm": "", "maxMm": ""}
                row = {"pass": passIndex, "target": targetName, "subjects": len(current), "failed": len(failed),
                       "meanMoveMm": _fmt(moved["meanMm"]), "maxMoveMm": _fmt(moved["maxMm"]),
                       "driftScale": _fmt(tb.scaleOf(drift), 4), "driftShiftMm": _fmt(float(np.linalg.norm(
                           drift[:3, 3])), 2),
                       "templateCorrelation": _fmt(tb.correlation(templateArray, previousTemplate), 5)
                       if previousTemplate is not None else "",
                       "seconds": f"{time.time() - start:.0f}", "failures": failed}
                rows.append(row)
                tb.writeIterationsLog(logPath, rows)
                logging.info(f"Static Neuro template: pass {passIndex} {row}")
                lastPass, lastAnatomy = passIndex, anatomyTemplates
                previous, previousTemplate = current, templateArray
                if passIndex > 0 and tb.converged(moved, stopMm):
                    isConverged = True
                    break
                if stopped:
                    break
                if target is not mniTemplate:
                    slicer.mrmlScene.RemoveNode(target)
                target = slicer.util.loadVolume(os.path.join(folder, relative),
                                                {"show": False, "name": f"{header['name']} template pass {passIndex}"})
                targetName = os.path.basename(relative)
                method = METHOD_PET_TEMPLATE
                stepSettings = dict(settings, initialization="Off")
        finally:
            if target is not mniTemplate and slicer.mrmlScene.IsNodePresent(target):
                slicer.mrmlScene.RemoveNode(target)
            # the last complete pass becomes the database's template, also when a later pass failed
            if lastPass is not None:
                try:
                    normaldb.useTemplate(folder, os.path.join(folder, tb.templateFile(lastPass)), n=len(previous),
                                         iteration=lastPass, built="iterative", passes=lastPass + 1,
                                         converged=isConverged, anatomy=lastAnatomy)
                except Exception:
                    logging.exception("Static Neuro template: could not set the template of the last pass")
        return rows, logPath, stopped

    def _templateTakeBack(self, folder, header, itemOf, byId, subjectIds, previous, passIndex, plan, gridTemplate,
                          gridPath, qcAnatomy, qcFolder):
        """A pass that failed half way taken back: its subjects already resampled are resampled again with the
        matrices of the pass before (previous), and the failed pass's matrix is removed from their transforms."""
        for subjectId in subjectIds:
            try:
                tb.dropPass(folder, subjectId, passIndex)
                if subjectId in previous:
                    tb.writeTransform(folder, subjectId, passIndex - 1, previous[subjectId],
                                      byId.get(subjectId, ""))
                    self._templateResample(folder, normaldb.readHeader(folder), itemOf[byId[subjectId]], subjectId,
                                           previous[subjectId], gridTemplate, gridPath, plan, qcAnatomy, qcFolder)
            except Exception:
                logging.exception(f"Static Neuro template: could not put {subjectId} back to pass {passIndex - 1}")

    def _templateRegister(self, folder, item, target, method, settings, mask, start, progress):
        """A subject's native image registered to target, starting from matrix start (None: the settings'
        initialization): its matrix (native to target, to parent); False when stopped."""
        volume = self._loadSource(folder, item)[0]
        result = None
        try:
            startMatrix = np.eye(4) if start is None else np.asarray(start, dtype=float)
            if start is not None:
                volume.SetIJKToRASMatrix(_vtkMatrix(startMatrix @ _ijkToRas(volume)))
            result = self.normalize(volume, target, method, None, settings, mask, progress)
            if result.get("stopped") or not result.get("steps"):
                return False
            return np.asarray(result["linear"], dtype=float) @ startMatrix
        finally:
            for node in (volume, (result or {}).get("bspline")):
                if node is not None and slicer.mrmlScene.IsNodePresent(node):
                    slicer.mrmlScene.RemoveNode(node)

    def _templateResample(self, folder, header, item, subjectId, matrix, gridTemplate, gridPath, plan, qcAnatomy,
                          qcFolder):
        """A subject's native image resampled once through matrix onto the template grid and replaced in the
        database (with its QC picture over qcAnatomy: the space's T1 as (rasArray, ijkToRas values), None: none)."""
        from DynamicPETRegistration import DynamicPETRegistrationLogic
        scene = slicer.mrmlScene
        nodes = []
        try:
            volume, isNative = self._loadSource(folder, item)
            nodes.append(volume)
            # the stored CT / MR (the native PET lies on it) goes through the same matrix
            anatomy, contrast, anatomyStored, placement = self._loadAnatomy(folder, item, volume)
            nodes.append(anatomy)
            native = None if isNative else self.storeNative(folder, subjectId, volume, item[0])
            anatomyNative = None
            if anatomy is not None:
                anatomyNative = (os.path.relpath(item[2]["anatomy"], folder).replace(os.sep, "/") if anatomyStored
                                 else self.storeAnatomy(folder, subjectId, anatomy, contrast,
                                                        (placement or {}).get("file", ""), placement))
            chain = DynamicPETRegistrationLogic.transformNode("vtkMRMLLinearTransformNode", "Template normalization")
            nodes.append(chain)
            chain.SetMatrixTransformToParent(_vtkMatrix(matrix))
            description = f"Iterative template, pass {plan.get('templatePass')} (linear)"
            _, normalized, _, normalizedAnatomy = self.harden(volume, gridTemplate, chain, reg.SAVE_SEPARATELY,
                                                              reg.OUTPUT_FIXED_GRID, None, "Linear", description,
                                                              plan["method"], anatomy)
            nodes.extend((normalized, normalizedAnatomy))
            normalized.SetAttribute(NORMALIZATION_ATTRIBUTE, json.dumps(plan))
            age, sex = self.ageAndSex(normalized)
            self.addSubject(folder, header, normalized, subjectId, gridPath, age, sex, plan, True, native=native,
                            anatomy=normalizedAnatomy, contrast=contrast, anatomyNative=anatomyNative,
                            placement=placement)
            if qcAnatomy is not None:
                try:
                    self.writeQcPicture(os.path.join(qcFolder, normaldb.safeName(subjectId) + "_qc.png"), normalized,
                                        qcAnatomy)
                except Exception:
                    logging.exception(f"Static Neuro template: no QC picture of {subjectId}")
        finally:
            for node in nodes:
                if node is not None and scene.IsNodePresent(node):
                    scene.RemoveNode(node)

    @staticmethod
    def _meanSubjectMap(folder, header, subjectIds):
        """(mean array, IJK-to-RAS 4x4) of the subjects' maps (SUVR or SUV by the map type), each smoothed with the
        database's FWHM."""
        mapKey = normaldb.subjectMapKey(header)
        wanted = set(subjectIds)
        running, geometry = normaldb.RunningStatistics(), None
        fwhm = float(header.get("smoothingFwhmMm") or 0)
        for subject in normaldb.readSubjects(folder):
            if subject["id"] not in wanted or not normaldb.subjectMapFile(subject, mapKey):
                continue
            node = slicer.util.loadVolume(os.path.join(folder, normaldb.subjectMapFile(subject, mapKey)),
                                         {"show": False})
            if node is None:
                continue
            try:
                geometry = _ijkToRas(node) if geometry is None else geometry
                running.add(normaldb.smoothMap(slicer.util.arrayFromVolume(node), fwhm,
                                               tuple(float(x) for x in node.GetSpacing())[::-1]))
            finally:
                slicer.mrmlScene.RemoveNode(node)
        if running.n < 2:
            raise ValueError("The template needs the maps of at least two subjects.")
        return np.asarray(running.mean(), dtype=np.float32), geometry

    @staticmethod
    def _meanAnatomyMaps(folder, header, subjectIds, brainMask=None):
        """{contrast: (mean array, IJK-to-RAS 4x4, n)} of the subjects' normalized CT / MR (unsmoothed; an MR scaled
        to its brain median first, normaldb.anatomyScale), for each contrast at least two subjects have."""
        wanted = set(subjectIds)
        maskArray = slicer.util.arrayFromVolume(brainMask) if brainMask is not None else None
        running, geometry = {}, {}
        for subject in normaldb.readSubjects(folder):
            contrast, path = subject.get("anatomy"), normaldb.anatomyPath(folder, subject)
            if subject["id"] not in wanted or not contrast or not path:
                continue
            node = slicer.util.loadVolume(path, {"show": False})
            if node is None:
                continue
            try:
                geometry.setdefault(contrast, _ijkToRas(node))
                running.setdefault(contrast, normaldb.RunningStatistics()).add(
                    normaldb.anatomyScale(slicer.util.arrayFromVolume(node), contrast, maskArray))
            except ValueError:
                logging.warning(f"Static Neuro: the {contrast} of {subject['id']} is not on the others' grid")
            finally:
                slicer.mrmlScene.RemoveNode(node)
        return {c: (np.asarray(r.mean(), dtype=np.float32), geometry[c], r.n) for c, r in running.items() if r.n >= 2}

    @staticmethod
    def _saveArray(array, ijkToRas, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "template (temporary)")
        try:
            node.SetIJKToRASMatrix(_vtkMatrix(ijkToRas))
            slicer.util.updateVolumeFromArray(node, array)
            if not slicer.util.saveNode(node, path):
                raise RuntimeError(f"Could not write {path}")
        finally:
            slicer.mrmlScene.RemoveNode(node)

    @staticmethod
    def writeQcPicture(path, volume, anatomy):
        """PNG of the volume over the anatomy ((rasArray, ijkToRas values) on the same grid): 3 slices."""
        from vtk.util import numpy_support
        anatomyArray, anatomyMatrix = anatomy
        if not np.allclose(np.asarray(anatomyMatrix).reshape(4, 4), _ijkToRas(volume), atol=1e-3):
            raise ValueError("the image is not on the T1 template grid")
        picture = sn.qcMontage(anatomyArray, sn.rasArray(slicer.util.arrayFromVolume(volume), _ijkToRas(volume)))
        height, width, _ = picture.shape
        image = vtk.vtkImageData()
        image.SetDimensions(width, height, 1)
        scalars = numpy_support.numpy_to_vtk(np.ascontiguousarray(picture[::-1]).reshape(-1, 3), deep=True,
                                             array_type=vtk.VTK_UNSIGNED_CHAR)
        scalars.SetNumberOfComponents(3)
        image.GetPointData().SetScalars(scalars)
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(path)
        writer.SetInputData(image)
        writer.Write()

    def _writeBatchLog(self, path, rows):
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.BATCH_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
