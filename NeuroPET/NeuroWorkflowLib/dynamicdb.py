"""
Dynamic normal databases: a patient's parametric maps, summed image and regional TAC fits added to a user defined
dynamic database (the Review/Compare step of the Dynamic Neuro workflow, DynamicNeuroPET.py).

A dynamic subject's folder (subjects/<id>/, normaldb):
    <id>_<map>.nii.gz       each map of the database's model (normaldb.kinetics maps), on the database grid
    <id>_summed.nii.gz      the summed image normalized, on the grid (the PET to PET template's source)
    <id>_fits.json          the regions' TAC fit values {"model", "k2ref", "reference", "settings", "regions":
                            {region: {map: value}}}
    <id>_regions.csv        the region fits as the Modeling export writes them, the TACs beside it
                            (<id>_regions_TACs.csv)
subjects.csv: suvFile = the summed image, mapFiles = {map: file} (JSON), fitsFile, tacsFile.
"""

import json
import logging
import os

import numpy as np
import slicer

from DICOMGatedDynamicPETLib import resampling
from NeuroWorkflowLib.base import _ijkToRas, _vtkMatrix
from NeuroPETLib import normaldb


class DynamicDatabaseLogicMixin:
    """The logic of adding a patient to a dynamic normal database (mixed into DynamicNeuroPETLogic)."""

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
                          scanDate="", replace=False):
        """Add a patient to a user defined dynamic database: maps {map name: volume} (every map of the database's
        model; normalized, made with its kinetic settings) and the summed image (normalized) resampled onto the
        database grid (gridPath: its reference mask, or the space's brain mask) and saved in subjects/<id>/; fits
        {region: {map: value}} the regions' TAC fit values (k2ref: the reference's k2' used, settings: the maps'
        kinetic settings, both kept with them); writeRegionsCsv(path) writes the region fits CSV (and its TACs).
        Returns the row written into subjects.csv."""
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
               "mapFiles": json.dumps(files, sort_keys=True), "fitsFile": fitsFile, "tacsFile": tacsFile}
        if old and replace:         # the old subject's files not written again go; its native image is kept
            for key in ("nativeFile", "scanDate", "source"):
                if not row.get(key) and old.get(key):
                    row[key] = old[key]
            stale = set(normaldb.subjectMapFiles(old).values()) | {old.get("suvFile"), old.get("suvrFile")}
            stale -= set(files.values()) | {summedFile}
            for relative in stale:
                path = os.path.join(folder, relative) if relative else ""
                if path and os.path.isfile(path):
                    os.remove(path)
        normaldb.addSubject(folder, row, normalization, replace)
        logging.info(f"Dynamic Neuro: '{subjectId}' added to {folder} ({', '.join(files)})")
        return row
