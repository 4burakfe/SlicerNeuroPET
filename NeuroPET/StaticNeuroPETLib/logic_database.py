"""
Normal databases: adding a patient, the native images of a database, its regions and their values, the
database folder.

Part of the Static Neuro PET/SPECT module (StaticNeuroPET.py): the methods of DatabaseLogicMixin, mixed into
StaticNeuroPETLogic there.
"""

import logging
import os

import numpy as np
import qt
import slicer

from DICOMGatedDynamicPETLib import resampling
from DynamicNeuroPET import DynamicNeuroPETLogic, NEURO_PREFIX, _ijkToRas, _vtkMatrix
from NeuroPETLib import neuro, normaldb, neuroreview as nr, staticneuro as sn

from StaticNeuroPETLib.constants import SETTINGS_DB_ROOT


class DatabaseLogicMixin:
    """The logic database methods of StaticNeuroPETLogic."""

    # --- adding a patient to a normal database --------------------------------------------------
    def addSubject(self, folder, header, volume, subjectId, gridPath, age, sex, normalization, replace=False,
                   segmentation=None, segmentId=None, referenceName="", native=None, anatomy=None, contrast="",
                   anatomyNative=None, placement=None):
        """Add a normalized volume to a user defined database: resampled onto the database grid (gridPath: its
        reference mask, or the MNI brain mask; MNI 2009c 1 mm) and saved as subjects/<id>/<id>_SUV.nii.gz and its
        SUVR map. The reference region (intensity normalization) is the segment segmentId of segmentation (the
        regions of step 4) or, without it, the mask gridPath. native: the image before the normalization, stored too
        (storeNative); a subject replaced without one keeps its stored native image. anatomy: the subject's CT / MR
        normalized with the same transform (optional, contrast 'CT' / 'T1' / ...), saved on the grid as
        subjects/<id>/<id>_<contrast>.nii.gz; anatomyNative: the CT / MR as it was (a volume, stored with
        storeAnatomy and placement, or the relative path of one stored already). A subject replaced without a CT / MR
        keeps its native one only when its native PET is kept too (native None: the stored PET lies on that CT / MR);
        with a new native PET the old CT / MR is removed (the new PET does not lie on it). Its normalized one (of the
        old transform) is removed either way. Returns the row written into subjects.csv."""
        # before anything is written: an ID already in (unless replace) or naming another subject's folder
        subjectId = normaldb.checkSubjectId(folder, subjectId, replace)
        scene = slicer.mrmlScene
        gridArray, gridNode = self._loadLabels(gridPath)
        nodes = [gridNode]
        try:
            dims = tuple(int(x) for x in gridNode.GetImageData().GetDimensions())
            image = resampling.Resampler(_ijkToRas(volume), dims, _ijkToRas(gridNode), None, True,
                                         "Linear").resample(volume.GetImageData())
            suvNode = scene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Neuro subject SUV (temporary)")
            nodes.append(suvNode)
            suvNode.SetIJKToRASMatrix(_vtkMatrix(_ijkToRas(gridNode)))
            suvNode.SetAndObserveImageData(image)
            values = slicer.util.arrayFromVolume(suvNode).astype(np.float32)
            if segmentation is not None and segmentId:
                mask = slicer.util.arrayFromSegmentBinaryLabelmap(segmentation, segmentId, gridNode)
                if mask is None:
                    raise ValueError("The reference segment is empty.")
                mask = np.asarray(mask) > 0
            else:
                mask = gridArray > 0
            ratio, mean = sn.ratioMap(values, mask)
            slicer.util.updateVolumeFromArray(suvNode, values)
            suvrNode = scene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Neuro subject SUVR (temporary)")
            nodes.append(suvrNode)
            suvrNode.SetIJKToRASMatrix(_vtkMatrix(_ijkToRas(gridNode)))
            slicer.util.updateVolumeFromArray(suvrNode, ratio)
            suvFile, suvrFile = normaldb.subjectFileNames(subjectId)
            os.makedirs(os.path.join(folder, normaldb.subjectDir(subjectId)), exist_ok=True)
            for node, relative in ((suvNode, suvFile), (suvrNode, suvrFile)):
                if not slicer.util.saveNode(node, os.path.join(folder, relative)):
                    raise RuntimeError(f"Could not write {relative}")
            anatomyFile = ""
            if anatomy is not None:
                contrast = contrast or self.anatomyContrast(anatomy)[0] or "T1"
                anatomyNode = scene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Neuro subject CT/MR (temporary)")
                nodes.append(anatomyNode)
                anatomyNode.SetIJKToRASMatrix(_vtkMatrix(_ijkToRas(gridNode)))
                anatomyNode.SetAndObserveImageData(self.resampleKeepingAir(anatomy, dims, _ijkToRas(gridNode)))
                anatomyFile = normaldb.anatomyFileName(subjectId, contrast)
                if not slicer.util.saveNode(anatomyNode, os.path.join(folder, anatomyFile)):
                    raise RuntimeError(f"Could not write {anatomyFile}")
        finally:
            for node in nodes:
                if node is not None and scene.IsNodePresent(node):
                    scene.RemoveNode(node)
        attributes = self.attributes(volume)
        scanDate = self.dicomValues(volume, ("0008,0020",))["0008,0020"] or ""
        if len(scanDate) == 8 and scanDate.isdigit():
            scanDate = f"{scanDate[:4]}-{scanDate[4:6]}-{scanDate[6:]}"
        row = {"id": subjectId, "age": "" if age is None else f"{age:g}", "sex": sex or "", "scanDate": scanDate,
               "suvFile": suvFile, "suvrFile": suvrFile, "referenceMean": f"{mean:.6g}",
               "reference": referenceName or "", "normalization": normaldb.normalizationText(normalization),
               "source": attributes.get(sn.IMPORTED_ATTRIBUTE) or volume.GetName()}
        old = next((s for s in normaldb.readSubjects(folder) if s.get("id", "").lower() == subjectId.lower()), None)
        if isinstance(native, str):        # stored already (storeNative)
            row["nativeFile"] = native
        elif native is not None:
            row["nativeFile"] = self.storeNative(folder, subjectId, native, row["source"])
        if anatomyFile:
            row.update(anatomy=contrast, anatomyFile=anatomyFile)
        if isinstance(anatomyNative, str):
            row["anatomyNativeFile"] = anatomyNative
        elif anatomyNative is not None:
            row["anatomyNativeFile"] = self.storeAnatomy(folder, subjectId, anatomyNative, contrast, "",
                                                         placement)
        if old and replace:        # kept from the subject's previous row: its native image, scan date, source
            for key in ("nativeFile", "scanDate", "source"):
                if not row.get(key) and old.get(key):
                    row[key] = old[key]
            oldAnatomy = old.get("anatomyNativeFile")
            if not row.get("anatomyNativeFile") and oldAnatomy:
                if native is None:          # the stored native PET is kept: its CT / MR still fits it
                    row["anatomyNativeFile"] = oldAnatomy
                    row["anatomy"] = row.get("anatomy") or old.get("anatomy", "")
                else:                       # a new native PET without a CT / MR: the old one does not fit it
                    for relative in (oldAnatomy, normaldb.anatomyInfoName(subjectId)):
                        path = os.path.join(folder, relative)
                        if os.path.isfile(path):
                            os.remove(path)
            previous = old.get("anatomyFile")      # the CT / MR of the old transform (or of another contrast)
            if previous and previous != row.get("anatomyFile") and os.path.isfile(os.path.join(folder, previous)):
                os.remove(os.path.join(folder, previous))
            for key in ("suvFile", "suvrFile"):     # an older layout's maps are not left behind
                previous = old.get(key)
                if previous and previous != row[key] and os.path.isfile(os.path.join(folder, previous)):
                    os.remove(os.path.join(folder, previous))
        normaldb.addSubject(folder, row, normalization, replace)
        logging.info(f"Static Neuro: '{subjectId}' added to {folder} (reference mean {mean:.4g})")
        return row

    # --- the native images of a normal database -------------------------------------------------------
    @staticmethod
    def _saveCopy(volume, folder, relative):
        """Save a copy of the volume (as it is, its storage node untouched) to folder/relative, unless it is that
        file already."""
        path = os.path.join(folder, relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if volume.GetParentTransformNode() is not None:
            raise ValueError(f"'{volume.GetName()}' is under a transform: harden it first.")
        storage = volume.GetStorageNode()
        if os.path.realpath((storage.GetFileName() if storage is not None else None) or "") == os.path.realpath(path):
            return
        copy = slicer.modules.volumes.logic().CloneVolume(slicer.mrmlScene, volume, "native (temporary)")
        try:
            if not slicer.util.saveNode(copy, path):
                raise RuntimeError(f"Could not write {relative}")
        finally:
            slicer.mrmlScene.RemoveNode(copy)

    def storeNative(self, folder, subjectId, volume, source=""):
        """Save the image as added (before the normalization) in the subject's folder with its tags
        (subjects/<id>/<id>_native.nii.gz and .json). Returns the relative path."""
        relative = normaldb.nativeFileName(subjectId)
        self._saveCopy(volume, folder, relative)
        scanDate = self.dicomValues(volume, ("0008,0020",))["0008,0020"] or ""
        normaldb.writeNativeInfo(folder, subjectId, {"id": subjectId, "source": source or volume.GetName(),
                                                      "scanDate": scanDate, "stored": normaldb.now(),
                                                      "attributes": self.attributes(volume)})
        return relative

    def storeAnatomy(self, folder, subjectId, volume, contrast, source="", placement=None):
        """Save the subject's CT / MR as it was (its native PET lies on it) in the subject's folder with its tags
        (subjects/<id>/<id>_anat_native.nii.gz and .json: contrast, source, placement - how the PET was put on it).
        Returns the relative path."""
        relative = normaldb.anatomyNativeFileName(subjectId)
        self._saveCopy(volume, folder, relative)
        normaldb.writeAnatomyInfo(folder, subjectId, {"id": subjectId, "contrast": contrast or "",
                                                       "source": source or volume.GetName(), "stored": normaldb.now(),
                                                       "placement": placement or {},
                                                       "attributes": self.attributes(volume)})
        return relative

    def loadAnatomyNative(self, folder, subjectId, path):
        """The stored native CT / MR of a subject as a volume, with its tags."""
        node = slicer.util.loadVolume(path, {"show": False, "name": f"{subjectId} CT-MR native"})
        if node is None:
            raise RuntimeError(f"Could not load {path}")
        for key, value in (normaldb.readAnatomyInfo(folder, subjectId).get("attributes") or {}).items():
            if value is not None:
                node.SetAttribute(key, str(value))
        return node

    def _loadAnatomy(self, folder, item, volume, align=True, progress=None):
        """The CT / MR of a batch item (extra "anatomy": its file, "contrast"): (volume or None, contrast, stored,
        placement). A stored native CT / MR is taken as it is (the stored PET lies on it); a file is loaded and the
        PET (volume) put on it: registered rigidly (align) and moved there (its geometry, no resampling), or taken
        as lying on it already (PET/CT, PET/MR)."""
        extra = item[2] if len(item) > 2 else None
        path = (extra or {}).get("anatomy")
        if not path:
            return None, "", False, None
        stored = os.path.realpath(path).startswith(os.path.realpath(os.path.join(folder, normaldb.SUBJECTS_DIR))
                                                   + os.sep)
        if stored:
            node = self.loadAnatomyNative(folder, item[1], path)
            return node, extra.get("contrast") or self.anatomyContrast(node)[0] or "T1", True, None
        if not os.path.isfile(path):
            raise ValueError(f"The CT/MR {path} does not exist.")
        node = slicer.util.loadVolume(path, {"show": False})
        if node is None:
            raise RuntimeError(f"Could not load the CT/MR {path}")
        contrast = extra.get("contrast") or self.anatomyContrast(node)[0] or "T1"
        if not align:
            return node, contrast, False, {"how": "taken as aligned", "file": path}
        result = None
        try:
            result = self.registerToAnatomy(volume, node, None, progress)
            if result.get("stopped") or not result.get("steps"):
                raise ValueError("the PET to CT/MR registration stopped")
            matrix = np.asarray(result["linear"], dtype=float)
            volume.SetIJKToRASMatrix(_vtkMatrix(matrix @ _ijkToRas(volume)))
        except Exception:
            slicer.mrmlScene.RemoveNode(node)
            raise
        finally:
            bspline = (result or {}).get("bspline")
            if bspline is not None and slicer.mrmlScene.IsNodePresent(bspline):
                slicer.mrmlScene.RemoveNode(bspline)
        return node, contrast, False, {"how": "registered (rigid)", "file": path,
                                       "matrix": [float(v) for v in matrix.ravel()]}

    def loadNative(self, folder, subjectId, path=None):
        """The stored native image of a subject as a volume, with its tags (as when it was added)."""
        path = path or normaldb.nativePath(folder, next((s for s in normaldb.readSubjects(folder)
                                                         if s.get("id", "").lower() == subjectId.lower()), None))
        if not path:
            raise ValueError(f"Subject '{subjectId}' has no native image in the database.")
        node = slicer.util.loadVolume(path, {"show": False, "name": f"{subjectId} native"})
        if node is None:
            raise RuntimeError(f"Could not load {path}")
        for key, value in (normaldb.readNativeInfo(folder, subjectId).get("attributes") or {}).items():
            if value is not None:
                node.SetAttribute(key, str(value))
        return node

    def _loadSource(self, folder, item):
        """A batch item as a volume: (path, subject ID, extra) where the path is a stored native image of the
        database (loadNative) or any image file (loadFromFile); extra {"age", "sex"} sets them when given."""
        path, subjectId, extra = item
        native = os.path.realpath(path).startswith(os.path.realpath(os.path.join(folder, normaldb.SUBJECTS_DIR))
                                                   + os.sep)
        volume = self.loadNative(folder, subjectId, path) if native else self.loadFromFile(path)
        if extra and (extra.get("age") or extra.get("sex")):
            age = sn.parseAge(extra.get("age"))
            current, currentSex = self.ageAndSex(volume)
            self.setAgeAndSex(volume, age if age is not None else current,
                              sn.parseSex(extra.get("sex")) or currentSex)
        return volume, native

    @staticmethod
    def batchItems(folder, files=None, table=None):
        """Batch items (path, subject ID, extra): the given image files (the BIDS subject, or the subject of table
        - normaldb.readSubjectTable - the file name fits, else the file name), or without files the database's
        stored native images."""
        if files is None:
            return [(normaldb.nativePath(folder, s), s["id"],
                     {"anatomy": normaldb.anatomyNativePath(folder, s), "contrast": s.get("anatomy", "")}
                     if normaldb.anatomyNativePath(folder, s) else None)
                    for s in normaldb.readSubjects(folder) if normaldb.nativePath(folder, s)]
        items = []
        for index, path in enumerate(files):
            stem = sn.splitImageName(os.path.basename(path))[0]
            row = normaldb.matchSubject(stem, table or {}) or normaldb.matchSubject(sn.bidsSubjectId(path),
                                                                                    table or {})
            subjectId = (row or {}).get("id") or sn.bidsSubjectId(path) or normaldb.safeName(stem) \
                or f"subject{index}"
            items.append((path, subjectId, row))
        return items

    # --- the regions of a normal database and their values ----------------------------------------
    @staticmethod
    def databaseRegionArguments(folder, header):
        """Keyword arguments of loadAtlasRegions() for the region settings of a database (as the Regions step uses
        them when locked)."""
        settings = sn.regionSettings(folder, header)
        arguments = {"targetKind": settings["regionAtlas"],
                     "referenceKind": neuro.mergedRegionAtlas(settings["regionAtlas"]), "referenceName": None,
                     "growMm": settings["growMm"], "erodeReferenceMm": settings["erodeReferenceMm"],
                     "growInto": settings["growInto"]}
        if settings["referenceMaskPath"]:
            arguments["referenceMaskPath"] = settings["referenceMaskPath"]
            arguments["referenceName"] = settings["referenceTitle"] or "Reference"
        elif settings["referenceAtlasRegion"]:
            arguments["referenceName"] = settings["referenceAtlasRegion"]
        else:       # as the Regions step: never the default reference region in place of the database's
            raise ValueError(f"The database '{header.get('name')}' has no copy of its reference mask "
                             f"({settings['referenceTitle']}).")
        return arguments

    def segmentMasks(self, segmentation, gridNode, dropped=True):
        """{segment name: nr.RegionMask (a boolean mask kept as its bounding box) on the grid of gridNode} of a
        segmentation (the optic chiasm left out with dropped)."""
        masks = {}
        shape = tuple(reversed(gridNode.GetImageData().GetDimensions()))
        for segmentId in self.segmentIds(segmentation):
            name = segmentation.GetSegmentation().GetSegment(segmentId).GetName()
            if dropped and nr.isDroppedSegment(name):
                continue
            array = slicer.util.arrayFromSegmentBinaryLabelmap(segmentation, segmentId, gridNode)
            masks[name] = nr.RegionMask.fromArray(np.asarray(array) > 0) if array is not None else \
                nr.RegionMask.empty(shape)
        return masks

    savedRegionsReference = None        # the reference segment's name of the regions databaseRegionMasks() saved

    def databaseRegionMasks(self, atlas, folder, header, gridNode, savePath=None):
        """The regions of a database's region settings (its region atlas) as masks on gridNode's grid.
        savePath: the segmentation is written there too (.seg.nrrd; not written when it cannot be: the file is then
        missing), its reference segment's name kept in savedRegionsReference."""
        segmentation, referenceId = self.loadAtlasRegions(atlas, **self.databaseRegionArguments(folder, header))
        try:
            masks = self.segmentMasks(segmentation, gridNode)
            if savePath:
                self.savedRegionsReference = None
                try:
                    os.makedirs(os.path.dirname(savePath), exist_ok=True)
                    if slicer.util.saveNode(segmentation, savePath) and referenceId:
                        self.savedRegionsReference = segmentation.GetSegmentation().GetSegment(referenceId).GetName()
                except Exception:
                    logging.exception("Static Neuro: could not save the database's regions")
            return masks
        finally:
            storage = segmentation.GetStorageNode()
            slicer.mrmlScene.RemoveNode(segmentation)
            if storage is not None and slicer.mrmlScene.IsNodePresent(storage):
                slicer.mrmlScene.RemoveNode(storage)

    def loadAtlasRegions(self, atlas, *args, storedRegions=None, **kwargs):
        """DynamicNeuroPETLogic.loadAtlasRegions(); storedRegions (path, reference segment name; sn.storedRegions):
        the regions stored with a normal database's statistics are loaded as they are instead."""
        if storedRegions:
            return self.loadStoredRegions(storedRegions[0], storedRegions[1], kwargs.get("target"))
        return DynamicNeuroPETLogic.loadAtlasRegions(self, atlas, *args, **kwargs)

    def loadStoredRegions(self, path, referenceName=None, target=None):
        """The regions stored with a database's statistics (a .seg.nrrd) as a segmentation: a new one, or target
        filled with them. Not tied to the file (never saved back onto the database). Returns (segmentation, reference
        segment ID or None)."""
        from DynamicNeuroPET import REGIONS_NAME
        scene = slicer.mrmlScene
        loaded = slicer.util.loadSegmentation(path)
        if loaded is None:
            raise RuntimeError(f"Could not load the database's regions ({path})")
        storage = loaded.GetStorageNode()
        if storage is not None:
            scene.RemoveNode(storage)
        if target is not None and scene.IsNodePresent(target) and target is not loaded:
            target.GetSegmentation().DeepCopy(loaded.GetSegmentation())
            scene.RemoveNode(loaded)
            segmentation = target
        else:
            segmentation = loaded
            segmentation.SetName(scene.GenerateUniqueName(REGIONS_NAME))
        segmentation.CreateDefaultDisplayNodes()
        segmentation.SetAttribute(NEURO_PREFIX + "Atlas", os.path.basename(path))
        display = segmentation.GetDisplayNode()
        if display is not None:
            display.SetOpacity2DFill(0.3)
            display.SetVisibility3D(False)
        referenceId = self.segmentIdByName(segmentation, referenceName) if referenceName else None
        logging.info(f"Static Neuro: '{segmentation.GetName()}' loaded from the database's regions ({path})")
        return segmentation, referenceId

    @staticmethod
    def subjectRegionValues(folder, header, masks, progress=None):
        """{subject ID: {region: mean}} of the subjects' maps (SUVR or SUV by the map type; not smoothed) of a user
        defined database. progress(index, total, text) returns False to stop (None returned)."""
        subjects = normaldb.readSubjects(folder)
        mapKey = normaldb.subjectMapKey(header)
        values = {}
        for index, subject in enumerate(subjects):
            if progress is not None and not progress(index, len(subjects), f"{subject['id']} ({index + 1} / "
                                                                           f"{len(subjects)})"):
                return None
            relative = subject.get(mapKey)
            path = os.path.join(folder, relative) if relative else ""
            if not path or not os.path.isfile(path):
                continue
            node = slicer.util.loadVolume(path, {"show": False})
            try:
                values[subject["id"]] = nr.regionMeans(slicer.util.arrayFromVolume(node), masks)
            finally:
                slicer.mrmlScene.RemoveNode(node)
        return values

    def regionReference(self, folder, header, masks, groupKey, locked, progress=None):
        """({region: [mean, SD, n]}, source text) of the normal regional values of a group: the database's own
        (stats/regions.json, made with the mean / SD maps) when the regions are the database's (locked); else, for a
        user defined database, from its subjects on the fly (cached in <database>/cache/); else the approximation from
        the mean / SD maps. None when stopped."""
        names = list(masks)
        version = header.get("version", 1)
        mapKey = normaldb.subjectMapKey(header)
        path = nr.regionsPath(folder, header)
        if locked and path:
            document = nr.readJson(path)
            if document and document.get("maps") == mapKey:
                found, missing = nr.matchRegions(document.get("groups"), groupKey, names)
                if found and len(missing) <= len(names) // 10:
                    return found, "database regional statistics"
        if header.get("kind") == normaldb.USER_DEFINED and normaldb.readSubjects(folder):
            ids = sorted(s.get("id", "") for s in normaldb.readSubjects(folder))
            signature = nr.masksSignature(masks, extra=ids)
            cache = nr.cachePath(folder, signature, mapKey, version)
            document = nr.readJson(cache)
            if document is None:
                values = self.subjectRegionValues(folder, header, masks, progress)
                if values is None:
                    return None
                groups = normaldb.statisticsGroups(header, normaldb.readSubjects(folder))
                document = nr.regionsDocument(nr.regionStatistics(values, groups), mapKey, version,
                                              signature=signature)
                try:
                    nr.writeJson(cache, document)
                except OSError:
                    logging.warning(f"Static Neuro: could not write {cache}", exc_info=True)
            found, _ = nr.matchRegions(document.get("groups"), groupKey, names)
            if found:
                return found, "the database's subjects (these regions)"
        return None, "approximate (from the mean / SD maps)"

    # --- normal databases ------------------------------------------------------------------------
    @staticmethod
    def databaseRoot():
        """The database folder of the Database Manager, else its default one (~/NeuroPET/Databases)."""
        return str(qt.QSettings().value(SETTINGS_DB_ROOT) or "") or normaldb.defaultRootFolder()

    @staticmethod
    def setDatabaseRoot(folder):
        qt.QSettings().setValue(SETTINGS_DB_ROOT, folder or "")

    @staticmethod
    def databases(root, info):
        """[(folder, header, rank, problems)] of the static databases of root, those fitting the image first."""
        found, errors = normaldb.listDatabases(root, normaldb.STATIC)
        for folder, error in errors:
            logging.warning(f"Static Neuro: database {folder} not readable: {error}")
        return sn.sortDatabases(found, info)
