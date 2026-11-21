"""
Review: the patient compared with the database and the 3D-SSP.

Part of the Static Neuro PET/SPECT module (StaticNeuroPET.py): the methods of ReviewLogicMixin, mixed into
StaticNeuroPETLogic there.
"""

import logging
import os

import numpy as np
import slicer
import vtk

from DICOMGatedDynamicPETLib import resampling
from NeuroWorkflowLib.base import NeuroWorkflowLogic, _ijkToRas, _vtkMatrix
from NeuroPETLib import neuro, normaldb, neuroreview as nr, staticneuro as sn, ssp

from StaticNeuroPETLib.constants import SSP_MAX_EDGE_MM, SSP_TARGET_POINTS


class ReviewLogicMixin:
    """The logic review methods of StaticNeuroPETLogic."""

    # --- review: the patient compared with the database ------------------------------------------
    def brainHull(self, union, atlas, gridNode):
        """The brain the 3D-SSP surface and the unfolded brain are made of: the regions (union) with every CerebrA
        structure on the grid of gridNode. A region atlas that leaves structures out (Broad (Compatible): no
        brainstem, no ventral diencephalon) would otherwise leave the brain hollow and open below, its holes not
        filled: a surface inside it, and the peel's depth measured from the inside too. The structures are those of
        the default segmentation (CerebrA); the union alone without it (or when it cannot be read)."""
        try:
            info = neuro.segmentationInfo(neuro.DEFAULT_SEGMENTATION)
            path = self.segmentationLabelMap(info) if info else None
        except Exception:       # noqa: BLE001 - the union is still a brain
            logging.warning("Static Neuro: no CerebrA structures for the brain hull", exc_info=True)
            path = None
        if not path:
            return union
        try:
            array, node = self._loadLabels(path)
            try:
                if array.shape == union.shape and np.allclose(_ijkToRas(node), _ijkToRas(gridNode), atol=1e-3):
                    structures = array > 0
                else:
                    # another grid: resampled as a 0 / 1 mask
                    slicer.util.updateVolumeFromArray(node, (array > 0).astype(np.uint8))
                    structures = self.resampledArray(node, gridNode) > 0.5
            finally:
                slicer.mrmlScene.RemoveNode(node)
        except Exception:
            logging.warning("Static Neuro: the CerebrA structures could not be added to the brain", exc_info=True)
            return union
        return union | structures if structures.shape == union.shape else union

    @staticmethod
    def resampledArray(volume, gridNode):
        """The volume's voxels resampled (linear) onto the grid of gridNode, as a float32 array (k, j, i)."""
        dims = tuple(int(x) for x in gridNode.GetImageData().GetDimensions())
        image = resampling.Resampler(_ijkToRas(volume), dims, _ijkToRas(gridNode), None, True,
                                     "Linear").resample(volume.GetImageData())
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Neuro review (temporary)")
        try:
            node.SetIJKToRASMatrix(_vtkMatrix(_ijkToRas(gridNode)))
            node.SetAndObserveImageData(image)
            return np.array(slicer.util.arrayFromVolume(node), dtype=np.float32)
        finally:
            slicer.mrmlScene.RemoveNode(node)

    @staticmethod
    def dropSegments(segmentation):
        """Remove the segments the review leaves out (the optic chiasm). Returns their names."""
        removed = []
        seg = segmentation.GetSegmentation()
        for segmentId in NeuroWorkflowLogic.segmentIds(segmentation):
            name = seg.GetSegment(segmentId).GetName()
            if nr.isDroppedSegment(name):
                seg.RemoveSegment(segmentId)
                removed.append(name)
        return removed

    def reviewData(self, volume, segmentation, folder=None, header=None, groupKey=normaldb.MASTER, atlas=None,
                   referenceSegmentId=None, locked=True, progress=None, meanMask=True):
        """The patient (a normalized volume and its regions) compared with a normal database:
        {"grid": the grid node (the group's mean map; without a database the volume), "z": voxel z map (negative
        below the mean; t = z / tFactor(n)) or None, "n", "names", "suv" {region: SUV mean}, "values" {region:
        value of the database's map type}, "reference" {region: [mean, SD, n]}, "source", "ratio" (map type SUVR /
        Ratio), "referenceMean", "fwhm", "notes", "meanMask"}. meanMask: the voxels scored only where the database
        mean is above MASK_FRACTION (5 %) of its maximum (else the brain mask alone). None when stopped."""
        notes = []
        result = {"z": None, "n": None, "reference": {}, "source": "", "ratio": False, "referenceMean": None,
                  "fwhm": 0.0, "notes": notes, "meanMask": bool(meanMask)}
        groups = ((header or {}).get("statistics") or {}).get("groups") or {}
        group = groups.get(groupKey) if header is not None else None
        if header is not None and not group:
            raise ValueError(f"The database '{header.get('name')}' has no mean / SD maps"
                             + (f" of the group '{groupKey}'." if groups else ": compute them in the Neuro PET "
                                "Database Manager."))
        if header is None:
            grid = volume
            values = suv = np.array(slicer.util.arrayFromVolume(volume), dtype=np.float32)
            ratio = None
            masks = self.segmentMasks(segmentation, grid)
            result.update(grid=grid, names=list(masks), suv=nr.regionMeans(suv, masks))
            result["labels"] = nr.labelMap(masks, values.shape)       # the region names at the mouse
            if referenceSegmentId:
                mask = slicer.util.arrayFromSegmentBinaryLabelmap(segmentation, referenceSegmentId, grid)
                if mask is not None and np.any(mask):
                    ratio, mean = sn.ratioMap(values, np.asarray(mask) > 0)
                    result.update(ratio=True, referenceMean=mean, values=nr.regionMeans(ratio, masks))
                else:
                    ratio = None
            result.setdefault("values", result["suv"])
            union = nr.unionMask(masks, values.shape)
            result.update(smoothed=ratio if result["ratio"] else values, union=union,
                          hull=self.brainHull(union, atlas, grid), hemispheres=ssp.hemisphereMasks(masks, values.shape))
            return result
        grid = self.loadAtlasVolume(os.path.join(folder, group["mean"]), name=f"{header['name']} mean ({groupKey})")
        sdNode = self.loadAtlasVolume(os.path.join(folder, group["sd"]), name=f"{header['name']} SD ({groupKey})")
        mean = np.array(slicer.util.arrayFromVolume(grid), dtype=np.float32)
        sd = np.array(slicer.util.arrayFromVolume(sdNode), dtype=np.float32)
        if sd.shape != mean.shape:
            raise ValueError("The mean and SD maps of the database are not on the same grid.")
        suv = self.resampledArray(volume, grid)
        values = suv
        ratio = header.get("mapType") in (normaldb.MAP_SUVR, normaldb.MAP_RATIO)
        if ratio:
            maskPath = sn.referenceMaskPath(folder, header)
            if maskPath:
                maskArray, maskNode = self._loadLabels(maskPath)
                slicer.mrmlScene.RemoveNode(maskNode)
                if maskArray.shape != mean.shape:
                    raise ValueError("The database's reference mask is not on the grid of its mean map.")
                referenceMask = maskArray > 0
            elif referenceSegmentId:
                referenceArray = slicer.util.arrayFromSegmentBinaryLabelmap(segmentation, referenceSegmentId, grid)
                if referenceArray is None:
                    raise ValueError("The reference region of the regions is empty.")
                referenceMask = np.asarray(referenceArray) > 0
                notes.append("The database has no reference mask: the regions' reference region is used.")
            else:
                raise ValueError("The database compares ratio maps but has no reference mask, and the regions have "
                                 "no reference region.")
            values, referenceMean = sn.ratioMap(suv, referenceMask)
            result["referenceMean"] = referenceMean
        masks = self.segmentMasks(segmentation, grid)
        result["labels"] = nr.labelMap(masks, mean.shape)             # the region names at the mouse
        result.update(grid=grid, names=list(masks), suv=nr.regionMeans(suv, masks),
                      values=nr.regionMeans(values, masks), ratio=ratio, n=group.get("n"))
        # voxel scores: the patient smoothed like the normals, in the mask, with the SD floor
        fwhm = float((header.get("statistics") or {}).get("smoothingFwhmMm", header.get("smoothingFwhmMm")) or 0)
        result["fwhm"] = fwhm
        spacing = tuple(float(x) for x in grid.GetSpacing())[::-1]
        brain = None
        if atlas and neuro.BRAIN_MASK in atlas:
            brainArray, brainNode = self._loadLabels(atlas[neuro.BRAIN_MASK])
            slicer.mrmlScene.RemoveNode(brainNode)
            if brainArray.shape == mean.shape:
                brain = brainArray > 0
            else:
                notes.append("The MNI brain mask is not on the database grid: the score mask is the mean map's.")
        mask = nr.scoreMask(mean, brain, nr.MASK_FRACTION if meanMask else None)
        smoothed = normaldb.smoothMap(values, fwhm, spacing)
        union = nr.unionMask(masks, mean.shape)
        result.update(smoothed=smoothed, union=union, hull=self.brainHull(union, atlas, grid),
                      hemispheres=ssp.hemisphereMasks(masks, mean.shape))
        result["z"] = nr.scoreMap(smoothed, mean, sd, mask, nr.SCORE_Z)
        result["brightness"] = nr.petBrightness(values)     # the patient's map (not smoothed) as brightness
        result["gm"] = None         # the MNI grey matter probability on the grid: the score slices' default mask
        if atlas and "GM" in atlas:
            try:
                result["gm"] = self.resampledArray(self.loadAtlasVolume(atlas["GM"], name="MNI ICBM152 2009c GM"), grid)
            except Exception as error:
                notes.append(f"No grey matter map ({error}): the score slices show the high normal uptake.")
        result.update(mask=mask, mean=mean)     # mean: the normal uptake (the score map shown where it is high)
        # regional normal values
        reference = self.regionReference(folder, header, masks, groupKey, locked, progress)
        if reference is None:
            return None
        found, source = reference
        if found is None:
            found = nr.approximateRegions(mean, sd, masks)
            notes.append("Regional normal values approximated from the mean / SD maps (the mean voxel SD is larger "
                         "than the SD of a regional mean: these scores are conservative).")
        result.update(reference=found, source=source)
        return result

    # --- 3D-SSP ------------------------------------------------------------------------------------
    @staticmethod
    def brainSurface(mask, gridNode, targetPoints=SSP_TARGET_POINTS):
        """The surface of a brain mask (numpy (k, j, i) on gridNode's grid; holes filled, smoothed 1 voxel, padded
        so that it is closed, the largest piece): contoured on a grid coarser by a step that gives about targetPoints
        vertices (edges at most SSP_MAX_EDGE_MM; no decimation): one closed surface, its normals consistent:
        {"points" (RAS mm), "normals" (outward, unit), "triangles"}."""
        from scipy import ndimage
        from vtk.util import numpy_support
        mask = ndimage.binary_fill_holes(np.asarray(mask, dtype=bool))
        if not mask.any():
            raise ValueError("The regions are empty: no brain surface.")
        smooth = np.pad(ndimage.gaussian_filter(mask.astype(np.float32), 1.0), 1)      # closed at the grid's edges
        ijkToRas = _ijkToRas(gridNode)
        spacing = float(np.min(np.linalg.norm(ijkToRas[:3, :3], axis=0))) or 1.0

        def contour(field):
            image = vtk.vtkImageData()
            image.SetDimensions(*field.shape[::-1])
            image.GetPointData().SetScalars(numpy_support.numpy_to_vtk(np.ascontiguousarray(field).ravel(), deep=True,
                                                                       array_type=vtk.VTK_FLOAT))
            surface = vtk.vtkFlyingEdges3D() if hasattr(vtk, "vtkFlyingEdges3D") else vtk.vtkMarchingCubes()
            surface.SetInputData(image)
            surface.SetValue(0, 0.5)
            largest = vtk.vtkPolyDataConnectivityFilter()
            largest.SetInputConnection(surface.GetOutputPort())
            largest.SetExtractionModeToLargestRegion()
            cleaned = vtk.vtkCleanPolyData()        # the points of the other pieces dropped
            cleaned.SetInputConnection(largest.GetOutputPort())
            cleaned.Update()
            return cleaned.GetOutput()

        poly = contour(smooth)
        step = 1.0
        count = poly.GetNumberOfPoints()
        if count > targetPoints:
            # vertices ~ area / step^2; a triangle's edge at most the cell's diagonal (sqrt(3) step)
            step = min(float(np.sqrt(count / float(targetPoints))), SSP_MAX_EDGE_MM / (np.sqrt(3.0) * spacing))
            if step > 1.05:
                grid = np.meshgrid(*[np.arange(0.0, n - 1 + 1e-6, step) for n in smooth.shape], indexing="ij")
                poly = contour(ndimage.map_coordinates(smooth, grid, order=1).astype(np.float32))
            else:
                step = 1.0
        # the contour's (i, j, k): the padded grid's, step apart
        matrix = ijkToRas @ np.array([[step, 0, 0, -1], [0, step, 0, -1], [0, 0, step, -1], [0, 0, 0, 1]], float)
        transform = vtk.vtkTransform()
        transform.SetMatrix(_vtkMatrix(matrix))
        moved = vtk.vtkTransformPolyDataFilter()
        moved.SetTransform(transform)
        moved.SetInputData(poly)
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(moved.GetOutputPort())
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOff()
        normals.SplittingOff()
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()       # outward (the surface closed and in one piece)
        normals.Update()
        poly = normals.GetOutput()
        points = numpy_support.vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
        pointNormals = numpy_support.vtk_to_numpy(poly.GetPointData().GetNormals()).astype(np.float64)
        polys = poly.GetPolys()
        if hasattr(polys, "GetConnectivityArray"):
            cells = numpy_support.vtk_to_numpy(polys.GetConnectivityArray()).reshape(-1, 3)
        else:
            cells = numpy_support.vtk_to_numpy(polys.GetData()).reshape(-1, 4)[:, 1:]
        out = {"points": points, "normals": pointNormals, "triangles": cells.astype(np.int64)}
        openEdges = ssp.openEdges(out["triangles"])
        if openEdges:
            logging.warning(f"Static Neuro: the 3D-SSP surface has {openEdges} open edges")
        return out

    @classmethod
    def reviewSurface(cls, mask, gridNode, targetPoints=SSP_TARGET_POINTS, hemispheres=None):
        """The 3D-SSP surface of a brain mask: the whole brain's and, for the medial views, each hemisphere's in one
        surface (ssp.combine). hemispheres: (left, right) masks of the regions' own sides (ssp.hemisphereMasks), each
        hemisphere's surface made from its own mask; None (bilateral regions): the brain's only, no medial views."""
        brain = cls.brainSurface(mask, gridNode, targetPoints)
        if hemispheres is None:
            return brain
        mask = np.asarray(mask, dtype=bool)
        halves = []
        for half in hemispheres:
            try:
                halves.append(cls.brainSurface(mask & np.asarray(half, dtype=bool), gridNode, targetPoints // 2))
            except ValueError:          # nothing on that side
                halves.append(None)
        return ssp.combine(brain, *halves)

    @staticmethod
    def surfacePolyData(surface, values=None, name="SSP"):
        """vtkPolyData of a surface dict, with the vertex values as its active scalars."""
        from vtk.util import numpy_support
        poly = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        points.SetData(numpy_support.numpy_to_vtk(np.asarray(surface["points"], np.float32), deep=True))
        poly.SetPoints(points)
        triangles = np.asarray(surface["triangles"], dtype=np.int64)
        cells = np.hstack([np.full((len(triangles), 1), 3, dtype=np.int64), triangles]).ravel()
        polys = vtk.vtkCellArray()
        polys.SetCells(len(triangles), numpy_support.numpy_to_vtkIdTypeArray(cells, deep=True))
        poly.SetPolys(polys)
        normals = numpy_support.numpy_to_vtk(np.asarray(surface["normals"], np.float32), deep=True)
        normals.SetName("Normals")
        poly.GetPointData().SetNormals(normals)
        if values is not None:
            array = numpy_support.numpy_to_vtk(np.asarray(values, np.float32), deep=True)
            array.SetName(name)
            poly.GetPointData().AddArray(array)
            poly.GetPointData().SetActiveScalars(name)
        return poly

    def sspFromSubjects(self, folder, header, surface, fwhm, progress=None, mask=None):
        """{group key: {"mean", "sd", "n", "name"}} of the normals' projections on a surface: each subject's map
        (SUVR or SUV by the map type) smoothed with fwhm and projected (each ray stopped where it leaves mask, the
        union of the regions on the database's grid); cached in <database>/cache/ (key: the surface,
        the projection, the FWHM, the map type, the database version and the subjects). None when stopped."""
        subjects = normaldb.readSubjects(folder)
        mapKey = normaldb.subjectMapKey(header)
        key = ssp.signature(surface["points"], surface["normals"], ssp.DEPTH_MM, ssp.STEP_MM,
                            extra=(fwhm, mapKey, header.get("version", 1), sorted(s.get("id", "") for s in subjects),
                                   "clip-brain"))
        cache = ssp.cachePath(folder, key)
        cached = ssp.read(cache)
        if cached is not None:
            return cached[1]
        groups = {k: g for k, g in normaldb.statisticsGroups(header, subjects).items() if len(g["ids"]) >= 2}
        accumulators = {k: normaldb.RunningStatistics() for k in groups}
        for index, subject in enumerate(subjects):
            if progress is not None and not progress(index, len(subjects), f"surface projection of {subject['id']} "
                                                                           f"({index + 1} / {len(subjects)})"):
                return None
            relative = subject.get(mapKey)
            path = os.path.join(folder, relative) if relative else ""
            if not path or not os.path.isfile(path):
                continue
            node = slicer.util.loadVolume(path, {"show": False})
            try:
                smoothed = normaldb.smoothMap(slicer.util.arrayFromVolume(node), fwhm,
                                              tuple(float(x) for x in node.GetSpacing())[::-1])
                clip = mask if mask is not None and np.shape(mask) == np.shape(smoothed) else None
                if mask is not None and clip is None:
                    logging.warning(f"Static Neuro: {subject['id']} is not on the regions' grid: 3D-SSP not clipped")
                projected = ssp.project(smoothed, _ijkToRas(node), surface["points"], surface["normals"], mask=clip)
            finally:
                slicer.mrmlScene.RemoveNode(node)
            for groupKey, group in groups.items():
                if subject["id"] in group["ids"]:
                    accumulators[groupKey].add(projected)
        out = {k: {"mean": a.mean(), "sd": a.sd(), "n": a.n, "name": groups[k]["name"]}
               for k, a in accumulators.items() if a.n >= 2}
        try:
            ssp.write(cache, surface, out, {"depthMm": ssp.DEPTH_MM, "stepMm": ssp.STEP_MM, "clipToBrain": True,
                                            "fwhmMm": fwhm,
                                            "maps": mapKey, "version": header.get("version", 1)})
        except OSError:
            logging.warning(f"Static Neuro: could not write {cache}", exc_info=True)
        return out

    def reviewSsp(self, result, folder=None, header=None, groupKey=normaldb.MASTER, locked=True, progress=None):
        """3D-SSP of a reviewData() result: {"surface", "patient" (the patient's projection), "mean", "sd", "n" (the
        normals' projections of the group, Minoshima; None when not available), "voxel" (the voxel z map projected,
        the most extreme along the normal; None without a database), "source", "notes"}. The surface: the database's
        (stats/ssp.npz) when the regions are the database's, else made from the union of the regions. None when
        stopped."""
        grid = result["grid"]
        notes = []
        out = {"mean": None, "sd": None, "n": None, "voxel": None, "source": "", "notes": notes}
        surface = groups = None
        mapKey = normaldb.subjectMapKey(header) if header is not None else ""
        path = ssp.sspPath(folder, header) if header is not None else None
        outdated = noMedial = False
        medial = result.get("hemispheres") is not None      # the segmentation's regions with sides (R / L)
        if path and locked:
            read = ssp.read(path)
            if read is not None and read[2].get("maps") == mapKey and not (
                    abs(float(read[2].get("depthMm") or 0) - ssp.DEPTH_MM) < 1e-6 and read[2].get("clipToBrain")
                    and int(read[2].get("surface") or 0) >= ssp.SURFACE_VERSION):
                outdated, read = True, None         # an older projection (13.5 mm, not stopped at the brain's edge,
                #                                     an older surface)
            if read is not None and read[2].get("maps") == mapKey and medial and not ssp.hasMedial(read[0]):
                noMedial, read = True, None         # the database's regions bilateral, the segmentation's R / L
            if read is not None and read[2].get("maps") == mapKey:
                surface, groups, meta = read
                if not medial and ssp.hasPart(surface, ssp.PART_LEFT):      # bilateral regions: no medial views
                    surface, index = ssp.brainPart(surface)
                    groups = {k: dict(g, mean=np.asarray(g["mean"])[index], sd=np.asarray(g["sd"])[index])
                              for k, g in groups.items()}
                out["source"] = "the database's surface projections"
                if abs(float(meta.get("fwhmMm") or 0) - result["fwhm"]) > 1e-6:
                    notes.append("The database's surface projections were made with another smoothing.")
        if surface is None:
            hull = result.get("hull")
            surface = self.reviewSurface(result["union"] if hull is None else hull, grid,
                                         hemispheres=result.get("hemispheres"))
        out["surface"] = surface
        ijkToRas = _ijkToRas(grid)
        brain = result["union"]                     # each ray stops where it leaves the brain
        out["patient"] = ssp.project(result["smoothed"], ijkToRas, surface["points"], surface["normals"], mask=brain)
        if header is None:
            return out
        if groups is None and header.get("kind") == normaldb.USER_DEFINED and normaldb.readSubjects(folder):
            groups = self.sspFromSubjects(folder, header, surface, result["fwhm"], progress, mask=brain)
            if groups is None:
                return None
            out["source"] = "the database's subjects projected on these regions"
            if outdated:
                notes.append("The database's 3D-SSP projections were made by an older version (another depth or "
                             "surface): projected from its subjects now; compute the mean / SD again in the Neuro "
                             "PET Database Manager to store them.")
            elif noMedial:
                notes.append("The database's 3D-SSP projections have no medial surfaces (its regions are "
                             "bilateral): its subjects projected on the hemispheres of these R / L regions now "
                             "(cached).")
        group = (groups or {}).get(groupKey)
        if group is not None:
            out.update(mean=group["mean"], sd=group["sd"], n=group.get("n") or result.get("n"))
        elif noMedial:
            notes.append("The database's 3D-SSP projections have no medial surfaces (its regions are bilateral) and "
                         "it has no subjects: the 3D-SSP shows the voxel score map projected.")
        elif outdated:
            notes.append("The database's 3D-SSP projections are from an older version (another depth, not stopped at "
                         "the brain's edge, or no medial surfaces): the 3D-SSP shows the voxel score map projected.")
        else:
            notes.append("No surface projections of the normals for this database: the 3D-SSP shows the voxel score "
                         "map projected.")
        if result.get("z") is not None:
            # the voxel scores of the grey matter (else of the high normal uptake): not the CSF's or the white matter's
            if result.get("gm") is not None:
                shown = nr.greyMatterMask(result["gm"], result["mask"])
            else:
                shown = nr.uptakeMask(result["mean"], result["mask"])
            out["voxel"] = ssp.project(np.where(shown, result["z"], 0.0), ijkToRas, surface["points"],
                                       surface["normals"], mode="extreme", mask=brain)
        return out
