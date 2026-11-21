"""
Review: the cut 3D-SSP (the surface cut at the slices).

Part of the Static Neuro PET/SPECT module (StaticNeuroPET.py): the methods of ReviewCutMixin, mixed into
StaticNeuroPETWidget there.
"""

import logging

import numpy as np
import qt
import slicer
import vtk

from NeuroWorkflowLib.base import _ijkToRas
from NeuroPETLib import neuroreview as nr, ssp

from StaticNeuroPETLib.constants import CUT_MODEL_ATTRIBUTE, CUT_MODEL_NAME, CUT_UPDATE_DELAY_MS, SSP_CAMERA_DISTANCE


class ReviewCutMixin:
    """The review cut methods of StaticNeuroPETWidget."""

    # the cut 3D-SSP layout: in each row the 3D-SSP cut at that row's score slice (the part between the slice and the
    # camera taken off), the cut face colored as the score slice, seen from the left and the right anterior superior
    def _cutViewNodes(self):
        """{(orientation, side, view set): (3D widget, view node)} of the cut 3D-SSP views of the layout."""
        layoutManager = slicer.app.layoutManager()
        out = {}
        if layoutManager is None:
            return out
        tags = nr.cutViewTags()
        for index in range(layoutManager.threeDViewCount):
            widget = layoutManager.threeDWidget(index)
            viewNode = widget.mrmlViewNode() if widget is not None else None
            if viewNode is not None and viewNode.GetSingletonTag() in tags:
                out[tags[viewNode.GetSingletonTag()]] = (widget, viewNode)
        return out

    def _cutSliceNodes(self):
        """{orientation: slice node} of the score slices (the planes of the cuts)."""
        layoutManager = slicer.app.layoutManager()
        out = {}
        if layoutManager is None:
            return out
        for orientation, _ in nr.ORIENTATIONS:
            widget = layoutManager.sliceWidget(nr.sliceViewName(nr.SCORE_ROW, orientation))
            if widget is not None and widget.mrmlSliceNode() is not None:
                out[orientation] = widget.mrmlSliceNode()
        return out

    @staticmethod
    def _slicePlane(sliceNode):
        """(origin, normal) RAS of a slice node's plane."""
        m = sliceNode.GetSliceToRAS()
        return (np.array([m.GetElement(r, 3) for r in range(3)], dtype=np.float64),
                np.array([m.GetElement(r, 2) for r in range(3)], dtype=np.float64))

    def _watchCutSlices(self, enable):
        for node, tag in getattr(self, "_cutSliceObservers", None) or []:
            try:
                node.RemoveObserver(tag)
            except Exception:
                pass
        self._cutSliceObservers = []
        if not enable:
            return
        for orientation, node in self._cutSliceNodes().items():
            tag = node.AddObserver(vtk.vtkCommand.ModifiedEvent,
                                   lambda caller, event, o=orientation: self._onCutSliceMoved(o))
            self._cutSliceObservers.append((node, tag))

    def _onCutSliceMoved(self, orientation):
        """A score slice modified: its row's cut face made again (if its plane moved) a moment later; the surfaces
        are clipped by Slicer itself (clip nodes on the slices), nothing to do for them."""
        pending = getattr(self, "_cutPending", None)
        if pending is None:
            pending = self._cutPending = set()
        first = not pending
        pending.add(orientation)
        if first:
            qt.QTimer.singleShot(CUT_UPDATE_DELAY_MS, self._afterCutSliceMoved)

    def _afterCutSliceMoved(self):
        orientations, self._cutPending = set(getattr(self, "_cutPending", None) or ()), set()
        try:
            if not getattr(self, "_nativeCut", True):
                self._updateCutSurfaces(orientations)
            self._updateCutFaces(orientations, onlyMoved=True)
        except Exception:
            logging.exception("Static Neuro: could not update the cut 3D-SSP")

    def _cutModelNode(self, key):
        scene = slicer.mrmlScene
        node = self.cutModels.get(key)
        if node is None or not scene.IsNodePresent(node):
            node = scene.AddNewNodeByClass("vtkMRMLModelNode", scene.GenerateUniqueName(
                f"{CUT_MODEL_NAME} {' '.join(key)}"))
            node.SetAttribute(CUT_MODEL_ATTRIBUTE, "/".join(key))
            node.CreateDefaultDisplayNodes()
            node.GetDisplayNode().SetVisibility2D(False)
            # no views = every view: hidden until its views are set
            self._setViewNodeIds(node.GetDisplayNode(), ["vtkMRMLViewNodeNone"])
            self.cutModels[key] = node
        return node

    def _dropStaleCutModels(self):
        """The cut models in the scene that are not this widget's removed (a module reloaded)."""
        mine = {node.GetID() for node in self.cutModels.values() if node is not None}
        for node in list(slicer.util.getNodesByClass("vtkMRMLModelNode")):
            if node.GetAttribute(CUT_MODEL_ATTRIBUTE) and node.GetID() not in mine:
                slicer.mrmlScene.RemoveNode(node)

    def _hideCutModel(self, key):
        node = self.cutModels.get(key)
        if node is not None and slicer.mrmlScene.IsNodePresent(node) and node.GetDisplayNode() is not None:
            self._setViewNodeIds(node.GetDisplayNode(), ["vtkMRMLViewNodeNone"])

    def _cutClipNode(self, orientation, side, sliceNode):
        """The clip node of a cut view (Slicer's own clipping, vtkMRMLClipNode): the model clipped by the row's score
        slice, the side towards the camera taken off, following the slice with no work here; None when this Slicer
        has no clip nodes driven by any slice node (then clipped here, vtkClipPolyData)."""
        scene = slicer.mrmlScene
        nodes = getattr(self, "_cutClipNodes", None)
        if nodes is None:
            nodes = self._cutClipNodes = {}
        node = nodes.get((orientation, side))
        if node is None or not scene.IsNodePresent(node):
            if (not hasattr(slicer, "vtkMRMLClipNode")
                    or not hasattr(slicer.vtkMRMLClipNode, "AddAndObserveClippingNodeID")
                    or not hasattr(slicer.vtkMRMLDisplayNode, "SetAndObserveClipNodeID")):
                return None
            node = scene.AddNewNodeByClass("vtkMRMLClipNode", scene.GenerateUniqueName(
                f"{CUT_MODEL_NAME} clip {orientation} {side}"))
            node.SetAttribute(CUT_MODEL_ATTRIBUTE, f"clip/{orientation}/{side}")
            nodes[(orientation, side)] = node
        _, normal = self._slicePlane(sliceNode)
        direction, _ = ssp.cutCamera(nr.cutDirection(side))
        # the slice's normal towards the camera: its positive side taken off, else the negative (Slicer's
        # ClipNegativeSpace keeps the negative side: it takes the positive one off)
        state = (slicer.vtkMRMLClipNode.ClipNegativeSpace if float(np.dot(normal, direction)) > 0
                 else slicer.vtkMRMLClipNode.ClipPositiveSpace)
        wasModifying = node.StartModify()
        current = [node.GetNthClippingNodeID(i) for i in range(node.GetNumberOfClippingNodes())] \
            if hasattr(node, "GetNumberOfClippingNodes") and hasattr(node, "GetNthClippingNodeID") else None
        if current != [sliceNode.GetID()]:
            if hasattr(node, "RemoveAllClippingNodeIDs"):
                node.RemoveAllClippingNodeIDs()
            node.AddAndObserveClippingNodeID(sliceNode.GetID())
        node.SetClippingNodeState(sliceNode.GetID(), state)
        if hasattr(node, "SetClippingMethod"):
            node.SetClippingMethod(slicer.vtkMRMLClipNode.Straight)
        if hasattr(node, "SetClipType"):            # one plane: union / intersection the same
            node.SetClipType(slicer.vtkMRMLClipNode.ClipUnion)
        node.EndModify(wasModifying)
        return node

    def _updateCutSurfaces(self, orientations=None):
        """The 3D-SSP surface of each cut view: the main 3D-SSP's (values, colors, window), clipped at the row's
        score slice by Slicer (a clip node; with an older Slicer, clipped here and done again when the slice moves),
        the side towards the camera taken off."""
        views = self._cutViewNodes()
        if not views:
            return
        planes = self._cutSliceNodes()
        self._dropStaleCutModels()
        for (orientation, side, viewSet), (widget, viewNode) in views.items():
            if orientations is not None and orientation not in orientations:
                continue
            # the main 3D-SSP's surface or the low scores' (the same clip node: the same slice and side)
            found = getattr(self, "_sspPolys", {}).get((viewSet, ssp.PART_BRAIN))
            display = getattr(self, "_sspLowDisplay" if viewSet == nr.SSP_LOW else "_sspMainDisplay", None)
            key = ("surface", orientation, side) + ((viewSet,) if viewSet else ())
            if found is None or display is None or orientation not in planes:
                self._hideCutModel(key)
                continue
            model = self._cutModelNode(key)
            modelDisplay = model.GetDisplayNode()
            clipNode = self._cutClipNode(orientation, side, planes[orientation])
            self._nativeCut = clipNode is not None
            if clipNode is not None:
                if model.GetPolyData() is not found[1]:       # the main surface itself: its values follow
                    model.SetAndObservePolyData(found[1])
                current = modelDisplay.GetClipNode() if hasattr(modelDisplay, "GetClipNode") else None
                if current is None or current.GetID() != clipNode.GetID():
                    modelDisplay.SetAndObserveClipNodeID(clipNode.GetID())
                modelDisplay.SetClipping(True)
            else:
                origin, normal = self._slicePlane(planes[orientation])
                direction, _ = ssp.cutCamera(nr.cutDirection(side))
                plane = vtk.vtkPlane()
                plane.SetOrigin(*origin)
                plane.SetNormal(*ssp.cutNormal(normal, direction))
                clip = vtk.vtkClipPolyData()
                clip.SetInputData(found[1])
                clip.SetClipFunction(plane)
                clip.Update()
                poly = vtk.vtkPolyData()
                poly.DeepCopy(clip.GetOutput())
                model.SetAndObservePolyData(poly)
            colorId, low, high = display
            if modelDisplay.GetColorNodeID() != colorId:
                modelDisplay.SetAndObserveColorNodeID(colorId)
            modelDisplay.SetActiveScalarName("SSP")
            modelDisplay.SetScalarVisibility(True)
            modelDisplay.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseManualScalarRange)
            modelDisplay.SetScalarRange(low, high)
            modelDisplay.SetBackfaceCulling(False)
            modelDisplay.SetVisibility2D(False)
            if hasattr(modelDisplay, "SetLighting"):
                modelDisplay.SetLighting(bool(self.sspLightingCheck.checked))
            self._setViewNodeIds(modelDisplay, [viewNode.GetID()])

    def _cutFaceColors(self, viewSet=nr.SSP_MAIN):
        """(volume, colorize) of the cut faces: the colored PET of the score row (RGB) when shown, else the score map
        through the score colors (outside the scored mask black); None without scores. The low scores' faces
        (viewSet SSP_LOW): the scores through the low scores' 3D-SSP colors (below the threshold, reversed, the rest
        grey)."""
        node = self._scoreColorsNode() if viewSet == nr.SSP_MAIN else None
        if node is not None and node.GetImageData() is not None:
            return slicer.util.arrayFromVolume(node), None
        score = self.scoreNode
        if score is None or not slicer.mrmlScene.IsNodePresent(score) or score.GetImageData() is None:
            return None
        if viewSet == nr.SSP_LOW:
            rgba, (low, high) = self._scoreColors(self.sspColorCombo.currentText, self.sspMaximumSpin.value,
                                                  nr.SIDE_THRESHOLD)
        else:
            rgba, (low, high) = self._scoreColors()
        table = np.clip(np.round(np.asarray(rgba, dtype=np.float64)[:, :3] * 255.0), 0, 255).astype(np.uint8)
        span = max(float(high) - float(low), 1e-6)

        def colorize(values):
            values = np.asarray(values, dtype=np.float64)
            index = np.clip(np.round((values - low) / span * (len(table) - 1)), 0, len(table) - 1).astype(np.int64)
            colors = table[index]
            colors[values <= nr.OUTSIDE / 2.0] = 0
            return colors
        return slicer.util.arrayFromVolume(score), colorize

    def _updateCutFaces(self, orientations=None, onlyMoved=False, viewSets=None):
        """The cut face of each row (in both its views): the brain (the regions' union) at the score slice, colored as
        the score slice (the low scores' cut views: as their 3D-SSP). onlyMoved: only the rows whose slice plane moved
        since their face was made; viewSets: only these sets' faces."""
        views = self._cutViewNodes()
        if not views:
            return
        for viewSet in sorted({key[2] for key in views}):
            if viewSets is None or viewSet in viewSets:
                self._updateCutFacesOf(views, viewSet, orientations, onlyMoved)

    def _updateCutFacesOf(self, views, viewSet, orientations, onlyMoved):
        from vtk.util import numpy_support
        result = self.reviewResult or {}
        planes = self._cutSliceNodes()
        source = self._cutFaceColors(viewSet) if result.get("labels") is not None and result.get("grid") is not None \
            else None
        mask = None
        if source is not None:
            labels = result["labels"][0]
            mask = labels > 0
            if mask.shape != source[0].shape[:3]:
                mask = None
        ijkToRas = _ijkToRas(result["grid"]) if mask is not None else None
        made = getattr(self, "_cutFacePlanes", None)
        if made is None:
            made = self._cutFacePlanes = {}
        for orientation, _ in nr.ORIENTATIONS:
            if orientations is not None and orientation not in orientations:
                continue
            key = ("face", orientation) + ((viewSet,) if viewSet else ())
            viewIds = [viewNode.GetID() for (o, _, vs), (_, viewNode) in views.items()
                       if o == orientation and vs == viewSet]
            face = None
            if mask is not None and orientation in planes and viewIds:
                origin, normal = self._slicePlane(planes[orientation])
                plane = (tuple(np.round(origin, 3)), tuple(np.round(normal, 4)))
                if onlyMoved and made.get((orientation, viewSet)) == plane:
                    continue        # modified, not moved (e.g. its field of view): the face as it is
                made[(orientation, viewSet)] = plane
                face = ssp.cutFace(source[0], mask, ijkToRas, normal, origin, source[1])
            if face is None:
                self._hideCutModel(key)
                continue
            poly = vtk.vtkPolyData()
            points = vtk.vtkPoints()
            points.SetData(numpy_support.numpy_to_vtk(np.asarray(face["points"], np.float32), deep=True))
            poly.SetPoints(points)
            quads = np.asarray(face["quads"], dtype=np.int64)
            cells = np.hstack([np.full((len(quads), 1), 4, dtype=np.int64), quads]).ravel()
            polys = vtk.vtkCellArray()
            polys.SetCells(len(quads), numpy_support.numpy_to_vtkIdTypeArray(cells, deep=True))
            poly.SetPolys(polys)
            colors = numpy_support.numpy_to_vtk(np.ascontiguousarray(face["colors"], dtype=np.uint8), deep=True)
            colors.SetName("Colors")
            poly.GetCellData().AddArray(colors)
            poly.GetCellData().SetActiveScalars("Colors")
            model = self._cutModelNode(key)
            model.SetAndObservePolyData(poly)
            display = model.GetDisplayNode()
            if hasattr(display, "SetActiveScalar"):
                display.SetActiveScalar("Colors", vtk.vtkAssignAttribute.CELL_DATA)
            else:
                display.SetActiveScalarName("Colors")
                display.SetActiveAttributeLocation(vtk.vtkAssignAttribute.CELL_DATA)
            display.SetScalarVisibility(True)
            if hasattr(slicer.vtkMRMLDisplayNode, "UseDirectMapping"):       # the RGB as it is (no color table)
                display.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseDirectMapping)
            display.SetBackfaceCulling(False)
            display.SetVisibility2D(False)
            if hasattr(display, "SetLighting"):         # flat: as on the slice
                display.SetLighting(False)
            else:
                display.SetAmbient(1.0)
                display.SetDiffuse(0.0)
            self._setViewNodeIds(display, viewIds)

    def _setupCutViews(self, fit=False):
        """The cut 3D-SSP views (if the layout has them): black, no box, parallel projection, each camera from its
        side (left / right anterior superior) at the 3D-SSP's zoom; the cuts made and the score slices watched."""
        views = self._cutViewNodes()
        self._watchCutSlices(bool(views))
        if not views:
            return
        data = (self.reviewResult or {}).get("ssp")
        if data:
            self._showSsp()         # the main 3D-SSP's values (its models hidden without its views): then the cuts
        self._updateCutFaces()
        center = self._sspCenter()
        scale = None
        if data and fit:
            points = np.asarray(data["surface"]["points"], dtype=np.float64)
            half = 0.5 * (points.max(axis=0) - points.min(axis=0)) if len(points) else np.full(3, 100.0)
            scale = 1.05 * float(max(half))
        cameras = slicer.modules.cameras.logic()
        for (orientation, side, _), (widget, viewNode) in views.items():
            widget.threeDController().setVisible(False)
            viewNode.SetBoxVisible(False)
            viewNode.SetAxisLabelsVisible(False)
            viewNode.SetBackgroundColor(0.0, 0.0, 0.0)
            viewNode.SetBackgroundColor2(0.0, 0.0, 0.0)
            viewNode.SetRenderMode(slicer.vtkMRMLViewNode.Orthographic)
            if hasattr(viewNode, "SetOrientationMarkerType"):
                viewNode.SetOrientationMarkerType(slicer.vtkMRMLAbstractViewNode.OrientationMarkerTypeCube)
                viewNode.SetOrientationMarkerSize(slicer.vtkMRMLAbstractViewNode.OrientationMarkerSizeSmall)
            camera = cameras.GetViewActiveCameraNode(viewNode)
            if camera is None:
                continue
            if scale:
                direction, up = ssp.cutCamera(nr.cutDirection(side))
                wasModifying = camera.StartModify()
                camera.SetFocalPoint(*center)
                camera.SetPosition(*(center + SSP_CAMERA_DISTANCE * direction))
                camera.SetViewUp(*up)
                camera.SetParallelScale(scale)
                camera.EndModify(wasModifying)
            self._clipSspCamera(widget, camera)
            widget.threeDView().forceRender()
