"""
Review: the 3D-SSP views (surface models, colours, legends, cameras, side markers).

Part of the Static Neuro PET/SPECT module (StaticNeuroPET.py): the methods of ReviewSspMixin, mixed into
StaticNeuroPETWidget there.
"""

import logging

import ctk
import numpy as np
import qt
import slicer
import vtk

from DICOMGatedDynamicPETLib import colormaps
from DynamicNeuroPET import _button
from NeuroPETLib import neuroreview as nr, ssp

from StaticNeuroPETLib.constants import (REVIEW_SSP_PATIENT_COLORMAP, SSP_CAMERA_DISTANCE, SSP_COLOR_NODE,
                                         SSP_LOW_COLOR_NODE, SSP_MODEL_ATTRIBUTE, SSP_MODEL_NAME, SSP_SIDE_MARKER_X)


class ReviewSspMixin:
    """The review ssp methods of StaticNeuroPETWidget."""

    # 3D-SSP
    def _buildSspBox(self, layout):
        box = ctk.ctkCollapsibleButton()
        box.text = "3D-SSP"
        box.collapsed = True
        form = qt.QFormLayout(box)
        self.sspMapCombo = qt.QComboBox()
        for key, title in ssp.MAPS:
            self.sspMapCombo.addItem(title, key)
        self.sspMapCombo.toolTip = ("The score (z / t, as picked above) on the brain surface, or the patient's own "
                                    "projected values")
        form.addRow("Map:", self.sspMapCombo)
        self.sspSourceCombo = qt.QComboBox()
        for key, title in ssp.SOURCES:
            self.sspSourceCombo.addItem(title, key)
        self.sspSourceCombo.toolTip = (
            "Minoshima: the patient's maximum along the inward normal (0 - 20 mm, stopped at the brain's edge) scored against the same projection "
            "of every normal (vertex mean / SD). Voxel: the voxel score map projected (its most extreme score along the "
            "normal); used when the database has no projections of its normals (predefined).")
        form.addRow("Score from:", self.sspSourceCombo)
        self.sspColorCombo = qt.QComboBox()
        self.sspPatientColorCombo = qt.QComboBox()
        for label in nr.SCORE_COLORMAPS:
            self.sspColorCombo.addItem(label)
        for label in colormaps.LABELS:
            self.sspPatientColorCombo.addItem(label)
        self.sspPatientColorCombo.setCurrentIndex(max(0, self.sspPatientColorCombo.findText(REVIEW_SSP_PATIENT_COLORMAP)))
        form.addRow("Score colors:", self.sspColorCombo)
        self.sspMaximumSpin = qt.QDoubleSpinBox()
        self.sspMaximumSpin.setRange(0.5, 50.0)
        self.sspMaximumSpin.decimals = 1
        self.sspMaximumSpin.singleStep = 0.5
        self.sspMaximumSpin.value = nr.DEFAULT_MAXIMUM
        self.sspMaximumSpin.prefix = "\u00b1 "
        self.sspMaximumSpin.toolTip = "Score window of the surface, symmetric about 0"
        form.addRow("Score window:", self.sspMaximumSpin)
        form.addRow("Patient colors:", self.sspPatientColorCombo)
        self.sspWindowRange = ctk.ctkRangeWidget()
        self.sspWindowRange.decimals = 2
        self.sspWindowRange.toolTip = "Display range of the patient's projected values"
        form.addRow("Patient window:", self.sspWindowRange)
        self.sspLightingCheck = qt.QCheckBox("Lighting")
        self.sspLightingCheck.checked = True
        self.sspLightingCheck.toolTip = "Shaded surface; off: flat colors at full brightness (as in the original 3D-SSP)"
        form.addRow(self.sspLightingCheck)
        self.sspLightingCheck.connect("toggled(bool)", lambda on: self._showSsp())
        self.sspOutlineCheck = qt.QCheckBox("Show the region outlines")
        self.sspOutlineCheck.checked = False
        self.sspOutlineCheck.toolTip = ("The borders of all the regions drawn on the surface (as the outline of the "
                                        "region under the mouse, which is always shown)")
        form.addRow(self.sspOutlineCheck)
        self.sspOutlineCheck.connect("toggled(bool)", lambda on: self._sspAllOutlines())
        self.restoreViewsButton = _button("Restore 3D views")
        self.restoreViewsButton.toolTip = "The 3D-SSP views back to their default direction, zoom and position"
        form.addRow(self.restoreViewsButton)
        self.restoreViewsButton.connect("clicked()", lambda: self._setupSspViews(fit=True))
        layout.addWidget(box)
        for combo in (self.sspMapCombo, self.sspSourceCombo, self.sspColorCombo, self.sspPatientColorCombo):
            combo.connect("currentIndexChanged(int)", lambda i: self._showSsp())
        self.sspMaximumSpin.connect("valueChanged(double)", lambda v: self._showSsp())
        self.sspWindowRange.connect("valuesChanged(double,double)", lambda a, b: self._showSsp())
        self.scoreSideCombo.connect("currentIndexChanged(int)", lambda i: self._showSsp())
        self.thresholdSpin.connect("valueChanged(double)", lambda v: self._showSsp())

    def _patientUnits(self):
        """Units of the patient's map compared (and projected): the database's map type, SUVR to the reference
        region without a database, else SUV."""
        result = self.reviewResult or {}
        if not result.get("ratio"):
            return "SUV"
        header = result.get("header")
        return (header or {}).get("mapType") or "SUVR"

    def _initSspWindow(self):
        data = (self.reviewResult or {}).get("ssp")
        if not data:
            return
        index = self.sspMapCombo.findData(ssp.MAP_PATIENT)
        if index >= 0:
            self.sspMapCombo.setItemText(index, f"Patient ({self._patientUnits()}, projected, smoothed)")
        values = np.asarray(data["patient"], dtype=np.float32)
        low, high = (float(v) for v in np.percentile(values, (1.0, 99.5))) if values.size else (0.0, 1.0)
        high = high if high > low else low + 1.0
        self.sspWindowRange.blockSignals(True)
        self.sspWindowRange.minimum = min(0.0, low)
        self.sspWindowRange.maximum = max(float(values.max()), high) if values.size else high
        self.sspWindowRange.singleStep = max(high / 200.0, 0.001)
        self.sspWindowRange.setValues(low, high)
        self.sspWindowRange.blockSignals(False)

    def _sspScores(self):
        """(vertex scores, what) of the 3D-SSP (Minoshima or the voxel map by "Score from"); (None, "") without."""
        result = self.reviewResult
        data = result.get("ssp")
        kind = self._scoreKind()
        n = data.get("n") or result.get("n")
        if self.sspSourceCombo.currentData == ssp.SOURCE_MINOSHIMA and data.get("mean") is not None:
            return ssp.scores(data["patient"], data["mean"], data["sd"], n, kind), f"{kind} (Minoshima)"
        if data.get("voxel") is not None:
            factor = nr.tFactor(result.get("n")) if kind == nr.SCORE_T else 1.0
            return np.asarray(data["voxel"], np.float32) / np.float32(factor), f"{kind} (voxel map)"
        return None, ""

    def _sspValues(self, viewSet=nr.SSP_MAIN):
        """(vertex values, color table RGBA, (low, high), what) of a set of 3D-SSP views: the main one as set in the
        3D-SSP box; the low scores' the scores below the threshold (colors reversed, the rest grey). None for
        the low scores without scores."""
        result = self.reviewResult
        data = result.get("ssp")
        score = what = None
        if self.sspMapCombo.currentData == ssp.MAP_SCORE or viewSet == nr.SSP_LOW:
            score, what = self._sspScores()
        if score is not None:
            side = nr.SIDE_THRESHOLD if viewSet == nr.SSP_LOW else None
            colors, window = self._scoreColors(self.sspColorCombo.currentText, self.sspMaximumSpin.value, side)
            return score, colors, window, what + (f" < {self.thresholdSpin.value:g}" if viewSet == nr.SSP_LOW else "")
        if viewSet == nr.SSP_LOW:
            return None
        low, high = self.sspWindowRange.minimumValue, self.sspWindowRange.maximumValue
        colors = ssp.surfaceColors(self._rgbOf(self.sspPatientColorCombo.currentText))
        return np.asarray(data["patient"], np.float32), colors, (low, max(high, low + 1e-6)), "patient"

    def _sspColorNode(self, rgba, name=SSP_COLOR_NODE):
        """The color table of a set of 3D-SSP views, written only when its colors change and then in one modification
        (each SetColor alone fires a Modified event that updates every model using the table in every 3D view: 256
        of them made a layout change take tens of seconds)."""
        scene = slicer.mrmlScene
        rgba = np.asarray(rgba, dtype=np.float64)
        node = scene.GetFirstNodeByName(name)
        if node is None or not node.IsA("vtkMRMLColorTableNode"):
            node = scene.CreateNodeByClass("vtkMRMLColorTableNode")
            node.UnRegister(None)
            node.SetName(name)
            node.SetTypeToUser()
            node.SetNumberOfColors(len(rgba))
            scene.AddNode(node)
        written = getattr(self, "_sspColorsWritten", None)
        if written is None:
            written = self._sspColorsWritten = {}
        key = (node.GetID(), rgba.shape, rgba.tobytes())
        if written.get(name) == key:
            return node
        wasModifying = node.StartModify()
        try:
            if node.GetNumberOfColors() != len(rgba):
                node.SetNumberOfColors(len(rgba))
            for i, (r, g, b, a) in enumerate(rgba):
                node.SetColor(i, f"{i}", float(r), float(g), float(b), float(a))
        finally:
            node.EndModify(wasModifying)
        written[name] = key
        return node

    def _sspPolyData(self, viewSet, part, surface, values):
        """The polydata of one part of the surface in one set of views with its vertex values: the geometry built once
        per surface (and the part's vertex indices kept), only the values copied in when they change; (polydata,
        changed)."""
        from vtk.util import numpy_support
        cache = getattr(self, "_sspPolys", None)
        if cache is None:
            cache = self._sspPolys = {}
        found = cache.get((viewSet, part))
        if found is None or found[0] is not surface:
            if part == ssp.PART_BRAIN and surface.get("parts") is None:
                piece, index = surface, None
            else:
                piece, index = ssp.piece(surface, part)
            count = len(piece["points"])
            poly = self.logic.surfacePolyData(piece, np.zeros(count, np.float32))
            found = cache[(viewSet, part)] = (surface, poly, index, None)
        surface, poly, index, last = found
        values = np.asarray(values if index is None else values[index], dtype=np.float32)
        if last is not None and last.shape == values.shape and np.array_equal(last, values):
            return poly, False
        array = poly.GetPointData().GetArray("SSP")
        numpy_support.vtk_to_numpy(array)[:] = values
        array.Modified()
        poly.Modified()
        cache[(viewSet, part)] = (surface, poly, index, values.copy())
        return poly, True

    @staticmethod
    def _setViewNodeIds(display, viewIds):
        """The views of a display node, changed only when they differ (a change redraws the model in every view)."""
        current = [display.GetNthViewNodeID(i) for i in range(display.GetNumberOfViewNodeIDs())]
        if current == list(viewIds):
            return
        wasModifying = display.StartModify()
        display.RemoveAllViewNodeIDs()
        for viewId in viewIds:
            display.AddViewNodeID(viewId)
        display.EndModify(wasModifying)

    def _sspViewNodes(self):
        """{(view set, view key): (3D widget, view node)} of the 3D-SSP views of the layout."""
        layoutManager = slicer.app.layoutManager()
        out = {}
        if layoutManager is None:
            return out
        tags = nr.sspViewTags()
        for index in range(layoutManager.threeDViewCount):
            widget = layoutManager.threeDWidget(index)
            viewNode = widget.mrmlViewNode() if widget is not None else None
            if viewNode is not None and viewNode.GetSingletonTag() in tags:
                out[tags[viewNode.GetSingletonTag()]] = (widget, viewNode)
        return out

    def _sspModelNode(self, viewSet, part):
        """The model of one part of the surface (the brain, or a hemisphere for the medial views) in one set of
        views."""
        scene = slicer.mrmlScene
        node = self.sspModels.get((viewSet, part))
        if node is None or not scene.IsNodePresent(node):
            suffix = {ssp.PART_LEFT: " left hemisphere", ssp.PART_RIGHT: " right hemisphere"}.get(part, "")
            name = SSP_MODEL_NAME + (" low scores" if viewSet else "") + suffix
            node = scene.AddNewNodeByClass("vtkMRMLModelNode", scene.GenerateUniqueName(name))
            node.SetAttribute(SSP_MODEL_ATTRIBUTE, f"{viewSet or 'main'}/{part}")
            node.CreateDefaultDisplayNodes()
            self.sspModels[(viewSet, part)] = node
        return node

    def _dropStaleSspModels(self):
        """The 3D-SSP models in the scene that are not this widget's removed: those of a module reloaded (or an older
        version) stay in their views, drawn inside the new ones (a nested, speckled surface)."""
        mine = {node.GetID() for node in self.sspModels.values() if node is not None}
        for node in list(slicer.util.getNodesByClass("vtkMRMLModelNode")):
            name = node.GetName() or ""
            if (name.startswith(SSP_MODEL_NAME) or node.GetAttribute(SSP_MODEL_ATTRIBUTE)) and node.GetID() not in mine:
                slicer.mrmlScene.RemoveNode(node)

    def _dropSspModel(self, viewSet, part):
        getattr(self, "_sspPolys", {}).pop((viewSet, part), None)
        node = self.sspModels.pop((viewSet, part), None)
        if node is not None and slicer.mrmlScene.IsNodePresent(node):
            slicer.mrmlScene.RemoveNode(node)

    def _showSsp(self):
        result = self.reviewResult
        data = (result or {}).get("ssp")
        if not data or not hasattr(self, "sspMapCombo"):
            return
        self._dropStaleSspModels()
        views = self._sspViewNodes()
        surface = data["surface"]
        for viewSet, colorName in ((nr.SSP_MAIN, SSP_COLOR_NODE), (nr.SSP_LOW, SSP_LOW_COLOR_NODE)):
            shownViews = {key: viewNode for (vs, key), (_, viewNode) in views.items() if vs == viewSet}
            found = self._sspValues(viewSet) if viewSet == nr.SSP_MAIN or shownViews else None
            if found is None:
                for part in (ssp.PART_BRAIN, ssp.PART_LEFT, ssp.PART_RIGHT):
                    self._dropSspModel(viewSet, part)
                continue
            values, rgba, (low, high), what = found
            colorNode = self._sspColorNode(rgba, colorName)
            if viewSet == nr.SSP_MAIN:
                self._sspMainDisplay = (colorNode.GetID(), float(low), float(high))
            else:
                self._sspLowDisplay = (colorNode.GetID(), float(low), float(high))
            values = np.asarray(values, np.float32)
            models = []
            for part in (ssp.PART_BRAIN, ssp.PART_LEFT, ssp.PART_RIGHT):
                if not ssp.hasPart(surface, part):
                    self._dropSspModel(viewSet, part)
                    continue
                model = self._sspModelNode(viewSet, part)
                poly, _ = self._sspPolyData(viewSet, part, surface, values)
                if model.GetPolyData() is not poly:
                    model.SetAndObservePolyData(poly)
                display = model.GetDisplayNode()
                display.SetAndObserveColorNodeID(colorNode.GetID())
                display.SetActiveScalarName("SSP")
                display.SetScalarVisibility(True)
                display.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseManualScalarRange)
                display.SetScalarRange(low, high)
                display.SetBackfaceCulling(False)
                display.SetVisibility2D(False)
                if hasattr(display, "SetLighting"):         # off: full brightness, no shading (flat colors)
                    display.SetLighting(bool(self.sspLightingCheck.checked))
                shown = [viewNode.GetID() for key, viewNode in shownViews.items() if ssp.viewPart(key) == part]
                # not in the review layout: hidden (shown in the SSP views only)
                self._setViewNodeIds(display, shown or ["vtkMRMLViewNodeNone"])
                models.append(model)
            if not models:
                continue
            rows = [shownViews[key].GetID() for key in (nr.SSP_VIEWS[0][0], nr.SSP_VIEWS[nr.SSP_COLUMNS][0])
                    if key in shownViews]
            if not rows:        # the cut layout: on its first oblique view
                rows = [shownViews[key].GetID() for key, _ in nr.SSP_OBLIQUE_VIEWS[:1] if key in shownViews]
            self._legend(models[0], rows, what if what != "patient" else self._patientUnits(), "%.1f")
        self._sspAllOutlines()
        self._updateCutSurfaces()
        if any(key[2] == nr.SSP_LOW for key in self._cutViewNodes()):
            self._updateCutFaces(viewSets=(nr.SSP_LOW,))     # their colors: the 3D-SSP's score colors and window

    def _legend(self, node, viewIds, title, labelFormat):
        """A color legend of node on the left of the views viewIds (hidden with none)."""
        try:
            colorLogic = slicer.modules.colors.logic()
            if not hasattr(colorLogic, "AddDefaultColorLegendDisplayNode"):
                return
            legend = colorLogic.GetColorLegendDisplayNode(node) if hasattr(colorLogic, "GetColorLegendDisplayNode") \
                else None
            if legend is None:
                if not viewIds:
                    return
                legend = colorLogic.AddDefaultColorLegendDisplayNode(node)
            wasModifying = legend.StartModify()
            legend.RemoveAllViewNodeIDs()
            for viewId in viewIds:
                legend.AddViewNodeID(viewId)
            legend.SetVisibility(bool(viewIds) and bool(self.legendCheck.checked))
            legend.SetTitleText(title)
            legend.SetLabelFormat(labelFormat)
            legend.SetPosition(0.02, 0.5)       # left, middle
            legend.SetSize(0.1, 0.6)
            legend.EndModify(wasModifying)
        except Exception:
            logging.warning("Static Neuro: no color legend", exc_info=True)

    def _updateLegends(self):
        """One color legend per row, on the left: the PET, the scores, the 3D-SSP (each in its row's first view)."""
        if not hasattr(self, "legendCheck"):
            return
        first = {}
        for row, widget in self._reviewSliceWidgets():
            first.setdefault(row, widget.mrmlSliceNode().GetID())
        volume = self.volume()
        if volume is not None:
            self._legend(volume, [first[nr.PET_ROW]] if nr.PET_ROW in first else [], self._petUnits(volume), "%.2f")
        if self.scoreNode is not None and slicer.mrmlScene.IsNodePresent(self.scoreNode):
            self._legend(self.scoreNode, [first[nr.SCORE_ROW]] if nr.SCORE_ROW in first else [], self._scoreKind(),
                         "%.1f")

    def _petUnits(self, volume):
        units = self.logic.imageInfo(volume).get("units") or ""
        return "SUV" if not units or "suv" in units.lower() or "g/ml" in units.lower() else units

    def _syncCameras(self, enable=True):
        """Zoom and pan of the 3D-SSP views linked (each its own rotation): camera node observers."""
        for cameraNode, tag in getattr(self, "_cameraObservers", []):
            try:
                cameraNode.RemoveObserver(tag)
            except Exception:
                pass
        self._cameraObservers = []
        if not enable:
            return
        logic = slicer.modules.cameras.logic()
        for key, (_, viewNode) in self._sspViewNodes().items():
            cameraNode = logic.GetViewActiveCameraNode(viewNode)
            if cameraNode is not None:
                tag = cameraNode.AddObserver(vtk.vtkCommand.ModifiedEvent,
                                             lambda caller, event, k=key: self._onCameraMoved(k))
                self._cameraObservers.append((cameraNode, tag))

    def _sspBoundsOf(self):
        """(centre, radius) of the 3D-SSP surface's vertices, worked out once per surface; None without one."""
        data = (self.reviewResult or {}).get("ssp")
        if not data:
            return None
        surface = data["surface"]
        found = getattr(self, "_sspBounds", None)
        if found is None or found[0] is not surface:
            points = np.asarray(surface["points"], dtype=np.float64)
            center = points.mean(axis=0) if len(points) else np.zeros(3)
            radius = float(np.max(np.linalg.norm(points - center, axis=1))) if len(points) else 100.0
            found = self._sspBounds = (surface, center, radius)
        return found[1], found[2]

    def _sspCenter(self):
        bounds = self._sspBoundsOf()
        return bounds[0].copy() if bounds else np.zeros(3)

    def _onCameraMoved(self, sourceKey):
        if getattr(self, "_syncingCameras", False):
            return
        views = self._sspViewNodes()
        if sourceKey not in views:
            return
        logic = slicer.modules.cameras.logic()
        source = logic.GetViewActiveCameraNode(views[sourceKey][1])
        if source is None:
            return
        center = self._sspCenter()
        camera = source.GetCamera()
        pan = ssp.panOffset(center, camera.GetFocalPoint(), camera.GetPosition(), camera.GetViewUp())
        scale = camera.GetParallelScale()
        self._syncingCameras = True
        try:
            for key, (widget, viewNode) in views.items():
                target = logic.GetViewActiveCameraNode(viewNode)
                if key == sourceKey or target is None:
                    continue
                other = target.GetCamera()
                focal, position = ssp.pannedCamera(center, pan, other.GetFocalPoint(), other.GetPosition(),
                                                   other.GetViewUp())
                if (np.allclose(focal, other.GetFocalPoint(), atol=1e-3)
                        and abs(other.GetParallelScale() - scale) < 1e-6):
                    continue
                wasModifying = target.StartModify()
                target.SetFocalPoint(*focal)
                target.SetPosition(*position)
                target.SetParallelScale(scale)
                target.EndModify(wasModifying)
                self._clipSspCamera(widget, target)
        finally:
            self._syncingCameras = False

    def _clipSspCamera(self, widget, cameraNode):
        """The near and far planes of a 3D-SSP camera set to hold the whole surface (with the hemispheres' views
        drawn before anything was rendered in them, the planes were left where they cut the surface: the view black
        or striped until zoomed)."""
        bounds = self._sspBoundsOf()
        camera = cameraNode.GetCamera() if cameraNode is not None else None
        if bounds is None or camera is None:
            return
        center, radius = bounds
        focal = np.asarray(camera.GetFocalPoint(), dtype=np.float64)
        distance = float(np.linalg.norm(np.asarray(camera.GetPosition()) - focal))
        # the surface's bounding sphere seen from the focal point (not every vertex again at each camera change)
        reach = radius + float(np.linalg.norm(focal - center))
        margin = 20.0
        camera.SetClippingRange(max(0.1, distance - reach - margin), distance + reach + margin)
        try:
            widget.threeDView().scheduleRender()
        except AttributeError:
            pass

    def _sspSideMarkers(self, widget, viewNode, key):
        """The R / L markers on the left and right of a superior, inferior, anterior or posterior 3D-SSP view (text
        actors in its renderer, made once per view)."""
        labels = ssp.sideLabels(key)
        markers = getattr(self, "_sspMarkers", None)
        if markers is None:
            markers = self._sspMarkers = {}
        try:
            renderer = widget.threeDView().renderWindow().GetRenderers().GetFirstRenderer()
        except Exception:
            return
        old = markers.get(viewNode.GetID())
        if old is not None and (old[0] is not renderer or old[2] != labels):
            for actor in old[1]:
                old[0].RemoveViewProp(actor)
            old = None
        if old is not None or labels is None or renderer is None:
            return
        actors = []
        for text, x in zip(labels, SSP_SIDE_MARKER_X):
            actor = vtk.vtkTextActor()
            actor.SetInput(text)
            prop = actor.GetTextProperty()
            prop.SetFontSize(20)
            prop.BoldOn()
            prop.SetColor(1.0, 1.0, 1.0)
            prop.SetJustificationToCentered()
            prop.SetVerticalJustificationToCentered()
            actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            actor.SetPosition(x, 0.5)
            renderer.AddViewProp(actor)
            actors.append(actor)
        markers[viewNode.GetID()] = (renderer, actors, labels)

    def _setupSspViews(self, fit=False):
        """The 3D-SSP views: black, no box, parallel projection, a box orientation marker, each camera looking at
        the surface from its side (top and bottom with the front up, the others with the top up; the medial views at
        one hemisphere from the other side), all with the same zoom and pan, and near / far planes holding the whole
        surface."""
        self._setupCutViews(fit)
        data = (self.reviewResult or {}).get("ssp")
        views = self._sspViewNodes()
        if not views:
            self._syncCameras(False)
            return
        self._showSsp()
        center = self._sspCenter()
        scale = None
        self._syncingCameras = True
        try:
            for key, (widget, viewNode) in views.items():
                widget.threeDController().setVisible(False)
                viewNode.SetBoxVisible(False)
                viewNode.SetAxisLabelsVisible(False)
                viewNode.SetBackgroundColor(0.0, 0.0, 0.0)
                viewNode.SetBackgroundColor2(0.0, 0.0, 0.0)
                viewNode.SetRenderMode(slicer.vtkMRMLViewNode.Orthographic)
                if hasattr(viewNode, "SetOrientationMarkerType"):      # a small box with the sides' letters
                    viewNode.SetOrientationMarkerType(slicer.vtkMRMLAbstractViewNode.OrientationMarkerTypeCube)
                    viewNode.SetOrientationMarkerSize(slicer.vtkMRMLAbstractViewNode.OrientationMarkerSizeSmall)
                if not data or not fit:
                    continue
                direction, up = ssp.CAMERAS[key[1]]
                camera = slicer.modules.cameras.logic().GetViewActiveCameraNode(viewNode)
                if camera is None:
                    continue
                camera.SetFocalPoint(*center)
                camera.SetPosition(*(center + SSP_CAMERA_DISTANCE * np.asarray(direction)))
                camera.SetViewUp(*up)
            if data and fit:
                # one zoom for all, centred: the whole surface's extent across the views (not ResetCamera, which
                # sees nothing in a view not drawn yet and centres each on its own model)
                points = np.asarray(data["surface"]["points"], dtype=np.float64)
                half = 0.5 * (points.max(axis=0) - points.min(axis=0)) if len(points) else np.full(3, 100.0)
                scale = 1.05 * float(max(half[1], half[2], half[0]))
            for key, (widget, viewNode) in views.items():
                camera = slicer.modules.cameras.logic().GetViewActiveCameraNode(viewNode)
                if camera is None:
                    continue
                if scale:
                    direction, _ = ssp.CAMERAS[key[1]]
                    camera.SetFocalPoint(*center)
                    camera.SetPosition(*(center + SSP_CAMERA_DISTANCE * np.asarray(direction)))
                    camera.SetParallelScale(scale)
                self._clipSspCamera(widget, camera)
                self._sspSideMarkers(widget, viewNode, key[1])
                widget.threeDView().forceRender()
        finally:
            self._syncingCameras = False
        self._syncCameras(True)
        self._watchMouse(True)
