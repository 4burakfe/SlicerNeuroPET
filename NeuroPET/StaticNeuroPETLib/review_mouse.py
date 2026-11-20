"""
Review: the mouse on the review views: region name at the mouse, right drag = window, 3D-SSP region
outlines.

Part of the Static Neuro PET/SPECT module (StaticNeuroPET.py): the methods of ReviewMouseMixin, mixed into
StaticNeuroPETWidget there.
"""

import logging

import ctk
import numpy as np
import qt
import slicer
import vtk

from DynamicNeuroPET import _ijkToRas
from NeuroPETLib import neuroreview as nr, ssp

from StaticNeuroPETLib.constants import (CLICK_TOLERANCE_PX, MOUSE_PRIORITY, SSP_OUTLINE_ATTRIBUTE,
                                         SSP_OUTLINE_COLOR, SSP_OUTLINE_LIFT_MM, SSP_OUTLINE_NAME, SSP_OUTLINE_WIDTH,
                                         SSP_PICK_MM, WINDOW_DRAG_MS)


class ReviewMouseMixin:
    """The review mouse methods of StaticNeuroPETWidget."""

    @staticmethod
    def _buildMouseHelp():
        """How to use the mouse on the review's slices and 3D-SSP views."""
        box = ctk.ctkCollapsibleButton()
        box.text = "Mouse controls"
        form = qt.QVBoxLayout(box)
        label = qt.QLabel(
            "<b>Slices</b><ul style='margin-left:-20px'>"
            "<li>Right click: the view's menu. Right drag: window. PET row: left / right the width, up / down the level. Score row: the score "
            "window (right / up wider).</li>"
            "<li>Wheel: next / previous slice. Ctrl + wheel: zoom. Middle drag: pan.</li>"
            "<li>Shift + move: the other views follow the mouse.</li>"
            "<li>Mouse over a region: its name, SUVmean and score.</li>"
            "<li>Click a region in the table: it is shown alone for a few seconds.</li></ul>"
            "<b>3D-SSP</b><ul style='margin-left:-20px'>"
            "<li>Left drag: rotate the view (Restore 3D views turns it back).</li>"
            "<li>Right click: the view's menu. Right drag: score window (the patient window with the Patient map).</li>"
            "<li>Wheel: zoom. Middle drag: pan. Zoom and pan are the same in all the 3D-SSP views.</li>"
            "<li>Mouse over the surface: the region's name, SUVmean and score; its outline is drawn in white in "
            "every 3D-SSP view.</li></ul>")
        label.wordWrap = True
        label.setTextFormat(qt.Qt.RichText)
        form.addWidget(label)
        return box

    # the name of the region under the mouse, on its left
    def _hoverNames(self, enable):
        crosshair = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLCrosshairNode")
        old = getattr(self, "_hoverObserver", None)
        if old is not None:
            try:
                old[0].RemoveObserver(old[1])
            except Exception:
                pass
        self._hoverObserver = None
        self._hideHoverName()
        if enable and crosshair is not None:
            tag = crosshair.AddObserver(slicer.vtkMRMLCrosshairNode.CursorPositionModifiedEvent,
                                        lambda caller, event: self._onCursorMoved(caller))
            self._hoverObserver = (crosshair, tag)

    def _hoverLabel(self, parent):
        label = getattr(self, "_hoverNameLabel", None)
        try:
            if label is not None and label.parent() is not parent:
                label.setParent(parent)
        except Exception:       # deleted with its view
            label = None
        if label is None:
            label = qt.QLabel(parent)
            label.setStyleSheet("QLabel { color: white; background-color: rgba(0, 0, 0, 170); padding: 1px 4px; "
                                "border-radius: 3px; }")
            label.setAttribute(qt.Qt.WA_TransparentForMouseEvents)
            self._hoverNameLabel = label
        return label

    def _hideHoverName(self):
        label = getattr(self, "_hoverNameLabel", None)
        try:
            if label is not None:
                label.hide()
        except Exception:
            self._hoverNameLabel = None

    def _onCursorMoved(self, crosshair):
        """The visible regions at the mouse on a review slice view, named left of the mouse."""
        try:
            xyz = [0.0, 0.0, 0.0]
            sliceNode = crosshair.GetCursorPositionXYZ(xyz)
            if sliceNode is None and getattr(self, "_mouseObservers", None):
                return          # the mouse on a 3D view: named there by _sspHover
            segmentation = self.node("Segmentation")
            display = segmentation.GetDisplayNode() if segmentation is not None else None
            names = []
            widget = None
            if sliceNode is not None and sliceNode.GetLayoutName() in nr.peelViewTags():
                ras = [0.0, 0.0, 0.0]
                crosshair.GetCursorPositionRAS(ras)
                self._peelHover(sliceNode.GetLayoutName(), ras)
                return
            if getattr(self, "_peelHighlighted", 0):
                self._peelHighlight(0)
            if sliceNode is not None and display is not None:
                layoutManager = slicer.app.layoutManager()
                widget = layoutManager.sliceWidget(sliceNode.GetLayoutName()) if layoutManager else None
            if widget is not None and any(w is widget for _, w in self._reviewSliceWidgets()):
                ras = [0.0, 0.0, 0.0]
                crosshair.GetCursorPositionRAS(ras)
                result = self.reviewResult or {}
                if result.get("labels") is not None and result.get("segmentation") is segmentation:
                    # the regions of the comparison: named also with the outlines hidden (a region hidden by a click
                    # in the table is not named)
                    labels, regionNames = result["labels"]
                    found = nr.labelAt(labels, _ijkToRas(result["grid"]), ras)
                    _, ids = self._segmentColors()
                    if found and display.GetSegmentVisibility(ids.get(regionNames[found - 1], "")):
                        names.append(regionNames[found - 1])
                elif display.GetVisibility() and display.GetVisibility2DOutline():
                    manager = widget.sliceView().displayableManagerByClassName(
                        "vtkMRMLSegmentationsDisplayableManager2D")
                    ids = vtk.vtkStringArray()
                    if manager is not None:
                        manager.GetVisibleSegmentsForPosition(ras, display, ids)
                    seg = segmentation.GetSegmentation()
                    for i in range(ids.GetNumberOfValues()):
                        segment = seg.GetSegment(ids.GetValue(i))
                        if segment is not None:
                            names.append(segment.GetName())
            if not names:
                self._hideHoverName()
                return
            self._showHoverText(widget.sliceView(), names, stacked=True)       # one value a line, as on the 3D-SSP
        except Exception:
            logging.debug("Static Neuro: no region name at the mouse", exc_info=True)
            self._hideHoverName()

    def _hoverText(self, name, stacked=False):
        """A region's name with its SUVmean and score (of the last comparison) for the label at the mouse."""
        suv, score = (getattr(self, "_hoverValues", None) or {}).get(name, (None, None))
        parts = [name]
        if suv is not None and np.isfinite(suv):
            parts.append(f"SUVmean {suv:.2f}")
        if score is not None and np.isfinite(score):
            parts.append(f"{getattr(self, '_hoverKind', '') or self._scoreKind()} {score:+.2f}")
        return ("\n" if stacked else "   ").join(parts)       # stacked (one value a line) on the 3D-SSP

    def _showHoverText(self, view, names, stacked=False):
        """The names (with their values) left of the mouse on view (right of it without room)."""
        label = self._hoverLabel(view)
        label.text = ("\n\n" if stacked else "\n").join(self._hoverText(name, stacked) for name in names)
        label.adjustSize()
        point = view.mapFromGlobal(qt.QCursor.pos())
        x = point.x() - label.width - 12
        if x < 2:           # no room on the left: on the right
            x = point.x() + 16
        label.move(x, max(2, point.y() - label.height // 2))
        label.show()
        label.raise_()

    # --- the mouse on the review views: right drag = window; the region under the mouse on the 3D-SSP -------------
    def _mouseTargets(self):
        """[(interactor, what, view widget)] of the review views: what "pet" / "score" (slice rows) or (view set, view
        key) of a 3D-SSP view."""
        out = []
        for row, widget in self._reviewSliceWidgets():
            out.append((widget.sliceView().interactor(), "pet" if row == nr.PET_ROW else "score", widget.sliceView()))
        for key, (widget, _) in self._sspViewNodes().items():
            out.append((widget.threeDView().interactor(), key, widget.threeDView()))
        for key, (widget, _) in self._cutViewNodes().items():
            out.append((widget.threeDView().interactor(), ("cut", key), widget.threeDView()))
        return out

    def _watchMouse(self, enable):
        """Observers on the review views' interactors (before Slicer's own, which they stop for the right button):
        right drag sets the window of the row (PET window, score window) or of the 3D-SSP; on the 3D-SSP the region
        under the mouse is named (SUVmean, score) and outlined on the surface."""
        targets = self._mouseTargets() if enable else []
        old = getattr(self, "_mouseObservers", None) or {}
        if enable and sorted(old) == sorted(id(t[0]) for t in targets) and all(
                old[id(t[0])][1] == t[1] for t in targets):
            return
        for interactor, _, _, tags in old.values():
            for tag in tags.values():
                try:
                    interactor.RemoveObserver(tag)
                except Exception:
                    pass
        self._mouseObservers = {}
        self._drag = None
        for interactor, what, view in targets:
            if interactor is None:
                continue
            tags = {}
            for event in ("RightButtonPressEvent", "MouseMoveEvent", "RightButtonReleaseEvent", "LeaveEvent"):
                tags[event] = interactor.AddObserver(
                    event, lambda caller, ev, w=what, v=view: self._onViewMouse(caller, ev, w, v), MOUSE_PRIORITY)
            self._mouseObservers[id(interactor)] = (interactor, what, view, tags)

    def _replayRightClick(self, interactor, position):
        """A right click (press and release, not moved) given to Slicer's interaction: its context menu."""
        self._replayingClick = True
        try:
            interactor.SetEventPosition(*position)
            interactor.InvokeEvent("RightButtonPressEvent")
            interactor.InvokeEvent("RightButtonReleaseEvent")
        except Exception:
            logging.debug("Static Neuro: no right click menu", exc_info=True)
        finally:
            self._replayingClick = False

    def _abortEvent(self, interactor, event):
        entry = (getattr(self, "_mouseObservers", None) or {}).get(id(interactor))
        tag = entry[3].get(event) if entry else None
        command = interactor.GetCommand(tag) if tag is not None else None
        if command is not None:
            command.SetAbortFlag(1)

    def _windowTarget(self, what):
        """(kind, widget) set by a right drag on a view: ("range", range widget) for a display range, ("maximum", spin
        box) for a score window symmetric about 0; None when the view has none."""
        if what == "pet":
            return "range", self.petWindowRange
        if what == "score":
            return "maximum", self.scoreMaximumSpin
        if self.sspMapCombo.currentData == ssp.MAP_PATIENT and what[0] == nr.SSP_MAIN:
            return "range", self.sspWindowRange
        return "maximum", self.sspMaximumSpin

    @staticmethod
    def _rightButtonDown():
        try:
            return bool(int(qt.QApplication.mouseButtons()) & int(qt.Qt.RightButton))
        except Exception:
            return True         # not known: keep the drag

    def _onViewMouse(self, interactor, event, what, view):
        if getattr(self, "_replayingClick", False):     # a right click given back to Slicer (its view menu)
            return
        try:
            drag = getattr(self, "_drag", None)
            if event == "RightButtonPressEvent":
                target = self._windowTarget(what)
                if target is None:
                    return
                kind, widget = target
                start = ((widget.minimumValue, widget.maximumValue) if kind == "range" else float(widget.value))
                span = (widget.maximum - widget.minimum) if kind == "range" else None
                self._drag = {"interactor": interactor, "position": tuple(interactor.GetEventPosition()),
                              "kind": kind, "widget": widget, "start": start, "span": span, "moved": False}
                self._abortEvent(interactor, event)
            elif event == "MouseMoveEvent":
                if drag is not None and not self._rightButtonDown():
                    self._drag = drag = None        # the release went elsewhere (e.g. outside the view)
                if drag is not None and drag["interactor"] is interactor:
                    if not drag["moved"]:
                        x0, y0 = drag["position"]
                        x, y = interactor.GetEventPosition()
                        drag["moved"] = abs(x - x0) + abs(y - y0) > CLICK_TOLERANCE_PX
                        if drag["moved"]:
                            self._hideHoverName()
                    if drag["moved"]:
                        self._dragWindow(drag, interactor)
                    self._abortEvent(interactor, event)
                elif not isinstance(what, str) and what[0] == "cut":
                    self._cutHover(interactor, view)
                elif not isinstance(what, str):
                    self._sspHover(interactor, view)
            elif event == "RightButtonReleaseEvent":
                if drag is not None and drag["interactor"] is interactor:
                    self._drag = None
                    self._abortEvent(interactor, event)
                    if drag["moved"]:
                        self._dragWindow(drag, interactor)
                        self._applyDragWindow()
                    else:       # a click, not a drag: Slicer's own right click (the view's menu)
                        position = drag["position"]
                        qt.QTimer.singleShot(0, lambda i=interactor, p=position: self._replayRightClick(i, p))
            elif event == "LeaveEvent":
                self._hideHoverName()
                if not isinstance(what, str):
                    self._sspOutline(None)
        except Exception:
            logging.debug("Static Neuro: mouse on a review view", exc_info=True)

    def _dragWindow(self, drag, interactor):
        """The window from the drag: left / right the width, down / up the level (a display range); for a symmetric
        score window right or up wider. Applied at most every WINDOW_DRAG_MS."""
        x0, y0 = drag["position"]
        x, y = interactor.GetEventPosition()
        size = interactor.GetRenderWindow().GetSize()
        scale = float(max(100, min(size[0], size[1])))
        dx, dy = (x - x0) / scale, (y - y0) / scale
        if drag["kind"] == "range":
            low, high = drag["start"]
            span = drag["span"] or max(high - low, 1e-6)
            width = max(span * 1e-3, (high - low) + dx * span)
            center = 0.5 * (low + high) - dy * 0.5 * span
            drag["value"] = (center - 0.5 * width, center + 0.5 * width)
        else:
            widget = drag["widget"]
            drag["value"] = float(min(widget.maximum, max(widget.minimum, drag["start"] * float(np.exp(2.0 * (dx + dy))))))
        self._pendingDrag = drag
        if not getattr(self, "_dragTimerRunning", False):
            self._dragTimerRunning = True
            qt.QTimer.singleShot(WINDOW_DRAG_MS, self._applyDragWindow)

    def _applyDragWindow(self):
        self._dragTimerRunning = False
        drag = getattr(self, "_pendingDrag", None)
        self._pendingDrag = None
        if drag is None or drag.get("value") is None:
            return
        widget = drag["widget"]
        if drag["kind"] == "range":
            low, high = drag["value"]
            widget.setValues(max(widget.minimum, low), min(widget.maximum, max(high, low + 1e-6)))
        else:
            widget.value = round(drag["value"], widget.decimals)

    def _sspVertexLabels(self):
        """(surface, KD tree of its vertices, region label of each vertex) of the last comparison's 3D-SSP; None
        without one."""
        result = self.reviewResult or {}
        data = result.get("ssp")
        if not data or result.get("labels") is None:
            return None
        surface = data["surface"]
        found = getattr(self, "_sspLabels", None)
        if found is not None and found[0] is surface and found[3] is result["labels"]:
            return found[:3]
        from scipy.spatial import cKDTree
        labels, _ = result["labels"]
        points = np.asarray(surface["points"], dtype=np.float64)
        vertexLabels = ssp.vertexLabels(labels, _ijkToRas(result["grid"]), points, surface["normals"])
        self._sspLabels = (surface, cKDTree(points), vertexLabels, result["labels"])
        return self._sspLabels[:3]

    def _sspHover(self, interactor, view):
        """The region under the mouse on a 3D-SSP view: its name (SUVmean, score) next to the mouse and its outline
        on the surface."""
        found = self._sspVertexLabels()
        if found is None:
            return
        surface, tree, vertexLabels = found
        x, y = interactor.GetEventPosition()
        renderer = interactor.GetRenderWindow().GetRenderers().GetFirstRenderer()
        if renderer is None:
            return
        picker = getattr(self, "_sspPicker", None)
        if picker is None:
            picker = self._sspPicker = vtk.vtkWorldPointPicker()
        picker.Pick(x, y, 0, renderer)
        distance, index = tree.query(np.asarray(picker.GetPickPosition(), dtype=np.float64))
        label = int(vertexLabels[index]) if distance < SSP_PICK_MM else 0
        _, names = (self.reviewResult or {}).get("labels")
        if label <= 0 or label > len(names):
            self._hideHoverName()
            self._sspOutline(None)
            return
        self._showHoverText(view, [names[label - 1]], stacked=True)
        self._sspOutline(label)

    def _cutHover(self, interactor, view):
        """The region under the mouse on a cut 3D-SSP view (the cut face: the voxel's region; the surface: its nearest
        vertex's): its name, SUVmean and score next to the mouse."""
        result = self.reviewResult or {}
        if result.get("labels") is None or result.get("grid") is None:
            return
        x, y = interactor.GetEventPosition()
        renderer = interactor.GetRenderWindow().GetRenderers().GetFirstRenderer()
        if renderer is None:
            return
        picker = getattr(self, "_sspPicker", None)
        if picker is None:
            picker = self._sspPicker = vtk.vtkWorldPointPicker()
        picker.Pick(x, y, 0, renderer)
        ras = np.asarray(picker.GetPickPosition(), dtype=np.float64)
        labels, names = result["labels"]
        label = nr.labelAt(labels, _ijkToRas(result["grid"]), ras)
        if not label:
            found = self._sspVertexLabels()
            if found is not None:
                _, tree, vertexLabels = found
                distance, index = tree.query(ras)
                label = int(vertexLabels[index]) if distance < SSP_PICK_MM else 0
        if label <= 0 or label > len(names):
            self._hideHoverName()
            return
        self._showHoverText(view, [names[label - 1]], stacked=True)

    def _sspOutline(self, label):
        """The outline of region label on the 3D-SSP surface (white, SSP_OUTLINE_WIDTH px, a little outside the
        surface so it is drawn over it) in every 3D-SSP view; hidden with None."""
        if getattr(self, "_sspOutlineLabel", None) == label:
            return
        self._sspOutlineLabel = label
        models = getattr(self, "_sspOutlineModels", None)
        if models is None:
            models = self._sspOutlineModels = {}
        found = self._sspVertexLabels() if label else None
        views = self._sspViewNodes() if found else {}
        for part in (ssp.PART_BRAIN, ssp.PART_LEFT, ssp.PART_RIGHT):
            node = models.get(part)
            if node is not None and not slicer.mrmlScene.IsNodePresent(node):
                node = models[part] = None
            shown = [viewNode.GetID() for (_, key), (_, viewNode) in views.items() if ssp.viewPart(key) == part]
            if not found or not shown or not ssp.hasPart(found[0], part):
                if node is not None and node.GetDisplayNode() is not None:
                    node.GetDisplayNode().SetVisibility(False)
                continue
            lines = self._sspOutlineLines(found[0], found[2], part, label)
            if node is None:
                scene = slicer.mrmlScene
                node = scene.AddNewNodeByClass("vtkMRMLModelNode", scene.GenerateUniqueName(SSP_OUTLINE_NAME))
                node.SetAttribute(SSP_OUTLINE_ATTRIBUTE, "1")
                node.SetHideFromEditors(True)
                node.CreateDefaultDisplayNodes()
                models[part] = node
            node.SetAndObservePolyData(lines)
            display = node.GetDisplayNode()
            wasModifying = display.StartModify()
            display.SetColor(*SSP_OUTLINE_COLOR)
            display.SetLineWidth(SSP_OUTLINE_WIDTH)
            display.SetScalarVisibility(False)
            display.SetVisibility2D(False)
            if hasattr(display, "SetLighting"):
                display.SetLighting(False)
            display.SetVisibility(True)
            display.EndModify(wasModifying)
            self._setViewNodeIds(display, shown)

    def _sspOutlineLines(self, surface, vertexLabels, part, label):
        """The outline of a region on one part of the surface: the 0.5 contour of its vertices (1 in, 0 out) on the
        part's surface moved SSP_OUTLINE_LIFT_MM outward."""
        from vtk.util import numpy_support
        cache = getattr(self, "_sspOutlineSurfaces", None)
        if cache is None:
            cache = self._sspOutlineSurfaces = {}
        found = cache.get(part)
        if found is None or found[0] is not surface:
            if part == ssp.PART_BRAIN and surface.get("parts") is None:
                piece, index = surface, None
            else:
                piece, index = ssp.piece(surface, part)
            normals = np.asarray(piece["normals"], dtype=np.float64)
            lengths = np.linalg.norm(normals, axis=1)
            lengths[lengths == 0] = 1.0
            lifted = dict(piece, points=np.asarray(piece["points"], np.float64)
                          + SSP_OUTLINE_LIFT_MM * normals / lengths[:, None])
            poly = self.logic.surfacePolyData(lifted, np.zeros(len(lifted["points"]), np.float32), name="Region")
            contour = vtk.vtkContourFilter()
            contour.SetInputData(poly)
            contour.SetValue(0, 0.5)
            found = cache[part] = (surface, poly, index, contour)
        _, poly, index, contour = found
        inside = (np.asarray(vertexLabels if index is None else vertexLabels[index]) == label).astype(np.float32)
        array = poly.GetPointData().GetArray("Region")
        numpy_support.vtk_to_numpy(array)[:] = inside
        array.Modified()
        poly.Modified()
        contour.Update()
        out = vtk.vtkPolyData()
        out.DeepCopy(contour.GetOutput())
        return out

    def _sspAllOutlines(self):
        """The borders of all the regions on the 3D-SSP surface in every 3D-SSP view ("Show the region outlines" of
        the 3D-SSP box): drawn as the outline of the region under the mouse, which stays apart from this setting."""
        models = getattr(self, "_sspAllOutlineModels", None)
        if models is None:
            models = self._sspAllOutlineModels = {}
        on = hasattr(self, "sspOutlineCheck") and bool(self.sspOutlineCheck.checked)
        found = self._sspVertexLabels() if on else None
        views = self._sspViewNodes() if found else {}
        for part in (ssp.PART_BRAIN, ssp.PART_LEFT, ssp.PART_RIGHT):
            node = models.get(part)
            if node is not None and not slicer.mrmlScene.IsNodePresent(node):
                node = models[part] = None
            shown = [viewNode.GetID() for (_, key), (_, viewNode) in views.items() if ssp.viewPart(key) == part]
            if not found or not shown or not ssp.hasPart(found[0], part):
                if node is not None and node.GetDisplayNode() is not None:
                    node.GetDisplayNode().SetVisibility(False)
                continue
            lines = self._sspBorderLines(found[0], found[2], part)
            if node is None:
                scene = slicer.mrmlScene
                node = scene.AddNewNodeByClass("vtkMRMLModelNode", scene.GenerateUniqueName(SSP_OUTLINE_NAME + " all"))
                node.SetAttribute(SSP_OUTLINE_ATTRIBUTE, "all")
                node.SetHideFromEditors(True)
                node.CreateDefaultDisplayNodes()
                models[part] = node
            if node.GetPolyData() is not lines:
                node.SetAndObservePolyData(lines)
            display = node.GetDisplayNode()
            wasModifying = display.StartModify()
            display.SetColor(*SSP_OUTLINE_COLOR)
            display.SetLineWidth(SSP_OUTLINE_WIDTH)
            display.SetScalarVisibility(False)
            display.SetVisibility2D(False)
            if hasattr(display, "SetLighting"):
                display.SetLighting(False)
            display.SetVisibility(True)
            display.EndModify(wasModifying)
            self._setViewNodeIds(display, shown)

    def _sspBorderLines(self, surface, vertexLabels, part):
        """The borders between all the regions on one part of the surface (ssp.labelBorders), SSP_OUTLINE_LIFT_MM
        outside it; kept until the surface or the regions change."""
        from vtk.util import numpy_support
        cache = getattr(self, "_sspBorderPolys", None)
        if cache is None:
            cache = self._sspBorderPolys = {}
        found = cache.get(part)
        if found is not None and found[0] is surface and found[1] is vertexLabels:
            return found[2]
        if part == ssp.PART_BRAIN and surface.get("parts") is None:
            piece, index = surface, None
        else:
            piece, index = ssp.piece(surface, part)
        normals = np.asarray(piece["normals"], dtype=np.float64)
        lengths = np.linalg.norm(normals, axis=1)
        lengths[lengths == 0] = 1.0
        lifted = np.asarray(piece["points"], np.float64) + SSP_OUTLINE_LIFT_MM * normals / lengths[:, None]
        labels = np.asarray(vertexLabels if index is None else vertexLabels[index])
        ends, segments = ssp.labelBorders(lifted, piece["triangles"], labels)
        poly = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        points.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(ends, np.float32), deep=True))
        poly.SetPoints(points)
        cells = np.empty((len(segments), 3), np.int64)
        cells[:, 0] = 2
        cells[:, 1:] = segments
        lines = vtk.vtkCellArray()
        lines.SetCells(len(segments), numpy_support.numpy_to_vtkIdTypeArray(cells.ravel(), deep=True))
        poly.SetLines(lines)
        cache[part] = (surface, vertexLabels, poly)
        return poly

    def _dropSspOutlines(self):
        """The outline models removed (also those left by a module reload)."""
        for node in list(slicer.util.getNodesByClass("vtkMRMLModelNode")):
            if node.GetAttribute(SSP_OUTLINE_ATTRIBUTE):
                slicer.mrmlScene.RemoveNode(node)
        self._sspOutlineModels = {}
        self._sspAllOutlineModels = {}
        self._sspOutlineLabel = None
