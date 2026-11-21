"""
Review: the unfolded brain (peel) layout.

Part of the Static Neuro PET/SPECT module (StaticNeuroPET.py): the methods of ReviewPeelMixin, mixed into
StaticNeuroPETWidget there.
"""

import logging

import numpy as np
import qt
import slicer
import vtk

from DICOMGatedDynamicPETLib import colormaps
from NeuroWorkflowLib.base import _ijkToRas, _vtkMatrix
from NeuroPETLib import neuroreview as nr, peel

from StaticNeuroPETLib.constants import MOUSE_PRIORITY, PEEL_ATTRIBUTE, PEEL_NODE_PREFIX, SSP_SETUP_DELAY_MS


class ReviewPeelMixin:
    """The review peel methods of StaticNeuroPETWidget."""

    # the unfolded brain layout: the brain peeled at one depth (curvilinear reformatting) and unfolded in one map;
    # rows the anatomy (the patient's normalized MR, else the MNI T1) with the region outlines, the PET, the scores
    def _peelViewWidgets(self):
        """{(row, view key): slice widget} of the peel views of the layout."""
        layoutManager = slicer.app.layoutManager()
        out = {}
        if layoutManager is None:
            return out
        for tag, key in nr.peelViewTags().items():
            widget = layoutManager.sliceWidget(tag)
            if widget is not None:
                out[key] = widget
        return out

    def _peelAnatomy(self):
        """The anatomy of the peel: the patient's normalized MR, else the MNI T1 (None without either)."""
        normalized = self.node("NormalizedAnatomy")
        if (normalized is not None and slicer.mrmlScene.IsNodePresent(normalized)
                and self.anatomy() is not None and self.contrast()[0] != "CT"):
            return normalized
        if "T1" in (self.atlas or {}):
            try:
                return self.logic.loadAtlasVolume(self.atlas["T1"], name="MNI ICBM152 2009c T1")
            except Exception:
                logging.debug("Static Neuro: no MNI T1 for the peel", exc_info=True)
        return None

    def _peelDepthMap(self):
        """(depth map mm, IJK to RAS) of the review grid's brain (the regions, holes filled), made once per
        comparison."""
        result = self.reviewResult or {}
        if result.get("labels") is None or result.get("grid") is None:
            return None
        cached = getattr(self, "_peelDepth", None)
        if cached is not None and cached[0] is result["labels"]:
            return cached[1], cached[2]
        grid = _ijkToRas(result["grid"])
        brain = np.asarray(result["labels"][0]) > 0
        hull = result.get("hull")
        if hull is not None and np.shape(hull) == brain.shape:
            brain = brain | hull        # every CerebrA structure: not hollow when the regions leave some out
        depth = peel.depthMap(peel.brainMask(brain, grid), grid)
        self._peelDepth = (result["labels"], depth, grid)
        self._peelRays = None
        return depth, grid

    def _peelRaysOf(self, depth, grid, center):
        """The rays of the unfolded maps (made once a brain and centre: a few seconds)."""
        cached = getattr(self, "_peelRays", None)
        if cached is not None and cached[0] is depth and cached[1] == center:
            return cached[2]
        slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            rays = peel.unfoldRays(depth, grid, center)
        finally:
            slicer.app.restoreOverrideCursor()
        self._peelRays = (depth, center, rays)
        return rays

    def _peelNode(self, name, className="vtkMRMLScalarVolumeNode"):
        scene = slicer.mrmlScene
        nodes = getattr(self, "peelNodes", None)
        if nodes is None:
            nodes = self.peelNodes = {}
        node = nodes.get(name)
        if node is None or not scene.IsNodePresent(node):
            node = scene.AddNewNodeByClass(className, f"{PEEL_NODE_PREFIX} {name}")
            node.SetAttribute(PEEL_ATTRIBUTE, name)
            node.SetHideFromEditors(True)
            nodes[name] = node
        return node

    @staticmethod
    def _peelGeometry(node, shape, rowMm, colMm):
        """A peel image's 1-slice volume in an axial view: its columns from the screen's left to right (RAS -x in
        Slicer's axial view), its rows from the top down (-y), centred at 0."""
        rows, cols = shape
        matrix = np.eye(4)
        matrix[0, 0] = -colMm
        matrix[1, 1] = -rowMm
        matrix[:3, 3] = [0.5 * (cols - 1) * colMm, 0.5 * (rows - 1) * rowMm, 0.0]
        node.SetIJKToRASMatrix(_vtkMatrix(matrix))

    def _peelImage(self, name, array, rowMm, colMm, className="vtkMRMLScalarVolumeNode"):
        node = self._peelNode(name, className)
        self._peelGeometry(node, array.shape[:2], rowMm, colMm)
        slicer.util.updateVolumeFromArray(node, np.ascontiguousarray(array[None]))
        if node.GetDisplayNode() is None:
            node.CreateDefaultDisplayNodes()
        return node

    @staticmethod
    def _copyDisplay(target, source):
        """The color table and window (and threshold) of source's display on target's."""
        display, original = target.GetDisplayNode(), source.GetDisplayNode() if source is not None else None
        if display is None or original is None:
            return
        display.SetAndObserveColorNodeID(original.GetColorNodeID())
        display.SetAutoWindowLevel(False)
        display.SetWindowLevelMinMax(original.GetWindowLevelMin(), original.GetWindowLevelMax())
        display.SetApplyThreshold(original.GetApplyThreshold())
        display.SetLowerThreshold(original.GetLowerThreshold())
        display.SetUpperThreshold(original.GetUpperThreshold())
        display.SetInterpolate(True)        # the maps drawn smooth (linear), not pixel by pixel

    def _peelLabelColors(self):
        """The color table of the peel's region labels (label i + 1: the region's segment color)."""
        result = self.reviewResult or {}
        names = ((result.get("labels") or (None, []))[1])
        colors, _ = self._segmentColors()
        scene = slicer.mrmlScene
        node = getattr(self, "_peelColorNode", None)
        if node is None or not scene.IsNodePresent(node):
            node = scene.CreateNodeByClass("vtkMRMLColorTableNode")
            node.UnRegister(None)
            node.SetName(f"{PEEL_NODE_PREFIX} region colors")
            node.SetTypeToUser()
            node.SetAttribute(PEEL_ATTRIBUTE, "colors")
            scene.AddNode(node)
            self._peelColorNode = node
        wasModifying = node.StartModify()
        node.SetNumberOfColors(len(names) + 1)
        node.SetColor(0, "background", 0.0, 0.0, 0.0, 0.0)
        for index, name in enumerate(names):
            r, g, b = colors.get(name, (1.0, 1.0, 1.0))
            node.SetColor(index + 1, name, float(r), float(g), float(b), 1.0)
        node.EndModify(wasModifying)
        return node

    @staticmethod
    def _peelVoxels(ras, hit, grid):
        """The voxels (k, j, i) of the review grid at the maps' points (the regions, the hover)."""
        inverse = np.linalg.inv(np.asarray(grid, dtype=np.float64))
        ijk = ras @ inverse[:3, :3].T + inverse[:3, 3]
        kji = np.floor(ijk[..., ::-1] + 0.5).astype(np.int64)
        kji[~hit] = 0
        return kji

    @staticmethod
    def _peelScores(values, grid, ras, hit):
        """The scores at the maps' points, trilinear (out of the scored voxels: nr.OUTSIDE, never blended in)."""
        values = np.asarray(values)
        outside = values <= nr.OUTSIDE / 2.0
        smooth = peel.sample(np.where(outside, 0.0, values).astype(np.float32), grid, ras, 0.0, order=1)
        nearest = peel.sample(values, grid, ras, nr.OUTSIDE)
        smooth = smooth.astype(values.dtype if np.issubdtype(values.dtype, np.floating) else np.float32)
        smooth[(nearest <= nr.OUTSIDE / 2.0) | ~hit] = nr.OUTSIDE
        return smooth

    def _applyPeelWindows(self):
        """The left panel's windows on the maps and the top row at once (no update of the maps needed)."""
        nodes = getattr(self, "peelNodes", None) or {}
        if not nodes:
            return
        pet = self.volume()
        anatomy = getattr(self, "_peelAnatomyNode", None)
        for key, _ in nr.PEEL_VIEWS:
            for name, source, petColors in ((f"{key} PET", pet, True), (f"{key} anatomy", anatomy, False)):
                node = nodes.get(name)
                if node is not None and source is not None and slicer.mrmlScene.IsNodePresent(node):
                    self._copyDisplay(node, source)
                    if petColors:
                        self._peelPetColors(node)
        node = nodes.get("PET (slices)")
        if node is not None and pet is not None and slicer.mrmlScene.IsNodePresent(node):
            self._copyDisplay(node, pet)
            self._peelPetColors(node)

    def _peelPetColors(self, node):
        """The colors of the unfolded maps' PET (its own choice, PET-DICOM by default; the window of the PET's)."""
        display = node.GetDisplayNode() if node is not None else None
        if display is None:
            return
        try:
            display.SetAndObserveColorNodeID(colormaps.colorNodeForLabel(self.peelPetColorCombo.currentText).GetID())
        except Exception:
            logging.exception("Static Neuro: no colors for the unfolded PET")

    def _applyPeelPetColors(self):
        for key, _ in nr.PEEL_VIEWS:
            self._peelPetColors((getattr(self, "peelNodes", None) or {}).get(f"{key} PET"))
        self._peelPetColors((getattr(self, "peelNodes", None) or {}).get("PET (slices)"))
        widgets = self._peelViewWidgets()
        if widgets:
            self._peelLegends(widgets)

    def _peelHoverColors(self):
        """The color table of the maps' hovered region: every region transparent but the one at the mouse."""
        result = self.reviewResult or {}
        names = ((result.get("labels") or (None, []))[1])
        full = self._peelLabelColors()
        scene = slicer.mrmlScene
        node = getattr(self, "_peelHoverColorNode", None)
        if node is None or not scene.IsNodePresent(node):
            node = scene.CreateNodeByClass("vtkMRMLColorTableNode")
            node.UnRegister(None)
            node.SetName(f"{PEEL_NODE_PREFIX} hovered region colors")
            node.SetTypeToUser()
            node.SetAttribute(PEEL_ATTRIBUTE, "hover colors")
            scene.AddNode(node)
            self._peelHoverColorNode = node
        wasModifying = node.StartModify()
        node.SetNumberOfColors(len(names) + 1)
        rgba = [0.0, 0.0, 0.0, 0.0]
        for index in range(len(names) + 1):
            full.GetColor(index, rgba)
            node.SetColor(index, full.GetColorName(index), rgba[0], rgba[1], rgba[2], 0.0)
        node.EndModify(wasModifying)
        self._peelHighlighted = 0
        return node

    def _peelHighlight(self, value):
        """The outline of the region value (0: none) on the PET, score and low score maps, as on the 3D-SSP."""
        node = getattr(self, "_peelHoverColorNode", None)
        previous = getattr(self, "_peelHighlighted", 0)
        if node is None or not slicer.mrmlScene.IsNodePresent(node) or value == previous:
            return
        rgba = [0.0, 0.0, 0.0, 0.0]
        wasModifying = node.StartModify()
        for index, alpha in ((previous, 0.0), (value, 1.0)):
            if 0 < index < node.GetNumberOfColors():
                node.GetColor(index, rgba)
                node.SetColor(index, node.GetColorName(index), rgba[0], rgba[1], rgba[2], alpha)
        node.EndModify(wasModifying)
        self._peelHighlighted = value

    def _peelSliceWidgets(self):
        """{orientation: slice widget} of the top row's slice views of the unfolded brain layout."""
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None:
            return {}
        out = {}
        for tag, orientation in nr.peelSliceTags().items():
            widget = layoutManager.sliceWidget(tag)
            if widget is not None:
                out[orientation] = widget
        return out

    def _peelShell(self, depth, grid, depthMm):
        """The label map (on the review grid) of the shell the maps are taken from: its outer line the brain's hull,
        its inner one the depth of the maps."""
        node = self._peelNode("shell", "vtkMRMLLabelMapVolumeNode")
        node.SetIJKToRASMatrix(_vtkMatrix(grid))
        shell = ((depth > 0) & (depth < max(float(depthMm), 1.0))).astype(np.int16)
        slicer.util.updateVolumeFromArray(node, shell)
        if node.GetDisplayNode() is None:
            node.CreateDefaultDisplayNodes()
        scene = slicer.mrmlScene
        colors = getattr(self, "_peelShellColorNode", None)
        if colors is None or not scene.IsNodePresent(colors):
            colors = scene.CreateNodeByClass("vtkMRMLColorTableNode")
            colors.UnRegister(None)
            colors.SetName(f"{PEEL_NODE_PREFIX} shell colors")
            colors.SetTypeToUser()
            colors.SetAttribute(PEEL_ATTRIBUTE, "shell colors")
            colors.SetNumberOfColors(2)
            colors.SetColor(0, "outside", 0.0, 0.0, 0.0, 0.0)
            colors.SetColor(1, "shell", 1.0, 0.9, 0.0, 1.0)
            scene.AddNode(colors)
            self._peelShellColorNode = colors
        node.GetDisplayNode().SetAndObserveColorNodeID(colors.GetID())
        return node

    def _peelPetVolume(self, pet):
        """The PET of the top row's fusion in the maps' PET colors: a volume sharing the PET's voxels (no copy) with a
        display of its own (the PET's own display keeps the review's colors)."""
        if pet is None or pet.GetImageData() is None:
            return None
        node = self._peelNode("PET (slices)")
        if node.GetImageData() is not pet.GetImageData():
            node.SetAndObserveImageData(pet.GetImageData())
        matrix = vtk.vtkMatrix4x4()
        pet.GetIJKToRASMatrix(matrix)
        node.SetIJKToRASMatrix(matrix)
        node.SetAndObserveTransformNodeID(pet.GetTransformNodeID())
        if node.GetDisplayNode() is None:
            node.CreateDefaultDisplayNodes()
        self._copyDisplay(node, pet)
        self._peelPetColors(node)
        return node

    def _updatePeelSlices(self, anatomy, shell, fit):
        """The top row: the PET over the anatomy (fusion opacity) with the shell's lines (outer: the hull, inner: the
        maps' depth); the anatomy alone, else the PET alone."""
        petNode = self._peelPetVolume(self.volume())
        background, foreground = (anatomy, petNode) if anatomy is not None else (petNode, None)
        for orientation, widget in self._peelSliceWidgets().items():
            composite = widget.mrmlSliceCompositeNode()
            current = composite.GetBackgroundVolumeID()
            composite.SetLinkedControl(False)
            composite.SetBackgroundVolumeID(background.GetID() if background is not None else None)
            composite.SetForegroundVolumeID(foreground.GetID() if foreground is not None else None)
            composite.SetForegroundOpacity(float(self.fusionOpacity.value) if foreground is not None else 0.0)
            self._alphaBlend(composite)
            composite.SetLabelVolumeID(shell.GetID() if shell is not None else None)
            composite.SetLabelOpacity(1.0)
            widget.mrmlSliceNode().SetUseLabelOutline(True)
            widget.sliceController().setVisible(False)
            if fit or (fit is None and current != (background.GetID() if background is not None else None)):
                widget.sliceLogic().FitSliceToAll()

    @staticmethod
    def _alphaBlend(composite):
        """The foreground blended over the background: (1 - opacity) x background + opacity x foreground colors."""
        try:
            composite.SetCompositing(getattr(composite, "Alpha", 0))
        except Exception:
            pass

    @staticmethod
    def _equalColumns(widget, count=4):
        """The splitter of widget's row (count views) to equal widths, whatever the views' own minimum sizes (their
        toolbars): the views' widths ignored by the splitter, equal stretch, equal sizes."""
        parent = widget.parent() if widget is not None else None
        while parent is not None and not (parent.inherits("QSplitter") and parent.count() == count
                                          and int(parent.orientation()) == int(qt.Qt.Horizontal)):
            parent = parent.parent()
        if parent is None:
            return False
        for index in range(count):
            child = parent.widget(index)
            if child is not None:
                policy = child.sizePolicy
                policy.setHorizontalPolicy(qt.QSizePolicy.Ignored)
                child.setSizePolicy(policy)
                child.setMinimumWidth(0)
            parent.setStretchFactor(index, 1)
        total = sum(parent.sizes())
        if total > 0:
            parent.setSizes([total // count] * count)
        return True

    def _peelTopThreeD(self):
        """The 3D-SSP view of the unfolded brain layout's top row."""
        layoutManager = slicer.app.layoutManager()
        for index in range(layoutManager.threeDViewCount if layoutManager is not None else 0):
            widget = layoutManager.threeDWidget(index)
            viewNode = widget.mrmlViewNode() if widget is not None else None
            if viewNode is not None and viewNode.GetSingletonTag() == nr.sspViewName("Top"):
                return widget
        return None

    def _equalPeelColumns(self):
        """Both rows' columns equal, then every view fitted to its image (the whole map seen, nothing cut)."""
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None or layoutManager.layout not in nr.PEEL_LAYOUTS:
            return
        # the superior 3D-SSP on the top row in the unfolded brain layout only (in the main window in the dual one)
        superior = layoutManager.layout == nr.LAYOUT_PEEL
        try:
            for widget in self._peelSliceWidgets().values():
                widget.sliceController().setVisible(False)
            for widget in self._peelViewWidgets().values():
                widget.sliceController().setVisible(False)
            threeD = self._peelTopThreeD() if superior else None
            if threeD is not None:
                threeD.threeDController().setVisible(False)
            slices = list(self._peelSliceWidgets().values())
            if slices:
                self._equalColumns(slices[0], len(nr.PEEL_SLICES) + (1 if superior else 0))
            maps = list(self._peelViewWidgets().values())
            if self._peelGrid():
                for widget in maps:         # 2 x 2: each row of 2
                    self._equalColumns(widget, 2)
            elif maps:
                self._equalColumns(maps[0])
        except Exception:
            logging.debug("Static Neuro: unfolded brain columns not equalled", exc_info=True)
        qt.QTimer.singleShot(50, self._fitPeelViews)        # once the views have their new sizes

    def _fitPeelViews(self):
        try:
            for widget in list(self._peelSliceWidgets().values()) + list(self._peelViewWidgets().values()):
                widget.sliceLogic().FitSliceToAll()
        except Exception:
            logging.debug("Static Neuro: unfolded brain views not fitted", exc_info=True)

    def _setupPeelViews(self, fit=False):
        widgets = self._peelViewWidgets()
        self._watchPeelWheel(bool(widgets))
        if not widgets:
            return
        self._equalPeelColumns()
        # again once the layout is loaded and drawn (the 3D-SSP view set up, its toolbar hidden)
        for delay in (0, SSP_SETUP_DELAY_MS + 300, 2 * SSP_SETUP_DELAY_MS + 500):
            qt.QTimer.singleShot(delay, self._equalPeelColumns)
        self._updatePeel(fit=True if fit else None)

    def _updatePeel(self, fit=None):
        """The peel images at the depth of the spin box in every peel view (fit: the views fitted to them; None: only
        when their size changed)."""
        widgets = self._peelViewWidgets()
        if not widgets:
            return
        found = self._peelDepthMap()
        if found is None:
            return
        depth, grid = found
        result = self.reviewResult or {}
        depthMm = float(self.peelDepthSpin.value)
        labels = result["labels"][0]
        anatomy = self._peelAnatomy()
        pet = self.volume()
        colors = self._cutFaceColors()
        score = self.scoreNode if self.scoreNode is not None and slicer.mrmlScene.IsNodePresent(self.scoreNode) \
            else None
        scoreValues = slicer.util.arrayFromVolume(score) if score is not None and score.GetImageData() else None
        colorTable = self._peelLabelColors()
        hoverTable = self._peelHoverColors()
        outlineAll = bool(self.outlineCheck.checked)
        self._peelMaps = {}
        center = self.peelCenterCombo.currentData or peel.UNFOLD_CENTERS[0][0]
        centerTitle = self.peelCenterCombo.currentText
        try:
            rays = self._peelRaysOf(depth, grid, center)
            for key, title in nr.PEEL_VIEWS:
                ras, hit, (rowMm, colMm) = peel.unfoldPoints(rays, depthMm)
                kji = self._peelVoxels(ras, hit, grid)
                self._peelMaps[key] = (kji, hit, rowMm, colMm)
                nodes = {}
                regionMap = peel.take(labels, kji, hit, 0).astype(np.int16)
                label = self._peelImage(f"{key} regions", regionMap, rowMm, colMm, "vtkMRMLLabelMapVolumeNode")
                label.GetDisplayNode().SetAndObserveColorNodeID(colorTable.GetID())
                hovered = self._peelImage(f"{key} hovered region", regionMap, rowMm, colMm,
                                          "vtkMRMLLabelMapVolumeNode")
                hovered.GetDisplayNode().SetAndObserveColorNodeID(hoverTable.GetID())
                if anatomy is not None:
                    values = peel.sample(slicer.util.arrayFromVolume(anatomy), _ijkToRas(anatomy), ras, order=1)
                    values[~hit] = 0
                    node = self._peelImage(f"{key} anatomy", values, rowMm, colMm)
                    self._copyDisplay(node, anatomy)
                    nodes[nr.ANATOMY_ROW] = (node, None)
                if pet is not None:
                    values = peel.sample(slicer.util.arrayFromVolume(pet), _ijkToRas(pet), ras, order=1)
                    values[~hit] = 0
                    node = self._peelImage(f"{key} PET", values, rowMm, colMm)
                    self._copyDisplay(node, pet)
                    self._peelPetColors(node)
                    nodes[nr.PET_ROW] = (node, None)
                    if nr.ANATOMY_ROW in nodes:
                        nodes[nr.FUSION_ROW] = (nodes[nr.ANATOMY_ROW][0], node)
                    else:
                        nodes[nr.FUSION_ROW] = (node, None)
                if scoreValues is not None and scoreValues.shape == labels.shape:
                    scoreMap = self._peelScores(scoreValues, grid, ras, hit)
                    scalar = self._peelImage(f"{key} scores", scoreMap, rowMm, colMm)
                    self._copyDisplay(scalar, score)
                    if colors is not None and colors[0].shape[:3] == labels.shape:
                        if colors[1] is None:
                            rgb = peel.sample(colors[0], grid, ras, 0, order=1)
                            rgb[~hit] = 0
                        else:
                            rgb = np.zeros(hit.shape + (3,), np.uint8)
                            values = scoreMap if colors[0] is scoreValues else \
                                self._peelScores(colors[0], grid, ras, hit)
                            rgb[hit] = colors[1](values[hit])
                        node = self._peelImage(f"{key} score colors", rgb.astype(np.uint8), rowMm, colMm,
                                               "vtkMRMLVectorVolumeNode")
                        node.GetDisplayNode().SetAutoWindowLevel(False)
                        node.GetDisplayNode().SetWindowLevelMinMax(0.0, 255.0)
                        nodes[nr.SCORE_ROW] = (node, scalar)       # the scores on top, invisible (legend, probe)
                    else:
                        nodes[nr.SCORE_ROW] = (scalar, None)
                    low = self._peelImage(f"{key} low scores", scoreMap, rowMm, colMm)
                    self._peelLowDisplay(low)
                    nodes[nr.LOW_ROW] = (nodes[nr.ANATOMY_ROW][0] if nr.ANATOMY_ROW in nodes else None, low)
                for row, _ in nr.PEEL_ROWS:
                    widget = widgets.get((row, key))
                    if widget is None:
                        continue
                    background, foreground = nodes.get(row, (None, None))
                    opacity = float(self.fusionOpacity.value) if row == nr.FUSION_ROW else \
                        1.0 if row == nr.LOW_ROW else 0.0
                    # the fusion: every region's outline (the segments); the others the region at the mouse (every
                    # region with the outlines on)
                    self._showPeel(widget, background, foreground,
                                   label if row == nr.FUSION_ROW or outlineAll else hovered, True, fit, opacity)
                    self._peelText(widget, f"{centerTitle}  {depthMm:g} mm")
        except ValueError as error:
            logging.warning(f"Static Neuro: no peel views: {error}")
            return
        self._peelAnatomyNode = anatomy
        self._updatePeelSlices(anatomy, self._peelShell(depth, grid, depthMm), fit)
        self._peelLegends(widgets)

    def _peelLowDisplay(self, node):
        """The low scores' map: the scores below the threshold in the reversed score colors (as the low score 3D-SSP
        views), the rest transparent (the anatomy under it)."""
        display = node.GetDisplayNode() if node is not None else None
        if display is None:
            return
        rgba, (low, high) = self._scoreColors(side=nr.SIDE_THRESHOLD)
        display.SetAndObserveColorNodeID(self._sspColorNode(rgba, f"{PEEL_NODE_PREFIX} low score colors").GetID())
        display.SetAutoWindowLevel(False)
        display.SetWindowLevelMinMax(low, high)
        display.SetApplyThreshold(True)
        display.SetLowerThreshold(nr.OUTSIDE / 2.0)
        display.SetUpperThreshold(min(float(self.thresholdSpin.value), 0.0))
        display.SetInterpolate(True)

    def _showPeel(self, widget, background, foreground, label, outlines, fit, opacity=0.0):
        composite = widget.mrmlSliceCompositeNode()
        sliceNode = widget.mrmlSliceNode()
        current = composite.GetBackgroundVolumeID()
        composite.SetLinkedControl(False)
        composite.SetBackgroundVolumeID(background.GetID() if background is not None else None)
        composite.SetForegroundVolumeID(foreground.GetID() if foreground is not None else None)
        composite.SetForegroundOpacity(opacity)
        self._alphaBlend(composite)
        composite.SetLabelVolumeID(label.GetID() if label is not None and outlines else None)
        composite.SetLabelOpacity(1.0)
        sliceNode.SetUseLabelOutline(True)
        widget.sliceController().setVisible(False)
        if fit or (fit is None and current != (background.GetID() if background is not None else None)):
            widget.sliceLogic().FitSliceToAll()

    def _peelText(self, widget, text):
        """The centre and the depth in the top left corner of an unfolded map."""
        texts = getattr(self, "_peelTexts", None)
        if texts is None:
            texts = self._peelTexts = {}
        try:
            view = widget.sliceView()
            renderer = view.renderWindow().GetRenderers().GetFirstRenderer()
        except Exception:
            return
        tag = widget.mrmlSliceNode().GetID()
        found = texts.get(tag)
        if found is None or found[0] is not renderer:
            actor = vtk.vtkTextActor()
            prop = actor.GetTextProperty()
            prop.SetFontSize(14)
            prop.SetColor(1.0, 1.0, 0.4)
            prop.SetVerticalJustificationToTop()
            actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            actor.SetPosition(0.02, 0.97)
            renderer.AddViewProp(actor)
            found = texts[tag] = (renderer, actor)
        found[1].SetInput(text)
        view.scheduleRender()

    def _peelLegends(self, widgets):
        """The color legends of the peel's PET, score and low score rows (on their first view)."""
        if not hasattr(self, "legendCheck"):
            return
        first = {}
        for key, _ in nr.PEEL_VIEWS:
            for row in (nr.PET_ROW, nr.SCORE_ROW, nr.LOW_ROW):
                if (row, key) in widgets and row not in first:
                    first[row] = (key, widgets[(row, key)].mrmlSliceNode().GetID())
        pet = (getattr(self, "peelNodes", None) or {}).get(f"{first[nr.PET_ROW][0]} PET") if nr.PET_ROW in first \
            else None
        if pet is not None and self.volume() is not None:
            self._legend(pet, [first[nr.PET_ROW][1]], self._petUnits(self.volume()), "%.2f")
        scores = (getattr(self, "peelNodes", None) or {}).get(f"{first[nr.SCORE_ROW][0]} scores") \
            if nr.SCORE_ROW in first else None
        if scores is not None:
            self._legend(scores, [first[nr.SCORE_ROW][1]], self._scoreKind(), "%.1f")
        low = (getattr(self, "peelNodes", None) or {}).get(f"{first[nr.LOW_ROW][0]} low scores") \
            if nr.LOW_ROW in first else None
        if low is not None:
            self._legend(low, [first[nr.LOW_ROW][1]], f"{self._scoreKind()} < {self.thresholdSpin.value:g}", "%.1f")

    def _peelHover(self, tag, ras):
        """The region (SUVmean, score) at the mouse on a peel view: the voxel its pixel was taken from."""
        row, key = nr.peelViewTags()[tag]
        found = (getattr(self, "_peelMaps", None) or {}).get(key)
        result = self.reviewResult or {}
        layoutManager = slicer.app.layoutManager()
        widget = layoutManager.sliceWidget(tag) if layoutManager else None
        if found is None or widget is None or result.get("labels") is None:
            self._hideHoverName()
            self._peelHighlight(0)
            return
        kji, hit, rowMm, colMm = found
        rows, cols = hit.shape
        column = int(np.floor((0.5 * (cols - 1) * colMm - ras[0]) / colMm + 0.5))
        line = int(np.floor((0.5 * (rows - 1) * rowMm - ras[1]) / rowMm + 0.5))
        if not (0 <= line < rows and 0 <= column < cols) or not hit[line, column]:
            self._hideHoverName()
            self._peelHighlight(0)
            return
        labels, names = result["labels"]
        voxel = kji[line, column]
        if int(qt.QApplication.keyboardModifiers()) & int(qt.Qt.ShiftModifier):
            self._peelJump(peel.rasOf(voxel, _ijkToRas(result["grid"])))     # Shift: the slices to that point
        value = int(labels[tuple(voxel)])
        if value <= 0 or value > len(names):
            self._hideHoverName()
            self._peelHighlight(0)
            return
        self._peelHighlight(value)
        self._showHoverText(widget.sliceView(), [names[value - 1]], stacked=True)

    def _peelJump(self, ras):
        """The top row's slices through the point of the map under the mouse (Shift + mouse move); with no top row
        (the low scores' map of a dual layout) the review's slices."""
        widgets = list(self._peelSliceWidgets().values()) or [widget for _, widget in self._reviewSliceWidgets()]
        for widget in widgets:
            widget.mrmlSliceNode().JumpSliceByCentering(float(ras[0]), float(ras[1]), float(ras[2]))

    def _watchPeelWheel(self, enable):
        """The mouse wheel on a peel view: deeper / shallower (not Slicer's slice scroll, which leaves the 1-slice
        image)."""
        for interactor, tags in getattr(self, "_peelWheelObservers", None) or []:
            for tag in tags:
                try:
                    interactor.RemoveObserver(tag)
                except Exception:
                    pass
        self._peelWheelObservers = []
        if not enable:
            return
        for widget in self._peelViewWidgets().values():
            interactor = widget.sliceView().interactor()
            tags = []
            for event, step in (("MouseWheelForwardEvent", -1.0), ("MouseWheelBackwardEvent", 1.0)):
                tags.append(interactor.AddObserver(
                    event, lambda caller, ev, s=step: self._onPeelWheel(caller, ev, s), MOUSE_PRIORITY))
            self._peelWheelObservers.append((interactor, tags))

    def _onPeelWheel(self, interactor, event, step):
        if interactor.GetControlKey() or interactor.GetShiftKey():
            return          # Ctrl + wheel: Slicer's zoom
        self.peelDepthSpin.value = max(0.0, min(peel.MAX_DEPTH_MM, float(self.peelDepthSpin.value) + step))
        for observed, tags in getattr(self, "_peelWheelObservers", None) or []:
            if observed is interactor:
                for tag in tags:
                    command = interactor.GetCommand(tag)
                    if command is not None:
                        command.SetAbortFlag(1)
