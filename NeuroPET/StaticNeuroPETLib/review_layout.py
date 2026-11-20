"""
Review: the layouts, the second monitor, entering and leaving the review views, outlines.

Part of the Static Neuro PET/SPECT module (StaticNeuroPET.py): the methods of ReviewLayoutMixin, mixed into
StaticNeuroPETWidget there.
"""

import logging

import qt
import slicer
import vtk

from NeuroPETLib import neuroreview as nr, peel, ssp

from StaticNeuroPETLib.constants import (MONITOR2_PLACE_DELAY_MS, REVIEW_OUTLINE_THICKNESS, SSP_SETUP_DELAY_MS,
                                         STEP_REVIEW)
from StaticNeuroPETLib.helpers import _mainViewportAttributes, _showOnSecondScreen


class ReviewLayoutMixin:
    """The review layout methods of StaticNeuroPETWidget."""

    # layout and views
    def _reviewLayout(self):
        return self.layoutCombo.currentData if hasattr(self, "layoutCombo") else nr.LAYOUT_SINGLE

    def _onLayoutPicked(self):
        if getattr(self, "current", None) == STEP_REVIEW and self.node("Segmentation") is not None:
            self._enterReviewViews()

    def _registerReviewLayouts(self, layoutManager):
        """The review layouts in Slicer (a description updated only when that layout is not on screen)."""
        layoutNode = layoutManager.layoutLogic().GetLayoutNode()
        attributes = _mainViewportAttributes()
        medial = self._sspMedial()
        self._peelGridShown = self._peelGrid()
        for layoutId in nr.LAYOUT_IDS:
            description = nr.layoutXml(layoutId, attributes, medial, self._tableFor(layoutId), self._peelGridShown)
            if not layoutNode.IsLayoutDescription(layoutId):
                layoutNode.AddLayoutDescription(layoutId, description)
            elif layoutNode.GetLayoutDescription(layoutId) != description:
                if layoutManager.layout == layoutId:      # with or without the medial views: left and built again
                    layoutManager.setLayout(self._reviewPreviousLayout
                                            or slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
                    slicer.app.processEvents()
                layoutNode.SetLayoutDescription(layoutId, description)

    def _tableFor(self, layoutId):
        """Whether layoutId shows the region table: as the Table button left it, else the setting."""
        shown = getattr(self, "_tableShown", None) or {}
        if layoutId in shown:
            return shown[layoutId]
        return nr.tableByDefault(layoutId, self.tableDefaultCheck.checked if hasattr(self, "tableDefaultCheck")
                                 else True)

    def _peelGrid(self):
        """Whether the unfolded brain's maps are shown 2 x 2: the wide (front-back poles) maps."""
        combo = getattr(self, "peelCenterCombo", None)
        return combo is not None and combo.currentData in peel.WIDE_MAPS

    def _onPeelCenter(self):
        """Another unfolded map: the layout built again when its maps go between a row of 4 and 2 x 2."""
        changed = self._peelGrid() != getattr(self, "_peelGridShown", None)
        layoutManager = slicer.app.layoutManager()
        if (changed and layoutManager is not None and layoutManager.layout in nr.PEEL_LAYOUTS
                and getattr(self, "current", None) == STEP_REVIEW and getattr(self, "_reviewViewsShown", False)):
            self._enterReviewViews()
        else:
            self._updatePeel(fit=True)

    def _toggleTable(self):
        """The toolbar's Table button: the region table of the layout on screen shown / hidden."""
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None or layoutManager.layout not in nr.LAYOUT_IDS:
            return
        layoutId = layoutManager.layout
        shown = getattr(self, "_tableShown", None)
        if shown is None:
            shown = self._tableShown = {}
        shown[layoutId] = not self._tableFor(layoutId)
        self._enterReviewViews()

    def _onTableDefault(self):
        self._tableShown = {}
        layoutManager = slicer.app.layoutManager()
        if (getattr(self, "current", None) == STEP_REVIEW and getattr(self, "_reviewViewsShown", False)
                and layoutManager is not None and layoutManager.layout in nr.LAYOUT_IDS):
            self._enterReviewViews()

    def _sspMedial(self):
        """Whether the 3D-SSP has the medial views (the regions with sides, R / L): the layouts' 2 x 4 views, else
        2 x 3."""
        data = (self.reviewResult or {}).get("ssp")
        return ssp.hasMedial(data["surface"]) if data else True

    def _enterReviewViews(self):
        if self.volume() is None:
            return
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None:
            return
        self._registerReviewLayouts(layoutManager)
        self._reviewViewsShown = True
        wanted = self._reviewLayout()
        if layoutManager.layout != wanted:
            if layoutManager.layout not in nr.LAYOUT_IDS:
                self._reviewPreviousLayout = layoutManager.layout
            layoutManager.setLayout(wanted)
            slicer.app.processEvents()
        if self.reviewResult is None:
            self._initPetWindow()
        self._assignReviewViews(fit=True)
        self._showTable()
        self._hoverNames(True)
        self._watchLayout(True)
        if wanted in nr.DUAL_LAYOUTS:
            qt.QTimer.singleShot(MONITOR2_PLACE_DELAY_MS, self._placeMonitor2)
        else:       # once the new views are drawn: their cameras and planes again
            qt.QTimer.singleShot(SSP_SETUP_DELAY_MS, self._setupSspViewsIfShown)

    def _setupSspViewsIfShown(self):
        """The delayed set-up of the 3D-SSP views, unless the review views were left meanwhile (it would put the
        camera, mouse and slice observers back after _exitReviewViews took them off)."""
        if getattr(self, "_reviewViewsShown", False):
            self._setupSspViews(fit=True)

    def _monitor2Window(self):
        """The floating window of the 3D-SSP views (dual monitor layout); None when docked or not shown."""
        candidates = [widget.window() for widget, _ in self._sspViewNodes().values()][:1]
        candidates += [w for w in qt.QApplication.topLevelWidgets() if w.windowTitle == nr.MONITOR2_TITLE]
        for window in candidates:
            if window is None or window.inherits("QMainWindow"):
                continue
            if window.inherits("QDockWidget") and not getattr(window, "floating", True):
                continue        # docked into the main window by the user: left there
            return window
        return None

    def _placeMonitor2(self):
        layoutManager = slicer.app.layoutManager()
        if (layoutManager is None or layoutManager.layout not in nr.DUAL_LAYOUTS
                or not getattr(self, "_reviewViewsShown", False)):
            return
        try:
            window = self._monitor2Window()
            if window is not None:
                _showOnSecondScreen(window)
                qt.QTimer.singleShot(3 * MONITOR2_PLACE_DELAY_MS, self._afterMonitor2Placed)
        except Exception:
            logging.exception("Static Neuro: could not place the 3D-SSP window; drag it to the second screen")

    def _afterMonitor2Placed(self):
        """The views of the second screen with their new sizes: the 3D-SSP set up again, the unfolded maps fitted."""
        self._setupSspViewsIfShown()
        if getattr(self, "_reviewViewsShown", False) and self._peelViewWidgets():
            self._equalPeelColumns()        # (the unfolded brain layouts only) then fitted
            qt.QTimer.singleShot(50, self._fitPeelViews)

    def _watchLayout(self, enable):
        """Observe the layout node: after a view is maximized (double click) and restored, Slicer builds the layout
        again; the review's view settings (no title bars, the embedded region table and its width, the 3D-SSP views,
        the legends) are then applied again."""
        old = getattr(self, "_layoutObserver", None)
        if old is not None:
            try:
                old[0].RemoveObserver(old[1])
            except Exception:
                pass
        self._layoutObserver = None
        layoutManager = slicer.app.layoutManager()
        if not enable or layoutManager is None:
            return
        layoutNode = layoutManager.layoutLogic().GetLayoutNode()
        tag = layoutNode.AddObserver(vtk.vtkCommand.ModifiedEvent, lambda caller, event: self._onLayoutModified())
        self._layoutObserver = (layoutNode, tag)

    def _onLayoutModified(self):
        if getattr(self, "_layoutRefreshPending", False):
            return
        self._layoutRefreshPending = True
        qt.QTimer.singleShot(200, self._afterLayoutModified)

    def _afterLayoutModified(self):
        self._layoutRefreshPending = False
        layoutManager = slicer.app.layoutManager()
        if (getattr(self, "current", None) != STEP_REVIEW or layoutManager is None
                or layoutManager.layout not in nr.LAYOUT_IDS):
            return
        try:
            self._assignReviewViews()
            self._showTable()
            self._watchMouse(True)
            if layoutManager.layout in nr.PEEL_LAYOUTS:
                qt.QTimer.singleShot(100, self._equalPeelColumns)
        except Exception:
            logging.exception("Static Neuro: could not restore the review views")

    def _exitReviewViews(self):
        self._reviewViewsShown = False
        self._watchLayout(False)
        self._watchCutSlices(False)
        self._watchPeelWheel(False)
        self._watchMouse(False)
        self._dropSspOutlines()
        self._hoverNames(False)
        self._endFlash()
        self._syncCameras(False)
        self._restoreSegmentationDisplay()
        layoutManager = slicer.app.layoutManager()
        if layoutManager is not None and layoutManager.layout in nr.LAYOUT_IDS and self._reviewPreviousLayout is not None:
            layoutManager.setLayout(self._reviewPreviousLayout)
        self._reviewPreviousLayout = None

    def _reviewSliceWidgets(self):
        layoutManager = slicer.app.layoutManager()
        out = []
        for row in (nr.PET_ROW, nr.SCORE_ROW):
            for orientation, _ in nr.ORIENTATIONS:
                widget = layoutManager.sliceWidget(nr.sliceViewName(row, orientation)) if layoutManager else None
                if widget is not None:
                    out.append((row, widget))
        return out

    def _assignReviewViews(self, fit=False):
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None or layoutManager.layout not in nr.LAYOUT_IDS:
            return
        layers = self._reviewLayers()
        sliceViews = []
        for row, widget in self._reviewSliceWidgets():
            composite = widget.mrmlSliceCompositeNode()
            composite.SetLinkedControl(False)
            background, foreground, opacity = layers(row)
            composite.SetBackgroundVolumeID(background.GetID() if background is not None else None)
            composite.SetForegroundVolumeID(foreground.GetID() if foreground is not None else None)
            composite.SetForegroundOpacity(opacity)
            composite.SetLabelVolumeID(None)
            composite.SetLinkedControl(True)        # all the review's slice views linked
            widget.sliceController().setVisible(False)
            sliceViews.append(widget.mrmlSliceNode().GetID())
            if fit:
                widget.sliceLogic().FitSliceToAll()
        self._outlineSegmentation(sliceViews)
        self._setupSspViews(fit)
        self._setupPeelViews(fit)
        self._updateLegends()

    def _reviewLayers(self):
        """A function row -> (background, foreground, opacity) of the review's slice views of that row."""
        pet = self.volume()
        anatomy = self._fusionVolume()
        score = self.scoreNode if self.scoreNode is not None and slicer.mrmlScene.IsNodePresent(self.scoreNode) \
            else None
        colored = self._scoreColorsNode() if score is not None else None
        greyPet = []        # made once, when a row needs it

        def layers(row):
            top = pet if row == nr.PET_ROW else score
            rowColored = colored if row == nr.SCORE_ROW else None
            if rowColored is not None and anatomy is not None:     # the colored PET over the fusion image
                return anatomy, rowColored, float(self.scoreOpacity.value)
            if rowColored is not None:   # the colored PET; the scores on top, invisible (data probe, legend)
                return rowColored, score, 0.0
            if anatomy is not None:
                return anatomy, top, float((self.fusionOpacity if row == nr.PET_ROW else self.scoreOpacity).value)
            if row == nr.SCORE_ROW and self.scoreBackgroundCombo.currentData == "pet" and score is not None:
                if not greyPet:
                    greyPet.append(self._greyPet())
                return greyPet[0], score, 1.0     # the scores over the PET in grey
            return top, None, 0.0
        return layers

    def _applyReviewOpacities(self):
        """An opacity slider moved: only the opacity of the review's slice views changes (the volumes, the 3D-SSP
        views and the legends stay); the whole assignment when a view does not show what it should."""
        for (row, _), widget in self._peelViewWidgets().items():
            if row == nr.FUSION_ROW:
                widget.mrmlSliceCompositeNode().SetForegroundOpacity(float(self.fusionOpacity.value))
        for widget in self._peelSliceWidgets().values():
            composite = widget.mrmlSliceCompositeNode()
            if composite.GetForegroundVolumeID():
                composite.SetForegroundOpacity(float(self.fusionOpacity.value))
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None or layoutManager.layout not in nr.LAYOUT_IDS:
            return
        layers = self._reviewLayers()
        updates = []
        for row, widget in self._reviewSliceWidgets():
            composite = widget.mrmlSliceCompositeNode()
            background, foreground, opacity = layers(row)
            if (composite.GetBackgroundVolumeID() != (background.GetID() if background is not None else None) or
                    composite.GetForegroundVolumeID() != (foreground.GetID() if foreground is not None else None)):
                self._assignReviewViews()
                return
            updates.append((composite, opacity))
        for composite, opacity in updates:
            if abs(composite.GetForegroundOpacity() - opacity) > 1e-9:
                linked = composite.GetLinkedControl()
                composite.SetLinkedControl(False)
                composite.SetForegroundOpacity(opacity)
                composite.SetLinkedControl(linked)

    def _outlineSegmentation(self, viewIds):
        """The regions as outlines (no fill, REVIEW_OUTLINE_THICKNESS px) on the PET and the score slices, hidden with
        the outline check box; the earlier display restored on leaving."""
        segmentation = self.node("Segmentation")
        display = segmentation.GetDisplayNode() if segmentation is not None else None
        if display is None or not viewIds:
            return
        if self._segmentationDisplay is None or self._segmentationDisplay[0] is not display:
            self._restoreSegmentationDisplay()
            self._segmentationDisplay = (display, [display.GetNthViewNodeID(i) for i in
                                                   range(display.GetNumberOfViewNodeIDs())],
                                         display.GetVisibility2DFill(), display.GetVisibility2DOutline(),
                                         display.GetSliceIntersectionThickness()
                                         if hasattr(display, "GetSliceIntersectionThickness") else None)
        self._outlineViewIds = list(viewIds)
        wasModifying = display.StartModify()
        display.SetVisibility(True)
        display.SetVisibility2DFill(False)
        display.SetVisibility2DOutline(True)
        if hasattr(display, "SetSliceIntersectionThickness"):
            display.SetSliceIntersectionThickness(REVIEW_OUTLINE_THICKNESS)
        self._showOutlines(display, bool(self.outlineCheck.checked) or self._flash is not None)
        display.EndModify(wasModifying)

    def _showOutlines(self, display, show):
        """The region outlines in the review's slice views, or in none (hidden: out of the views, not fill and outline
        both off, which Slicer's slice controller warns about at every update)."""
        viewIds = (getattr(self, "_outlineViewIds", None) or []) if show else []
        self._setViewNodeIds(display, viewIds or ["vtkMRMLViewNodeNone"])

    def _restoreSegmentationDisplay(self):
        saved, self._segmentationDisplay = self._segmentationDisplay, None
        if saved is None:
            return
        display, viewIds, fill, outline, thickness = saved
        if not slicer.mrmlScene.IsNodePresent(display):
            return
        display.RemoveAllViewNodeIDs()
        for viewId in viewIds:
            display.AddViewNodeID(viewId)
        display.SetVisibility2DFill(fill)
        display.SetVisibility2DOutline(outline)
        if thickness is not None and hasattr(display, "SetSliceIntersectionThickness"):
            display.SetSliceIntersectionThickness(thickness)
