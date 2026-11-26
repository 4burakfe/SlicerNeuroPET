"""
The Modeling step's own toolbar row (Dynamic Neuro PET): the report and screenshots, then view settings of the
embedded PET4D Modeling layout (the curves, segments, orientation, colors, the results table). Nothing here fits or
computes: every button sets a check or a combo of the Modeling panel (or calls its layout), so the panel and the
toolbar stay the same.

Part of the Dynamic Neuro PET module (DynamicNeuroPET.py): the methods of ModelingToolbarMixin, mixed into
DynamicNeuroPETWidget there.
"""

import logging
import os

import ctk
import qt
import slicer

from NeuroWorkflowLib.base import STEP_MODEL
from DICOMGatedDynamicPETLib import colormaps
from StaticNeuroPETLib.constants import ICON_PATH


class ModelingToolbarMixin:
    """The Modeling step's toolbar row of DynamicNeuroPETWidget."""

    def _modelingWidget(self):
        return self.stepWidget(STEP_MODEL) if hasattr(self, "stepWidget") else None

    def _modelingSession(self):
        modeling = self._modelingWidget()
        return getattr(modeling, "session", None) if modeling is not None else None

    # --- the row --------------------------------------------------------------------------------------------------
    def _buildModelingRow(self):
        """The row under the workflow row, shown on the Modeling step only. A row left by a developer reload is
        taken away first."""
        window = slicer.util.mainWindow()
        name = self.TOOLBAR_NAME + "Modeling"
        for old in [w for w in window.findChildren(qt.QToolBar) if w.objectName == name]:
            try:
                window.removeToolBarBreak(old)
                window.removeToolBar(old)
                old.hide()
                old.objectName = name + "Old"
            except Exception:
                pass
        row = qt.QToolBar("Dynamic Neuro modeling", window)
        row.objectName = name
        row.setIconSize(qt.QSize(24, 24))
        row.setToolButtonStyle(qt.Qt.ToolButtonTextBesideIcon)
        row.hide()
        window.addToolBarBreak()
        window.addToolBar(row)
        self.modelingToolbar = row
        folder = os.path.join(os.path.dirname(ICON_PATH), "Review")

        def action(icon, label, tip):
            item = qt.QAction(label, row)
            item.toolTip = tip
            if icon:
                item.setIcon(qt.QIcon(os.path.join(folder, icon + ".svg")))
            row.addAction(item)
            row.widgetForAction(item).setFocusPolicy(qt.Qt.NoFocus)
            return item

        def caption(text):
            title = qt.QLabel(text)
            title.setStyleSheet("QLabel { font-weight: bold; padding: 0 2px 0 6px; }")
            row.addWidget(title)

        def menuButton(icon, label, tip, fill):
            item = action(icon, label, tip)
            menu = qt.QMenu(label, row)
            menu.connect("aboutToShow()", lambda m=menu, f=fill: f(m))
            item.setMenu(menu)
            row.widgetForAction(item).setPopupMode(qt.QToolButton.InstantPopup)
            return item

        caption("Report:")
        pdf = action("ReportPDF", "PDF", "Save the kinetic modeling report as a PDF (fit all segments first)")
        pdf.connect("triggered()", lambda: self.onSaveModelingReport(False))
        dicom = action("ReportDICOM", "DICOM", "Save the kinetic modeling report's pages as images straight into the "
                                               "DICOM database (the PET's study)")
        dicom.connect("triggered()", lambda: self.onSaveModelingReport(True))
        row.addSeparator()
        caption("Screenshot:")
        copy = action("ScreenshotClipboard", "Clipboard", "Copy a screenshot of the views and the curves to the "
                                                          "clipboard")
        copy.connect("triggered()", self.onScreenshotClipboard)
        shot = action("ScreenshotDICOM", "DICOM", "Save a screenshot of the views and the curves into the DICOM "
                                                  "database (the PET's study)")
        shot.connect("triggered()", self.onScreenshotDicom)
        row.addSeparator()
        menuButton(None, "Plot", "Time-activity curves or the Patlak / Logan plot of the target", self._fillPlotMenu)
        reset = action(None, "Reset zoom", "The curves' whole time and value range again")
        reset.connect("triggered()", self._resetPlotZoom)
        curves = action(None, "All curves", "The other regions' curves (faint) behind the target and the reference")
        curves.setCheckable(True)
        curves.connect("triggered(bool)", lambda on: self._setModelingCheck("allCurvesCheck", on))
        self._allCurvesAction = curves
        segments = action("Segments", "Segments", "The region outlines on the summed image (on / off)")
        segments.setCheckable(True)
        segments.connect("triggered(bool)", lambda on: self._setModelingCheck("segmentsCheck", on))
        self._modelSegmentsAction = segments
        row.addSeparator()
        menuButton("Slices", "Orientation", "The orientation of the frames and the summed image",
                   self._fillOrientationMenu)
        menuButton("Colors", "Colors", "The color maps of the frames and of the summed image, the frames' windows",
                   self._fillModelingColorMenu)
        table = action("Table", "Table", "Show or hide the results table right of the curves")
        table.setCheckable(True)
        table.setChecked(True)
        table.connect("triggered(bool)", self._showResultsTable)
        self._resultsTableAction = table
        self._updateModelingTools()
        return row

    def _removeModelingRow(self):
        row = getattr(self, "modelingToolbar", None)
        if row is not None:
            window = slicer.util.mainWindow()
            window.removeToolBarBreak(row)
            window.removeToolBar(row)
            row.deleteLater()
            self.modelingToolbar = None

    def _updateModelingTools(self, shown=None):
        """The row shown with the workflow row on the Modeling step; its check buttons as the panel's checks."""
        row = getattr(self, "modelingToolbar", None)
        if row is None:
            return
        if shown is None:
            shown = getattr(self, "current", None) == STEP_MODEL
        main = getattr(self, "toolbar", None)
        try:
            row.setVisible(bool(shown and main is not None and main.visible))
        except Exception:
            logging.debug("Neuro: the Modeling toolbar is gone", exc_info=True)
            return
        if shown:
            self._connectModelingChecks()
            self._syncModelingActions()

    def _connectModelingChecks(self):
        """The toolbar's check buttons follow the Modeling panel's checks (once its widget exists)."""
        modeling = self._modelingWidget()
        if modeling is None or getattr(self, "_modelingChecksConnected", None) is modeling:
            return
        for name in ("allCurvesCheck", "segmentsCheck"):
            check = getattr(modeling, name, None)
            if check is not None:
                check.connect("toggled(bool)", lambda on: self._syncModelingActions())
        self._modelingChecksConnected = modeling

    def _syncModelingActions(self):
        modeling = self._modelingWidget()
        for action, name in ((getattr(self, "_allCurvesAction", None), "allCurvesCheck"),
                             (getattr(self, "_modelSegmentsAction", None), "segmentsCheck")):
            check = getattr(modeling, name, None) if modeling is not None else None
            if action is None:
                continue
            action.enabled = check is not None
            on = bool(check is not None and check.checked)
            if action.checked != on:
                action.setChecked(on)
        action = getattr(self, "_resultsTableAction", None)
        if action is not None:
            table = getattr(self._modelingSession(), "table", None)
            action.enabled = table is not None
            on = bool(table is not None and table.visible) if table is not None else action.checked
            if action.checked != on:
                action.setChecked(on)

    def _setModelingCheck(self, name, on):
        modeling = self._modelingWidget()
        check = getattr(modeling, name, None) if modeling is not None else None
        if check is not None:
            check.checked = bool(on)
        self._syncModelingActions()

    # --- menus and actions ----------------------------------------------------------------------------------------
    def _fillPlotMenu(self, menu):
        menu.clear()
        session = self._modelingSession()
        tac, graphical = getattr(session, "tacRadio", None), getattr(session, "graphicalRadio", None)
        for title, radio in (("Time-activity curves", tac), ("Patlak / Logan plot (target)", graphical)):
            item = self._menuChoice(menu, title, radio is not None and radio.checked,
                                    lambda r=radio: setattr(r, "checked", True) if r is not None else None)
            item.enabled = radio is not None and radio.enabled
        if graphical is not None and not graphical.enabled:
            menu.addSection("Patlak / Logan: a graphical method's fit first")

    def _resetPlotZoom(self):
        plot = getattr(self._modelingSession(), "plot", None)
        if plot is not None:
            plot.resetZoom()

    def _fillOrientationMenu(self, menu):
        menu.clear()
        modeling = self._modelingWidget()
        combo = getattr(modeling, "orientationCombo", None) if modeling is not None else None
        if combo is None:
            menu.addAction("Open the Modeling step first").enabled = False
            return
        for index in range(combo.count):
            title = combo.itemText(index)
            self._menuChoice(menu, title, combo.currentIndex == index,
                             lambda i=index: combo.setCurrentIndex(i))

    def _fillModelingColorMenu(self, menu):
        menu.clear()
        modeling = self._modelingWidget()
        if modeling is None:
            menu.addAction("Open the Modeling step first").enabled = False
            return
        for title, name in (("Frames", "frameColorCombo"), ("Summed image", "summedColorCombo")):
            combo = getattr(modeling, name, None)
            if combo is None:
                continue
            sub = menu.addMenu(title)
            for label in colormaps.LABELS:
                self._menuChoice(sub, label, combo.currentText == label,
                                 lambda c=combo, t=label: c.setCurrentIndex(max(0, c.findText(t))))
        check = getattr(modeling, "frameWindowCheck", None)
        if check is not None:
            menu.addSeparator()
            self._menuChoice(menu, check.text, check.checked, lambda: setattr(check, "checked", not check.checked))

    def _showResultsTable(self, on):
        table = getattr(self._modelingSession(), "table", None)
        if table is not None:
            table.setVisible(bool(on))
        self._syncModelingActions()

    # --- screenshots: the curves too on the Modeling step -------------------------------------------------------
    def _screenImages(self):
        from StaticNeuroPETLib.report import ReportMixin
        images = ReportMixin._screenImages(self)
        if getattr(self, "current", None) != STEP_MODEL:
            return images
        dock = getattr(self._modelingSession(), "dock", None)
        try:
            if dock is not None and dock.visible and dock.widget() is not None:
                image = ctk.ctkWidgetsUtils.grabWidget(dock.widget())
                if image is not None and not image.isNull():
                    images.append(image)
        except Exception:
            logging.debug("Neuro: the curves were not captured", exc_info=True)
        return images
