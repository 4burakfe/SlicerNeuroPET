"""
The workflow toolbar: the steps, the patient right of the module name, the review tools and their menus.

Part of the Static Neuro PET/SPECT module (StaticNeuroPET.py): the methods of ToolbarMixin, mixed into
StaticNeuroPETWidget there.
"""

import os

import qt
import slicer

from DICOMGatedDynamicPETLib import colormaps
from NeuroWorkflowLib.base import NeuroWorkflowWidget
from NeuroPETLib import neuroreview as nr, peel

from StaticNeuroPETLib.constants import ICON_PATH, PEEL_DEPTHS, REVIEW_SCORE_WINDOWS, STEP_ICON_SIZE, STEP_REVIEW
from StaticNeuroPETLib.helpers import _stepIcon


class ToolbarMixin:
    """The toolbar methods of StaticNeuroPETWidget."""

    # --- workflow toolbar: the patient right of the module's name, the review tools ----------------------------
    def _stepButton(self, index, key, title, explanation):
        button = NeuroWorkflowWidget._stepButton(self, index, key, title, explanation)
        button.setIconSize(qt.QSize(2 * STEP_ICON_SIZE, STEP_ICON_SIZE))
        return button

    def _stepIcon(self, key, state, number):
        return _stepIcon(key, state, number)

    def _buildToolbar(self):
        toolbar = NeuroWorkflowWidget._buildToolbar(self)
        if toolbar is None:
            return None
        toolbar.setIconSize(qt.QSize(2 * STEP_ICON_SIZE, STEP_ICON_SIZE))
        actions = list(toolbar.actions())
        self.patientLabel = qt.QLabel()
        self.patientLabel.setStyleSheet("QLabel { color: #2f7ed8; font-weight: bold; padding: 0 6px; }")
        if len(actions) > 1:            # right of the module's name, before the steps
            self._patientAction = toolbar.insertWidget(actions[1], self.patientLabel)
        else:
            self._patientAction = toolbar.addWidget(self.patientLabel)
        self._patientAction.setVisible(False)
        window = slicer.util.mainWindow()
        self.reviewToolbar = qt.QToolBar("Static Neuro review", window)
        self.reviewToolbar.objectName = self.TOOLBAR_NAME + "Review"
        self.reviewToolbar.setIconSize(qt.QSize(24, 24))
        self.reviewToolbar.setToolButtonStyle(qt.Qt.ToolButtonTextBesideIcon)
        self.reviewToolbar.hide()
        window.addToolBarBreak()
        window.addToolBar(self.reviewToolbar)
        self._addReviewTools(toolbar, self.reviewToolbar)
        toolbar.connect("visibilityChanged(bool)", lambda visible: self._updateReviewTools())
        return toolbar

    def _addReviewTools(self, toolbar, reviewToolbar=None):
        """Static review has its own row. Dynamic Compare uses its workflow row."""
        def action(row, icon, label, tip):
            item = qt.QAction(label, row)
            item.toolTip = tip
            if reviewToolbar is not None:
                folder = os.path.join(os.path.dirname(ICON_PATH), "Review")
                item.setIcon(qt.QIcon(os.path.join(folder, icon + ".svg")))
            row.addAction(item)
            button = row.widgetForAction(item)
            button.setFocusPolicy(qt.Qt.NoFocus)
            if row is toolbar:
                button.setToolButtonStyle(qt.Qt.ToolButtonTextBesideIcon)
                button.setIconSize(qt.QSize(24, 24))
            return item

        toolbar.addSeparator()
        panel = action(toolbar, "Panel", "Panel", "Show or hide the module panel on the left")
        panel.connect("triggered()", self._toggleModulePanel)

        row = reviewToolbar if reviewToolbar is not None else toolbar
        self._reviewToolActions = []
        def label(text):
            caption = qt.QLabel(text)
            caption.setStyleSheet("QLabel { font-weight: bold; padding: 0 2px 0 6px; }")
            row.addWidget(caption)

        if reviewToolbar is not None:
            label("Report:")
            pdf = action(row, "ReportPDF", "PDF", "Save the review report as a PDF")
            pdf.connect("triggered()", self.onSaveReport)
            dicom = action(row, "ReportDICOM", "DICOM", "Save the report as DICOM and add it to the DICOM database")
            dicom.connect("triggered()", self.onSaveDicomReport)
            dicom.toolTip = "Save the report pages as images straight into the DICOM database (the PET's study)"
            row.addSeparator()
            label("Screenshot:")
            copy = action(row, "ScreenshotClipboard", "Clipboard", "Copy a screenshot of the views to the clipboard")
            copy.connect("triggered()", self.onScreenshotClipboard)
            shot = action(row, "ScreenshotDICOM", "DICOM",
                          "Save a screenshot of the views into the DICOM database (the PET's study)")
            shot.connect("triggered()", self.onScreenshotDicom)
        self._reviewToolActions.append(row.addSeparator())
        for icon, label, tip, fill in (
                ("SingleLayout", "Single Layouts", "Review layouts on one monitor",
                 lambda m: self._fillLayoutMenu(m, dual=False)),
                ("DualLayouts", "Dual Layouts", "Review layouts on two monitors",
                 lambda m: self._fillLayoutMenu(m, dual=True)),
                ("Slices", "Slices", "Slice windows, scores, fusion and unfolded brain settings", self._fillSliceMenu),
                ("Surface", "3D-SSP", "3D-SSP windows, scores, lighting, outlines and views", self._fillSspMenu),
                ("Colors", "Colors", "Color maps of the PET, scores and 3D-SSP", self._fillColorMenu)):
            item = action(row, icon, label, tip)
            menu = qt.QMenu(label, row)
            menu.connect("aboutToShow()", lambda m=menu, f=fill: f(m))
            item.setMenu(menu)
            row.widgetForAction(item).setPopupMode(qt.QToolButton.InstantPopup)
            self._reviewToolActions.append(item)
        restore = action(row, "Restore3D", "Restore 3D", "Restore the 3D-SSP views (rotation, zoom, pan)")
        restore.connect("triggered()", lambda: self._setupSspViews(fit=True))
        self._reviewToolActions.append(restore)
        fusion = action(row, "Fusion", "Fusion", "Show the PET and the scores over the anatomical image (on / off)")
        fusion.setCheckable(True)
        fusion.connect("triggered(bool)", self._setFusion)
        self._fusionAction = fusion
        self._reviewToolActions.append(fusion)
        segments = action(row, "Segments", "Segments", "Show or hide the region outlines on the slices and the 3D-SSP")
        segments.setCheckable(True)
        segments.connect("triggered(bool)", self._showAllOutlines)
        self._segmentsAction = segments
        self._connectSegmentsAction()
        self._reviewToolActions.append(segments)
        table = action(row, "Table", "Table", "Show or hide the region table on the right")
        table.connect("triggered()", self._toggleTable)
        self._reviewToolActions.append(table)
        self._updateReviewTools()

    def _outlineChecks(self):
        return [check for check in (getattr(self, "outlineCheck", None), getattr(self, "sspOutlineCheck", None))
                if check is not None]

    def _showAllOutlines(self, on):
        """The Segments button: the region outlines on the slices and on the 3D-SSP together."""
        for check in self._outlineChecks():
            check.checked = bool(on)
        self._syncSegmentsAction()

    def _connectSegmentsAction(self):
        """The Segments and Fusion buttons follow the panel's checks (once they exist: the toolbar is built before
        the review page)."""
        if getattr(self, "_segmentsConnected", False) or len(self._outlineChecks()) < 2:
            return
        for check in self._outlineChecks() + [getattr(self, "fusionCheck", None)]:
            if check is not None:
                check.connect("toggled(bool)", lambda on: self._syncSegmentsAction())
        self._segmentsConnected = True
        self._syncSegmentsAction()

    def _syncSegmentsAction(self):
        """The Segments button pressed while either outline setting (slices, 3D-SSP) is on; the Fusion button as the
        fusion check (off and greyed out without an anatomical image)."""
        action = getattr(self, "_segmentsAction", None)
        if action is not None:
            on = any(check.checked for check in self._outlineChecks())
            if action.checked != on:
                action.setChecked(on)
        action, check = getattr(self, "_fusionAction", None), getattr(self, "fusionCheck", None)
        if action is not None and check is not None:
            action.enabled = bool(check.enabled)
            on = bool(check.checked and check.enabled)
            if action.checked != on:
                action.setChecked(on)

    def _setFusion(self, on):
        check = getattr(self, "fusionCheck", None)
        if check is not None and check.enabled:
            check.checked = bool(on)
        self._syncSegmentsAction()

    @staticmethod
    def _toggleModulePanel():
        """The module panel (the dock on the left) shown / hidden."""
        window = slicer.util.mainWindow()
        dock = window.findChild(qt.QDockWidget, "PanelDockWidget") if window is not None else None
        if dock is not None:
            dock.setVisible(not dock.visible)

    def _updateReviewTools(self):
        if getattr(self, "reviewToolbar", None) is not None:
            main = getattr(self, "toolbar", None)
            self.reviewToolbar.setVisible(bool(main and main.visible and getattr(self, "current", None) == STEP_REVIEW))
            return
        for action in getattr(self, "_reviewToolActions", []):
            try:
                action.setVisible(getattr(self, "current", None) == STEP_REVIEW)
            except Exception:
                pass

    @staticmethod
    def _menuChoice(menu, title, checked, callback):
        action = menu.addAction(title)
        action.setCheckable(True)
        action.setChecked(bool(checked))
        action.connect("triggered()", callback)
        return action

    def _fillLayoutMenu(self, menu, dual=False):
        """The single monitor layouts, or (dual) the dual monitor layouts."""
        menu.clear()
        current = self.layoutCombo.currentData
        for key, title in nr.LAYOUTS:
            if (key in nr.DUAL_LAYOUTS) != dual:
                continue
            self._menuChoice(menu, title, key == current, lambda k=key: self._pickLayout(k))

    def _pickLayout(self, key):
        index = self.layoutCombo.findData(key)
        if index >= 0:
            self.layoutCombo.setCurrentIndex(index)
            self._onLayoutPicked()

    def _setScoreWindows(self, maximum):
        """The score window of the slices and of the 3D-SSP: -maximum .. +maximum."""
        self.scoreMaximumSpin.value = maximum
        self.sspMaximumSpin.value = maximum

    def _scoreWindowChoices(self, menu, spin):
        current = float(spin.value)
        for maximum in REVIEW_SCORE_WINDOWS:
            self._menuChoice(menu, f"\u00b1 {maximum:g} SD", abs(current - maximum) < 1e-6,
                             lambda v=maximum, w=spin: setattr(w, "value", v))

    def _scoreSideChoices(self, menu):
        """The "Scores colored" choices (one setting for the slices and the 3D-SSP, as on the left panel)."""
        menu.addSection("Scores colored (slices and 3D-SSP)")
        for key, title in nr.SCORE_SIDES:
            if key == nr.SIDE_THRESHOLD:
                title = f"{title}: {float(self.thresholdSpin.value):g}"
            self._menuChoice(menu, title, self.scoreSideCombo.currentData == key,
                             lambda k=key: self.scoreSideCombo.setCurrentIndex(max(0, self.scoreSideCombo.findData(k))))

    def _fillSliceMenu(self, menu):
        menu.clear()
        menu.addSection("Score window of the slices")
        self._scoreWindowChoices(menu, self.scoreMaximumSpin)
        self._scoreSideChoices(menu)
        menu.addSection("Image windows")
        reset = menu.addAction("PET: automatic (0 - 99.5th percentile)")
        reset.connect("triggered()", lambda: self._initPetWindow(percentile=True))
        full = menu.addAction("PET: 0 - maximum")
        full.connect("triggered()", lambda: self.petWindowRange.setValues(0.0, self.petWindowRange.maximum))
        fusion = menu.addAction("Fusion image: automatic")
        fusion.connect("triggered()", self._resetAnatomyWindow)
        fusion.enabled = self._fusionVolume() is not None
        sub = menu.addMenu("Unfolded brain depth")
        for depth in PEEL_DEPTHS:
            self._menuChoice(sub, f"{depth:g} mm", abs(float(self.peelDepthSpin.value) - depth) < 1e-6,
                             lambda d=depth: setattr(self.peelDepthSpin, "value", d))
        sub = menu.addMenu("Unfolded brain map")
        for index, (key, title, _) in enumerate(peel.UNFOLD_CENTERS):
            self._menuChoice(sub, title, self.peelCenterCombo.currentData == key,
                             lambda i=index: setattr(self.peelCenterCombo, "currentIndex", i))

    def _fillSspMenu(self, menu):
        menu.clear()
        menu.addSection("Score window of the 3D-SSP")
        self._scoreWindowChoices(menu, self.sspMaximumSpin)
        self._scoreSideChoices(menu)
        menu.addSection("3D-SSP")
        patient = menu.addAction("Patient values window: automatic")
        patient.connect("triggered()", lambda: (self._initSspWindow(), self._showSsp()))
        patient.enabled = bool((self.reviewResult or {}).get("ssp"))
        for title, check in (("Lighting", self.sspLightingCheck), ("Show the region outlines", self.sspOutlineCheck)):
            self._menuChoice(menu, title, check.checked, lambda c=check: setattr(c, "checked", not c.checked))
        restore = menu.addAction("Restore 3D views")
        restore.connect("triggered()", lambda: self._setupSspViews(fit=True))

    def _resetAnatomyWindow(self):
        self._anatomyWindowFor = None
        self._onFusionChanged()

    def _fillColorMenu(self, menu):
        menu.clear()
        layoutManager = slicer.app.layoutManager()
        shown = layoutManager.layout if layoutManager is not None else None
        pets = ((("PET (unfolded maps)", self.peelPetColorCombo, colormaps.LABELS),) if shown in nr.PEEL_LAYOUTS
                else ())
        if shown != nr.LAYOUT_PEEL:         # the review's own PET slices (also in the dual unfolded brain layout)
            pets = (("PET", self.petColorCombo, colormaps.LABELS),) + pets
        for title, combo, labels in pets + (("Scores (slices)", self.scoreColorCombo, nr.SCORE_COLORMAPS),
                                            ("3D-SSP scores", self.sspColorCombo, nr.SCORE_COLORMAPS),
                                            ("3D-SSP patient values", self.sspPatientColorCombo, colormaps.LABELS)):
            sub = menu.addMenu(title)
            for label in labels:
                self._menuChoice(sub, label, combo.currentText == label,
                                 lambda c=combo, t=label: c.setCurrentIndex(max(0, c.findText(t))))
        menu.addSection("Score slices")
        for key, title in nr.SCORE_STYLES:
            self._menuChoice(menu, title, self.scoreStyleCombo.currentData == key,
                             lambda k=key: self.scoreStyleCombo.setCurrentIndex(
                                 max(0, self.scoreStyleCombo.findData(k))))
        if self.fusionCheck.enabled:
            menu.addSeparator()
            self._menuChoice(menu, "Fusion with an anatomical image", self.fusionCheck.checked,
                             lambda: setattr(self.fusionCheck, "checked", not self.fusionCheck.checked))
