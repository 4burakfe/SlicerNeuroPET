"""
The workflow toolbar: the steps, the patient right of the module name, the review tools and their menus.

Part of the Static Neuro PET/SPECT module (StaticNeuroPET.py): the methods of ToolbarMixin, mixed into
StaticNeuroPETWidget there.
"""

import qt
import slicer

from DICOMGatedDynamicPETLib import colormaps
from NeuroWorkflowLib.base import NeuroWorkflowWidget
from NeuroPETLib import neuroreview as nr, peel

from StaticNeuroPETLib.constants import PEEL_DEPTHS, REVIEW_SCORE_WINDOWS, STEP_ICON_SIZE, STEP_REVIEW
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
        panelButton = qt.QToolButton()
        panelButton.setText("Panel")
        panelButton.toolTip = "Show / hide the module panel on the left (more room for the views)"
        panelButton.setToolButtonStyle(qt.Qt.ToolButtonTextOnly)
        panelButton.setFocusPolicy(qt.Qt.NoFocus)
        panelButton.connect("clicked()", self._toggleModulePanel)
        toolbar.addSeparator()
        toolbar.addWidget(panelButton)
        self._reviewToolActions = [toolbar.addSeparator()]
        for text, tip, fill in (
                ("Single Layouts", "Review layouts on one monitor (slices, 3D-SSP, cut 3D-SSP, unfolded brain)",
                 lambda m: self._fillLayoutMenu(m, dual=False)),
                ("Dual Layouts", "Review layouts on two monitors (the 3D-SSP views on the second screen)",
                 lambda m: self._fillLayoutMenu(m, dual=True)),
                ("Slices", "The slices: score window, scores colored, the PET and fusion windows",
                 self._fillSliceMenu),
                ("3D-SSP", "The 3D-SSP: score window, scores colored, the patient window, lighting, outlines, views",
                 self._fillSspMenu),
                ("Colors", "Color maps of the PET, the scores and the 3D-SSP", self._fillColorMenu)):
            button = qt.QToolButton()
            button.setText(text)
            button.toolTip = tip
            button.setToolButtonStyle(qt.Qt.ToolButtonTextOnly)
            button.setPopupMode(qt.QToolButton.InstantPopup)
            button.setFocusPolicy(qt.Qt.NoFocus)
            menu = qt.QMenu(button)
            button.setMenu(menu)
            menu.connect("aboutToShow()", lambda m=menu, f=fill: f(m))
            self._reviewToolActions.append(toolbar.addWidget(button))
        tableButton = qt.QToolButton()
        tableButton.setText("Table")
        tableButton.toolTip = "Show / hide the region table on the right of this layout"
        tableButton.setToolButtonStyle(qt.Qt.ToolButtonTextOnly)
        tableButton.setFocusPolicy(qt.Qt.NoFocus)
        tableButton.connect("clicked()", self._toggleTable)
        self._reviewToolActions.append(toolbar.addWidget(tableButton))
        for action in self._reviewToolActions:
            action.setVisible(False)
        return toolbar

    @staticmethod
    def _toggleModulePanel():
        """The module panel (the dock on the left) shown / hidden."""
        window = slicer.util.mainWindow()
        dock = window.findChild(qt.QDockWidget, "PanelDockWidget") if window is not None else None
        if dock is not None:
            dock.setVisible(not dock.visible)

    def _updateReviewTools(self):
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
