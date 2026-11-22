"""
Step 5: review: the page and its boxes, display settings, Compare, and adding the patient to a database.

Part of the Static Neuro PET/SPECT module (StaticNeuroPET.py): the methods of ReviewPageMixin, mixed into
StaticNeuroPETWidget there.
"""

import json
import logging

import ctk
import qt
import slicer

from DICOMGatedDynamicPETLib import colormaps, registration as reg
from NeuroWorkflowLib.base import ATLAS_FILE_ATTRIBUTE, NEURO_PREFIX, _button, _vtkMatrix
from NeuroPETLib import credits, neuro, normaldb, neuroreview as nr, peel, staticneuro as sn

from StaticNeuroPETLib.constants import (FUSION_TEMPLATES, NORMALIZATION_ATTRIBUTE, REVIEW_FUSION_OPACITY,
                                         REVIEW_OUTLINE_THICKNESS, REVIEW_PEEL_PET_COLORMAP, REVIEW_PET_COLORMAP,
                                         REVIEW_SCORE_OPACITY, SETTINGS_DISPLAY, SOURCE_ATTRIBUTE, STEP_REVIEW,
                                         WARNING_STYLE)
from StaticNeuroPETLib.helpers import _sameRegion, _screenCount


class ReviewPageMixin:
    """The review page methods of StaticNeuroPETWidget."""

    REVIEW_MEAN_MASK = True     # default of "Score only where the normal mean > 5 %" (Dynamic Compare: off)
    DISPLAY_SETTINGS_KEY = SETTINGS_DISPLAY     # the saved display settings (each module its own)
    NO_DATABASE_TEXT = "No database: the regional values (SUV, SUVR to the reference region) only."
    COMPARE_TEXT = ("Compare with the database", "Compute the regional values")     # (with, without a database)

    # --- hooks: what the review compares (the Dynamic Neuro Compare step: a parametric map) ---------------------
    def reviewVolume(self):
        """The patient's image the review compares and shows (the normalized PET / SPECT here)."""
        return self.volume()

    def _reviewHeader(self, folder):
        """The database header the review compares with (None: regional values only)."""
        return self._header(folder)

    def _reviewAgeAndSex(self):
        return self.logic.ageAndSex(self.reviewVolume())

    def _reviewReferenceId(self):
        """The reference segment the patient's values are divided by without a database (None: not divided)."""
        referenceId = self._parameter().GetParameter(NEURO_PREFIX + "ReferenceSegment") or None
        segmentation = self.node("Segmentation")
        if referenceId and (segmentation is None or segmentation.GetSegmentation().GetSegment(referenceId) is None):
            return None
        return referenceId

    def _adjustReviewResult(self, result):
        """The comparison's result before it is shown (the Dynamic Compare step: the regions' TAC fit values)."""
        return result

    def _reviewReady(self, volume, segmentation, header):
        """Whether Compare can run."""
        normalized = volume is not None and bool(volume.GetAttribute(NEURO_PREFIX + "Template"))
        return normalized and segmentation is not None and (
            header is None or bool((header.get("statistics") or {}).get("groups")))

    def _reviewImageLine(self, volume):
        return (f"Image: {volume.GetName() if volume else 'none'}"
                + (" (normalized)" if volume is not None and volume.GetAttribute(NEURO_PREFIX + "Template") else ""))

    @staticmethod
    def _defaultReviewLayout():
        return nr.defaultLayout(_screenCount())

    def _regionValueTitle(self):
        """The region table's column (and the mouse text) of the patient's regional value."""
        return "SUVmean"

    def _isFusionExcluded(self, node):
        """A normalized volume that is not offered as the fusion image (the static images)."""
        return self.logic.isStaticImage(node)

    # --- page 5: review (placeholder) and adding the patient to a database ------------------------------
    def _buildReviewPage(self):
        page = qt.QWidget()
        layout = qt.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        self._buildReviewBoxes(layout)

        addBox = qt.QGroupBox("Add this patient to a normal database")
        addBox.toolTip = ("For building a user defined database from normal subjects: the normalized image is "
                          "resampled onto the database grid (MNI 2009c, 1 mm) and saved as its SUV map and its SUVR "
                          "map (divided by the mean in the database's reference mask) in subjects/; a row is added "
                          "to subjects.csv. The mean / SD maps are then out of date.")
        form = qt.QFormLayout(addBox)
        self.addDbCombo = qt.QComboBox()
        self.addDbCombo.toolTip = "User defined databases of the database folder (step 3)"
        form.addRow("Database:", self.addDbCombo)
        self.addDbCredits = credits.infoBox()
        form.addRow(self.addDbCredits)
        self.addReferenceCombo = qt.QComboBox()
        self.addReferenceCombo.toolTip = ("Reference region of the SUVR map (intensity normalization): a segment of "
                                          "the regions (step 4; by default their reference region) or the database's "
                                          "reference mask. Warns when it is not the database's reference.")
        form.addRow("Reference region:", self.addReferenceCombo)
        self.subjectIdEdit = qt.QLineEdit()
        self.subjectIdEdit.toolTip = "ID of the subject in the database (file names and subjects.csv)"
        form.addRow("Subject ID:", self.subjectIdEdit)
        self.addAnatomyCheck = qt.QCheckBox("Add the CT/MR too")
        self.addAnatomyCheck.checked = True
        self.addAnatomyCheck.toolTip = ("The patient's CT / MR (step 1), normalized with the same transform at Harden, "
                                        "is stored with the subject (as it was and normalized) and goes into the "
                                        "database's CT / MR means.")
        form.addRow(self.addAnatomyCheck)
        self.addSubjectButton = _button("Add to database")
        form.addRow(self.addSubjectButton)
        self.addLabel = qt.QLabel()
        self.addLabel.wordWrap = True
        self.addLabel.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        form.addRow(self.addLabel)
        layout.addWidget(addBox)
        layout.addStretch(1)
        self.addSubjectButton.connect("clicked()", self.onAddToDatabase)
        self.addDbCombo.connect("activated(int)", lambda i: self._onAddDatabasePicked())
        self.addReferenceCombo.connect("activated(int)", lambda i: setattr(self, "_addReferenceChosen", True))
        return page

    # --- review: comparison, views and the region table -------------------------------------------
    def _buildReviewBoxes(self, layout):
        box = qt.QGroupBox("Comparison with the normal database")
        form = qt.QFormLayout(box)
        self.reviewLabel = qt.QLabel()
        self.reviewLabel.wordWrap = True
        form.addRow(self.reviewLabel)
        self.groupCombo = qt.QComboBox()
        self.groupCombo.toolTip = ("Mean / SD maps the patient is compared with: automatically the patient's age group "
                                   "(the master, all normals, when the age is not known or the group has no maps)")
        form.addRow("Normals:", self.groupCombo)
        self.scoreCombo = qt.QComboBox()
        for key, title in nr.SCORES:
            self.scoreCombo.addItem(title, key)
        self.scoreCombo.toolTip = ("z = (patient - normal mean) / SD; Crawford-Howell t = z / sqrt(1 + 1/n), for a "
                                   "small number of normals n. Negative: the patient below the normals "
                                   "(hypometabolism / hypoperfusion). Shown in the score row and the region table.")
        form.addRow("Score:", self.scoreCombo)
        percent = f"{nr.MASK_FRACTION * 100:g}"
        self.meanMaskCheck = qt.QCheckBox(f"Score only where the normal mean > {percent} % of its maximum")
        self.meanMaskCheck.checked = self.REVIEW_MEAN_MASK
        self.meanMaskCheck.toolTip = (
            f"On: the voxels scored are those in the MNI brain mask where the database mean is above {percent} % of "
            "its maximum (the ventricles and the CSF, near 0 in the normals, left out). Off: the brain mask alone "
            "(for maps whose values are low in much of the brain, e.g. a striatal tracer's BPnd in the cortex). "
            "Changing it compares again.")
        form.addRow(self.meanMaskCheck)
        self.compareButton = _button("Compare with the database")
        form.addRow(self.compareButton)
        self.compareLabel = qt.QLabel()
        self.compareLabel.wordWrap = True
        self.compareLabel.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        form.addRow(self.compareLabel)
        layout.addWidget(box)

        viewBox = ctk.ctkCollapsibleButton()
        viewBox.text = "Display"
        viewBox.collapsed = True
        view = qt.QFormLayout(viewBox)
        self.layoutCombo = qt.QComboBox()
        for key, title in nr.LAYOUTS:
            self.layoutCombo.addItem(title, key)
        self.layoutCombo.setCurrentIndex(max(0, self.layoutCombo.findData(self._defaultReviewLayout())))
        self.layoutCombo.toolTip = ("Single monitor: the slices, the 3D-SSP and the region table together. Dual "
                                    "monitor (the default with two screens): the slices and the table here, the "
                                    "3D-SSP views in a window on the second screen. Or the slices and the table, or "
                                    "the 3D-SSP views, alone.")
        view.addRow("Layout:", self.layoutCombo)
        self.peelDepthSpin = qt.QDoubleSpinBox()
        self.peelDepthSpin.setRange(0.0, peel.MAX_DEPTH_MM)
        self.peelDepthSpin.decimals = 0
        self.peelDepthSpin.singleStep = 1.0
        self.peelDepthSpin.value = peel.DEFAULT_DEPTH_MM
        self.peelDepthSpin.suffix = " mm"
        self.peelDepthSpin.toolTip = ("Unfolded brain layout: the depth under the brain's surface shown "
                                      "(curvilinear reformatting; the mouse wheel on a map too)")
        view.addRow("Unfolded depth:", self.peelDepthSpin)
        self.peelDepthSpin.connect("valueChanged(double)", lambda v: self._updatePeel())
        self.peelCenterCombo = qt.QComboBox()
        for key, title, _ in peel.UNFOLD_CENTERS:
            self.peelCenterCombo.addItem(title, key)
        self.peelCenterCombo.toolTip = "Unfolded brain layout: the map (rectangular, or a world map centred on a side)"
        view.addRow("Unfolded map:", self.peelCenterCombo)
        self.peelCenterCombo.connect("currentIndexChanged(int)", lambda i: self._onPeelCenter())
        self.peelPetColorCombo = qt.QComboBox()
        for label in colormaps.LABELS:
            self.peelPetColorCombo.addItem(label)
        self.peelPetColorCombo.setCurrentIndex(max(0, self.peelPetColorCombo.findText(REVIEW_PEEL_PET_COLORMAP)))
        self.peelPetColorCombo.toolTip = "Unfolded brain layout: the colors of the PET and fusion maps"
        view.addRow("Unfolded PET colors:", self.peelPetColorCombo)
        self.peelPetColorCombo.connect("currentIndexChanged(int)", lambda i: self._applyPeelPetColors())
        self.petColorCombo = qt.QComboBox()
        self.scoreColorCombo = qt.QComboBox()
        for label in colormaps.LABELS:
            self.petColorCombo.addItem(label)
        self.petColorCombo.setCurrentIndex(max(0, self.petColorCombo.findText(REVIEW_PET_COLORMAP)))
        for label in nr.SCORE_COLORMAPS:
            self.scoreColorCombo.addItem(label)
        view.addRow("PET colors:", self.petColorCombo)
        self.petWindowRange = ctk.ctkRangeWidget()
        self.petWindowRange.decimals = 2
        self.petWindowRange.toolTip = "Display range of the normalized PET / SPECT (top row)"
        view.addRow("PET window:", self.petWindowRange)
        self.scoreColorCombo.toolTip = ("Diverging colors over the score window, 0 in the middle: below the normals "
                                        "(negative) cold, above them hot. Outside the brain mask black.")
        view.addRow("Score colors:", self.scoreColorCombo)
        self.scoreMaximumSpin = qt.QDoubleSpinBox()
        self.scoreMaximumSpin.setRange(0.5, 50.0)
        self.scoreMaximumSpin.decimals = 1
        self.scoreMaximumSpin.singleStep = 0.5
        self.scoreMaximumSpin.value = nr.DEFAULT_MAXIMUM
        self.scoreMaximumSpin.prefix = "\u00b1 "
        self.scoreMaximumSpin.toolTip = "Score window, symmetric about 0: from minus to plus this value"
        view.addRow("Score window:", self.scoreMaximumSpin)
        self.scoreSideCombo = qt.QComboBox()
        for key, title in nr.SCORE_SIDES:
            self.scoreSideCombo.addItem(title, key)
        self.scoreSideCombo.toolTip = (
            "Which scores are colored (score slices and 3D-SSP): all; only those below the normals "
            "(hypometabolism / hypoperfusion); or only those below the threshold. In the last two the color scale is "
            "reversed across minus the window .. 0 (the lowest scores at its top end) and the rest is grey; maps "
            "with a neutral middle (IceFire, CoolWarm, GreyZero, PurpleOrange, DivergingBlueRed) stay as they are.")
        view.addRow("Scores colored:", self.scoreSideCombo)
        self.thresholdSpin = qt.QDoubleSpinBox()
        self.thresholdSpin.setRange(-10.0, 0.0)
        self.thresholdSpin.decimals = 1
        self.thresholdSpin.singleStep = 0.5
        self.thresholdSpin.value = nr.DEFAULT_THRESHOLD
        self.thresholdSpin.toolTip = ("Below the threshold only: the scores below this value colored, the rest grey; "
                                      "also the low score 3D-SSP views of the 16 view layouts")
        view.addRow("Threshold:", self.thresholdSpin)
        self.scoreStyleCombo = qt.QComboBox()
        for key, title in nr.SCORE_STYLES:
            self.scoreStyleCombo.addItem(title, key)
        self.scoreStyleCombo.toolTip = (
            "Score as color, PET as brightness (default): each voxel colored by its score and as bright as the "
            "patient's uptake there, with no threshold or mask (the PET's own pattern in the score's colors; over the "
            "fusion image at the score opacity). Score map: the scores alone, in a mask and with a threshold (below).")
        view.addRow("Score slices:", self.scoreStyleCombo)
        self.scoreMaskCombo = qt.QComboBox()
        for key, title in nr.SCORE_MASKS:
            self.scoreMaskCombo.addItem(title, key)
        self.scoreMaskCombo.toolTip = (
            "Where the score slices show the scores. Grey matter (default): the MNI grey matter probability, the same "
            "for every tracer (the cortex also with a striatal tracer). High normal uptake: the database mean above a "
            "fraction of its maximum. Elsewhere the PET (grey) or the MR shows.")
        view.addRow("Scores shown in:", self.scoreMaskCombo)
        self.gmSpin = qt.QDoubleSpinBox()
        self.gmSpin.setRange(0.05, 0.95)
        self.gmSpin.decimals = 2
        self.gmSpin.singleStep = 0.05
        self.gmSpin.value = nr.GM_MINIMUM
        self.gmSpin.toolTip = "Grey matter probability from which the scores are shown"
        view.addRow("Grey matter >=", self.gmSpin)
        self.uptakeSpin = qt.QSpinBox()
        self.uptakeSpin.setRange(int(round(nr.MASK_FRACTION * 100)), 95)
        self.uptakeSpin.singleStep = 5
        self.uptakeSpin.value = int(round(nr.UPTAKE_FRACTION * 100))
        self.uptakeSpin.suffix = " % of the normal maximum"
        self.uptakeSpin.toolTip = ("The score slices show only where the normals' uptake is high: the database mean "
                                   "above this fraction of its maximum (cortex, deep grey matter). White matter and "
                                   "CSF, low in the normals anyway, are hidden (black, or the MR in fusion).")
        view.addRow("Normal uptake >", self.uptakeSpin)
        self.hideBelowSpin = qt.QDoubleSpinBox()
        self.hideBelowSpin.setRange(0.0, 10.0)
        self.hideBelowSpin.decimals = 1
        self.hideBelowSpin.singleStep = 0.5
        self.hideBelowSpin.value = nr.HIDE_BELOW
        self.hideBelowSpin.toolTip = ("Scores closer to 0 than this are not colored (the PET or the MR under them "
                                      "shows): only the clusters below / above the normals. 0: all colored.")
        view.addRow("Hide |score| below:", self.hideBelowSpin)
        self.scoreBackgroundCombo = qt.QComboBox()
        self.scoreBackgroundCombo.addItem("PET (grey)", "pet")
        self.scoreBackgroundCombo.addItem("Black", "black")
        self.scoreBackgroundCombo.toolTip = ("Under the score slices (without the MR fusion): the PET in grey, or "
                                             "black")
        view.addRow("Score background:", self.scoreBackgroundCombo)
        self.outlineCheck = qt.QCheckBox("Show the region outlines")
        self.outlineCheck.checked = False
        self.outlineCheck.toolTip = (f"The regions as {REVIEW_OUTLINE_THICKNESS} px outlines on the PET and the score "
                                     "slices. Click a region in the table to show it alone for a few seconds.")
        view.addRow(self.outlineCheck)
        self.legendCheck = qt.QCheckBox("Show the color legends")
        self.legendCheck.checked = True
        self.legendCheck.toolTip = "A color legend on the left of each row (PET, scores, 3D-SSP)"
        view.addRow(self.legendCheck)
        self.tableDefaultCheck = qt.QCheckBox("Show the region table in the layouts")
        self.tableDefaultCheck.checked = True
        self.tableDefaultCheck.toolTip = ("The region table on the right of the layouts at start (in the main window "
                                          "of the dual monitor layouts; not in the unfolded brain and cut 3D-SSP "
                                          "layouts). The toolbar's Table button shows / hides it in a layout.")
        view.addRow(self.tableDefaultCheck)
        layout.addWidget(viewBox)
        self._buildSspBox(layout)
        layout.addWidget(self._buildMouseHelp())

        fusionBox = qt.QGroupBox("Fusion")
        fusion = qt.QFormLayout(fusionBox)
        self.fusionCheck = qt.QCheckBox("Show the PET and the scores over an anatomical image")
        fusion.addRow(self.fusionCheck)
        self.fusionSourceCombo = qt.QComboBox()
        self.fusionSourceCombo.toolTip = ("The patient's CT / MR and the other sequences of step 1 (normalized with "
                                          "the image in step 3), or an MNI template (T1, T2, PD)")
        fusion.addRow("Image:", self.fusionSourceCombo)
        self.fusionOpacity = ctk.ctkSliderWidget()
        self.fusionOpacity.minimum, self.fusionOpacity.maximum = 0.0, 1.0
        self.fusionOpacity.singleStep, self.fusionOpacity.decimals = 0.05, 2
        self.fusionOpacity.value = REVIEW_FUSION_OPACITY
        self.fusionOpacity.toolTip = "Opacity of the PET over the MR"
        fusion.addRow("PET opacity:", self.fusionOpacity)
        self.scoreOpacity = ctk.ctkSliderWidget()
        self.scoreOpacity.minimum, self.scoreOpacity.maximum = 0.0, 1.0
        self.scoreOpacity.singleStep, self.scoreOpacity.decimals = 0.05, 2
        self.scoreOpacity.value = REVIEW_SCORE_OPACITY
        self.scoreOpacity.toolTip = "Opacity of the z / t scores over the MR"
        fusion.addRow("Score opacity:", self.scoreOpacity)
        self.anatomyWindowRange = ctk.ctkRangeWidget()
        self.anatomyWindowRange.decimals = 1
        self.anatomyWindowRange.toolTip = "Display range of the fusion image (CT / MR / template, grey)"
        fusion.addRow("Image window:", self.anatomyWindowRange)
        layout.addWidget(fusionBox)
        settingsRow = qt.QHBoxLayout()
        self.saveDisplayButton = _button("Save display settings")
        self.saveDisplayButton.toolTip = ("Keep the display, 3D-SSP and fusion settings (colors, score windows, score "
                                          "slices, outlines, legends, opacities; not the image windows or the layout) "
                                          "as the defaults of the next sessions")
        self.restoreDisplayButton = _button("Restore defaults", "revert")
        self.restoreDisplayButton.toolTip = "The display settings back to the saved ones (the factory ones if none)"
        self.factoryDisplayButton = _button("Restore factory", "revert")
        self.factoryDisplayButton.toolTip = "The display settings back to the module's own; the saved ones forgotten"
        for button in (self.saveDisplayButton, self.restoreDisplayButton, self.factoryDisplayButton):
            settingsRow.addWidget(button)
        layout.addLayout(settingsRow)

        self.compareButton.connect("clicked()", self.onCompare)
        self.meanMaskCheck.connect("toggled(bool)", lambda on: self._onMeanMaskChanged())
        self.groupCombo.connect("activated(int)", lambda i: setattr(self, "_groupChosen", True))
        self.scoreCombo.connect("currentIndexChanged(int)", lambda i: self._showScores())
        self.petColorCombo.connect("currentIndexChanged(int)", lambda i: self._applyPetDisplay())
        self.petWindowRange.connect("valuesChanged(double,double)", lambda a, b: self._applyPetDisplay())
        for widget in (self.scoreColorCombo,):
            widget.connect("currentIndexChanged(int)", lambda i: self._applyScoreDisplay())
        self.scoreMaximumSpin.connect("valueChanged(double)", lambda v: self._applyScoreDisplay())
        self.scoreSideCombo.connect("currentIndexChanged(int)", lambda i: (self._applyScoreDisplay(),
                                                                         self._updateLegends()))
        self.thresholdSpin.connect("valueChanged(double)", lambda v: self._applyScoreDisplay())
        self.fusionCheck.connect("toggled(bool)", lambda on: self._onFusionChanged())
        self.fusionSourceCombo.connect("currentIndexChanged(int)", lambda i: self._onFusionChanged())
        self.fusionOpacity.connect("valueChanged(double)", lambda v: self._applyReviewOpacities())
        self.scoreOpacity.connect("valueChanged(double)", lambda v: self._applyReviewOpacities())
        self.layoutCombo.connect("activated(int)", lambda i: self._onLayoutPicked())
        self.uptakeSpin.connect("valueChanged(int)", lambda v: self._showScoreMap())
        self.gmSpin.connect("valueChanged(double)", lambda v: self._showScoreMap())
        self.hideBelowSpin.connect("valueChanged(double)", lambda v: self._showScoreMap())
        self.scoreMaskCombo.connect("currentIndexChanged(int)", lambda i: self._showScoreMap())
        self.scoreStyleCombo.connect("currentIndexChanged(int)", lambda i: (self._showScoreMap(),
                                                                          self._assignReviewViews()))
        self.scoreBackgroundCombo.connect("currentIndexChanged(int)", lambda i: self._assignReviewViews())
        self.outlineCheck.connect("toggled(bool)", lambda on: self._assignReviewViews())
        self.legendCheck.connect("toggled(bool)", lambda on: (self._updateLegends(), self._showSsp()))
        self.tableDefaultCheck.connect("toggled(bool)", lambda on: self._onTableDefault())
        self.anatomyWindowRange.connect("valuesChanged(double,double)", lambda a, b: self._applyAnatomyDisplay())
        self.reviewResult = None
        self.sspModels = {}                 # (view set, surface part): model node
        self.cutModels = {}                 # ("surface", orientation, side[, low]) / ("face", orientation[, low])
        self.scoreNode = None
        self.reviewTable = None
        self._reviewPreviousLayout = None
        self._segmentationDisplay = None
        self.regionTable = None             # QTableWidget in the layout's table view
        self._regionTableHost = None
        self._flash = None                  # (token, {segment ID: visible}) while a region is shown alone
        self._autoCompared = None
        self._factoryDisplay = self._displayValues()
        saved = self._savedDisplay()
        if saved:
            self._setDisplayValues(saved, quiet=True)
        self.saveDisplayButton.connect("clicked()", self.onSaveDisplay)
        self.restoreDisplayButton.connect("clicked()", self.onRestoreDisplay)
        self.factoryDisplayButton.connect("clicked()", self.onFactoryDisplay)

    def _resetReviewState(self):
        """A closed scene: the review's observers taken off the views and its nodes and caches forgotten."""
        # the views' interactors and the slice and camera nodes of the layout outlive the scene: the review's
        # observers are taken off them before they are forgotten below (else every closed scene left one more
        # set on them and the next review handled each event once more)
        for stop in (self._watchCutSlices, self._watchMouse, self._syncCameras, self._watchPeelWheel):
            try:
                stop(False)
            except Exception:
                logging.exception("Static Neuro: could not remove the review's observers")
        self.reviewResult = self.scoreNode = self.reviewTable = self.greyPetNode = self.rainbowPetNode = None
        self._brightnessCache = None
        self.sspModels = {}
        self.cutModels = {}
        self.peelNodes = {}
        self._peelDepth = None
        self._peelRays = None
        self._peelHighlighted = 0
        self._peelMaps = {}
        self._peelWheelObservers = []
        self._peelTexts = {}
        self._cutSliceObservers = []
        self._cutClipNodes = {}
        self._cutFacePlanes = {}
        self._sspMainDisplay = None
        self._sspLowDisplay = None
        self._sspPolys = {}
        self._mouseObservers = {}
        self._drag = None
        self._sspLabels = None
        self._sspOutlineModels = {}
        self._sspOutlineSurfaces = {}
        self._sspAllOutlineModels = {}
        self._sspBorderPolys = {}
        self._sspOutlineLabel = None
        self._sspColorsWritten = {}
        self._sspBounds = None
        self.scoreRgbNode = None
        self._cameraObservers = []
        self._segmentationDisplay = None
        self._anatomyWindowFor = None
        self._flash = self._autoCompared = None
        self._regionTableData = None
        if getattr(self, "regionTable", None) is not None:
            try:
                self.regionTable.setRowCount(0)
            except Exception:
                self.regionTable = None
        if hasattr(self, "compareLabel"):
            self.compareLabel.text = ""

    # saved display settings
    def _displayWidgets(self):
        """{key: widget} of the review display settings that are saved (with the layout, not the image windows)."""
        return {"layout": self.layoutCombo, "petColors": self.petColorCombo, "scoreColors": self.scoreColorCombo,
                "scoreMaximum": self.scoreMaximumSpin, "scoreSide": self.scoreSideCombo,
                "threshold": self.thresholdSpin,
                "scoreStyle": self.scoreStyleCombo, "scoreMask": self.scoreMaskCombo, "greyMatter": self.gmSpin,
                "uptake": self.uptakeSpin, "hideBelow": self.hideBelowSpin, "meanMask": self.meanMaskCheck,
                "scoreBackground": self.scoreBackgroundCombo, "outlines": self.outlineCheck,
                "legends": self.legendCheck, "table": self.tableDefaultCheck, "sspMap": self.sspMapCombo, "sspSource": self.sspSourceCombo,
                "sspColors": self.sspColorCombo, "sspMaximum": self.sspMaximumSpin,
                "sspPatientColors": self.sspPatientColorCombo, "sspLighting": self.sspLightingCheck,
                "sspOutlines": self.sspOutlineCheck,
                "fusionOpacity": self.fusionOpacity, "scoreOpacity": self.scoreOpacity,
                "peelPetColors": self.peelPetColorCombo}

    def _displayValues(self):
        values = {}
        for key, widget in self._displayWidgets().items():
            if isinstance(widget, qt.QComboBox):
                values[key] = widget.currentText
            elif isinstance(widget, qt.QCheckBox):
                values[key] = bool(widget.checked)
            else:
                values[key] = widget.value
        return values

    def _setDisplayValues(self, values, quiet=False):
        """The display widgets set to values (unknown keys and labels skipped); quiet: without their signals."""
        for key, widget in self._displayWidgets().items():
            if key not in values:
                continue
            value = values[key]
            if quiet:
                widget.blockSignals(True)
            try:
                if isinstance(widget, qt.QComboBox):
                    index = widget.findText(str(value))
                    if index >= 0:
                        widget.setCurrentIndex(index)
                elif isinstance(widget, qt.QCheckBox):
                    widget.checked = bool(value)
                elif isinstance(widget, qt.QSpinBox):
                    widget.value = int(value)
                else:
                    widget.value = float(value)
            except (TypeError, ValueError):
                logging.warning(f"Static Neuro: display setting {key} = {value!r} skipped")
            finally:
                if quiet:
                    widget.blockSignals(False)

    def _savedDisplay(self):
        try:
            return json.loads(str(qt.QSettings().value(self.DISPLAY_SETTINGS_KEY) or "") or "{}")
        except ValueError:
            return {}

    def onSaveDisplay(self):
        qt.QSettings().setValue(self.DISPLAY_SETTINGS_KEY, json.dumps(self._displayValues()))
        slicer.util.showStatusMessage("Static Neuro: display settings saved as the defaults", 3000)

    def onRestoreDisplay(self):
        self._setDisplayValuesAndLayout(self._savedDisplay() or self._factoryDisplay)

    def onFactoryDisplay(self):
        qt.QSettings().remove(self.DISPLAY_SETTINGS_KEY)
        self._setDisplayValuesAndLayout(self._factoryDisplay)

    def _setDisplayValuesAndLayout(self, values):
        """The display settings set; the views rebuilt when the layout changed (the combo sends no activated())."""
        layout = self._reviewLayout()
        self._setDisplayValues(values)
        if self._reviewLayout() != layout:
            self._onLayoutPicked()

    def _updateReviewInfo(self, volume, segmentation, header, age, sex):
        lines = [self._reviewImageLine(volume),
                 "Age: " + (f"{age:g}" if age is not None else "unknown") + ", sex: "
                 + (dict(sn.SEXES).get(sex) if sex else "unknown"),
                 f"Regions: {segmentation.GetName() if segmentation else 'not loaded (step 4)'}"]
        if header is not None:
            stats = header.get("statistics") or {}
            lines.append(f"Database: {header['name']} ({header.get('mapType', '')} maps"
                         + (f", smoothing {float(stats.get('smoothingFwhmMm', header.get('smoothingFwhmMm')) or 0):g} "
                            "mm" if stats.get("groups") else "") + ")")
            if not stats.get("groups"):
                lines.append(WARNING_STYLE.format("The database has no mean / SD maps: compute them in the Neuro PET "
                                                  "Database Manager."))
            elif stats.get("stale"):
                lines.append(WARNING_STYLE.format("Subjects were added after the mean / SD maps were computed."))
        else:
            lines.append(self.NO_DATABASE_TEXT)
        result = self.reviewResult
        if result is not None and (result.get("volume") is not volume or result.get("header") != header):
            lines.append(WARNING_STYLE.format("The image or the database changed: compare again."))
        self.reviewLabel.text = "<br>".join(lines)
        self._fillGroups(header, age)
        self.groupCombo.enabled = header is not None and self.groupCombo.count > 1
        self.compareButton.text = self.COMPARE_TEXT[0 if header is not None else 1]
        self.compareButton.enabled = bool(self._reviewReady(volume, segmentation, header))
        self._fillFusionSources()

    def _fillGroups(self, header, age):
        key = (json.dumps((header or {}).get("statistics"), sort_keys=True, default=str), age)
        if getattr(self, "_groupsFor", None) == key:
            return
        self._groupsFor = key
        current = self.groupCombo.currentData
        chosen = getattr(self, "_groupChosen", False)
        self.groupCombo.blockSignals(True)
        self.groupCombo.clear()
        if header is not None:
            automatic, why = nr.patientGroup(header, age)
            names = {k: (name, n) for k, name, n in nr.statisticsGroups(header)}
            title = names.get(automatic, (automatic, None))[0]
            title = normaldb.MASTER_TITLE if automatic == normaldb.MASTER else title
            self.groupCombo.addItem(f"Automatic: {title} ({why})", "")
            for k, (name, n) in names.items():
                self.groupCombo.addItem((normaldb.MASTER_TITLE if k == normaldb.MASTER else f"Age group {name}")
                                        + (f" (n = {n})" if n else ""), k)
        index = self.groupCombo.findData(current) if chosen and current else -1
        self.groupCombo.setCurrentIndex(max(0, index))
        self.groupCombo.blockSignals(False)

    def _groupKey(self, header, age):
        key = self.groupCombo.currentData if self.groupCombo.count else ""
        return key or nr.patientGroup(header, age)[0]

    def _fusionCandidates(self):
        """The patient's volumes in the template space for the fusion: the normalized CT / MR first, then the other
        normalized sequences (and any other normalized anatomical volume)."""
        first = self.node("NormalizedAnatomy")
        skip = {n.GetID() for n in (self.reviewVolume(), getattr(self, "greyPetNode", None),
                                    getattr(self, "rainbowPetNode", None), self.scoreNode)
                if n is not None}
        nodes = []
        for node in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
            if (node.GetID() in skip or node.IsA("vtkMRMLLabelMapVolumeNode") or node.GetHideFromEditors()
                    or node.GetAttribute(ATLAS_FILE_ATTRIBUTE) or not node.GetAttribute(NEURO_PREFIX + "Template")
                    or self._isFusionExcluded(node)):
                continue
            nodes.append(node)
        nodes.sort(key=lambda n: n is not first)
        return nodes

    def _fillFusionSources(self):
        wanted = []
        for node in self._fusionCandidates():
            source = node.GetAttribute(SOURCE_ATTRIBUTE)
            what = "CT" if source and source == (self.anatomy().GetName() if self.anatomy() else None) and \
                self.contrast()[0] == "CT" else "Patient"
            wanted.append((f"node:{node.GetID()}", f"{what}: {node.GetName()}"))
        for key, title, _ in neuro.TEMPLATES:
            if key in FUSION_TEMPLATES and key in (self.atlas or {}):
                wanted.append((f"template:{key}", f"MNI template {title}"))
        if [(self.fusionSourceCombo.itemData(i), self.fusionSourceCombo.itemText(i))
                for i in range(self.fusionSourceCombo.count)] == wanted:
            return
        current = self.fusionSourceCombo.currentData
        self.fusionSourceCombo.blockSignals(True)
        self.fusionSourceCombo.clear()
        for key, title in wanted:
            self.fusionSourceCombo.addItem(title, key)
        self.fusionSourceCombo.setCurrentIndex(max(0, self.fusionSourceCombo.findData(current)))
        self.fusionSourceCombo.blockSignals(False)
        self.fusionCheck.enabled = bool(wanted)

    def _fusionVolume(self):
        if not self.fusionCheck.checked or not self.fusionCheck.enabled:
            return None
        key = self.fusionSourceCombo.currentData or ""
        if key.startswith("node:"):
            node = slicer.mrmlScene.GetNodeByID(key[len("node:"):])
            return node if node is not None and slicer.mrmlScene.IsNodePresent(node) else None
        if key.startswith("template:") and key[len("template:"):] in (self.atlas or {}):
            name = key[len("template:"):]
            return self.logic.loadAtlasVolume(self.atlas[name], name=f"MNI ICBM152 2009c {name}")
        return None

    def onCompare(self):
        volume, segmentation = self.reviewVolume(), self.node("Segmentation")
        if volume is None or segmentation is None:
            return
        folder = self._database()
        header = self._reviewHeader(folder)
        removed = self.logic.dropSegments(segmentation)
        age, _ = self._reviewAgeAndSex()
        groupKey = self._groupKey(header, age) if header is not None else normaldb.MASTER
        referenceId = self._reviewReferenceId()
        dialog = slicer.util.createProgressDialog(windowTitle="Review", labelText="Comparing...", maximum=0)

        def progress(index, total, text):
            dialog.maximum = total
            dialog.setValue(index)
            dialog.setLabelText(f"The normals: {text}")
            slicer.app.processEvents()
            return not dialog.wasCanceled

        slicer.app.processEvents()
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            result = self.logic.reviewData(volume, segmentation, folder if header is not None else None, header,
                                           groupKey, self.atlas, referenceId, self._locked(), progress,
                                           meanMask=bool(self.meanMaskCheck.checked))
            if result is not None:
                result = self._adjustReviewResult(result)
            if result is not None:
                try:
                    result["ssp"] = self.logic.reviewSsp(result, folder, header, groupKey, self._locked(), progress)
                    if result["ssp"] is None:
                        result = None
                    else:
                        result["notes"].extend(result["ssp"]["notes"])
                except Exception as error:
                    logging.exception("Static Neuro: the 3D-SSP failed")
                    result["ssp"] = None
                    result["notes"].append(f"No 3D-SSP: {error}")
        except Exception as error:
            logging.exception("Static Neuro: the review failed")
            slicer.util.errorDisplay(f"Could not compare the patient:\n{error}", windowTitle="Review")
            return
        finally:
            qt.QApplication.restoreOverrideCursor()
            dialog.close()
            dialog.deleteLater()
        if result is None:
            self.compareLabel.text = "Stopped."
            return
        result.update(volume=volume, header=header, groupKey=groupKey, segmentation=segmentation)
        self.reviewResult = result
        self._brightnessCache = None        # of the comparison before
        self._makeScoreNode()
        self._initPetWindow()
        self._initSspWindow()
        self._showScores()
        self._enterReviewViews()
        lines = []
        if header is not None:
            group = next((name for k, name, _ in nr.statisticsGroups(header) if k == groupKey), groupKey)
            lines.append(f"Compared with {normaldb.MASTER_TITLE if groupKey == normaldb.MASTER else group} "
                         f"(n = {result.get('n')}); patient smoothed {result['fwhm']:g} mm FWHM like the normals.")
            lines.append(f"Regional normal values: {result['source']}.")
        if result.get("referenceMean"):
            lines.append(f"Reference region mean: {result['referenceMean']:.4g}.")
        if removed:
            lines.append("Left out: " + ", ".join(removed) + ".")
        lines.extend(result["notes"])
        self.compareLabel.text = " ".join(lines)
        self.refresh()

    def _onMeanMaskChanged(self):
        """The 5 % mean mask changes the voxels scored: compare again when there is a comparison (with a database)."""
        result = self.reviewResult
        if (result is not None and result.get("header") is not None
                and result.get("meanMask", True) != bool(self.meanMaskCheck.checked)
                and getattr(self, "current", None) == STEP_REVIEW and self.compareButton.enabled):
            self.onCompare()

    def _autoCompare(self):
        """Compare on entering the review when there is no comparison of this image, regions and database yet (once
        per image, regions and database: not again after a failure or a stop)."""
        if getattr(self, "current", None) != STEP_REVIEW or not self.compareButton.enabled:
            return
        volume, segmentation = self.reviewVolume(), self.node("Segmentation")
        folder = self._database()
        header = self._reviewHeader(folder)
        result = self.reviewResult
        meanMask = bool(self.meanMaskCheck.checked)
        if (result is not None and result.get("volume") is volume and result.get("header") == header
                and result.get("segmentation") is segmentation
                and (header is None or result.get("meanMask", True) == meanMask)):
            return
        key = (volume.GetID() if volume else "", segmentation.GetID() if segmentation else "", folder,
               json.dumps(header, sort_keys=True, default=str), meanMask)
        if self._autoCompared == key:
            return
        self._autoCompared = key
        self.onCompare()

    def _fillAddDatabases(self):
        if not hasattr(self, "addDbCombo"):
            return
        current = self.addDbCombo.currentData or ""
        chosen = getattr(self, "_addDbChosen", False)
        self.addDbCombo.blockSignals(True)
        self.addDbCombo.clear()
        for folder, header, rank, _ in self.databaseRows:
            if header.get("kind") != normaldb.USER_DEFINED:
                continue
            self.addDbCombo.addItem(("! " if rank else "") + f"{header.get('name')} (n = "
                                    f"{normaldb.subjectCount(folder, header)})", folder)
        wanted = current if chosen and current else self._database()
        self.addDbCombo.setCurrentIndex(max(0, self.addDbCombo.findData(wanted)))
        self.addDbCombo.blockSignals(False)
        self._updateAddDbCredits()

    def _updateAddDbCredits(self):
        if hasattr(self, "addDbCredits"):
            self.addDbCredits.setHtml(credits.databaseHtml(self._header(self.addDbCombo.currentData or ""),
                                                           "No user defined database in the folder."))

    def _onAddDatabasePicked(self):
        self._addDbChosen = True
        self._updateAddDbCredits()
        self._fillAddReferences()

    def _fillAddReferences(self):
        """The reference regions to pick from: the reference mask of the database picked for adding first (picked by
        default, as the batch add uses it), then the segments of the regions (their reference region first)."""
        if not hasattr(self, "addReferenceCombo"):
            return
        segmentation = self.node("Segmentation")
        folder = self.addDbCombo.currentData or ""
        header = self._header(folder)
        referenceId = self._parameter().GetParameter(NEURO_PREFIX + "ReferenceSegment") or ""
        ids = self.logic.segmentIds(segmentation) if segmentation is not None else []
        key = (segmentation.GetID() if segmentation else "", tuple(ids), referenceId, folder)
        if getattr(self, "_addReferencesFor", None) == key:
            return
        self._addReferencesFor = key
        current = self.addReferenceCombo.currentData
        chosen = getattr(self, "_addReferenceChosen", False)
        self.addReferenceCombo.blockSignals(True)
        self.addReferenceCombo.clear()
        if header is not None and sn.referenceMaskPath(folder, header):
            title = sn.regionSettings(folder, header)["referenceTitle"] or "reference"
            self.addReferenceCombo.addItem(f"Database reference mask ({title})", "")
        if referenceId in ids:
            ids.remove(referenceId)
            ids.insert(0, referenceId)
        for segmentId in ids:
            self.addReferenceCombo.addItem(segmentation.GetSegmentation().GetSegment(segmentId).GetName(), segmentId)
        index = self.addReferenceCombo.findData(current) if chosen and current is not None else -1
        self.addReferenceCombo.setCurrentIndex(index if index >= 0 else 0)
        self.addReferenceCombo.blockSignals(False)

    def _addReference(self, folder, header):
        """(segmentation, segment ID, name) of the reference region picked for adding; (None, None, name) for the
        database's reference mask."""
        segmentId = self.addReferenceCombo.currentData if self.addReferenceCombo.count else None
        segmentation = self.node("Segmentation")
        if segmentId and segmentation is not None:
            name = segmentation.GetSegmentation().GetSegment(segmentId).GetName()
            return segmentation, segmentId, name
        return None, None, sn.regionSettings(folder, header)["referenceTitle"] or "Database reference mask"

    def _updateReviewPage(self):
        if not hasattr(self, "reviewLabel"):
            return
        volume = self.reviewVolume()
        segmentation = self.node("Segmentation")
        header = self._reviewHeader(self._database())
        age, sex = self._reviewAgeAndSex()
        self._updateReviewInfo(volume, segmentation, header, age, sex)
        if hasattr(self, "subjectIdEdit"):
            key = volume.GetID() if volume is not None else ""
            if getattr(self, "_subjectIdFor", None) != key:
                self._subjectIdFor = key
                self.subjectIdEdit.text = (sn.suggestedSubjectId(self.logic.attributes(volume), volume.GetName())
                                           if volume is not None else "")
            self._fillAddReferences()
            normalized = volume is not None and bool(volume.GetAttribute(NEURO_PREFIX + "Template"))
            self.addSubjectButton.enabled = (normalized and self.addDbCombo.count > 0
                                             and self.addReferenceCombo.count > 0)
            anatomy = self._normalizedAnatomy()
            self.addAnatomyCheck.enabled = anatomy is not None
            self.addAnatomyCheck.text = (f"Add the CT/MR too ('{anatomy.GetName()}')" if anatomy is not None else
                                         "Add the CT/MR too (none normalized with the image)")
            if volume is not None and not normalized:
                self.addLabel.text = "Normalize the image (step 3) to add it to a database."
            elif self.addDbCombo.count == 0 and not self.addLabel.text:
                self.addLabel.text = ("No user defined database in the database folder (create one in the Neuro PET "
                                      "Database Manager).")

    def _nativeImage(self, volume):
        """(the image before its normalization, temporary): from the normalization's Harden (its undo record: the
        voxels an Overwrite replaced, or the original of a Save separately); (None, False) when not known."""
        record = getattr(self, "normUndo", None) or {}
        undo = record.get("undo") or {}
        if undo.get("action") == reg.OVERWRITE and undo.get("node") is volume and undo.get("image") is not None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", f"{volume.GetName()} native")
            node.SetIJKToRASMatrix(_vtkMatrix(undo["matrix"]))
            node.SetAndObserveImageData(undo["image"])
            for key in volume.GetAttributeNames() or ():
                node.SetAttribute(key, volume.GetAttribute(key))
            for key, value in (undo.get("attributes") or {}).items():     # as before the Harden
                if value is None:
                    node.RemoveAttribute(key)
                else:
                    node.SetAttribute(key, value)
            node.RemoveAttribute(NORMALIZATION_ATTRIBUTE)
            return node, True
        previous = record.get("previous")
        if previous is not None and previous is not volume and slicer.mrmlScene.IsNodePresent(previous):
            return previous, False
        return None, False

    def _normalizedAnatomy(self):
        """The patient's CT / MR normalized with the image at Harden (still in the scene), else None."""
        node = (getattr(self, "normUndo", None) or {}).get("anatomy")
        return node if node is not None and slicer.mrmlScene.IsNodePresent(node) else None

    def onAddToDatabase(self):
        volume = self.volume()
        folder = self.addDbCombo.currentData or ""
        header = self._header(folder)
        if volume is None or header is None:
            return
        subjectId = self.subjectIdEdit.text.strip()
        if not subjectId:
            slicer.util.warningDisplay("Type the subject ID.", windowTitle="Add to database")
            return
        segmentation, segmentId, referenceName = self._addReference(folder, header)
        gridPath = sn.referenceMaskPath(folder, header) or (self.atlas or {}).get(neuro.BRAIN_MASK)
        if not gridPath:
            slicer.util.errorDisplay(f"'{header['name']}' has no reference mask and the space has no brain mask: no "
                                     "database grid.", windowTitle="Add to database")
            return
        if segmentId is None and not sn.referenceMaskPath(folder, header):
            slicer.util.errorDisplay("Pick the reference region (load the regions in step 4).",
                                     windowTitle="Add to database")
            return
        age, sex = self.logic.ageAndSex(volume)
        info = self.logic.imageInfo(volume)
        notes = []
        rank, problems = sn.databaseMatch(header, info)
        if problems:
            notes.append("The database differs from the image: " + "; ".join(problems) + ".")
        notes.extend(sn.unitWarnings(info, header))
        used = self.usedNormalization(volume)
        if used is None:
            notes.append("The image's normalization settings are not known.")
        else:
            differences = normaldb.normalizationDifferences(header.get("normalization"), used)
            if differences:
                notes.append("Normalized otherwise than the database: " + "; ".join(differences) + ".")
        expected = sn.regionSettings(folder, header)["referenceTitle"]
        if segmentId and expected and not _sameRegion(referenceName, expected):
            notes.append(f"The reference region '{referenceName}' is not the database's ({expected}).")
        if age is None:
            notes.append("The age is not known" + (" (the database has age groups)." if header.get("ageStratified")
                                                   else "."))
        replace = normaldb.hasSubject(folder, subjectId)
        if replace:
            notes.append(f"Subject '{subjectId}' is already in the database: it will be replaced.")
        normalizedAnatomy = self._normalizedAnatomy() if self.addAnatomyCheck.checked else None
        anatomy = self.anatomy() if normalizedAnatomy is not None else None
        if normalizedAnatomy is not None and anatomy is None:
            notes.append("The CT/MR of step 1 is not in the scene any more: only its normalized copy is added.")
        contrast = self.contrast()[0] if normalizedAnatomy is not None else ""
        if normalizedAnatomy is not None:
            notes.append(f"With its {contrast or 'CT/MR'} ('{normalizedAnatomy.GetName()}').")
        text = f"Add '{volume.GetName()}' to '{header['name']}' as '{subjectId}'?"
        if notes:
            text += "\n\n" + "\n".join("- " + n for n in notes)
        if not slicer.util.confirmOkCancelDisplay(text, windowTitle="Add to database"):
            return
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        native, temporary = None, False
        try:
            native, temporary = self._nativeImage(volume)
            placement = None
            if anatomy is not None:
                placement = {"how": ("registered in step 2" if self.logic.petInAnatomySpace(native or volume, anatomy)
                                     else "taken as aligned (PET/CT, PET/MR)"), "file": anatomy.GetName()}
            row = self.logic.addSubject(folder, header, volume, subjectId, gridPath, age, sex, used, replace,
                                        segmentation, segmentId, referenceName, native=native,
                                        anatomy=normalizedAnatomy, contrast=contrast, anatomyNative=anatomy,
                                        placement=placement)
        except Exception as error:
            logging.exception("Static Neuro: could not add the subject")
            slicer.util.errorDisplay(f"Could not add the patient to the database:\n{error}")
            return
        finally:
            qt.QApplication.restoreOverrideCursor()
            if temporary and native is not None and slicer.mrmlScene.IsNodePresent(native):
                slicer.mrmlScene.RemoveNode(native)
        self._fillDatabases(force=True)
        count = normaldb.subjectCount(folder, normaldb.readHeader(folder))
        self.addLabel.text = (f"'{subjectId}' added to '{header['name']}' (now {count} subjects; reference mean "
                              f"{float(row['referenceMean']):.4g}). Its mean / SD maps are out of date: compute them "
                              "again in the Neuro PET Database Manager.")
