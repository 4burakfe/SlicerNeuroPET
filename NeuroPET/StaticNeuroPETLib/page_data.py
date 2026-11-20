"""
Step 1: data (the image, the CT/MR, import from a file, patient, age and sex).

Part of the Static Neuro PET/SPECT module (StaticNeuroPET.py): the methods of DataPageMixin, mixed into
StaticNeuroPETWidget there.
"""

import logging
import os

import qt
import slicer

from DynamicNeuroPET import ATLAS_FILE_ATTRIBUTE, DynamicNeuroPETWidget, NEURO_PREFIX, REG_PREFIX, STEP_DATA
from NeuroPETLib import staticneuro as sn

from StaticNeuroPETLib.constants import SETTINGS_IMPORT_FOLDER, WARNING_STYLE


class DataPageMixin:
    """The page data methods of StaticNeuroPETWidget."""

    # --- page 1: data --------------------------------------------------------------------------------
    def _buildDataPage(self):
        page = qt.QWidget()
        layout = qt.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        patientBox = qt.QGroupBox("Patient")
        patientForm = qt.QFormLayout(patientBox)
        self.patientNameEdit = qt.QLineEdit()
        self.patientNameEdit.toolTip = ("Name of the patient (from the DICOM tags; edit it if it is missing or wrong). "
                                        "Shown on the toolbar from the next step on.")
        patientForm.addRow("Name:", self.patientNameEdit)
        self.patientIdEdit = qt.QLineEdit()
        self.patientIdEdit.toolTip = ("Patient ID (from the DICOM tags, or the BIDS subject of an imported file; edit "
                                      "it if it is missing or wrong)")
        patientForm.addRow("Patient ID:", self.patientIdEdit)
        ageRow = qt.QHBoxLayout()
        self.ageSpin = qt.QDoubleSpinBox()
        self.ageSpin.setRange(-1.0, 130.0)
        self.ageSpin.setDecimals(0)
        self.ageSpin.specialValueText = "Unknown"
        self.ageSpin.suffix = " years"
        self.ageSpin.toolTip = "Age of the patient at the scan (from the tags; edit it if it is missing or wrong)"
        self.sexCombo = qt.QComboBox()
        for key, title in sn.SEXES:
            self.sexCombo.addItem(title, key)
        self.sexCombo.toolTip = "Sex of the patient (from the tags; edit it if it is missing or wrong)"
        ageRow.addWidget(self.ageSpin)
        ageRow.addWidget(qt.QLabel("Sex:"))
        ageRow.addWidget(self.sexCombo)
        patientForm.addRow("Age:", ageRow)
        box = qt.QGroupBox("Images")
        form = qt.QFormLayout(box)
        self.sequenceSelector = self._nodeSelector(["vtkMRMLScalarVolumeNode"],
                                                   "Static PET (PET4D DICOM plugin, e.g. '<series> static SUVbw') or "
                                                   "SPECT (DICOM SPECT reconstruction) of the brain")
        self.sequenceSelector.hideChildNodeTypes = ["vtkMRMLLabelMapVolumeNode"]
        form.addRow("PET / SPECT:", self.sequenceSelector)
        importButton = qt.QPushButton("Import from file...")
        importButton.toolTip = ("Load a PET / SPECT image file (NIfTI, NRRD, ...). BIDS names (sub-<id>_ses-<session>_"
                                "pet.nii.gz) are read with their .json sidecar (tracer, nuclide, units, timing) and "
                                "the participants.tsv of the dataset (age, sex); a session 'SUV' means SUV units.")
        importButton.connect("clicked()", self.onImportFile)
        form.addRow("", importButton)
        self.sequenceInfo = qt.QLabel()
        self.sequenceInfo.wordWrap = True
        form.addRow(self.sequenceInfo)
        self.anatomySelector = self._nodeSelector(["vtkMRMLScalarVolumeNode"],
                                                  "CT or MR of the same patient (optional: registration in step 2; "
                                                  "normalized with the image in step 3, and an MR may be the one "
                                                  "normalized)")
        self.anatomySelector.hideChildNodeTypes = ["vtkMRMLLabelMapVolumeNode"]
        form.addRow("CT / MR:", self.anatomySelector)
        self.anatomyInfo = qt.QLabel()
        self.anatomyInfo.wordWrap = True
        form.addRow(self.anatomyInfo)
        self.contrastCombo = qt.QComboBox()
        for key, title in (("", "Automatic"), ("T1", "T1"), ("T2", "T2"), ("PD", "PD"), ("CT", "CT")):
            self.contrastCombo.addItem(title, key)
        self.contrastCombo.toolTip = ("Contrast of the CT/MR, read from the DICOM tags. It picks the template of "
                                      "step 3 when the MR is normalized (a CT is not: the PET/SPECT is).")
        form.addRow("Contrast:", self.contrastCombo)
        self.extraSelector = slicer.qMRMLCheckableNodeComboBox()
        self.extraSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.extraSelector.hideChildNodeTypes = ["vtkMRMLLabelMapVolumeNode"]
        self.extraSelector.setMRMLScene(slicer.mrmlScene)
        self.extraSelector.toolTip = ("Other sequences of the patient lying on the CT/MR (e.g. FLAIR, T2, the CT of a "
                                      "PET/CT next to the MR): normalized with the same transform in step 3, for the "
                                      "fusion of the review")
        form.addRow("Other sequences:", self.extraSelector)
        layout.addWidget(patientBox)
        layout.addWidget(box)
        dicomButton = qt.QPushButton("Open the DICOM browser")
        dicomButton.toolTip = "Load the static PET / SPECT and the CT/MR from DICOM"
        dicomButton.connect("clicked()", lambda: slicer.util.selectModule("DICOM"))
        layout.addWidget(dicomButton)
        note = qt.QLabel("Static brain PET (SUV) or SPECT (counts, Bq/ml or SUV): optionally registered to the CT/MR, "
                         "normalized to the MNI ICBM 152 2009c space, the CerebrA regions loaded and compared with a "
                         "normal database. Steps can be skipped; without a database the review gives the regional "
                         "values only.")
        note.wordWrap = True
        note.setStyleSheet("color: #7d868d")
        layout.addWidget(note)
        layout.addStretch(1)
        self.sequenceSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onDataChanged)
        self.anatomySelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onDataChanged)
        self.contrastCombo.connect("currentIndexChanged(int)", lambda i: self.onDataChanged())
        self.ageSpin.connect("valueChanged(double)", lambda v: self.onAgeSexChanged())
        self.sexCombo.connect("currentIndexChanged(int)", lambda i: self.onAgeSexChanged())
        self.patientNameEdit.connect("textEdited(QString)", lambda text: self.onPatientChanged())
        self.patientIdEdit.connect("textEdited(QString)", lambda text: self.onPatientChanged())
        return page

    def extraVolumes(self):
        """The other sequences picked on the Data page (not the image or the CT/MR)."""
        if not hasattr(self, "extraSelector"):
            return []
        try:
            nodes = list(self.extraSelector.checkedNodes())
        except Exception:
            return []
        return [n for n in nodes if n is not None and n is not self.volume() and n is not self.anatomy()]

    def selectInputs(self):
        if self.volume() is None:
            nodes = [n for n in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode") if self.logic.isStaticImage(n)]
            original = [n for n in nodes if not n.GetAttribute(REG_PREFIX + "Fixed")
                        and not n.GetAttribute(NEURO_PREFIX + "Template")]
            if nodes:
                self.setNode("Sequence", (original or nodes)[-1])
        if self.anatomy() is None:
            from DynamicPETRegistration import fixedCandidates
            candidates = [n for n in fixedCandidates() if not n.GetAttribute(ATLAS_FILE_ATTRIBUTE)
                          and not n.GetAttribute(NEURO_PREFIX + "Template") and n is not self.volume()
                          and not self.logic.imageInfo(n)["modality"]]
            if candidates:
                self.setNode("Anatomy", candidates[0])
        self._syncDataPage()

    def onImportFile(self):
        settings = qt.QSettings()
        folder = str(settings.value(SETTINGS_IMPORT_FOLDER) or "")
        paths = qt.QFileDialog.getOpenFileNames(slicer.util.mainWindow(), "Import PET / SPECT image", folder,
                                                "Images (*.nii *.nii.gz *.nrrd *.nhdr *.mha *.mhd);;All files (*)")
        paths = [str(p) for p in (paths or [])]
        if not paths:
            return
        settings.setValue(SETTINGS_IMPORT_FOLDER, os.path.dirname(paths[0]))
        node = None
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            for path in paths:
                node = self.logic.loadFromFile(path)
        except Exception as error:
            logging.exception("Static Neuro: import failed")
            slicer.util.errorDisplay(f"Could not import the image:\n{error}")
        finally:
            qt.QApplication.restoreOverrideCursor()
        if node is not None:
            self.sequenceSelector.setCurrentNode(node)
            slicer.util.setSliceViewerLayers(background=node, fit=True)

    def onDataChanged(self, *args):
        if self._updating:
            return
        if self.sequenceSelector.currentNode() is not self.volume() or \
                self.anatomySelector.currentNode() is not self.anatomy():
            self._clearRegistration()
            self.regUndo = None
        DynamicNeuroPETWidget.onDataChanged(self, *args)
        self._databaseKey = None
        self._syncAgeSex()

    def _syncDataPage(self):
        DynamicNeuroPETWidget._syncDataPage(self)
        self._syncAgeSex()

    def _syncAgeSex(self):
        volume = self.volume()
        age, sex = self.logic.ageAndSex(volume)
        name, patientId = self.logic.patientNameAndId(volume)
        self._updating = True
        try:
            self.ageSpin.value = age if age is not None else -1.0
            self.sexCombo.setCurrentIndex(max(0, self.sexCombo.findData(sex or "")))
            self.patientNameEdit.text = name
            self.patientIdEdit.text = patientId
            for widget in (self.ageSpin, self.sexCombo, self.patientNameEdit, self.patientIdEdit):
                widget.enabled = volume is not None
        finally:
            self._updating = False
        if volume is not None:
            self._patientTexts[volume.GetID()] = sn.patientText(name, patientId)
        self._updatePatientLabel()

    def onPatientChanged(self):
        volume = self.volume()
        if self._updating or volume is None:
            return
        self.logic.setPatientNameAndId(volume, self.patientNameEdit.text, self.patientIdEdit.text)
        self._patientTexts[volume.GetID()] = sn.patientText(self.patientNameEdit.text, self.patientIdEdit.text)
        self._updatePatientLabel()

    def _patientText(self):
        """'Name (ID)' of the image's patient (read once per volume)."""
        volume = self.volume()
        if volume is None:
            return ""
        if volume.GetID() not in self._patientTexts:
            self._patientTexts[volume.GetID()] = sn.patientText(*self.logic.patientNameAndId(volume))
        return self._patientTexts[volume.GetID()]

    def _updatePatientLabel(self):
        """The patient's name and ID right of the module's name on the toolbar, from the second step on (under the
        step title when there is no toolbar)."""
        text = self._patientText() if getattr(self, "current", STEP_DATA) != STEP_DATA else ""
        label = getattr(self, "patientLabel", None)
        if label is not None:
            label.text = text
            label.toolTip = f"Patient: {text}" if text else ""
            self._patientAction.setVisible(bool(text))
        elif text and hasattr(self, "titleLabel") and "Patient:" not in self.titleLabel.text:
            self.titleLabel.text += f"<br><span style='color:#2f7ed8'>Patient: {text}</span>"

    def onAgeSexChanged(self):
        if self._updating or self.volume() is None:
            return
        age = float(self.ageSpin.value)
        self.logic.setAgeAndSex(self.volume(), age if age >= 0 else None, self.sexCombo.currentData or "")

    def _updateDataInfo(self):
        volume = self.volume()
        if volume is None:
            self.sequenceInfo.text = "Pick the static PET / SPECT, or import it from a file."
        else:
            info = self.logic.imageInfo(volume)
            what = " ".join(t for t in (info["modality"] or "Image", info["nuclide"], info["tracer"]) if t)
            parts = [what, f"units {info['units'] or 'unknown'}"]
            if info["source"]:
                parts.append(f"from {info['source']}")
            done = []
            if volume.GetAttribute(REG_PREFIX + "Fixed"):
                done.append(f"registered to '{volume.GetAttribute(REG_PREFIX + 'Fixed')}'")
            if volume.GetAttribute(NEURO_PREFIX + "Template"):
                done.append("normalized")
            text = ", ".join(parts) + (f"; {', '.join(done)}" if done else "") + "."
            warnings = sn.unitWarnings(info)
            if warnings:
                text += "<br>" + WARNING_STYLE.format(" ".join(warnings))
            self.sequenceInfo.text = text
        anatomy = self.anatomy()
        if anatomy is None:
            self.anatomyInfo.text = "No CT/MR: step 2 is skipped and the PET/SPECT is normalized."
        else:
            contrast, reason = self.contrast()
            if contrast == "CT":
                self.anatomyInfo.text = (f"Recognised as <b>CT</b> ({reason}): by default the PET/SPECT is "
                                         "normalized and the CT follows it; 'CT/MR to MR template' registers the "
                                         "CT to the CT template of the space (CT) when it has one.")
            else:
                self.anatomyInfo.text = (f"Recognised as <b>{contrast}</b> ({reason})." if contrast else
                                         f"Contrast not recognised ({reason}); the T1 template is used unless you "
                                         "pick one.")
