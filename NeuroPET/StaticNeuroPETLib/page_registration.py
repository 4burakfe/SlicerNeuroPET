"""
Step 2: CT/MR registration (optional).

Part of the Static Neuro PET/SPECT module (StaticNeuroPET.py): the methods of RegistrationPageMixin, mixed into
StaticNeuroPETWidget there.
"""

import logging

import ctk
import numpy as np
import qt
import slicer

from DICOMGatedDynamicPETLib import registration as reg
from DICOMGatedDynamicPETLib.hardendialog import hardenDialog
from NeuroWorkflowLib.base import _button, _ijkToRas, _percentiles

from StaticNeuroPETLib.constants import REG_LINEAR


class RegistrationPageMixin:
    """The page registration methods of StaticNeuroPETWidget."""

    # --- page 2: CT/MR registration --------------------------------------------------------------
    def _buildRegPage(self):
        from DynamicPETRegistration import iterativeOptions
        page = qt.QWidget()
        layout = qt.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        box = qt.QGroupBox("Registration to the CT / MR")
        form = qt.QFormLayout(box)
        self.regInfo = qt.QLabel()
        self.regInfo.wordWrap = True
        form.addRow(self.regInfo)
        # the patient's own CT / MR: rigid only (iterative multi-resolution; no scaling, no deformable), the other
        # engines and stages not offered
        self.regEngineCombo = qt.QComboBox()
        for title, key in reg.ENGINES:
            self.regEngineCombo.addItem(title, key)
        self.regEngineCombo.setCurrentIndex(self.regEngineCombo.findData(reg.ENGINE_ITERATIVE))
        self.regEngineCombo.visible = False
        self.regIterativeBox = qt.QWidget()
        iterativeForm = qt.QFormLayout(self.regIterativeBox)
        iterativeForm.setContentsMargins(0, 0, 0, 0)
        self.regIterative = iterativeOptions(iterativeForm, linear=REG_LINEAR, deformable=False)
        for key in ("linear", "deformable", "grid"):        # rigid, no deformable passes
            self._setRowVisible(self.regIterative[key], False)
        form.addRow(self.regIterativeBox)
        self.regInitCombo = qt.QComboBox()
        for title, mode in reg.INITIALIZATIONS:
            self.regInitCombo.addItem(title, mode)
        self.regInitCombo.setCurrentIndex(self.regInitCombo.findData(reg.DEFAULT_INITIALIZATION))
        form.addRow("Initialization:", self.regInitCombo)
        self.registerButton = _button("Register")
        self.registerButton.toolTip = "Register the image to the CT/MR and show it in the registration layout"
        form.addRow(self.registerButton)
        row = qt.QHBoxLayout()
        self.regManualButton = qt.QPushButton("Correct manually")
        self.regManualButton.checkable = True
        self.regRevertButton = _button("Revert", "revert")
        self.regRevertButton.toolTip = ("Before Harden: drop the registration. After Harden: remove the registered "
                                        "image (or put an overwritten one back).")
        self.regHardenButton = _button("Harden", "harden")
        self.regHardenButton.toolTip = "Resample the image onto the CT/MR field of view"
        row.addWidget(self.regManualButton)
        row.addWidget(self.regRevertButton)
        row.addWidget(self.regHardenButton)
        form.addRow(row)
        self.regLabel = qt.QLabel("Not registered. This step is optional (e.g. skip it for a PET/MR, or when "
                                  "normalizing the PET/SPECT itself).")
        self.regLabel.wordWrap = True
        self.regLabel.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        form.addRow(self.regLabel)
        layout.addWidget(box)
        viewBox = ctk.ctkCollapsibleButton()
        viewBox.text = "View"
        viewBox.collapsed = True
        view = qt.QFormLayout(viewBox)
        self.regOpacity = ctk.ctkSliderWidget()
        self.regOpacity.minimum, self.regOpacity.maximum = 0.0, 1.0
        self.regOpacity.singleStep, self.regOpacity.decimals = 0.05, 2
        self.regOpacity.value = 0.5
        view.addRow("Fusion opacity:", self.regOpacity)
        self.regCheckerCheck = qt.QCheckBox("Checkerboard column")
        view.addRow(self.regCheckerCheck)
        layout.addWidget(viewBox)
        layout.addStretch(1)
        self.regEngineCombo.connect("currentIndexChanged(int)", lambda i: self._updateRegEngine())
        self.registerButton.connect("clicked()", self.onRegister)
        self.regManualButton.connect("toggled(bool)", lambda on: self.regSession and
                                     self.regSession.setManualHandles(on))
        self.regRevertButton.connect("clicked()", self.onRegRevert)
        self.regHardenButton.connect("clicked()", self.onRegHarden)
        self.regOpacity.connect("valueChanged(double)", lambda v: self.regSession and
                                self.regSession.setFusionOpacity(v))
        self.regCheckerCheck.connect("toggled(bool)", lambda on: self.regSession and
                                     self.regSession.setShowCheckerboard(on))
        self._updateRegEngine()
        return page

    def _updateRegEngine(self):
        self.regIterativeBox.visible = True         # iterative rigid only

    def regSettings(self):
        from DynamicPETRegistration import DynamicPETRegistrationLogic, iterativeSettings
        settings = DynamicPETRegistrationLogic.defaultSettings()
        settings["initialization"] = self.regInitCombo.currentData
        # the patient's own CT / MR: iterative rigid only (no scaling, no deformable)
        settings.update(iterativeSettings(self.regIterative), engine=reg.ENGINE_ITERATIVE,
                        iterativeLinear=REG_LINEAR, iterativeDeformable=False)
        settings["steps"] = reg.iterativeSteps(REG_LINEAR, False)
        return settings

    def onRegister(self):
        volume, anatomy = self.volume(), self.anatomy()
        if volume is None or anatomy is None:
            slicer.util.warningDisplay("Pick the PET / SPECT and the CT/MR in step 1.", windowTitle="Registration")
            return
        settings = self.regSettings()
        self._clearRegistration()
        dialog = slicer.util.createProgressDialog(windowTitle="Registration", labelText="Registering...",
                                                  maximum=len(settings["steps"]))

        def progress(index, total, text):
            dialog.setValue(index)
            dialog.setLabelText(f"Registering: {text}")
            slicer.app.processEvents()
            return not dialog.wasCanceled

        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            result = self.logic.registerToAnatomy(volume, anatomy, settings, progress)
        except Exception as error:
            logging.exception("Static Neuro: registration failed")
            slicer.util.errorDisplay(f"Registration failed:\n{error}")
            return
        finally:
            qt.QApplication.restoreOverrideCursor()
            dialog.close()
            dialog.deleteLater()
        if not result["steps"]:
            self._dropResult(result)
            self.regLabel.text = "Stopped before the first step."
            self._updateRegButtons()
            return
        self.regResult = result
        self._openRegSession()
        lines = reg.summaryLines(result["steps"], result["linear"], result["center"], result["deformation"])
        notes = result.get("notes") or []
        self.regLabel.text = (f"'{volume.GetName()}' registered to '{anatomy.GetName()}'"
                              + (" (stopped)" if result["stopped"] else "") + ":\n" + "\n".join(lines)
                              + ("\n" + " ".join(notes) if notes else "")
                              + "\nCheck it in the views, correct it by hand if needed, then Harden.")
        self._updateRegButtons()

    @staticmethod
    def _dropResult(result):
        bspline = (result or {}).get("bspline")
        if bspline is not None and slicer.mrmlScene.IsNodePresent(bspline):
            slicer.mrmlScene.RemoveNode(bspline)

    def _openRegSession(self):
        from DynamicPETRegistration import RegistrationSession, DynamicPETRegistrationWidget
        self._closeRegSession()
        volume, anatomy = self.volume(), self.anatomy()
        if self.regResult is None or volume is None or anatomy is None:
            return
        if DynamicPETRegistrationWidget._closeOtherModules() is False:
            return              # Review/Edit has unsaved edits and the user keeps it open
        session = None
        try:
            session = RegistrationSession(anatomy)
            session.fusionOpacity = self.regOpacity.value
            session.showCheckerboard = self.regCheckerCheck.checked
            session.open(volume)
            session.setRegistration(self.regResult["linear"], self.regResult["bspline"])
            session.setFixedWindow(*_percentiles(anatomy))
            self.regSession = session
        except Exception as error:
            logging.exception("Static Neuro: could not open the registration layout")
            if session is not None:
                self._closeHalfOpen(session)
            slicer.util.errorDisplay(f"Could not open the registration layout:\n{error}")

    def _closeRegSession(self):
        if self.regSession is not None:
            if self.regManualButton.checked:
                self.regManualButton.blockSignals(True)
                self.regManualButton.checked = False
                self.regManualButton.blockSignals(False)
            self.regSession.close()
            self.regSession = None

    def _clearRegistration(self):
        """Drop a pending (not hardened) registration."""
        if not hasattr(self, "regLabel"):
            return
        self._closeRegSession()
        if self.regResult is not None:
            self._dropResult(self.regResult)
            self.regResult = None
            self.regLabel.text = "Not registered."
        self._updateRegButtons()

    def _updateRegButtons(self):
        if not hasattr(self, "regHardenButton"):
            return
        pending = self.regResult is not None and self.regSession is not None
        self.regHardenButton.enabled = pending
        self.regManualButton.enabled = pending
        self.regRevertButton.enabled = pending or self.regUndo is not None
        self.registerButton.enabled = self.volume() is not None and self.anatomy() is not None
        volume, anatomy = self.volume(), self.anatomy()
        if volume is None:
            self.regInfo.text = "Pick the PET / SPECT in step 1."
        elif anatomy is None:
            self.regInfo.text = "No CT/MR picked in step 1: skip this step."
        elif self.logic.petInAnatomySpace(volume, anatomy):
            self.regInfo.text = f"'{volume.GetName()}' is registered to '{anatomy.GetName()}'."
        else:
            self.regInfo.text = f"Register '{volume.GetName()}' to '{anatomy.GetName()}'."

    def onRegHarden(self):
        volume, anatomy = self.volume(), self.anatomy()
        if self.regResult is None or self.regSession is None or volume is None or anatomy is None:
            return
        text = (f"Resample '{volume.GetName()}' through the registration onto the field of view of "
                f"'{anatomy.GetName()}', grown to the image's where it is larger (not with the CT/MR grid option)."
                "\n\nSave separately (default) keeps the original next to the registered image.")
        choice = hardenDialog("Harden registration", text, reg.OUTPUT_GRIDS, reg.gridSpacing(_ijkToRas(volume)),
                              [("Save separately", reg.SAVE_SEPARATELY), ("Overwrite", reg.OVERWRITE)],
                              slicer.util.mainWindow())
        if choice is None:
            return
        description = (f"Registered to {anatomy.GetName()} ({reg.stepsText(self.regResult['steps'])}"
                       + ("" if np.allclose(self.regSession.manualMatrix(), np.eye(4)) else ", corrected by hand")
                       + ")")
        self.regManualButton.checked = False
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            undo, node = self.logic.hardenRegistration(volume, anatomy, self.regSession.linear, choice["action"],
                                                       choice["grid"], choice["spacing"], choice["interpolation"],
                                                       description)
        except Exception as error:
            logging.exception("Static Neuro: could not harden the registration")
            slicer.util.errorDisplay(f"Could not harden the registration:\n{error}")
            return
        finally:
            qt.QApplication.restoreOverrideCursor()
        self._clearRegistration()
        self.regUndo = {"undo": undo, "previous": volume}
        self.setNode("Sequence", node)
        self._clearNormalization()
        self.regLabel.text = (description + ".\n" + (f"Overwrote '{volume.GetName()}'" if choice["action"] ==
                                                     reg.OVERWRITE else f"Saved as '{node.GetName()}'") + ".")
        try:
            slicer.util.setSliceViewerLayers(background=anatomy, foreground=node, foregroundOpacity=0.5, fit=True)
        except Exception:
            pass
        self._updateRegButtons()
        self.refresh()

    def onRegRevert(self):
        if self.regResult is not None:
            self._clearRegistration()
            self.regLabel.text = "Not registered."
            return
        if self.regUndo is None:
            return
        record = self.regUndo
        try:
            self.logic.revertHarden(record["undo"])
        except Exception as error:
            logging.exception("Static Neuro: could not revert the registration")
            slicer.util.errorDisplay(f"Could not revert the registration:\n{error}")
            return
        if slicer.mrmlScene.IsNodePresent(record["previous"]):
            self.setNode("Sequence", record["previous"])
        self.regUndo = None
        self._clearNormalization()
        self.regLabel.text = "Reverted: the image is as it was."
        self._updateRegButtons()
        self.refresh()
