"""
Step 3: normalization: the method (CT/MR or PET/SPECT), the template, the normal database and the
normalization settings stored with a database.

Part of the Static Neuro PET/SPECT module (StaticNeuroPET.py): the methods of NormalizationPageMixin, mixed into
StaticNeuroPETWidget there; the normal database part (the pick, its settings as the default, the PET template) is
NeuroWorkflowLib.database.NormalizationDatabaseMixin, shared with the Dynamic Neuro module.
"""

import json

from NeuroWorkflowLib.base import NeuroWorkflowWidget, METHOD_ANATOMY, METHOD_PET
from NeuroWorkflowLib.database import NormalizationDatabaseMixin
from NeuroPETLib import staticneuro as sn

from StaticNeuroPETLib.constants import NORMALIZATION_ATTRIBUTE


class NormalizationPageMixin(NormalizationDatabaseMixin):
    """The page normalization methods of StaticNeuroPETWidget (the normal database: NeuroWorkflowLib.database)."""

    def _databaseUnitWarnings(self, info, header):
        return sn.unitWarnings(info, header)

    def _databasesChanged(self):
        self._fillAddDatabases()

    # --- the CT/MR: an MR is normalized to the MR template as it is, a CT as its brain only (skull stripped) -----
    def contrast(self):
        """(contrast, reason) of the CT/MR: the user's choice, else read from the DICOM tags (once per CT/MR)."""
        chosen = self.contrastCombo.currentData if hasattr(self, "contrastCombo") else ""
        if chosen:
            return chosen, "chosen"
        anatomy = self.anatomy()
        key = anatomy.GetID() if anatomy is not None else ""
        if self._contrastFor is None or self._contrastFor[0] != key:
            self._contrastFor = (key, self.logic.anatomyContrast(anatomy))
        return self._contrastFor[1]

    def _anatomyIsMr(self):
        """Whether the CT/MR is an MR: then it is the default way to the MR template (a CT can be picked: its brain
        only)."""
        return self.anatomy() is not None and self.contrast()[0] != "CT"

    def _updateMethodItems(self):
        """The CT/MR to MR template method offered with a CT/MR (a CT is registered as its brain only, skull stripped,
        to the skull-stripped template); without one the PET/SPECT is normalized."""
        if not hasattr(self, "methodCombo"):
            return
        allowed = self.anatomy() is not None
        index = self.methodCombo.findData(METHOD_ANATOMY)
        item = self.methodCombo.model().item(index) if index >= 0 else None
        if item is not None:
            item.setEnabled(allowed)
            item.setToolTip("Pick the CT/MR in step 1." if not allowed else (
                "With a CT template in the space (CT, with CT_GM) the CT is registered to it (CT to CT), its MNI "
                "space put on the space's through CT_GM; else with its bone (over 100 HU) black as in the MR."
                if self._anatomyIsCt() else ""))
        if not allowed and self.method() == METHOD_ANATOMY and self.normResult is None:
            self.methodCombo.setCurrentIndex(self.methodCombo.findData(METHOD_PET))

    def _defaultMethod(self):
        """The MR when the study has one, else the PET/SPECT (to the MR template, or to the database's template when
        its settings say so); until the method is picked by hand."""
        self._updateMethodItems()
        if getattr(self, "_methodChosen", False) or not hasattr(self, "methodCombo") or self.normResult is not None:
            return
        wanted = METHOD_ANATOMY if self._anatomyIsMr() else METHOD_PET
        if self.methodCombo.currentData != wanted:
            self.methodCombo.setCurrentIndex(self.methodCombo.findData(wanted))

    @staticmethod
    def _normHardenText(volume, template, method):
        return (f"Resample '{volume.GetName()}' through the normalization onto the field of view of the template "
                f"'{template.GetName()}'.\nThe patient's CT/MR (and the other sequences of step 1) are normalized "
                "too with the same transform (on the template grid)."
                "\n\nOverwrite (default) replaces the image in the patient space (Revert puts it back); Save "
                "separately keeps it next to the normalized one.")

    def _hardenNormalization(self, volume, template, choice, description, method):
        return self.logic.harden(volume, template, self.normSession.linear, choice["action"], choice["grid"],
                                 choice["spacing"], choice["interpolation"], description, method, self.anatomy(),
                                 self.extraVolumes())

    def onNormHarden(self):
        settings = (self.normResult or {}).get("settings")
        undo = self.normUndo
        method = (self.normResult or {}).get("method")
        NeuroWorkflowWidget.onNormHarden(self)
        volume = self.volume()
        if self.normUndo is not undo and volume is not None:
            if settings:
                volume.SetAttribute(NORMALIZATION_ATTRIBUTE, json.dumps(settings))
            notes = []
            extras = (self.normUndo.get("undo") or {}).get("extras") or []
            if extras:
                notes.append("Other sequences: " + ", ".join(f"'{n.GetName()}'" for n in extras) + ".")
            anatomy = self.anatomy()
            if (anatomy is not None and method != METHOD_ANATOMY
                    and not self.logic.petInAnatomySpace(self.normUndo["previous"], anatomy)):
                notes.append(f"'{anatomy.GetName()}' was not registered in step 2: it is taken as lying on the image "
                             "(PET/CT, PET/MR).")
            if notes:
                self.normLabel.text += "\n" + " ".join(notes)
        self._updateDatabaseInfo()
