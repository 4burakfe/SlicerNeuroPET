"""
Step 3: normalization: the method (CT/MR or PET/SPECT), the template, the normal database and the
normalization settings stored with a database.

Part of the Static Neuro PET/SPECT module (StaticNeuroPET.py): the methods of NormalizationPageMixin, mixed into
StaticNeuroPETWidget there.
"""

import json
import logging

import ctk
import qt
import slicer

from DICOMGatedDynamicPETLib import registration as reg
from NeuroWorkflowLib.base import (NeuroWorkflowWidget, METHOD_ANATOMY, METHOD_PET, METHOD_PET_TEMPLATE, NEURO_PREFIX,
                                   PET_TEMPLATE_ATTRIBUTE)
from NeuroPETLib import credits, neuro, normaldb, staticneuro as sn

from StaticNeuroPETLib.constants import DATABASE_PARAMETER, NORMALIZATION_ATTRIBUTE, WARNING_STYLE


class NormalizationPageMixin:
    """The page normalization methods of StaticNeuroPETWidget."""

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

    # --- page 3: normalization (the Dynamic Neuro page with the normal database and the PET template) -----
    def _buildNormPage(self):
        page = NeuroWorkflowWidget._buildNormPage(self)
        box = qt.QGroupBox("Normal database")
        form = qt.QFormLayout(box)
        self.dbRootPath = ctk.ctkPathLineEdit()
        self.dbRootPath.filters = ctk.ctkPathLineEdit.Dirs
        self.dbRootPath.settingKey = ""
        self.dbRootPath.currentPath = self.logic.databaseRoot()
        self.dbRootPath.toolTip = "Root folder of the normal databases (the same as in the Neuro PET Database Manager)"
        form.addRow("Database folder:", self.dbRootPath)
        dbRow = qt.QHBoxLayout()
        self.dbCombo = qt.QComboBox()
        self.dbCombo.toolTip = ("The normal database the patient is compared with (and may be added to). Those of the "
                                "image's modality, nuclide and tracer are listed first (and picked); the others are "
                                "marked with '!'. Its normalization settings are put on this page as the default.")
        refreshButton = qt.QPushButton("Refresh")
        refreshButton.toolTip = "Read the database folder again"
        dbRow.addWidget(self.dbCombo, 1)
        dbRow.addWidget(refreshButton)
        form.addRow("Database:", dbRow)
        self.dbInfo = qt.QLabel()
        self.dbInfo.wordWrap = True
        form.addRow(self.dbInfo)
        # the database's description (data source, citation, license) and the atlases it is built on (read only)
        self.dbCredits = credits.infoBox(150)
        form.addRow(self.dbCredits)
        page.layout().insertWidget(0, box)
        self.dbRootPath.connect("currentPathChanged(QString)", self.onDatabaseRootChanged)
        refreshButton.connect("clicked()", lambda: self._fillDatabases(force=True))
        self.dbCombo.connect("activated(int)", lambda i: self._setDatabase(self.dbCombo.currentData or "",
                                                                           chosen=True))
        for combo in [self.methodCombo, self.templateCombo, self.normEngineCombo] + list(self.normStepCombos):
            combo.connect("currentIndexChanged(int)", lambda i: self._updateDatabaseInfo())
        self.maskCheck.connect("toggled(bool)", lambda on: self._updateDatabaseInfo())
        for widget in self.normIterative.values():
            signal = "toggled(bool)" if isinstance(widget, qt.QCheckBox) else "currentIndexChanged(int)"
            try:
                widget.connect(signal, lambda *args: self._updateDatabaseInfo())
            except Exception:
                pass
        return page

    def _setRowVisible(self, widget, visible):
        widget.visible = visible
        try:
            label = widget.parentWidget().layout().labelForField(widget)
            if label is not None:
                label.visible = visible
        except Exception:
            pass

    def _updateTemplateChoice(self):
        if not hasattr(self, "templateCombo"):
            return
        petTemplate = self.method() == METHOD_PET_TEMPLATE
        if not petTemplate:
            NeuroWorkflowWidget._updateTemplateChoice(self)
            return
        self._fillTemplateCombo()          # the database's PET template ('From DB: <name> - PET template')
        self._updateTemplateCredits()
        folder = self._database()
        header = self._header(folder)
        if header is None:
            self.templateNote.text = ("Pick a normal database above: its template (a mean map frozen as the "
                                      "template, else its master mean map) is the template.")
        elif not sn.petTemplatePath(folder, header):
            self.templateNote.text = (f"'{header['name']}' has no template or master mean map yet (statistics not "
                                      "computed): pick another method or database.")
        else:
            frozen = normaldb.templatePath(folder, header)
            self.templateNote.text = (f"Registered to the {'template' if frozen else 'master mean map'} of "
                                      f"'{header['name']}' (a "
                                      f"{header.get('tracer') or header.get('nuclide') or ''} template in the MNI "
                                      "2009c space).")

    def _databaseTemplates(self):
        """The volumes of the normal database picked (step 3): its PET template and its CT / MR means."""
        if not hasattr(self, "dbCombo") or not hasattr(self, "databaseRows"):
            return {}, ""
        folder = self._database()
        header = self._header(folder)
        if header is None:
            return {}, ""
        return normaldb.databaseVolumes(folder, header), header.get("name", "")

    def _extraSpaces(self):
        """'From DB (name)' when the space of the normal database picked is not in the atlas folders."""
        if not hasattr(self, "dbCombo") or not hasattr(self, "databaseRows"):
            return []
        header = self._header(self._database())
        space = (header or {}).get("space")
        if not space or neuro.spaceInfo(space):
            return []
        return [(f"From DB ({header.get('name')})", normaldb.DB_TEMPLATE_PREFIX + space)]

    def _updateTemplateCredits(self):
        if not hasattr(self, "templateCredits"):
            return
        if self.method() != METHOD_PET_TEMPLATE:
            NeuroWorkflowWidget._updateTemplateCredits(self)
            return
        try:
            header = self._header(self._database()) if hasattr(self, "dbCombo") else None
        except Exception:       # noqa: BLE001 - while the page is built
            header = None
        self.templateCredits.setHtml(
            "<p>Template: the PET template of the normal database.</p>"
            + credits.databaseHtml(header, "Pick a normal database: its template is registered to."))

    def _template(self):
        if self.method() != METHOD_PET_TEMPLATE:
            return NeuroWorkflowWidget._template(self)
        folder = self._database()
        header = self._header(folder)
        path = sn.petTemplatePath(folder, header) if header else None
        if not path:
            raise ValueError("Pick a normal database with a template (or a master mean map).")
        template = self.logic.loadAtlasVolume(path, name=f"{header['name']} template")
        template.SetAttribute(PET_TEMPLATE_ATTRIBUTE, "1")      # a PET: shown in inverted grey
        self.setNode("Template", template)
        return template

    def _updateNormButtons(self):
        NeuroWorkflowWidget._updateNormButtons(self)
        if hasattr(self, "dbCombo") and self.method() == METHOD_PET_TEMPLATE:
            folder = self._database()
            header = self._header(folder)
            self.normalizeButton.enabled = (self.volume() is not None and header is not None
                                            and bool(sn.petTemplatePath(folder, header)))

    @staticmethod
    def _normHardenText(volume, template, method):
        return (f"Resample '{volume.GetName()}' through the normalization onto the field of view of the template "
                f"'{template.GetName()}'.\nThe patient's CT/MR (and the other sequences of step 1) are normalized "
                "too with the same transform (on the template grid)."
                "\n\nOverwrite (default) replaces the image in the patient space (Revert puts it back); Save "
                "separately keeps it next to the normalized one.")

    # --- normalization settings (stored with a database and on the normalized image) ----------------------
    def normalizationSettings(self):
        """The normalization settings on the page, in the categories of normaldb.NORM_KEYS."""
        from DynamicPETRegistration import iterativeSettings
        method = self.method()
        engine = self.normEngine()
        if engine == reg.ENGINE_ITERATIVE:
            iterative = iterativeSettings(self.normIterative)
            linear, deformable = iterative["iterativeLinear"], bool(iterative["iterativeDeformable"])
        else:
            steps = [c.currentData for c in self.normStepCombos if c.currentData]
            linear = next((s for s in steps if s != "BSpline"), "")
            deformable = "BSpline" in steps
        return {"method": method, "template": "" if method == METHOD_PET_TEMPLATE else
                (self.templateCombo.currentData or ""), "engine": engine, "linear": linear,
                "deformable": deformable, "brainMask": bool(self.maskCheck.checked and self.maskCheck.enabled)}

    @staticmethod
    def usedNormalization(volume):
        """The normalization settings a volume was normalized with (stored at Harden), None when not known."""
        text = volume.GetAttribute(NORMALIZATION_ATTRIBUTE) if volume is not None else None
        try:
            return json.loads(text) if text else None
        except ValueError:
            return None

    def onNormalize(self):
        settings = self.normalizationSettings()
        NeuroWorkflowWidget.onNormalize(self)
        if self.normResult is not None:
            self.normResult["settings"] = settings
        self._updateDatabaseInfo()

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

    def _applyNormalizationSettings(self, header):
        """A database's normalization settings put on the page as the default (not when the image is normalized or
        a normalization is pending); the user may change them."""
        norm = header.get("normalization") or {}
        if not normaldb.isSpecified(norm) or self.normResult is not None:
            return
        volume = self.volume()
        if volume is not None and volume.GetAttribute(NEURO_PREFIX + "Template"):
            return
        if header.get("space") and not self.setSpace(header["space"]):
            logging.warning(f"Static Neuro: the database's space {header['space']} is not offered")
        if norm.get("method"):
            index = self.methodCombo.findData(norm["method"])
            if index >= 0:
                self.methodCombo.setCurrentIndex(index)
                self._methodChosen = True
        if norm.get("template") and self.method() in (METHOD_PET, METHOD_ANATOMY):
            index = self.templateCombo.findData(norm["template"])
            if index >= 0 and self._templateAvailable(norm["template"]):
                self.templateCombo.setCurrentIndex(index)
                self._templateChosen = True
        if norm.get("engine"):
            index = self.normEngineCombo.findData(norm["engine"])
            if index >= 0:
                self.normEngineCombo.setCurrentIndex(index)
        if self.normEngine() == reg.ENGINE_ITERATIVE:
            if norm.get("linear"):
                combo = self.normIterative["linear"]
                index = combo.findData(norm["linear"])
                if index >= 0:
                    combo.setCurrentIndex(index)
            if norm.get("deformable") is not None:
                self.normIterative["deformable"].checked = bool(norm["deformable"])
        else:
            if norm.get("linear"):
                index = self.normStepCombos[0].findData(norm["linear"])
                if index >= 0:
                    self.normStepCombos[0].setCurrentIndex(index)
            if norm.get("deformable") is not None and len(self.normStepCombos) > 1:
                combo = self.normStepCombos[1]
                combo.setCurrentIndex(max(0, combo.findData("BSpline" if norm["deformable"] else "")))
        if norm.get("brainMask") is not None and self.maskCheck.enabled:
            self.maskCheck.checked = bool(norm["brainMask"])
        self._updateTemplateChoice()

    # --- normal database ---------------------------------------------------------------------------
    def onDatabaseRootChanged(self, folder):
        self.logic.setDatabaseRoot(folder)
        self._fillDatabases(force=True)

    def _database(self):
        return self._parameter().GetParameter(DATABASE_PARAMETER) or ""

    def _header(self, folder):
        for rowFolder, header, _, _ in self.databaseRows:
            if rowFolder == folder:
                return header
        return None

    def _fillDatabases(self, force=False):
        if not hasattr(self, "dbCombo"):
            return
        info = self.logic.imageInfo(self.volume())
        root = self.logic.databaseRoot()
        key = (root, info["modality"], info["nuclide"], info["tracer"])
        if key == self._databaseKey and not force:
            return
        self._databaseKey = key
        try:
            self.databaseRows = self.logic.databases(root, info)
        except Exception:
            logging.warning("Static Neuro: could not list the databases", exc_info=True)
            self.databaseRows = []
        current = self._database()
        self.dbCombo.blockSignals(True)
        self.dbCombo.clear()
        self.dbCombo.addItem("No database (regional values only)", "")
        for folder, header, rank, problems in self.databaseRows:
            n = normaldb.subjectCount(folder, header)
            text = (("! " if rank else "") + f"{header.get('name')} - {header.get('modality', 'PET')} "
                    f"{header.get('tracer') or header.get('nuclide')}, n = {n}"
                    + (" (predefined)" if header.get("kind") == normaldb.PREDEFINED else ""))
            self.dbCombo.addItem(text, folder)
            self.dbCombo.setItemData(self.dbCombo.count - 1, "Differs from the image: " + "; ".join(problems)
                                     if problems else "Fits the image", qt.Qt.ToolTipRole)
        self.dbCombo.setCurrentIndex(max(0, self.dbCombo.findData(current)))
        self.dbCombo.blockSignals(False)
        if not self._databaseChosen and not current:
            best = next((folder for folder, _, rank, _ in self.databaseRows if rank == 0), "")
            if best:
                self._setDatabase(best, chosen=False)
        elif current and self._header(current) is None:
            self._setDatabase("", chosen=False)
        self._fillAddDatabases()
        self._updateDatabaseInfo()
        self._updateTemplateChoice()

    def _setDatabase(self, folder, chosen=True):
        """Pick the normal database (the folder, '' for none): its normalization settings become the default of the
        Normalization page, its region settings are applied (locked)."""
        self._databaseChosen = self._databaseChosen or chosen
        self._parameter().SetParameter(DATABASE_PARAMETER, folder or "")
        if hasattr(self, "dbCombo"):
            self.dbCombo.blockSignals(True)
            self.dbCombo.setCurrentIndex(max(0, self.dbCombo.findData(folder or "")))
            self.dbCombo.blockSignals(False)
        header = self._header(folder)
        if hasattr(self, "spaceCombo"):
            self.refreshAtlasLibrary(save=False)        # a database whose space is not installed: 'From DB (name)'
        if header is not None:
            rank, problems = sn.databaseMatch(header, self.logic.imageInfo(self.volume()))
            if chosen and rank:
                slicer.util.warningDisplay(f"'{header['name']}' differs from the image: " + "; ".join(problems) + ".",
                                           windowTitle="Normal database")
            self._applyNormalizationSettings(header)
            self._applyDatabaseSettings(folder, header)
            self.unlockCheck.blockSignals(True)
            self.unlockCheck.checked = False
            self.unlockCheck.blockSignals(False)
        self._applyLock()
        self._fillAddDatabases()
        self._updateTemplateChoice()
        self._updateNormButtons()
        self._updateDatabaseInfo()

    def _normalizationWarnings(self, header):
        """Lines about the normalization against the database's: its settings, and how the image (normalized) or
        the page (not yet) differs. Never blocking."""
        norm = header.get("normalization") or {}
        if not normaldb.isSpecified(norm):
            return ["Normalization of the database: not specified (the first subject added sets it)."]
        lines = [f"Normalization of the database: {normaldb.normalizationText(norm)}."]
        volume = self.volume()
        normalized = volume is not None and bool(volume.GetAttribute(NEURO_PREFIX + "Template"))
        if normalized:
            used = self.usedNormalization(volume)
            if used is None:
                lines.append(WARNING_STYLE.format("The image's normalization settings are not known."))
                return lines
            differences = normaldb.normalizationDifferences(norm, used)
            if differences:
                lines.append(WARNING_STYLE.format("The image was normalized otherwise: " + "; ".join(differences)
                                                  + "."))
        else:
            differences = normaldb.normalizationDifferences(norm, self.normalizationSettings())
            if differences:
                lines.append(WARNING_STYLE.format("The settings on this page differ: " + "; ".join(differences)
                                                  + " (allowed, but the patient is not normalized like the "
                                                  "database's subjects)."))
        return lines

    def _updateDatabaseInfo(self):
        if not hasattr(self, "dbInfo") or not hasattr(self, "regionsDbInfo"):
            return
        root = self.logic.databaseRoot()
        folder = self._database()
        header = self._header(folder)
        info = self.logic.imageInfo(self.volume())
        if not root:
            text = "Pick the database folder, or go on without a database (the review gives the regional values only)."
            regionsText = "No normal database: the region settings below are free."
        elif header is None:
            text = (f"{len(self.databaseRows)} static database(s) in the folder. Without a database the review gives "
                    "the regional values only.")
            regionsText = "No normal database: the region settings below are free."
        else:
            lines = []
            rank, problems = sn.databaseMatch(header, info)
            if problems:
                lines.append(WARNING_STYLE.format("Differs from the image: " + "; ".join(problems)))
            warnings = sn.unitWarnings(info, header)
            if warnings:
                lines.append(WARNING_STYLE.format(" ".join(warnings)))
            if not sn.petTemplatePath(folder, header):
                lines.append("Its mean / SD maps are not computed yet.")
            lines.extend(self._normalizationWarnings(header))
            text = "<br>".join(lines)
            settings = sn.regionSettings(folder, header)
            regionLines = [f"Database '{header['name']}':"] + sn.settingsText(settings)
            if self._locked() and not settings["referenceMaskPath"] and not settings["referenceAtlasRegion"]:
                regionLines.append(WARNING_STYLE.format("It has no reference mask copy: unlock to pick the "
                                                        "reference region."))
            regionLines.append("Settings of the database (locked)." if self._locked() else
                               WARNING_STYLE.format("Unlocked: the regions may differ from the database's."))
            regionsText = "<br>".join(regionLines)
        self.dbInfo.text = text
        self.regionsDbInfo.text = regionsText
        self.dbCredits.setHtml(credits.databaseHtml(header, "No normal database picked."))
        self._updateTemplateCredits()
        self._updateRegionsCredits()
