"""
The normal database in the Neuro workflows (shared by DynamicNeuroPET and StaticNeuroPET): its folder and the
database list (logic), the database picked on the Normalization page with its normalization settings as the default
and its PET template ('PET to PET'), the Regions page locked to the database's region settings (Unlock) and the
regions step views (the PET alone, on the MNI T1 and on the normalization's template, axial and coronal).

The mixins are put before NeuroWorkflowWidget / NeuroWorkflowLogic in a module's class. DATABASE_MODE (normaldb
STATIC / DYNAMIC) picks the databases listed; the hooks _databaseImageInfo, _databaseUnitWarnings, _petTemplatePath,
_meanComputed and _databasesChanged are a module's own. volume() is the image the database is matched with and the regions views
show (Static: the PET / SPECT; Dynamic: the series' summed image).
"""

import json
import logging

import ctk
import qt
import slicer

from DICOMGatedDynamicPETLib import registration as reg
from NeuroWorkflowLib.base import (NeuroWorkflowWidget, METHOD_ANATOMY, METHOD_PET, METHOD_PET_TEMPLATE, NEURO_PREFIX,
                                   NORM_PATIENT_PET_COLOR_ID, NORM_TEMPLATE_PET_COLOR_ID, PET_TEMPLATE_ATTRIBUTE,
                                   PREFIX)
from NeuroPETLib import credits, neuro, normaldb, staticneuro as sn

# application settings key: kept from the PET4D extension (the neuro modules came from it), so the folder stays picked
SETTINGS_DB_ROOT = "DICOMGatedDynamicPET/NeuroDatabase/RootFolder"     # shared with the Neuro PET Database Manager
DATABASE_PARAMETER = NEURO_PREFIX + "Database"
NORMALIZATION_ATTRIBUTE = NEURO_PREFIX + "Normalization"    # JSON of the settings the image was normalized with
WARNING_STYLE = "<span style='color:#c0392b'>{}</span>"
# the regions step layout: columns (view name prefix, view label, what lies under the PET), rows axial / coronal
REGIONS_LAYOUT_ID = 7830
REGIONS_VIEW_GROUP = 94
REGIONS_COLUMNS = (("NeuroRegionsPet", "P", "pet"), ("NeuroRegionsT1", "M", "t1"),
                   ("NeuroRegionsTemplate", "T", "template"))
REGIONS_ORIENTATIONS = ("Axial", "Coronal")
REGIONS_ORIENTATION_COLORS = {"Axial": "#F34A33", "Coronal": "#6EB04B"}


def seriesInfo(sequence):
    """staticneuro.imageInfo() of a dynamic PET series: a PET, its units, nuclide and tracer from the PET4D attributes
    (DynamicPET.*) of the sequence, else of its first frame."""
    info = sn.imageInfo({})
    if sequence is None:
        return info
    nodes = [sequence]
    if sequence.IsA("vtkMRMLSequenceNode") and sequence.GetNumberOfDataNodes():
        nodes.append(sequence.GetNthDataNode(0))

    def value(key):
        for node in nodes:
            text = node.GetAttribute(PREFIX + key) if node is not None else None
            if text:
                return text
        return ""

    radiopharmaceutical = value("Radiopharmaceutical")
    info.update(modality=normaldb.PET, source="PET4D dynamic series", static=False, units=value("Units"),
                nuclide=(sn.identifyNuclide(value("Radionuclide")) or sn.identifyNuclide(radiopharmaceutical)
                         or ""),
                tracer=sn.identifyTracer(radiopharmaceutical) or "")
    info["unitsKind"] = sn.unitsKind(info["units"])
    if not info["nuclide"] and info["tracer"]:
        info["nuclide"] = next((n for n, names in normaldb.NUCLIDE_TRACERS[normaldb.PET] if info["tracer"] in names),
                               "")
    return info


class DatabaseFolderLogicMixin:
    """The database folder (that of the Neuro PET Database Manager) and its databases of DATABASE_MODE."""

    DATABASE_MODE = normaldb.STATIC

    @staticmethod
    def databaseRoot():
        """The database folder of the Database Manager, else its default one (~/NeuroPET/Databases)."""
        return str(qt.QSettings().value(SETTINGS_DB_ROOT) or "") or normaldb.defaultRootFolder()

    @staticmethod
    def setDatabaseRoot(folder):
        qt.QSettings().setValue(SETTINGS_DB_ROOT, folder or "")

    @classmethod
    def databases(cls, root, info):
        """[(folder, header, rank, problems)] of the databases of root of DATABASE_MODE, those fitting the image
        first."""
        found, errors = normaldb.listDatabases(root, cls.DATABASE_MODE)
        for folder, error in errors:
            logging.warning(f"Neuro: database {folder} not readable: {error}")
        return sn.sortDatabases(found, info)


class NormalizationDatabaseMixin:
    """The Normalization page with the normal database (picked here; '' for none, never blocking) and the PET
    template method."""

    DATABASE_MODE = normaldb.STATIC
    COMPARED_IN = "the review"      # where the patient is compared (the step's name in the page texts)

    # --- hooks of a module -------------------------------------------------------------------------
    def _databaseImageInfo(self):
        """staticneuro.imageInfo() of the image the databases are matched with (modality, nuclide, tracer)."""
        return self.logic.imageInfo(self.volume())

    def _databaseUnitWarnings(self, info, header):
        """Warnings about the image's units for a database."""
        return []

    @staticmethod
    def _petTemplatePath(folder, header):
        """The template of the PET to PET normalization: a static database's frozen template, else its master mean
        map; a dynamic database's frozen template (of its summed images) only (its mean maps are parametric)."""
        if (header or {}).get("mode") == normaldb.DYNAMIC:
            return normaldb.templatePath(folder, header)
        return sn.petTemplatePath(folder, header)

    def _meanComputed(self, folder, header):
        """Whether a database has its mean / SD maps (a static one: its PET template or master mean map)."""
        if (header or {}).get("mode") == normaldb.DYNAMIC:
            return bool(((header or {}).get("statistics") or {}).get("groups"))
        return bool(sn.petTemplatePath(folder, header))

    def _databasesChanged(self):
        """The database list or the database picked changed (Static: the Add to database list of the review)."""

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
        elif not self._petTemplatePath(folder, header):
            self.templateNote.text = (
                f"'{header['name']}' has no PET template of its subjects' summed images yet: pick another method or "
                "database." if header.get("mode") == normaldb.DYNAMIC else
                f"'{header['name']}' has no template or master mean map yet (statistics not computed): pick another "
                "method or database.")
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
        path = self._petTemplatePath(folder, header) if header else None
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
                                            and bool(self._petTemplatePath(folder, header)))

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
            logging.warning(f"Neuro: the database's space {header['space']} is not offered")
        # the database's CT/MR method only when the patient's CT/MR fits it (an MR, or a CT with the database's mean
        # CT); else the default (PET to the synthetic PET) is kept
        anatomyFits = norm.get("method") != METHOD_ANATOMY or self._anatomyMethodFits()
        if not anatomyFits:
            logging.info("Neuro: the database normalizes the CT/MR, the patient has no MR and the database no mean "
                         "CT: the PET is registered to the synthetic PET")
        if norm.get("method") and anatomyFits:
            index = self.methodCombo.findData(norm["method"])
            if index >= 0:
                self.methodCombo.setCurrentIndex(index)
                self._methodChosen = True
        if norm.get("template") and anatomyFits and self.method() in (METHOD_PET, METHOD_ANATOMY):
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
        info = self._databaseImageInfo()
        root = self.logic.databaseRoot()
        key = (root, info["modality"], info["nuclide"], info["tracer"])
        if key == self._databaseKey and not force:
            return
        self._databaseKey = key
        try:
            self.databaseRows = self.logic.databases(root, info)
        except Exception:
            logging.warning("Neuro: could not list the databases", exc_info=True)
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
        self._databasesChanged()
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
            rank, problems = sn.databaseMatch(header, self._databaseImageInfo())
            if chosen and rank:
                slicer.util.warningDisplay(f"'{header['name']}' differs from the image: " + "; ".join(problems) + ".",
                                           windowTitle="Normal database")
            self._applyNormalizationSettings(header)
            self._applyDatabaseSettings(folder, header)
            self.unlockCheck.blockSignals(True)
            self.unlockCheck.checked = False
            self.unlockCheck.blockSignals(False)
        self._applyLock()
        self._databasesChanged()
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
        info = self._databaseImageInfo()
        if not root:
            text = (f"Pick the database folder, or go on without a database ({self.COMPARED_IN} gives the regional "
                    "values only).")
            regionsText = "No normal database: the region settings below are free."
        elif header is None:
            mode = dict(normaldb.MODES)[self.DATABASE_MODE].lower()
            text = (f"{len(self.databaseRows)} {mode} database(s) in the folder. Without a database {self.COMPARED_IN} "
                    "gives the regional values only.")
            regionsText = "No normal database: the region settings below are free."
        else:
            lines = []
            rank, problems = sn.databaseMatch(header, info)
            if problems:
                lines.append(WARNING_STYLE.format("Differs from the image: " + "; ".join(problems)))
            warnings = self._databaseUnitWarnings(info, header)
            if warnings:
                lines.append(WARNING_STYLE.format(" ".join(warnings)))
            if not self._meanComputed(folder, header):
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


class RegionsDatabaseMixin:
    """The Regions page locked to the normal database's region settings, and the regions step views."""

    # --- page 4: regions (with the database's region settings) ------------------------------------------
    def _buildRegionsPage(self):
        page = NeuroWorkflowWidget._buildRegionsPage(self)
        box = qt.QGroupBox("Region settings of the normal database")
        form = qt.QFormLayout(box)
        self.regionsDbInfo = qt.QLabel()
        self.regionsDbInfo.wordWrap = True
        form.addRow(self.regionsDbInfo)
        self.unlockCheck = qt.QCheckBox("Unlock the region settings")
        self.unlockCheck.toolTip = ("The regions are drawn as in the database (atlas, growth, reference region) so the "
                                    "patient's values compare with the database's. Unlock to change them.")
        form.addRow(self.unlockCheck)
        page.layout().insertWidget(0, box)
        self.unlockCheck.connect("toggled(bool)", self.onUnlockToggled)
        self.growSpin.connect("valueChanged(double)", lambda v: self._applyLock())
        return page

    def _applyDatabaseSettings(self, folder, header):
        settings = sn.regionSettings(folder, header)
        if not self.setRegionAtlas(settings["regionAtlas"]):
            logging.warning(f"Neuro: the database's region atlas {settings['regionAtlas']} is not in the atlas "
                            "folders")
        self.growSpin.value = settings["growMm"]
        self.growIntoCombo.setCurrentIndex(max(0, self.growIntoCombo.findData(settings["growInto"])))
        self.erodeSpin.value = settings["erodeReferenceMm"]
        if settings["referenceAtlasRegion"]:
            self._fillReferenceRegions()
            index = self.referenceRegionCombo.findText(settings["referenceAtlasRegion"])
            if index >= 0:
                self.referenceRegionCombo.setCurrentIndex(index)

    def _updateRegionsCredits(self):
        """Locked to a database: the atlas and the reference region of the database's regions."""
        if not hasattr(self, "regionsCredits"):
            return
        try:
            header = self._header(self._database()) if hasattr(self, "dbCombo") else None
        except Exception:       # noqa: BLE001 - while the page is built
            header = None
        if header is None or not hasattr(self, "unlockCheck") or not self._locked():
            NeuroWorkflowWidget._updateRegionsCredits(self)
            return
        self.regionsCredits.setHtml(credits.regionsHtml(header.get("regionAtlas"),
                                                        normaldb.referenceTitle(header),
                                                        normaldb.regionAtlasTitle(header.get("regionAtlas"))
                                                        + " (as in the database)"))

    def _locked(self):
        return bool(self._header(self._database())) and not self.unlockCheck.checked

    def _applyLock(self):
        if not hasattr(self, "unlockCheck"):
            return
        locked = self._locked()
        for widget in (self.segmentationCombo, self.groupingCombo, self.referenceRegionCombo, self.growSpin,
                       self.erodeSpin):
            widget.enabled = not locked
        self.growIntoCombo.enabled = not locked and self.growSpin.value > 0
        self.unlockCheck.enabled = bool(self._header(self._database()))

    def onUnlockToggled(self, on):
        if on and not slicer.util.confirmOkCancelDisplay(
                "With other region settings than the database's, the patient's regional values are not drawn the "
                "way the normal values were, so the comparison may be biased. Unlock?", windowTitle="Regions"):
            self.unlockCheck.blockSignals(True)
            self.unlockCheck.checked = False
            self.unlockCheck.blockSignals(False)
        elif not on:
            folder = self._database()
            header = self._header(folder)
            if header is not None:
                self._applyDatabaseSettings(folder, header)
        self._applyLock()
        self._updateDatabaseInfo()

    def _regionsArguments(self):
        arguments = NeuroWorkflowWidget._regionsArguments(self)
        folder = self._database()
        header = self._header(folder)
        if header is None or not self._locked():
            return arguments
        stored = sn.storedRegions(folder, header)
        if stored:              # the regions the database's statistics were made on, as they are
            arguments["storedRegions"] = stored
            return arguments
        settings = sn.regionSettings(folder, header)
        if settings["referenceMaskPath"]:
            arguments["referenceMaskPath"] = settings["referenceMaskPath"]
            arguments["referenceName"] = settings["referenceTitle"] or "Reference"
            arguments.update(sn.referenceMaskDrop(settings))
        elif settings["referenceAtlasRegion"]:
            arguments["referenceKind"] = neuro.mergedRegionAtlas(settings["regionAtlas"])
            arguments["referenceName"] = settings["referenceAtlasRegion"]
        else:
            raise ValueError(f"The database '{header['name']}' has no copy of its reference mask "
                             f"({settings['referenceTitle']}). Unlock the region settings to pick a reference region.")
        return arguments

    def _updateRegionsPage(self):
        NeuroWorkflowWidget._updateRegionsPage(self)
        if hasattr(self, "dbCombo"):
            self._fillDatabases()
            self._applyLock()
            folder = self._database()
            header = self._header(folder)
            if header is not None and self._locked() and sn.storedRegions(folder, header) \
                    and self.volume() is not None:
                # the database's own regions: the segmentation files are not needed
                self.loadRegionsButton.enabled = True
                normalized = bool(self.volume().GetAttribute(NEURO_PREFIX + "Template"))
                segmentation = self.node("Segmentation")
                self.regionsInfo.text = " ".join(
                    ([] if normalized else ["<span style='color:#c0392b'>The image is not normalized yet "
                                            "(Normalization step).</span>"])
                    + [f"The regions stored with '{header['name']}' are loaded (the ones its statistics were made "
                       "on)."]
                    + ([f"Loaded: '{segmentation.GetName()}'."] if segmentation is not None else []))

    # --- regions step views: the PET alone, the PET on the MNI T1, the PET on the normalization's template ------
    def _regionsLayoutXml(self):
        return regionsLayoutXml()

    def _usedTemplate(self, volume):
        """The template the image was normalized to (by the name it keeps), loaded again when it left the scene;
        None when it cannot be found."""
        name = volume.GetAttribute(NEURO_PREFIX + "Template") or ""
        if not name:
            return None
        picked = self.node("Template")
        if picked is not None and picked.GetName() == name and slicer.mrmlScene.IsNodePresent(picked):
            return picked
        for node in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
            if node.GetName() == name and node is not volume and not node.IsA("vtkMRMLLabelMapVolumeNode"):
                return node
        try:
            prefix = "MNI ICBM152 2009c "
            if name.startswith(prefix) and name[len(prefix):] in (self.atlas or {}):
                return self.logic.loadAtlasVolume(self.atlas[name[len(prefix):]], name=name)
            folder = self._database()
            header = self._header(folder)
            if header is not None and name == f"{header['name']} template" and self._petTemplatePath(folder, header):
                node = self.logic.loadAtlasVolume(self._petTemplatePath(folder, header), name=name)
                node.SetAttribute(PET_TEMPLATE_ATTRIBUTE, "1")
                return node
        except Exception:
            logging.debug("Neuro: could not load the normalization's template", exc_info=True)
        return None

    def _regionsLayers(self):
        """{column: (background, foreground)} of the regions step views."""
        volume = self.volume()
        t1 = None
        if "T1" in (self.atlas or {}):
            try:
                t1 = self.logic.loadAtlasVolume(self.atlas["T1"], name="MNI ICBM152 2009c T1")
            except Exception:
                logging.debug("Neuro: could not load the MNI T1", exc_info=True)
        template = self._usedTemplate(volume)
        if template is not None and self._isPetTemplate(template):     # the synthetic / mean PET in inverted grey
            self._invertedGrey(template)
        petOnPet = None
        if template is not None and volume is not None and self._isPetTemplate(template):
            try:
                petOnPet = self._regionsPetOnPet(volume, template)
            except Exception:
                logging.exception("Neuro: could not color the PET on the PET template")
        layers = {}
        for column, _, kind in REGIONS_COLUMNS:
            if kind not in ("pet", "t1") and petOnPet is not None:
                layers[column] = petOnPet       # PET on a PET template: Blue under Red, as in the normalization
                continue
            under = None if kind == "pet" else t1 if kind == "t1" else template
            layers[column] = (under, volume) if under is not None else (volume, None)
        return layers

    def _regionsPetOnPet(self, volume, template):
        """(background, foreground) of the PET on a PET template (synthetic / database mean PET): the template in
        Blue and the patient's PET in Red, as in the normalization view. Hidden helper volumes that share the voxels,
        so the colors of the volumes themselves stay as they were; made again each time the step is entered."""
        import DynamicPETRegistration as dpr
        helpers = getattr(self, "_regionsHelpers", None)
        if helpers and helpers["key"] == (volume.GetID(), template.GetID()) and all(
                slicer.mrmlScene.IsNodePresent(n) for n in (helpers["template"], helpers["pet"])):
            return helpers["template"], helpers["pet"]
        self._dropRegionsHelpers()
        logic = dpr.DynamicPETRegistrationLogic
        under = logic.helperVolume(f"{template.GetName()} regions view", template)
        pet = logic.helperVolume(f"{volume.GetName()} regions view", volume)
        self._regionsHelpers = {"key": (volume.GetID(), template.GetID()), "template": under, "pet": pet}
        for node, source in ((under, template), (pet, volume)):
            node.SetAndObserveTransformNodeID(source.GetTransformNodeID())
            node.CreateDefaultDisplayNodes()
            node.GetDisplayNode().SetInterpolate(True)
        self._setColor(under, NORM_TEMPLATE_PET_COLOR_ID)
        self._setColor(pet, NORM_PATIENT_PET_COLOR_ID)
        lower, upper = _windowOf(template)
        display = under.GetDisplayNode()
        display.AutoWindowLevelOff()
        display.SetWindowLevelMinMax(lower, upper)
        # the window of the normalization view: 0 to the 99.9th percentile, below 3 % of it transparent
        upper = dpr._percentile(slicer.util.arrayFromVolume(volume))
        display = pet.GetDisplayNode()
        display.AutoWindowLevelOff()
        display.SetWindowLevelMinMax(0.0, upper if upper > 0 else 1.0)
        display.SetApplyThreshold(True)
        display.SetLowerThreshold(dpr.THRESHOLD_FRACTION * upper)
        display.SetUpperThreshold(max(abs(upper), 1.0) * 1e6)
        return under, pet

    def _dropRegionsHelpers(self):
        helpers = getattr(self, "_regionsHelpers", None)
        self._regionsHelpers = None
        for node in (helpers or {}).values():
            if not isinstance(node, tuple) and node is not None and slicer.mrmlScene.IsNodePresent(node):
                slicer.mrmlScene.RemoveNode(node)

    def _enterRegionsViews(self):
        """The regions step layout: axial (top) and coronal (bottom) of the patient's PET alone (left), on the MNI
        T1 MR template (middle) and on the template of the normalization (right), to check the normalization."""
        volume = self.volume()
        if volume is None or not volume.GetAttribute(NEURO_PREFIX + "Template"):
            return
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None:
            return
        self._dropRegionsHelpers()      # made again from the volumes as they are now
        try:
            layoutNode = layoutManager.layoutLogic().GetLayoutNode()
            description = self._regionsLayoutXml()
            if not layoutNode.IsLayoutDescription(REGIONS_LAYOUT_ID):
                layoutNode.AddLayoutDescription(REGIONS_LAYOUT_ID, description)
            elif layoutNode.GetLayoutDescription(REGIONS_LAYOUT_ID) != description \
                    and layoutManager.layout != REGIONS_LAYOUT_ID:
                layoutNode.SetLayoutDescription(REGIONS_LAYOUT_ID, description)
            if layoutManager.layout != REGIONS_LAYOUT_ID:
                self._regionsPreviousLayout = layoutManager.layout
                layoutManager.setLayout(REGIONS_LAYOUT_ID)
                slicer.app.processEvents()
            self._assignRegionsViews(fit=True)
        except Exception:
            logging.exception("Neuro: could not open the regions layout")

    def _assignRegionsViews(self, fit=False):
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None or layoutManager.layout != REGIONS_LAYOUT_ID:
            return
        layers = self._regionsLayers()
        opacity = float(self.normOpacity.value) if hasattr(self, "normOpacity") else 0.5
        widgets = []
        for column, _, _ in REGIONS_COLUMNS:
            background, foreground = layers[column]
            for orientation in REGIONS_ORIENTATIONS:
                widget = layoutManager.sliceWidget(column + orientation)
                if widget is None:
                    continue
                composite = widget.mrmlSliceCompositeNode()
                composite.SetLinkedControl(False)
                composite.SetBackgroundVolumeID(background.GetID() if background is not None else None)
                composite.SetForegroundVolumeID(foreground.GetID() if foreground is not None else None)
                composite.SetForegroundOpacity(opacity if foreground is not None else 0.0)
                composite.SetLabelVolumeID(None)
                widgets.append(widget)
        for widget in widgets:          # linked once the layers are set: the same slice and zoom in all the views
            widget.mrmlSliceCompositeNode().SetLinkedControl(True)
            if fit:
                widget.sliceLogic().FitSliceToAll()

    def _exitRegionsViews(self):
        layoutManager = slicer.app.layoutManager()
        previous = getattr(self, "_regionsPreviousLayout", None)
        if layoutManager is not None and layoutManager.layout == REGIONS_LAYOUT_ID:
            for column, _, _ in REGIONS_COLUMNS:
                for orientation in REGIONS_ORIENTATIONS:
                    widget = layoutManager.sliceWidget(column + orientation)
                    if widget is not None:
                        widget.mrmlSliceCompositeNode().SetLinkedControl(False)
            layoutManager.setLayout(previous if previous is not None
                                    else slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        self._regionsPreviousLayout = None
        self._dropRegionsHelpers()


def regionsLayoutXml():
    """The regions step layout: 3 columns (patient PET, on the MNI T1, on the normalization's template) x 2 rows
    (axial, coronal)."""
    def view(column, orientation, label):
        color = REGIONS_ORIENTATION_COLORS[orientation]
        return (f'<item><view class="vtkMRMLSliceNode" singletontag="{column}{orientation}">'
                f'<property name="orientation" action="default">{orientation}</property>'
                f'<property name="viewlabel" action="default">{label}{orientation[0]}</property>'
                f'<property name="viewcolor" action="default">{color}</property>'
                f'<property name="viewgroup" action="default">{REGIONS_VIEW_GROUP}</property>'
                '</view></item>')
    columns = "".join('<item><layout type="vertical" split="true">'
                      + "".join(view(column, o, label) for o in REGIONS_ORIENTATIONS) + '</layout></item>'
                      for column, label, _ in REGIONS_COLUMNS)
    return f'<layout type="horizontal" split="true">{columns}</layout>'


def registerRegionsLayout():
    """The regions step layout made known to Slicer (at startup: a scene saved in it opens its views)."""
    from NeuroWorkflowLib import scenestate
    scenestate.addLayout(REGIONS_LAYOUT_ID, regionsLayoutXml())


def _windowOf(node):
    """(lower, upper) of a volume's 1st and 99.5th percentiles (the window of the template in the normalization)."""
    from NeuroWorkflowLib.base import _percentiles
    return _percentiles(node)
