"""
Step 4: regions (with the database's region settings) and the regions step views.

Part of the Static Neuro PET/SPECT module (StaticNeuroPET.py): the methods of RegionsPageMixin, mixed into
StaticNeuroPETWidget there.
"""

import logging

import qt
import slicer

from NeuroWorkflowLib.base import (NeuroWorkflowWidget, NEURO_PREFIX, NORM_PATIENT_PET_COLOR_ID,
                                   NORM_TEMPLATE_PET_COLOR_ID, PET_TEMPLATE_ATTRIBUTE)
from NeuroPETLib import credits, neuro, normaldb, staticneuro as sn

from StaticNeuroPETLib.constants import (REGIONS_COLUMNS, REGIONS_LAYOUT_ID, REGIONS_ORIENTATIONS,
                                         REGIONS_ORIENTATION_COLORS, REGIONS_VIEW_GROUP)


class RegionsPageMixin:
    """The page regions methods of StaticNeuroPETWidget."""

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
            logging.warning(f"Static Neuro: the database's region atlas {settings['regionAtlas']} is not in the atlas "
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
            if header is not None and name == f"{header['name']} template" and sn.petTemplatePath(folder, header):
                node = self.logic.loadAtlasVolume(sn.petTemplatePath(folder, header), name=name)
                node.SetAttribute(PET_TEMPLATE_ATTRIBUTE, "1")
                return node
        except Exception:
            logging.debug("Static Neuro: could not load the normalization's template", exc_info=True)
        return None

    def _regionsLayers(self):
        """{column: (background, foreground)} of the regions step views."""
        volume = self.volume()
        t1 = None
        if "T1" in (self.atlas or {}):
            try:
                t1 = self.logic.loadAtlasVolume(self.atlas["T1"], name="MNI ICBM152 2009c T1")
            except Exception:
                logging.debug("Static Neuro: could not load the MNI T1", exc_info=True)
        template = self._usedTemplate(volume)
        if template is not None and self._isPetTemplate(template):     # the synthetic / mean PET in inverted grey
            self._invertedGrey(template)
        petOnPet = None
        if template is not None and volume is not None and self._isPetTemplate(template):
            try:
                petOnPet = self._regionsPetOnPet(volume, template)
            except Exception:
                logging.exception("Static Neuro: could not color the PET on the PET template")
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
            logging.exception("Static Neuro: could not open the regions layout")

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


def _windowOf(node):
    """(lower, upper) of a volume's 1st and 99.5th percentiles (the window of the template in the normalization)."""
    from NeuroWorkflowLib.base import _percentiles
    return _percentiles(node)
