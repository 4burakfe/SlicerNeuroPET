"""
The Review/Compare step of the Dynamic Neuro workflow (DynamicNeuroPET): the patient's parametric maps (BPnd, R1, k2 of
SRTM2 by default) shown and compared like the static image of the Static Neuro Review step (its review mixins, the map
as the review's image).

    maps        made here with the Modeling step's settings (PET4D DynamicPETModelingLogic.parametricMaps, in the
                template's brain mask, no threshold), or taken from the Modeling step when it made them with the same
                settings (asked first); a map of the set is picked to compare (the model's primary map first)
    regions     the regional value of a region: its TAC fitted with the model (default; the Modeling logic's region
                fits, the same as the Modeling step's) or the mean of the voxel map in it
    database    a dynamic normal database's comparison needs its mean / SD maps per map (later); its kinetic
                settings are compared with the Modeling step's here (warnings, the comparison still runs)

The mixin is put before the review mixins (StaticNeuroPETLib.review_*) in DynamicNeuroPETWidget; the page is the
review page with the 'Parametric maps' box on top.
"""

import json
import logging

import qt
import slicer

from NeuroWorkflowLib.base import NEURO_PREFIX, _button
from NeuroWorkflowLib.database import NORMALIZATION_ATTRIBUTE, WARNING_STYLE
from NeuroPETLib import dynamicneuro as dn, neuro, neuroreview as nr, normaldb

KM_PREFIX = "DynamicPET.Modeling."              # the PET4D Modeling module's attributes of its maps and tables
COMPARE_MAP_ATTRIBUTE = NEURO_PREFIX + "CompareMap"     # a map the Review/Compare step made ("1")
COMPARE_MAP_ROLE = "CompareMap"                 # parameter node reference: the map compared (saved scenes)
SETTINGS_COMPARE_DISPLAY = "DICOMGatedDynamicPET/DynamicNeuro/DisplaySettings"   # the Review/Compare step's display settings


def _modelingModule():
    """The PET4D Dynamic PET Modeling module file (DynamicPETModeling), imported."""
    import DynamicPETModeling
    return DynamicPETModeling


class ComparePageMixin:
    """The Review/Compare step's page and its parametric maps (see the module docstring)."""

    REVIEW_MEAN_MASK = False        # a striatal tracer's BPnd is low in the cortex: the brain mask alone
    DISPLAY_SETTINGS_KEY = SETTINGS_COMPARE_DISPLAY
    NO_DATABASE_TEXT = ("No comparison with a normal database yet (it needs the database's mean / SD maps of the "
                        "parametric map): the maps and the regional values.")
    COMPARE_TEXT = ("Compare with the database", "Make the maps and the regional values")

    # --- page ----------------------------------------------------------------------------------------
    def _buildComparePage(self):
        page = qt.QWidget()
        layout = qt.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        box = qt.QGroupBox("Parametric maps")
        form = qt.QFormLayout(box)
        self.kineticsLabel = qt.QLabel()
        self.kineticsLabel.wordWrap = True
        self.kineticsLabel.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        self.kineticsLabel.toolTip = ("The maps are made with the Modeling step's settings (model, reference region, "
                                      "weighting, time zero, k2', t*, window): change them there.")
        form.addRow(self.kineticsLabel)
        self.mapCombo = qt.QComboBox()
        self.mapCombo.toolTip = ("The parametric map compared and shown (the model's maps: BPnd, R1, k2 of SRTM2 / "
                                 "SRTM; DVR, BPnd of the Logan plot; SUVR)")
        form.addRow("Map:", self.mapCombo)
        self.regionValueCombo = qt.QComboBox()
        for key, title in dn.REGION_VALUES:
            self.regionValueCombo.addItem(title, key)
        self.regionValueCombo.toolTip = (
            "The regional value of the table: the model fitted to the region's mean time-activity curve (the same "
            "fit as Fit all of the Modeling step; less noisy), or the mean of the voxel map in the region")
        form.addRow("Regional values:", self.regionValueCombo)
        self.remakeMapsButton = _button("Make the maps again")
        self.remakeMapsButton.toolTip = ("Make the parametric maps again with the Modeling step's settings (else the "
                                         "maps of the same settings are used again)")
        form.addRow(self.remakeMapsButton)
        self.mapsLabel = qt.QLabel()
        self.mapsLabel.wordWrap = True
        self.mapsLabel.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        form.addRow(self.mapsLabel)
        layout.addWidget(box)
        self._buildReviewBoxes(layout)
        layout.addStretch(1)
        self.compareMaps = {}           # {map name: volume node} of the set compared
        self._compareMapsKey = None     # the settings (JSON) of that set
        self._regionFits = None         # (key, {segment ID: fit result}, {segment ID: name}, k2' note)
        self._askedModelingMaps = {}    # {settings key: True (taken) / False (declined)}
        self.mapCombo.connect("activated(int)", lambda i: self._onCompareMapPicked())
        self.regionValueCombo.connect("activated(int)", lambda i: self._recompare())
        self.remakeMapsButton.connect("clicked()", self.onRemakeMaps)
        return page

    def _resetCompareState(self):
        self.compareMaps = {}
        self._compareMapsKey = None
        self._regionFits = None
        self._askedModelingMaps = {}
        if hasattr(self, "mapsLabel"):
            self.mapsLabel.text = ""

    # --- the Modeling step's settings ------------------------------------------------------------------
    def _modelingWidget(self):
        """The Modeling panel (the PET4D module's widget) with the workflow's series, regions and reference."""
        widget = self.stepWidget(self._modelStep())
        if widget is None:
            module = getattr(slicer.modules, "dynamicpetmodeling", None)
            if module is None:
                raise RuntimeError("The PET4D Dynamic PET Modeling module is not loaded")
            widget = module.widgetRepresentation().self()
        sequence = self.sequence()
        if sequence is not None and widget.sequenceSelector.currentNode() is not sequence:
            widget.sequenceSelector.setCurrentNode(sequence)
        self._prepareModeling(widget)
        return widget

    @staticmethod
    def _modelStep():
        from NeuroWorkflowLib.base import STEP_MODEL
        return STEP_MODEL

    def _referenceName(self, widget):
        segmentation = self.node("Segmentation")
        referenceId = widget.referenceId()
        segment = segmentation.GetSegmentation().GetSegment(referenceId) if segmentation and referenceId else None
        return segment.GetName() if segment is not None else ""

    def _compareKinetics(self, widget=None):
        """(fit state, settings (dynamicneuro.mapSettings), error text or '') of the Modeling step."""
        try:
            widget = widget or self._modelingWidget()
        except Exception as error:
            return None, None, str(error)
        state = widget.fitState("maps")
        model = state.get("model")
        if model not in dn.COMPARE_MODELS:
            return state, None, (f"The Modeling step's model is {model}: the Review/Compare step makes the maps of a "
                                 f"reference tissue model ({', '.join(dn.COMPARE_MODELS)}); pick one there.")
        reference = self._referenceName(widget)
        if not reference:
            return state, None, "Pick the reference region in the Modeling step."
        state["referenceName"] = reference
        return state, dn.mapSettings(state, reference), ""

    def _settingsKey(self, settings):
        """A key of the settings that change the maps (and of the series and regions)."""
        sequence, segmentation = self.sequence(), self.node("Segmentation")
        relevant = {k: settings.get(k) for k in dn.relevantKeys(settings.get("model"))} if settings else {}
        return json.dumps([sequence.GetID() if sequence else "", segmentation.GetID() if segmentation else "",
                           relevant], sort_keys=True, default=str)

    def _kineticsText(self, settings):
        if not settings:
            return ""
        model = settings["model"]
        text = normaldb.kineticsText(dict(settings, maps=list(normaldb.KINETIC_MAPS[model]),
                                          primaryMap=normaldb.PRIMARY_MAPS[model]))
        return f"{text}; reference {settings.get('reference')}" + (
            "" if settings.get("clipNegative", True) else "; negative values kept")

    # --- the maps -------------------------------------------------------------------------------------
    def _sceneMaps(self, settings, own):
        """{map name: node} of the maps in the scene made from this series and regions with these settings: own (the
        Review/Compare step's) or the Modeling step's (made in the whole brain); the latest of each map."""
        sequence, segmentation = self.sequence(), self.node("Segmentation")
        if sequence is None or segmentation is None or not settings:
            return {}
        found = {}
        for node in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
            if (node.GetAttribute(KM_PREFIX + "Sequence") != sequence.GetID()
                    or node.GetAttribute(KM_PREFIX + "Model") != settings["model"]
                    or bool(node.GetAttribute(COMPARE_MAP_ATTRIBUTE)) != own):
                continue
            if node.GetAttribute(KM_PREFIX + "Segmentation") not in (None, "", segmentation.GetID()):
                continue
            state = node.GetAttribute(KM_PREFIX + "FitState")
            ok, _ = dn.mapsAcceptable(state)
            made = dn.mapSettings(state, node.GetAttribute(KM_PREFIX + "Reference"))
            if not ok or made is None or not dn.sameSettings(settings, made, self._lastEndMinutes()):
                continue
            name = node.GetAttribute(KM_PREFIX + "Map")
            if name:
                found[name] = node          # the scene's order: the latest one wins
        return found

    def _lastEndMinutes(self):
        """The end of the series' last frame, minutes from its first frame (the default SUVR window)."""
        sequence = self.sequence()
        if sequence is None:
            return None
        key = (sequence.GetID(), sequence.GetMTime())
        cached = getattr(self, "_lastEndCache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            intervals = _modelingModule().DynamicPETModelingLogic.intervals(sequence)
            value = (intervals[-1]["end"] - intervals[0]["start"]) / 60.0
        except Exception:       # noqa: BLE001 - the window then compared as given
            value = None
        self._lastEndCache = (key, value)
        return value

    def _mapsPresent(self):
        return bool(self.compareMaps) and all(slicer.mrmlScene.IsNodePresent(n) for n in self.compareMaps.values())

    def _adoptMaps(self, maps, key):
        """maps (map name: node) the set compared: tagged as normalized like the series (the review's checks)."""
        sequence = self.sequence()
        for node in maps.values():
            for attribute in (NEURO_PREFIX + "Template", NORMALIZATION_ATTRIBUTE):
                value = sequence.GetAttribute(attribute) if sequence is not None else None
                if value and not node.GetAttribute(attribute):
                    node.SetAttribute(attribute, value)
        self.compareMaps = dict(maps)
        self._compareMapsKey = key
        self._fillMapCombo()
        shown = self.reviewVolume()
        self.setNode(COMPARE_MAP_ROLE, shown)

    def _fillMapCombo(self):
        current = self.mapCombo.currentData
        names = list(self.compareMaps)
        model = next((n.GetAttribute(KM_PREFIX + "Model") for n in self.compareMaps.values()), None)
        order = [m for m in normaldb.KINETIC_MAPS.get(model, ()) if m in names] + \
            [m for m in names if m not in normaldb.KINETIC_MAPS.get(model, ())]
        self.mapCombo.blockSignals(True)
        self.mapCombo.clear()
        for name in order:
            self.mapCombo.addItem(name, name)
        index = self.mapCombo.findData(current) if current else -1
        if index < 0:
            index = self.mapCombo.findData(dn.defaultMap(model, order))
        self.mapCombo.setCurrentIndex(max(0, index))
        self.mapCombo.blockSignals(False)

    def _selectedMapName(self):
        return self.mapCombo.currentData if hasattr(self, "mapCombo") and self.mapCombo.count else None

    def reviewVolume(self):
        """The parametric map compared (None before the maps are made)."""
        name = self._selectedMapName()
        node = (getattr(self, "compareMaps", None) or {}).get(name) if name else None
        return node if node is not None and slicer.mrmlScene.IsNodePresent(node) else None

    def _progressDialog(self, title, text):
        dialog = slicer.util.createProgressDialog(windowTitle=title, labelText=text, maximum=0)

        def progress(index, total, label=None):
            dialog.maximum = max(1, int(total))
            dialog.setValue(int(index))
            if label:
                dialog.setLabelText(f"{text} {label}")
            slicer.app.processEvents()
            return not dialog.wasCanceled
        return dialog, progress

    def _ensureCompareMaps(self, remake=False):
        """The maps of the Modeling step's settings ready (made, found again, or the Modeling step's taken after
        asking) and the regions' fits for the TAC fit values. False when they could not be made (or were
        stopped)."""
        widget = self._modelingWidget()
        state, settings, error = self._compareKinetics(widget)
        if error:
            slicer.util.errorDisplay(error, windowTitle="Review/Compare")
            self.mapsLabel.text = WARNING_STYLE.format(error)
            return False
        key = self._settingsKey(settings)
        if not remake and self._compareMapsKey == key and self._mapsPresent():
            return self._ensureRegionFits(widget, state, settings, key)
        own = {} if remake else self._sceneMaps(settings, own=True)
        if own:
            self._adoptMaps(own, key)
            self.mapsLabel.text = "The maps made before with these settings: " + ", ".join(
                n.GetName() for n in own.values()) + "."
            return self._ensureRegionFits(widget, state, settings, key)
        modeling = {} if remake else self._sceneMaps(settings, own=False)
        if modeling and self._askedModelingMaps.get(key) is not False:
            take = self._askedModelingMaps.get(key)
            if take is None:
                names = ", ".join(n.GetName() for n in modeling.values())
                take = slicer.util.confirmOkCancelDisplay(
                    f"The Modeling step made the maps of this series with the same settings:\n{names}\n\nUse them? "
                    "(Cancel: the Review/Compare step makes its own maps.)", windowTitle="Review/Compare")
                self._askedModelingMaps[key] = bool(take)
            if take:
                self._adoptMaps(modeling, key)
                self.mapsLabel.text = "The Modeling step's maps: " + ", ".join(n.GetName() for n in modeling.values())
                return self._ensureRegionFits(widget, state, settings, key)
        # the fits first: SRTM2's automatic k2' (the median of the regions' SRTM fits) is then the maps' too
        if not self._ensureRegionFits(widget, state, settings, key, forMaps=True):
            return False
        return self._makeMaps(widget, state, settings, key)

    def _makeMaps(self, widget, state, settings, key):
        sequence = self.sequence()
        analysis = widget.ensureAnalysis()
        if analysis is None:
            self.mapsLabel.text = "The time-activity curves of the regions could not be read."
            return False
        model = settings["model"]
        options = widget.options()
        options["clipNegative"] = bool(widget.clipCheck.checked)
        if self._regionFits and self._regionFits[3]:
            options["k2ref"] = self._regionFits[3][0]
        brainMask = widget.brainMask()
        if brainMask is None and neuro.BRAIN_MASK in (self.atlas or {}):
            brainMask = self.logic.loadAtlasVolume(self.atlas[neuro.BRAIN_MASK], name=f"{self.spaceName()} brain mask")
        if brainMask is None:
            logging.warning("Neuro Compare: no brain mask, the maps are made in the whole field of view")
        dialog, progress = self._progressDialog("Review/Compare", f"The {model} maps...")
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            maps = widget.logic.parametricMaps(analysis, model, widget.referenceId(), options, None, 0.0, False,
                                               lambda i, n: progress(i, n), brainMask=brainMask, settings=state)
        except Exception as error:
            logging.exception("Neuro Compare: the parametric maps failed")
            slicer.util.errorDisplay(f"Could not make the parametric maps:\n{error}", windowTitle="Review/Compare")
            return False
        finally:
            qt.QApplication.restoreOverrideCursor()
            dialog.close()
            dialog.deleteLater()
        if not maps:
            self.mapsLabel.text = "Stopped: no maps made."
            return False
        # the Review/Compare step's maps of this series made before (other settings) go: a set takes a lot of memory
        keep = {n.GetID() for n in maps.values()}
        for node in list(slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")):
            if (node.GetAttribute(COMPARE_MAP_ATTRIBUTE) and node.GetID() not in keep
                    and node.GetAttribute(KM_PREFIX + "Sequence") == sequence.GetID()):
                slicer.mrmlScene.RemoveNode(node)
        for node in maps.values():
            node.SetAttribute(COMPARE_MAP_ATTRIBUTE, "1")
        self._adoptMaps(maps, key)
        note = f" (reference k2' {options['k2ref']:.4g}/min)" if model == "SRTM2" and options.get("k2ref") else ""
        self.mapsLabel.text = "Made " + ", ".join(n.GetName() for n in maps.values()) + note + "."
        return True

    def _ensureRegionFits(self, widget, state, settings, key, forMaps=False):
        """The regions' fits of these settings: the Modeling step's own (Fit all with the same settings), else made
        now (the Modeling logic). Needed for the TAC fit values, and before SRTM2 maps with the automatic k2' (the
        median of the regions' SRTM fits, the same for the maps)."""
        automaticK2 = settings["model"] == "SRTM2" and settings.get("k2ref") is None
        if self.regionValueCombo.currentData != dn.REGION_TAC_FIT and not (forMaps and automaticK2):
            return True
        if self._regionFits is not None and self._regionFits[0] == key:
            return True
        analysis = widget.ensureAnalysis()
        if analysis is None:
            self.mapsLabel.text = "The time-activity curves of the regions could not be read."
            return False
        model = settings["model"]
        saved = widget._savedFit()[0] if hasattr(widget, "_savedFit") else {}
        results = None
        if (getattr(widget, "results", None) and getattr(widget, "resultsModel", None) == model
                and saved.get("kind") == "all" and getattr(widget, "analysis", None) is analysis
                and dn.sameSettings(settings, dn.mapSettings(saved, settings.get("reference")),
                                    self._lastEndMinutes())):
            results = widget.results
        k2ref = None
        if results is None:
            dialog, progress = self._progressDialog("Review/Compare", "Fitting the regions:")
            qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
            try:
                options = widget.options()
                widget.logic.lastK2ref = None
                results = widget.logic.fitSegments(analysis, model, widget.referenceId(), options,
                                                   progress=lambda i, n, name: progress(i, n, name))
                if model == "SRTM2" and not options.get("k2ref"):
                    k2ref = getattr(widget.logic, "lastK2ref", None)
            except Exception as error:
                logging.exception("Neuro Compare: the region fits failed")
                slicer.util.errorDisplay(f"Could not fit the regions:\n{error}", windowTitle="Review/Compare")
                return False
            finally:
                qt.QApplication.restoreOverrideCursor()
                dialog.close()
                dialog.deleteLater()
        elif model == "SRTM2" and not widget.options().get("k2ref"):
            k2ref = getattr(widget.logic, "lastK2ref", None)
        names = {segmentId: analysis.name(segmentId) for segmentId in results}
        self._regionFits = (key, results, names, k2ref)
        return True

    # --- the comparison --------------------------------------------------------------------------------
    def onCompare(self):
        if self.sequence() is None or self.node("Segmentation") is None:
            return
        try:
            ready = self._ensureCompareMaps()
        except Exception as error:
            logging.exception("Neuro Compare: the maps failed")
            slicer.util.errorDisplay(f"Could not make the parametric maps:\n{error}", windowTitle="Review/Compare")
            ready = False
        if not ready:
            self.refresh()
            return
        super().onCompare()

    def onRemakeMaps(self):
        if self.sequence() is None or self.node("Segmentation") is None:
            return
        self._regionFits = None
        if self._ensureCompareMaps(remake=True):
            super().onCompare()
        self.refresh()

    def _recompare(self):
        """Another map or regional value: compared again when there is a comparison on screen."""
        from StaticNeuroPETLib.constants import STEP_REVIEW
        if getattr(self, "current", None) == STEP_REVIEW and self.compareButton.enabled and (
                self.reviewResult is not None or self._mapsPresent()):
            self.onCompare()
        else:
            self.refresh()

    def _onCompareMapPicked(self):
        node = self.reviewVolume()
        if node is not None:
            self.setNode(COMPARE_MAP_ROLE, node)
        self._recompare()

    def _adjustReviewResult(self, result):
        """The regions' TAC fit values in place of the voxel means (kept as 'voxelMeans') when picked."""
        mapName = self._selectedMapName()
        result["map"] = mapName
        result["regionValues"] = self.regionValueCombo.currentData
        if self.regionValueCombo.currentData != dn.REGION_TAC_FIT or not self._regionFits:
            result["notes"].append(f"Regional values: the mean of the {mapName} map in each region.")
            return result
        _, results, names, k2ref = self._regionFits
        fitted = dn.regionFitValues(results, names, mapName)
        result["voxelMeans"] = dict(result["suv"])
        values = {name: fitted.get(name) for name in result["suv"]}
        result["suv"] = result["values"] = values
        missing = [name for name, value in values.items() if value is None and not nr.isReferenceRegion(name)]
        result["notes"].append(f"Regional values: {mapName} of the model fitted to each region's time-activity "
                               "curve" + (f" (reference k2' {k2ref[0]:.4g}/min, the median of {k2ref[1]} SRTM "
                                          "fits)" if k2ref else "") + ".")
        warnings = dn.fitWarnings({s: r for s, r in results.items() if names.get(s) in values}, names, mapName)
        if warnings:
            result["notes"].append("Fit warnings: " + "; ".join(warnings[:8])
                                   + (f" and {len(warnings) - 8} more" if len(warnings) > 8 else "") + ".")
        if missing:
            result["notes"].append("No fit value: " + ", ".join(missing[:8]) + ".")
        return result

    # --- hooks of the review mixins ---------------------------------------------------------------------
    def _reviewHeader(self, folder):
        # a dynamic database's mean / SD maps per parametric map: not yet (the comparison with it comes later)
        return None

    def _reviewAgeAndSex(self):
        return self.logic.ageAndSex(self.sequence())

    def _reviewReferenceId(self):
        return None         # BPnd, R1, k2, DVR and SUVR are relative to the reference already

    def _reviewReady(self, volume, segmentation, header):
        sequence = self.sequence()
        return (sequence is not None and bool(sequence.GetAttribute(NEURO_PREFIX + "Template"))
                and segmentation is not None)

    def _reviewImageLine(self, volume):
        sequence = self.sequence()
        line = f"Series: {sequence.GetName() if sequence else 'none'}" + (
            " (normalized)" if sequence is not None and sequence.GetAttribute(NEURO_PREFIX + "Template") else "")
        return line + (f"<br>Map: {volume.GetName()}" if volume is not None else "<br>Map: not made yet")

    def _regionValueTitle(self):
        mapName = self._selectedMapName() or "Value"
        how = self.regionValueCombo.currentData if hasattr(self, "regionValueCombo") else dn.REGION_TAC_FIT
        result = getattr(self, "reviewResult", None) or {}
        if result.get("map") == mapName and result.get("regionValues"):
            how = result["regionValues"]
        return dn.regionValueTitle(mapName, how)

    def _isFusionExcluded(self, node):
        """The series' summed image and the parametric maps are not fusion images."""
        return node is self.volume() or bool(node.GetAttribute(KM_PREFIX + "Map"))

    def _petUnits(self, volume):
        return self._selectedMapName() or ""

    def _patientUnits(self):
        return self._selectedMapName() or ""

    @staticmethod
    def _defaultReviewLayout():
        return nr.LAYOUT_SLICES

    def _initPetWindow(self, percentile=True):
        """The map's window: 0 to its 99.5th percentile (BPnd is clipped at 100 in a few voxels)."""
        from StaticNeuroPETLib.review_colors import ReviewColorsMixin
        ReviewColorsMixin._initPetWindow(self, True)

    # --- the page's state -----------------------------------------------------------------------------
    def _updateCompareBox(self):
        if not hasattr(self, "kineticsLabel"):
            return
        sequence, segmentation = self.sequence(), self.node("Segmentation")
        if sequence is None or segmentation is None:
            self.kineticsLabel.text = "Load the regions first (Regions step)."
            self.remakeMapsButton.enabled = False
            return
        widget = self.stepWidget(self._modelStep())
        if widget is None:
            self.kineticsLabel.text = ("The Modeling step's settings (open it to change them; default SRTM2 with the "
                                       "regions' reference).")
            self.remakeMapsButton.enabled = True
            return
        _, settings, error = self._compareKinetics(widget)
        lines = [WARNING_STYLE.format(error)] if error else [self._kineticsText(settings)]
        header = self._header(self._database())
        expected = normaldb.kinetics(header) if header else None
        if expected and settings:
            differences = dn.settingsDifferences(expected, settings, self._lastEndMinutes(), skip=("reference",))
            if differences:
                lines.append(WARNING_STYLE.format(f"Not the kinetic settings of '{header.get('name')}': "
                                                  + "; ".join(differences) + "."))
        if settings and self.compareMaps and self._compareMapsKey != self._settingsKey(settings):
            lines.append(WARNING_STYLE.format("The Modeling step's settings changed: the maps are made again."))
        self.kineticsLabel.text = "<br>".join(lines)
        self.remakeMapsButton.enabled = not error
        self.mapCombo.enabled = self.mapCombo.count > 1

    def _adoptCompareMap(self):
        """A saved scene: the map compared and the maps made with it (same series, model and settings) taken back."""
        node = self.node(COMPARE_MAP_ROLE)
        if node is None or not slicer.mrmlScene.IsNodePresent(node):
            return
        state = node.GetAttribute(KM_PREFIX + "FitState")
        sequenceId = node.GetAttribute(KM_PREFIX + "Sequence")
        maps = {}
        for other in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
            if (other.GetAttribute(KM_PREFIX + "FitState") == state and other.GetAttribute(KM_PREFIX + "Sequence")
                    == sequenceId and other.GetAttribute(KM_PREFIX + "Model") == node.GetAttribute(KM_PREFIX + "Model")
                    and other.GetAttribute(KM_PREFIX + "Map")):
                maps[other.GetAttribute(KM_PREFIX + "Map")] = other
        maps[node.GetAttribute(KM_PREFIX + "Map")] = node
        self.compareMaps = maps
        self._compareMapsKey = None         # the settings checked again on Compare (the IDs of a loaded scene)
        self._fillMapCombo()
        index = self.mapCombo.findData(node.GetAttribute(KM_PREFIX + "Map"))
        if index >= 0:
            self.mapCombo.setCurrentIndex(index)
