"""
The reports of the Dynamic Neuro module (PDF, or the pages as images in the DICOM database):

- Review/Compare: the Static Neuro report (StaticNeuroPETLib.report) on the compared parametric map, with the dynamic
  series' patient, timing and frames (DynamicReportMixin overrides its hooks).
- Modeling: the kinetic modelling of the regions (the Modeling step's fits): the patient and the series, the model and
  its settings, the time-activity curves (the selected target and the reference with their fits, the other regions
  faint behind them, as the Modeling step draws them; more target / reference pairs picked in a dialog), every frame
  drawn from the series (inverted grey, the whole field of view) and the regions' results sorted by name.

Part of the Dynamic Neuro module (DynamicNeuroPET.py): the methods of DynamicReportMixin, mixed into its widget.
"""

import json
import logging
import os

import ctk
import numpy as np
import qt
import slicer

from NeuroWorkflowLib.base import NEURO_PREFIX, STEP_MODEL, _button, _ijkToRas
from NeuroPETLib import neuroreport as rp

from StaticNeuroPETLib.constants import REPORT_BANNER_PATH

DYNAMIC_REPORT_TITLE = "Dynamic brain PET report"
MODELING_REPORT_TITLE = "Dynamic brain PET kinetic modeling report"
SETTINGS_EXTRA_TACS = "DynamicNeuro/ReportExtraTacs"     # JSON: the region names last picked for more TAC plots
FRAME_COLUMNS = 6
FRAME_PERCENTILE = 99.5         # each frame's window: 0 to this percentile of its slice (early frames are faint)
TAC_PLOT_MM = 105.0             # the main TAC plot's height on the page
EXTRA_TAC_MM = 55.0             # the plots of more target / reference pairs, two in a row
# the rows of the patient table a kinetic report does not have (no comparison with a database)
NOT_IN_MODELING = ("Normal database", "Normals compared", "Score", "Image values", "Smoothing", "Regions",
                   "Image", "Reference region")


class DynamicReportMixin:
    """The report methods of DynamicNeuroPETWidget (with StaticNeuroPETLib.report.ReportMixin)."""

    REPORT_TITLE = DYNAMIC_REPORT_TITLE
    REPORT_FILE_SUFFIX = "NeuroPET_Dynamic"

    @property
    def REPORT_IMAGE_TITLE(self):
        return self._selectedMapName() or "Map"

    # --- the series: patient, timing, frames ---------------------------------------------------------------------
    def _reportSourceNode(self):
        """A frame of the series that came from the DICOM database (its DICOM values, its study), else the summed
        image."""
        sequence = self.sequence()
        if sequence is None:
            return None
        for index in range(sequence.GetNumberOfDataNodes()):
            node = sequence.GetNthDataNode(index)
            if node is not None and node.GetAttribute("DICOM.instanceUIDs"):
                return node
        return self.logic.displayImage(sequence)

    @staticmethod
    def _intervals(sequence):
        try:
            from DynamicPETModeling import DynamicPETModelingLogic
            return DynamicPETModelingLogic.intervals(sequence), DynamicPETModelingLogic
        except Exception:
            logging.warning("Neuro: the frame times of the report could not be read", exc_info=True)
            return [], None

    def _reportSubject(self):
        """(patient, study, scan, dicom values) of the dynamic series."""
        from NeuroPETLib import staticneuro as sn
        sequence = self.sequence()
        source = self._reportSourceNode()
        dicom = self._reportDicomValues(source)
        name, patientId = self.logic.patientNameAndId(sequence)
        age, sex = self.logic.ageAndSex(sequence)
        weight = None
        try:
            weight = float(dicom.get("PatientWeight")) if dicom.get("PatientWeight") else None
        except ValueError:
            weight = None
        patient = {"name": name or rp.formatPersonName(dicom.get("PatientName")),
                   "id": patientId or dicom.get("PatientID", ""),
                   "birthDate": rp.parseDate(dicom.get("PatientBirthDate")), "age": age,
                   "sex": dict(sn.SEXES).get(sex) if sex else "", "weight": weight}
        study = rp.studyInfo(dicom)
        intervals, logic = self._intervals(sequence)
        if intervals:
            first, last = float(intervals[0]["start"]), float(intervals[-1]["end"])
            study["durationMinutes"] = (last - first) / 60.0           # the whole series, not one frame
            injection = logic.injectionTimeS(intervals) if logic is not None else None
            if injection is not None:
                study["uptakeMinutes"] = (first - float(injection)) / 60.0
            study["frames"] = [float(i["end"]) - float(i["start"]) for i in intervals]
        info = self.logic.seriesInfo(sequence) or {}
        tracer = dicom.get("Radiopharmaceutical") or ""
        if info.get("tracer") and info["tracer"].lower() not in tracer.lower():
            tracer = f"{info['tracer']} ({tracer})" if tracer else info["tracer"]
        scanner = " ".join(v for v in (dicom.get("Manufacturer"), dicom.get("ManufacturerModelName")) if v)
        scan = {"modality": info.get("modality") or dicom.get("Modality", ""),
                "radionuclide": info.get("nuclide") or dicom.get("RadionuclideCodeMeaning", ""),
                "radiopharmaceutical": tracer, "scanner": scanner,
                "image": sequence.GetName() if sequence is not None else ""}
        return patient, study, scan, dicom

    def _reportComparison(self):
        from StaticNeuroPETLib.report import ReportMixin, NORMALIZATION_TITLES
        comparison = ReportMixin._reportComparison(self)
        sequence = self.sequence()
        method = sequence.GetAttribute(NEURO_PREFIX + "Method") if sequence is not None else None
        template = sequence.GetAttribute(NEURO_PREFIX + "Template") if sequence is not None else None
        if method or template:
            comparison["normalization"] = (NORMALIZATION_TITLES.get(method, method or "")
                                           + (f" ({template})" if template else "")).strip()
        comparison["units"] = self._selectedMapName() or ""
        comparison["smoothing"] = None if not (self.reviewResult or {}).get("header") else comparison.get("smoothing")
        return comparison

    def _frameRows(self, study):
        frames = study.get("frames") or []
        return [("Frames", f"{len(frames)}: {rp.frameSchemeText(frames)}")] if frames else []

    def _reportInfoRows(self, patient, study, scan, comparison):
        rows = self._frameRows(study)
        modeling = self.stepWidget(STEP_MODEL)
        if modeling is not None:
            try:
                rows.append(("Kinetic model", self._modelTitle(modeling)))
            except Exception:
                pass
        return rp.infoRows(patient, study, scan, comparison, extra=rows)

    # --- the Modeling step's report: its box ---------------------------------------------------------------------
    def _modelingReportBox(self):
        """The report box under the embedded Modeling panel (made once; shown only while the step is shown)."""
        box = getattr(self, "modelReportBox", None)
        if box is not None:
            return box
        panel = self.embedded.get(STEP_MODEL) if hasattr(self, "embedded") else None
        layout = panel.layout() if panel is not None else None
        if layout is None:
            return None
        box = ctk.ctkCollapsibleButton()
        box.text = "Report (Mergen Neuro PET)"
        box.collapsed = True
        form = qt.QFormLayout(box)
        self.modelReportComment = qt.QPlainTextEdit()
        self.modelReportComment.setFixedHeight(60)
        self.modelReportComment.placeholderText = "Comment / conclusion (optional, at the end of the report)"
        form.addRow("Comment:", self.modelReportComment)
        self.modelReportOpenCheck = qt.QCheckBox("Open the PDF when saved")
        self.modelReportOpenCheck.checked = bool(self._reportSettings().get("open", True))
        form.addRow(self.modelReportOpenCheck)
        self.modelReportPdfButton = _button("Save PDF report...")
        self.modelReportPdfButton.toolTip = (
            "A4 PDF: the patient and the series, the kinetic model and its settings, the time-activity curves (the "
            "selected target and the reference with their fits; more regions picked in a dialog), every frame of "
            "the series and the regions' results sorted by name. Fit all first.")
        form.addRow(self.modelReportPdfButton)
        self.modelReportDicomButton = _button("Save to the DICOM database")
        self.modelReportDicomButton.toolTip = ("The same report's pages as images (secondary capture), straight into "
                                               "Slicer's DICOM database, in the patient's study")
        form.addRow(self.modelReportDicomButton)
        self.modelReportLabel = qt.QLabel()
        self.modelReportLabel.wordWrap = True
        self.modelReportLabel.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        form.addRow(self.modelReportLabel)
        last = layout.itemAt(layout.count() - 1) if layout.count() else None
        if last is not None and last.spacerItem() is not None:
            layout.insertWidget(layout.count() - 1, box)       # above the panel's closing stretch
        else:
            layout.addWidget(box)
        self.modelReportPdfButton.connect("clicked()", lambda: self.onSaveModelingReport(False))
        self.modelReportDicomButton.connect("clicked()", lambda: self.onSaveModelingReport(True))
        self.modelReportBox = box
        return box

    def _showModelingReportBox(self, shown):
        """The box belongs to the workflow: hidden when the Modeling module is used on its own."""
        try:
            box = self._modelingReportBox() if shown else getattr(self, "modelReportBox", None)
            if box is not None:
                box.visible = bool(shown)
        except Exception:
            logging.warning("Neuro: the Modeling report box could not be shown", exc_info=True)

    # --- the Modeling step's report: content ---------------------------------------------------------------------
    @staticmethod
    def _kin():
        from DICOMGatedDynamicPETLib import kinetics as kin
        return kin

    def _modelTitle(self, modeling):
        kin = self._kin()
        key = getattr(modeling, "resultsModel", None) or modeling.modelKey()
        return kin.model(key).title

    def _modelingReady(self, modeling):
        return (modeling is not None and getattr(modeling, "analysis", None) is not None
                and bool(getattr(modeling, "results", None)))

    def _modelRows(self, modeling):
        """(label, value) of the model and its settings."""
        kin = self._kin()
        analysis = modeling.analysis
        key = getattr(modeling, "resultsModel", None) or modeling.modelKey()
        spec = kin.model(key)
        options = modeling.options() or {}
        rows = [("Model", spec.title)]
        reference = modeling.referenceId()
        if spec.inputKind == "reference":
            rows.append(("Reference region", analysis.name(reference)))
        elif modeling.usesPlasmaFile():
            rows.append(("Input", "Plasma curve (file)"))
        else:
            rows.append(("Input (blood pool)", analysis.name(reference)))
        rows.append(("Target region (plots)", analysis.name(modeling.targetId())))
        if key == "SRTM2":
            k2ref = options.get("k2ref")
            note = getattr(modeling.logic, "lastK2ref", None)
            if k2ref:
                rows.append(("Reference k2'", f"{kin.formatNumber(k2ref)} 1/min (fixed)"))
            elif note:
                rows.append(("Reference k2'", f"{kin.formatNumber(note[0])} 1/min (median of {note[1]} SRTM fits)"))
        if spec.method == "graphical":
            rows.append(("t*", "automatic" if options.get("autoTStar") else f"{options.get('tStar'):g} min"))
        if spec.method == "ratio" and options.get("window"):
            start, end = options["window"]
            rows.append(("Time window", f"{start:g}–{end:g} min"))
        weighting = getattr(modeling, "weightingCombo", None)
        if weighting is not None:
            rows.append(("Weighting", weighting.currentText))
        timeZero = getattr(modeling, "timeZeroCombo", None)
        if timeZero is not None:
            rows.append(("Time zero", timeZero.currentText))
        fitted = sum(1 for r in modeling.results.values() if not r.get("error"))
        rows.append(("Regions fitted", f"{fitted} of {len(analysis.segments)}"))
        rows.append(("Frame values", analysis.units or rp.NOT_KNOWN))
        return rows

    def _curve(self, modeling, segmentId, emphasis, role=""):
        analysis = modeling.analysis
        frames = analysis.frames
        mids = np.where(frames.valid, frames.mid, np.nan)
        segment = analysis.segments[segmentId]
        curve = {"name": segment["name"], "color": segment["color"], "times": mids,
                 "values": np.where(frames.valid, segment["values"], np.nan), "emphasis": emphasis, "role": role}
        result = modeling.results.get(segmentId)
        if result is not None and not result.get("error"):          # every curve's fit, as the module draws them
            if result.get("curve") is not None:
                curve["fitTimes"], curve["fitValues"] = result["curve"]
            elif result.get("fitted") is not None:
                curve["fitTimes"], curve["fitValues"] = mids, result["fitted"]
        return curve

    def _markers(self, modeling):
        kin = self._kin()
        key = getattr(modeling, "resultsModel", None) or modeling.modelKey()
        spec = kin.model(key)
        result = modeling.results.get(modeling.targetId())
        tStar = None
        if spec.method == "graphical":          # the fit's t*, else the fixed one (as the module's plot)
            options = modeling.options() or {}
            tStar = (result["values"].get("t*") if result and not result.get("error")
                     else (None if options.get("autoTStar") else options.get("tStar")))
        window = (modeling.options() or {}).get("window") if spec.method == "ratio" else None
        return tStar, window

    def _tacImages(self, modeling, extraIds):
        """(main plot, [more plots]) as QImages: the target and the reference (the other regions faint), then one
        plot per picked region with the reference."""
        kin = self._kin()
        analysis = modeling.analysis
        target, reference = modeling.targetId(), modeling.referenceId()
        plasma = kin.model(getattr(modeling, "resultsModel", None) or modeling.modelKey()).inputKind == "plasma"
        referenceRole = "blood pool" if plasma else "reference"
        tStar, window = self._markers(modeling)
        curves = []
        for segmentId in analysis.segments:
            role = "target" if segmentId == target else (referenceRole if segmentId == reference else "")
            curves.append(self._curve(modeling, segmentId, bool(role), role))
        inputCurve = None
        if modeling.usesPlasmaFile():               # the plasma file is the input, as in the module's plot
            try:
                times, values = modeling.plasma()
                inputCurve = {"name": "Plasma (file)", "color": "#d02020", "times": np.asarray(times, dtype=float),
                              "values": np.asarray(values, dtype=float), "emphasis": True, "role": "input"}
                curves.append(inputCurve)
                referenceRole = "plasma input"
            except Exception:
                logging.warning("Neuro: the plasma curve could not be read for the report", exc_info=True)
        main = rp.tacPlotImage(curves, analysis.units, title=f"{analysis.name(target)} and the {referenceRole}",
                               tStar=tStar, window=window)
        extra = []
        for segmentId in extraIds:
            if segmentId not in analysis.segments or segmentId in (target, reference):
                continue
            partner = inputCurve or self._curve(modeling, reference, True, referenceRole)
            pair = [self._curve(modeling, segmentId, True), partner]
            extra.append(rp.tacPlotImage(pair, analysis.units, width=1400, height=900,
                                         title=analysis.name(segmentId), window=window))
        return main, extra

    def _resultTable(self, modeling):
        """(columns, rows, swatches) of the regions' results sorted by name."""
        kin = self._kin()
        analysis = modeling.analysis
        key = getattr(modeling, "resultsModel", None) or modeling.modelKey()
        columns = ["Region"] + [kin.columnTitle(c) for c in kin.resultColumns(key)] + ["Notes"]
        reference = modeling.referenceId()
        rows, colors = [], {}
        for segmentId, segment in analysis.segments.items():
            result = modeling.results.get(segmentId)
            note = "reference" if segmentId == reference else ""
            if result is None or result.get("error"):
                values = [""] * (len(columns) - 2)
                note = "; ".join(v for v in (note, (result or {}).get("error", "")) if v)
            else:
                values = [kin.formatNumber(v) for v in kin.resultRow(result)]
                note = "; ".join(v for v in [note] + list(result.get("warnings") or []) if v)
            rows.append([segment["name"]] + values + [note])
            colors[segment["name"]] = segment["color"]
        rows = rp.sortedTableRows(rows)
        swatches = []
        for row in rows:
            color = qt.QColor(colors.get(row[0]) or "#808080")
            swatches.append((color.redF(), color.greenF(), color.blueF()))
        return columns, rows, swatches

    def _targetPoint(self, modeling, frame):
        """RAS point of the target region's centre (the frames' slice), else the grid's centre."""
        try:
            segmentation = modeling.analysis.segmentation
            mask = slicer.util.arrayFromSegmentBinaryLabelmap(segmentation, modeling.targetId(), frame)
            if mask is not None and mask.any():
                k, j, i = (float(v.mean()) for v in np.nonzero(mask))
                return (np.asarray(_ijkToRas(frame)) @ np.array([i, j, k, 1.0]))[:3]
        except Exception:
            logging.debug("Neuro: target centre for the frames", exc_info=True)
        array = slicer.util.arrayFromVolume(frame)
        low, high = rp.gridBox(array.shape, _ijkToRas(frame))
        return 0.5 * (low + high)

    def _frameImages(self, modeling, work):
        """[(QImage, caption)] of every frame: an axial slice through the target region, the whole field of view,
        inverted grey from 0 to its own 99.5th percentile."""
        analysis = modeling.analysis
        sequence = analysis.sequence
        count = sequence.GetNumberOfDataNodes()
        if not count:
            return []
        first = sequence.GetNthDataNode(0)
        matrix = _ijkToRas(first)
        box = rp.gridBox(slicer.util.arrayFromVolume(first).shape, matrix)
        point = self._targetPoint(modeling, first)
        spacing = max(0.5, min(first.GetSpacing()))
        points = rp.planePoints(box, point, (-1, 0, 0), (0, 1, 0), spacing)     # Slicer's axial (patient's right left)
        frames = analysis.frames
        out = []
        for index in range(count):
            node = sequence.GetNthDataNode(index)
            values = rp.sampleVolume(slicer.util.arrayFromVolume(node), _ijkToRas(node), points)
            positive = values[np.isfinite(values) & (values > 0)]
            high = float(np.percentile(positive, FRAME_PERCENTILE)) if positive.size else 1.0
            path = rp.writePng(rp.colorize(values, rp.INVERTED_GREY, 0.0, high if high > 0 else 1.0),
                               os.path.join(work, f"frame_{index + 1:03d}.png"))
            caption = (rp.frameCaption(index + 1, float(frames.rawStart[index]), float(frames.rawEnd[index]))
                       if index < len(frames.rawStart) else str(index + 1))
            out.append((qt.QImage(path), caption))
        return out

    # --- the extra TACs dialog -----------------------------------------------------------------------------------
    def _askExtraTacs(self, modeling):
        """The regions picked for more TAC plots (each with the reference); None when cancelled."""
        analysis = modeling.analysis
        skip = {modeling.targetId(), modeling.referenceId()}
        choices = sorted(((analysis.name(s), s) for s in analysis.segments if s not in skip),
                         key=lambda item: rp._naturalKey(item[0]))
        try:
            remembered = set(json.loads(str(qt.QSettings().value(SETTINGS_EXTRA_TACS) or "[]")))
        except ValueError:
            remembered = set()
        dialog = qt.QDialog(slicer.util.mainWindow())
        dialog.setWindowTitle("Modeling report")
        layout = qt.QVBoxLayout(dialog)
        text = qt.QLabel(f"The report plots {analysis.name(modeling.targetId())} with the reference "
                         f"({analysis.name(modeling.referenceId())}). More regions to plot, each with the "
                         "reference:")
        text.wordWrap = True
        layout.addWidget(text)
        combo = None
        try:
            combo = ctk.ctkCheckableComboBox()
            for name, segmentId in choices:
                combo.addItem(name, segmentId)
            model = combo.checkableModel() if hasattr(combo, "checkableModel") else combo.model()
            for row, (name, _) in enumerate(choices):
                combo.setCheckState(model.index(row, 0), qt.Qt.Checked if name in remembered else qt.Qt.Unchecked)
            layout.addWidget(combo)
        except Exception:
            logging.debug("Neuro: no ctkCheckableComboBox, a list instead", exc_info=True)
            combo = None
        listWidget = None
        if combo is None:
            listWidget = qt.QListWidget()
            for name, segmentId in choices:
                item = qt.QListWidgetItem(name)
                item.setFlags(item.flags() | qt.Qt.ItemIsUserCheckable)
                item.setCheckState(qt.Qt.Checked if name in remembered else qt.Qt.Unchecked)
                item.setData(qt.Qt.UserRole, segmentId)
                listWidget.addItem(item)
            listWidget.setMinimumHeight(220)
            layout.addWidget(listWidget)
        buttons = qt.QDialogButtonBox()         # buttons added one by one (PythonQt: the flags constructor is unsure)
        buttons.addButton("Make the report", qt.QDialogButtonBox.AcceptRole)
        buttons.addButton("Cancel", qt.QDialogButtonBox.RejectRole)
        buttons.connect("accepted()", dialog.accept)
        buttons.connect("rejected()", dialog.reject)
        layout.addWidget(buttons)
        dialog.resize(420, dialog.sizeHint.height())
        if not dialog.exec_():
            return None
        picked = []
        if combo is not None:
            model = combo.checkableModel() if hasattr(combo, "checkableModel") else combo.model()
            for row, (name, segmentId) in enumerate(choices):
                if combo.checkState(model.index(row, 0)) == qt.Qt.Checked:
                    picked.append((name, segmentId))
        else:
            for row in range(listWidget.count):
                item = listWidget.item(row)
                if item.checkState() == qt.Qt.Checked:
                    picked.append((item.text(), item.data(qt.Qt.UserRole)))
        qt.QSettings().setValue(SETTINGS_EXTRA_TACS, json.dumps([name for name, _ in picked]))
        return [segmentId for _, segmentId in picked]

    # --- the Modeling step's report: drawing and saving ----------------------------------------------------------
    def _renderModelingReport(self, path, work, modeling, extraIds, patient, study, scan, comparison,
                              imageFolder=None):
        report = rp.PdfReport(path, MODELING_REPORT_TITLE, REPORT_BANNER_PATH)
        report.heading("Patient and series")
        report.keyValues(rp.infoRows(patient, study, scan, comparison, extra=self._frameRows(study),
                                     leaveOut=NOT_IN_MODELING))
        report.paragraph(rp.DISCLAIMER, size=9, bold=True, color=rp.DISCLAIMER_COLOR)     # first page, red
        report.keepWithNext()
        report.heading("Kinetic model")
        report.keyValues(self._modelRows(modeling))
        main, extra = self._tacImages(modeling, extraIds)
        report.keepWithNext()
        report.heading("Time-activity curves")
        report.imageRow([main], maxHeightMm=TAC_PLOT_MM)
        report.paragraph("Solid lines and points: the regions' mean values of the frames; dashed: the model's fits. "
                         "The other regions are drawn faint behind them.", size=7, italic=True)
        for index in range(0, len(extra), 2):
            report.imageRow(extra[index:index + 2], maxHeightMm=EXTRA_TAC_MM)
        frames = []
        try:
            frames = self._frameImages(modeling, work)
        except Exception:
            logging.exception("Neuro: the frames of the report could not be drawn")
        if frames:
            report.pageBreak()
            report.heading(f"Frames ({len(frames)}, axial slice through {modeling.analysis.name(modeling.targetId())})")
            first = frames[0][0]
            aspect = first.height() / float(max(1, first.width()))
            report.imageGrid([image for image, _ in frames], FRAME_COLUMNS, captions=[c for _, c in frames],
                             background=(255, 255, 255), cellAspect=aspect)
            report.paragraph(f"Inverted grey; each frame from 0 to its own {FRAME_PERCENTILE:g}th percentile. Times "
                             "after time zero (min).", size=7, italic=True)
        columns, rows, swatches = self._resultTable(modeling)
        report.keepWithNext()
        report.heading("Regional results (sorted by region)")
        report.table(columns, rows, swatches=swatches)
        comment = self.modelReportComment.toPlainText().strip() if hasattr(self, "modelReportComment") else ""
        if comment:
            report.keepWithNext()
            report.heading("Comment")
            report.paragraph(comment, size=9.5)
        if imageFolder:
            return report.renderImages(imageFolder)
        return report.finish()

    def onSaveModelingReport(self, asDicom=False):
        title = "Modeling report"
        modeling = self.stepWidget(STEP_MODEL)
        if not self._modelingReady(modeling):
            slicer.util.warningDisplay("Fit the regions first (Fit all in the Modeling step).", windowTitle=title)
            return
        extraIds = self._askExtraTacs(modeling)
        if extraIds is None:
            return
        self._modelingReportBox()
        label = self.modelReportLabel

        def render(path, work, patient, study, scan, comparison, imageFolder=None):
            return self._renderModelingReport(path, work, modeling, extraIds, patient, study, scan, comparison,
                                              imageFolder=imageFolder)
        if asDicom:
            self._saveDicomReport("DICOM " + title.lower(), render, MODELING_REPORT_TITLE, label)
            return
        patient, study, scan, dicom = self._reportSubject()
        comparison = {}
        self._savePdfReport("PDF " + title.lower(), lambda path, work: render(path, work, patient, study, scan,
                                                                               comparison),
                            patient, study, "NeuroPET_Modeling", label, self.modelReportOpenCheck)
