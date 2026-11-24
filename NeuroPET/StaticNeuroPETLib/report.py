"""
Review: the PDF report (the patient and the scan, the slices on screen, the 3D-SSP in 16 views on white, the region
table sorted by name).

Part of the Static Neuro PET/SPECT module (StaticNeuroPET.py): the methods of ReportMixin, mixed into
StaticNeuroPETWidget there. The layout of the PDF and its drawing are in NeuroPETLib.neuroreport.
"""

import json
import logging
import os
import shutil
import tempfile

import ctk
import numpy as np
import qt
import slicer

from NeuroWorkflowLib.base import METHOD_ANATOMY, METHOD_PET, METHOD_PET_TEMPLATE, NEURO_PREFIX, _button
from NeuroPETLib import neuroreport as rp, neuroreview as nr, normaldb, ssp, staticneuro as sn

from StaticNeuroPETLib.constants import NO_SCORE_COLORMAP, REPORT_BANNER_PATH, SETTINGS_REPORT, SSP_CAMERA_DISTANCE

# DICOM keywords read for the report (the image's first instance); those of the radiopharmaceutical from the first
# item of its RadiopharmaceuticalInformationSequence
REPORT_DICOM_KEYWORDS = ("Modality", "PatientName", "PatientID", "PatientBirthDate", "PatientSex", "PatientAge",
                         "PatientWeight", "StudyDate", "SeriesDate", "SeriesTime", "AcquisitionDate",
                         "AcquisitionTime", "AcquisitionDateTime", "ActualFrameDuration", "NumberOfFrames",
                         "Manufacturer", "ManufacturerModelName", "InstitutionName", "StudyDescription")
REPORT_RADIOPHARMACEUTICAL_KEYWORDS = ("Radiopharmaceutical", "RadiopharmaceuticalStartTime",
                                       "RadiopharmaceuticalStartDateTime", "RadionuclideTotalDose")
REPORT_SLICE_MM = 50.0      # a slice's width in the PDF: the info table, both slice rows and their color bars on page 1
NORMALIZATION_TITLES = {METHOD_PET: "PET/SPECT to MR template", METHOD_ANATOMY: "CT/MR to MR template",
                        METHOD_PET_TEMPLATE: "PET/SPECT to PET template"}


class ReportMixin:
    """The PDF report methods of StaticNeuroPETWidget."""

    REPORT_TITLE = rp.TITLE
    REPORT_FILE_SUFFIX = "NeuroPET"

    def _buildReportBox(self, layout):
        box = ctk.ctkCollapsibleButton()
        box.text = "PDF report"
        box.collapsed = True
        form = qt.QFormLayout(box)
        self.reportCommentEdit = qt.QPlainTextEdit()
        self.reportCommentEdit.setFixedHeight(70)
        self.reportCommentEdit.placeholderText = "Comment / conclusion (optional, at the end of the report)"
        form.addRow("Comment:", self.reportCommentEdit)
        self.reportOpenCheck = qt.QCheckBox("Open the PDF when saved")
        self.reportOpenCheck.checked = True
        form.addRow(self.reportOpenCheck)
        self.reportButton = _button("Save PDF report...")
        self.reportButton.toolTip = (
            "A4 PDF under the Mergen banner: the patient and the scan (DICOM), the comparison; the slices as on screen (PET and scores); the "
            "3D-SSP in 16 views on a white background (the scores, then those below the threshold); the region "
            "table sorted by name.")
        form.addRow(self.reportButton)
        self.reportLabel = qt.QLabel()
        self.reportLabel.wordWrap = True
        self.reportLabel.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        form.addRow(self.reportLabel)
        layout.addWidget(box)
        saved = self._reportSettings()
        self.reportOpenCheck.checked = bool(saved.get("open", True))
        self.reportButton.connect("clicked()", self.onSaveReport)

    @staticmethod
    def _reportSettings():
        try:
            return json.loads(str(qt.QSettings().value(SETTINGS_REPORT) or "") or "{}")
        except ValueError:
            return {}

    def _saveReportSettings(self, folder):
        values = self._reportSettings()
        values.update(open=bool(self.reportOpenCheck.checked), folder=folder)
        qt.QSettings().setValue(SETTINGS_REPORT, json.dumps(values))

    # --- what the report says ------------------------------------------------------------------------------
    @staticmethod
    def _reportDicomValues(volume):
        """{DICOM keyword: text} of the image's first instance (REPORT_DICOM_KEYWORDS, the radiopharmaceutical's);
        {} when the image did not come from the DICOM database."""
        values = {}
        try:
            uids = (volume.GetAttribute("DICOM.instanceUIDs") or "").split() if volume is not None else []
            if not uids or slicer.dicomDatabase is None:
                return values
            path = slicer.dicomDatabase.fileForInstance(uids[0])
            if not path:
                return values
            import pydicom
            dataset = pydicom.dcmread(path, stop_before_pixels=True)
            for keyword in REPORT_DICOM_KEYWORDS:
                value = dataset.get(keyword)
                if value not in (None, ""):
                    values[keyword] = str(value)
            items = dataset.get("RadiopharmaceuticalInformationSequence")
            if items:
                item = items[0]
                for keyword in REPORT_RADIOPHARMACEUTICAL_KEYWORDS:
                    value = item.get(keyword)
                    if value not in (None, ""):
                        values[keyword] = str(value)
                codes = item.get("RadionuclideCodeSequence")
                if codes and codes[0].get("CodeMeaning"):
                    values["RadionuclideCodeMeaning"] = str(codes[0].get("CodeMeaning"))
        except Exception:
            logging.warning("Static Neuro: could not read the DICOM values of the report", exc_info=True)
        return values

    def _reportContent(self):
        """(patient, study, scan, comparison, dicom values) of the report (neuroreport.infoRows)."""
        result = self.reviewResult or {}
        volume = self.reviewVolume()
        attributes = self.logic.attributes(volume)
        dicom = self._reportDicomValues(volume)
        name, patientId = self.logic.patientNameAndId(volume)
        age, sex = self._reviewAgeAndSex()
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
        prefixes = (sn.STATIC_PREFIX, sn.SPECT_PREFIX)
        for prefix in prefixes:         # an imported (BIDS) image: its sidecar's timing
            if study["uptakeMinutes"] is None and attributes.get(prefix + "StartFromInjectionS"):
                try:
                    study["uptakeMinutes"] = float(attributes[prefix + "StartFromInjectionS"]) / 60.0
                except ValueError:
                    pass
            if study["durationMinutes"] is None and attributes.get(prefix + "DurationS"):
                try:
                    study["durationMinutes"] = float(attributes[prefix + "DurationS"]) / 60.0
                except ValueError:
                    pass
        info = self.logic.imageInfo(volume)
        nuclide = info.get("nuclide") or dicom.get("RadionuclideCodeMeaning", "")
        tracer = (dicom.get("Radiopharmaceutical") or attributes.get(sn.STATIC_PREFIX + "Radiopharmaceutical")
                  or attributes.get(sn.SPECT_PREFIX + "IsotopeDicomText") or "")
        if info.get("tracer") and info["tracer"].lower() not in tracer.lower():
            tracer = f"{info['tracer']} ({tracer})" if tracer else info["tracer"]
        scanner = " ".join(v for v in (dicom.get("Manufacturer"), dicom.get("ManufacturerModelName")) if v)
        if not scanner:
            scanner = " ".join(v for v in (attributes.get(p + k) for p in prefixes
                                           for k in ("Manufacturer", "ScannerModel")) if v)
        scan = {"modality": info.get("modality") or dicom.get("Modality", ""), "radionuclide": nuclide,
                "radiopharmaceutical": tracer, "scanner": scanner,
                "image": volume.GetName() if volume is not None else ""}
        header = result.get("header")
        comparison = {"score": dict(nr.SCORES).get(self._scoreKind(), self._scoreKind()) if header else "",
                      "units": self._patientUnits(), "smoothing": result.get("fwhm") if header else None}
        if header is not None:
            comparison["database"] = header.get("name", "")
            groupKey = result.get("groupKey") or normaldb.MASTER
            group = next((title for k, title, _ in nr.statisticsGroups(header) if k == groupKey), groupKey)
            comparison["normals"] = ((normaldb.MASTER_TITLE if groupKey == normaldb.MASTER else f"Age group {group}")
                                     + (f" (n = {result.get('n')})" if result.get("n") else ""))
            comparison["reference"] = normaldb.referenceTitle(header)
        else:
            comparison["database"] = "none (regional values only)"
            referenceId = self._reviewReferenceId()
            segmentation = self.node("Segmentation")
            if referenceId and segmentation is not None:
                comparison["reference"] = segmentation.GetSegmentation().GetSegment(referenceId).GetName()
        segmentation = result.get("segmentation") or self.node("Segmentation")
        comparison["regions"] = segmentation.GetName() if segmentation is not None else ""
        method = attributes.get(NEURO_PREFIX + "Method")
        template = attributes.get(NEURO_PREFIX + "Template")
        if method or template:
            comparison["normalization"] = (NORMALIZATION_TITLES.get(method, method or "")
                                           + (f" ({template})" if template else "")).strip()
        return patient, study, scan, comparison, dicom

    # --- the images ----------------------------------------------------------------------------------------
    @staticmethod
    def _withAlpha(rgb):
        rgb = np.asarray(rgb, dtype=float)
        if rgb.shape[1] == 4:
            return rgb
        return np.hstack([rgb, np.ones((len(rgb), 1))])

    def _reportSlices(self, report):
        """The review's slices as on screen (PET row, score row) with a colour bar under each row."""
        layoutManager = slicer.app.layoutManager()
        widgets = self._reviewSliceWidgets() if layoutManager is not None else []
        report.keepWithNext()
        report.heading("Slices")
        if not widgets:
            report.paragraph("The slices are not in the layout on screen: pick a layout with the slices to have "
                             "them in the report.", italic=True)
            return
        legends = [node for node in slicer.util.getNodesByClass("vtkMRMLColorLegendDisplayNode")
                   if node.GetVisibility()]
        for node in legends:        # the report's own colour bars instead
            node.SetVisibility(False)
        images, captions, titles = [], [], []
        try:
            slicer.app.processEvents()
            titles_ = {nr.PET_ROW: "PET", nr.SCORE_ROW: self._scoreKind() + " score"}
            for row in (nr.PET_ROW, nr.SCORE_ROW):
                rowWidgets = [widget for r, widget in widgets if r == row]
                if not rowWidgets:
                    continue
                titles.append(titles_[row])
                for (orientation, _), widget in zip(nr.ORIENTATIONS, rowWidgets):
                    view = widget.sliceView()
                    view.forceRender()
                    images.append(ctk.ctkWidgetsUtils.grabWidget(view))
                    captions.append(orientation)
        finally:
            for node in legends:
                node.SetVisibility(True)
        aspects = [image.height() / float(image.width()) for image in images if image.width() > 0]
        aspect = float(np.clip(np.median(aspects), 0.6, 1.3)) if aspects else 1.0
        volume = self.reviewVolume()
        if titles and titles[0] == "PET":
            report.imageGrid(images[:3], 3, captions=captions[:3], rowTitles=["PET"], cellAspect=aspect,
                             maxCellMm=REPORT_SLICE_MM)
            low, high = self.petWindowRange.minimumValue, self.petWindowRange.maximumValue
            report.colorbar(self._withAlpha(self._rgbOf(self.petColorCombo.currentText)), low, high,
                            self._petUnits(volume), "{:.2f}")
            images, captions, titles = images[3:], captions[3:], titles[1:]
        if images:
            scored = self.scoreNode is not None and slicer.mrmlScene.IsNodePresent(self.scoreNode)
            report.imageGrid(images, 3, captions=captions, rowTitles=[titles[0] if scored else "PET (colors)"],
                             cellAspect=aspect,
                             maxCellMm=REPORT_SLICE_MM)
            if scored:
                rgba, (low, high) = self._scoreColors()
                report.colorbar(rgba, low, high, self._scoreKind())
            else:
                low, high = self.petWindowRange.minimumValue, self.petWindowRange.maximumValue
                report.colorbar(self._withAlpha(self._rgbOf(NO_SCORE_COLORMAP)), low, high, self._petUnits(volume),
                                "{:.2f}")

    def _reportSspViews(self, values, keys):
        """[(key, polydata, camera, parallel scale)] of the 3D-SSP views keys with vertex values (as the review's
        views: the same cameras, zoom and parts)."""
        surface = self.reviewResult["ssp"]["surface"]
        values = np.asarray(values, np.float32)
        center = self._sspCenter()
        points = np.asarray(surface["points"], dtype=np.float64)
        half = 0.5 * (points.max(axis=0) - points.min(axis=0)) if len(points) else np.full(3, 100.0)
        scale = 1.05 * float(max(half))
        polys = {}
        out = []
        for key in keys:
            part = ssp.viewPart(key)
            if part not in polys:
                if part == ssp.PART_BRAIN and surface.get("parts") is None:
                    piece, index = surface, None
                else:
                    piece, index = ssp.piece(surface, part)
                polys[part] = self.logic.surfacePolyData(piece, values if index is None else values[index])
            direction, up = ssp.CAMERAS[key]
            position = center + SSP_CAMERA_DISTANCE * np.asarray(direction, dtype=np.float64)
            out.append((key, polys[part], (tuple(center), tuple(position), tuple(up)), scale))
        return out

    def _reportSsp(self, report, work):
        """The 3D-SSP in 16 views (2 x 4 of the scores as set in the 3D-SSP box, 2 x 4 of those below the
        threshold; 12 without the medial views) on a white background, drawn off screen."""
        data = (self.reviewResult or {}).get("ssp")
        if not data:
            return
        views, columns = nr.sspViews(self._sspMedial())
        keys = [key for key, _ in views]
        titles = [title for _, title in views]
        sides = [ssp.sideLabels(key) for key in keys]
        report.pageBreak()
        report.keepWithNext()
        report.heading("3D-SSP")
        drawn = False
        for viewSet in (nr.SSP_MAIN, nr.SSP_LOW):
            found = self._sspValues(viewSet)
            if found is None:
                continue
            values, rgba, (low, high), what = found
            try:
                paths = rp.renderSurfaceViews(self._reportSspViews(values, keys), rgba, (low, high), work,
                                              prefix=f"ssp{viewSet or 'main'}",
                                              lighting=bool(self.sspLightingCheck.checked))
                images = [qt.QImage(path) for path in paths]
            except Exception:
                logging.exception("Static Neuro: the report's 3D-SSP could not be drawn off screen")
                images = self._grabSspViews(viewSet, keys)
                if not images:
                    continue
            report.imageGrid(images, columns, captions=titles, background=None, sideLabels=sides)
            title = what if what != "patient" else self._patientUnits()
            report.colorbar(rgba, low, high, title, "{:.1f}" if what != "patient" else "{:.2f}")
            drawn = True
        if not drawn:
            report.paragraph("The 3D-SSP could not be drawn.", italic=True)

    def _grabSspViews(self, viewSet, keys):
        """The 3D-SSP views of the layout as on screen (black) when they cannot be drawn off screen; [] when the
        layout does not show them all."""
        views = self._sspViewNodes()
        images = []
        for key in keys:
            found = views.get((viewSet, key))
            if found is None:
                return []
            found[0].threeDView().forceRender()
            images.append(ctk.ctkWidgetsUtils.grabWidget(found[0].threeDView()))
        return images

    # --- the table ---------------------------------------------------------------------------------------------
    def _reportTable(self, report):
        """The region table (its columns as in the review) sorted by the regions' names; the scores at or beyond
        the threshold bold (below the normals blue, above red)."""
        data = getattr(self, "_regionTableData", None)
        if not data:
            return
        columns, rows = data
        result = self.reviewResult or {}
        scoreColumn = 1 if result.get("header") is not None and len(columns) > 1 else None
        colors, _ = self._segmentColors()
        threshold = float(self.thresholdSpin.value)
        texts, swatches, emphasis = [], [], []
        for row in rp.sortedTableRows(rows):
            cells = []
            for c, value in enumerate(row):
                if value is None:
                    cells.append("")
                elif isinstance(value, str):
                    cells.append(value)
                elif c == scoreColumn:
                    cells.append(f"{value:.2f}")
                else:
                    cells.append(f"{value:g}")
            texts.append(cells)
            swatches.append(colors.get(row[0]))
            marks = {}
            if scoreColumn is not None:
                side = rp.scoreEmphasis(row[scoreColumn], threshold)
                if side:
                    marks[scoreColumn] = rp.LOW_SCORE_COLOR if side < 0 else rp.HIGH_SCORE_COLOR
            emphasis.append(marks)
        report.keepWithNext()
        report.heading("Regional values" + (f" ({self._scoreKind()} score, sorted by region)" if scoreColumn else ""))
        report.table(list(columns), texts, swatches=swatches, emphasis=emphasis)
        if scoreColumn is not None:
            report.paragraph(f"Bold: |{self._scoreKind()}| ≥ {abs(threshold):g} (blue below the normals, red "
                             "above). Reference regions are not listed.", size=7, italic=True)

    # --- saving ------------------------------------------------------------------------------------------------
    def onSaveReport(self):
        result = self.reviewResult
        if result is None:
            slicer.util.warningDisplay("Compare the patient first (Review step).", windowTitle="PDF report")
            return
        patient, study, scan, comparison, dicom = self._reportContent()
        folder = self._reportSettings().get("folder") or os.path.expanduser("~")
        name = rp.reportFileName(patient.get("name"), study.get("date"), self.REPORT_FILE_SUFFIX)
        path = qt.QFileDialog.getSaveFileName(slicer.util.mainWindow(), "Save the PDF report",
                                              os.path.join(folder, name), "PDF (*.pdf)")
        if not path:
            return
        path = str(path)
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        self._saveReportSettings(os.path.dirname(path))
        work = tempfile.mkdtemp(prefix="NeuroReport")
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            report = rp.PdfReport(path, self.REPORT_TITLE, REPORT_BANNER_PATH)
            report.heading("Patient and scan")
            report.keyValues(rp.infoRows(patient, study, scan, comparison))
            report.paragraph(rp.DISCLAIMER, size=9, bold=True, color=rp.DISCLAIMER_COLOR)     # first page, red
            self._reportSlices(report)
            self._reportSsp(report, work)
            self._reportTable(report)
            comment = self.reportCommentEdit.toPlainText().strip()
            if comment:
                report.keepWithNext()
                report.heading("Comment")
                report.paragraph(comment, size=9.5)
            notes = rp.plainText(self.compareLabel.text)
            if notes:
                report.paragraph("Comparison: " + notes, size=7, italic=True)
            pages = report.finish()
        except Exception as error:
            logging.exception("Static Neuro: the PDF report failed")
            slicer.util.errorDisplay(f"Could not write the PDF report:\n{error}", windowTitle="PDF report")
            return
        finally:
            qt.QApplication.restoreOverrideCursor()
            shutil.rmtree(work, ignore_errors=True)
        self.reportLabel.text = f"Saved: {path} ({pages} page{'s' if pages != 1 else ''})"
        slicer.util.showStatusMessage(f"Static Neuro: PDF report saved ({path})", 5000)
        if self.reportOpenCheck.checked:
            qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(path))
