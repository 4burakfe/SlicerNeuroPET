"""
Review: the PDF report (the patient and the scan, the slices on screen, the 3D-SSP in 16 views on white, the region
table sorted by name).

Part of the Static Neuro PET/SPECT module (StaticNeuroPET.py): the methods of ReportMixin, mixed into
StaticNeuroPETWidget there. The layout of the PDF and its drawing are in NeuroPETLib.neuroreport.
"""

import json
import logging
import os
import tempfile

import ctk
import numpy as np
import qt
import slicer

from NeuroWorkflowLib.base import METHOD_ANATOMY, METHOD_PET, METHOD_PET_TEMPLATE, NEURO_PREFIX, _button, _ijkToRas
from NeuroPETLib import neuroreport as rp, neuroreview as nr, normaldb, ssp, staticneuro as sn

from StaticNeuroPETLib.constants import NO_SCORE_COLORMAP, REPORT_BANNER_PATH, SETTINGS_REPORT, SSP_CAMERA_DISTANCE

# DICOM keywords read for the report (the image's first instance); those of the radiopharmaceutical from the first
# item of its RadiopharmaceuticalInformationSequence
REPORT_DICOM_KEYWORDS = ("Modality", "PatientName", "PatientID", "PatientBirthDate", "PatientSex", "PatientAge",
                         "PatientWeight", "StudyDate", "SeriesDate", "SeriesTime", "AcquisitionDate",
                         "AcquisitionTime", "AcquisitionDateTime", "ActualFrameDuration", "NumberOfFrames",
                         "Manufacturer", "ManufacturerModelName", "InstitutionName", "StudyDescription",
                         "StudyTime", "StudyID", "AccessionNumber", "ReferringPhysicianName", "IssuerOfPatientID",
                         "StudyInstanceUID", "SOPClassUID")
REPORT_RADIOPHARMACEUTICAL_KEYWORDS = ("Radiopharmaceutical", "RadiopharmaceuticalStartTime",
                                       "RadiopharmaceuticalStartDateTime", "RadionuclideTotalDose")
REPORT_SLICE_MM = 50.0      # a slice row at most this high (mm): the info table, both rows and color bars on page 1
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
        self.reportDicomButton = _button("Save to the DICOM database")
        self.reportDicomButton.toolTip = ("The report's pages as images (secondary capture), straight into Slicer's "
                                          "DICOM database, in the patient's study (no file to pick)")
        form.addRow(self.reportDicomButton)
        self.reportLabel = qt.QLabel()
        self.reportLabel.wordWrap = True
        self.reportLabel.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        form.addRow(self.reportLabel)
        layout.addWidget(box)
        saved = self._reportSettings()
        self.reportOpenCheck.checked = bool(saved.get("open", True))
        self.reportButton.connect("clicked()", self.onSaveReport)
        self.reportDicomButton.connect("clicked()", self.onSaveDicomReport)

    @staticmethod
    def _reportSettings():
        try:
            return json.loads(str(qt.QSettings().value(SETTINGS_REPORT) or "") or "{}")
        except ValueError:
            return {}

    def _saveReportSettings(self, folder, openCheck=None):
        values = self._reportSettings()
        check = openCheck if openCheck is not None else self.reportOpenCheck
        values.update(open=bool(check.checked), folder=folder)
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

    REPORT_IMAGE_TITLE = "PET"      # the title of the slices' first row

    def _reportSourceNode(self):
        """The image the report is about: its DICOM values, its study (the DICOM report filed in it)."""
        return self.reviewVolume()

    def _reportInfoRows(self, patient, study, scan, comparison):
        return rp.infoRows(patient, study, scan, comparison)

    def _reportContent(self):
        """(patient, study, scan, comparison, dicom values) of the report (neuroreport.infoRows)."""
        patient, study, scan, dicom = self._reportSubject()
        return patient, study, scan, self._reportComparison(), dicom

    def _reportSubject(self):
        """(patient, study, scan, dicom values) of the report's image."""
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
        return patient, study, scan, dicom

    def _reportComparison(self):
        """The comparison's rows of the report (neuroreport.infoRows)."""
        result = self.reviewResult or {}
        volume = self.reviewVolume()
        attributes = self.logic.attributes(volume) if volume is not None else {}
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
        return comparison

    # --- the images ----------------------------------------------------------------------------------------
    @staticmethod
    def _withAlpha(rgb):
        rgb = np.asarray(rgb, dtype=float)
        if rgb.shape[1] == 4:
            return rgb
        return np.hstack([rgb, np.ones((len(rgb), 1))])

    def _reportSlicePoint(self, box):
        """(RAS point the slices go through, {orientation: (screen x, screen y)}) from the review's slice views (their
        slice nodes, also when the layout does not show them): the slices of the screen; the box's centre without."""
        point = 0.5 * (np.asarray(box[0]) + np.asarray(box[1]))
        axes = {}
        for orientation, _ in nr.ORIENTATIONS:
            node = None
            for row in (nr.PET_ROW, nr.SCORE_ROW):
                node = slicer.mrmlScene.GetSingletonNode(nr.sliceViewName(row, orientation), "vtkMRMLSliceNode")
                if node is not None:
                    break
            if node is None:
                continue
            matrix = node.GetSliceToRAS()
            column = [np.array([matrix.GetElement(r, c) for r in range(3)]) for c in range(4)]
            axes[orientation] = (column[0], column[1])
            normal = column[2] / (np.linalg.norm(column[2]) or 1.0)
            point = point + (float((column[3] - point) @ normal)) * normal       # onto that slice's plane
        defaults = {"Axial": ((-1, 0, 0), (0, 1, 0)), "Coronal": ((-1, 0, 0), (0, 0, 1)),
                    "Sagittal": ((0, -1, 0), (0, 0, 1))}      # Slicer's own (the patient's right on the left)
        for orientation, _ in nr.ORIENTATIONS:
            axes.setdefault(orientation, tuple(np.array(v, float) for v in defaults[orientation]))
        return point, axes

    def _reportSliceImages(self, work):
        """{row: [(orientation, PNG path)]} of the slices drawn from the images themselves (not grabbed from the
        views): the PET in its colors and window; the scores as shown (colored PET, or the score map over the PET in
        grey / black; the PET in rainbow colors without scores). Cut to the head (no black around it), at the image's
        own aspect. The fusion image is not drawn under them."""
        volume = self.reviewVolume()
        pet = slicer.util.arrayFromVolume(volume)
        petMatrix = _ijkToRas(volume)
        box = rp.contentBox(pet, petMatrix)
        point, axes = self._reportSlicePoint(box)
        spacing = max(0.5, min(volume.GetSpacing()) / 2.0)
        low, high = self.petWindowRange.minimumValue, self.petWindowRange.maximumValue
        petColors = self._rgbOf(self.petColorCombo.currentText)
        grey = np.tile(np.linspace(0, 1, 256)[:, None], (1, 3))
        score = self.scoreNode if self.scoreNode is not None and slicer.mrmlScene.IsNodePresent(self.scoreNode) \
            else None
        rgbNode = self._scoreColorsNode() if score is not None else None
        out = {nr.PET_ROW: [], nr.SCORE_ROW: []}
        for orientation, _ in nr.ORIENTATIONS:
            xAxis, yAxis = axes[orientation]
            points = rp.planePoints(box, point, xAxis, yAxis, spacing)
            petSlice = rp.sampleVolume(pet, petMatrix, points)
            images = {nr.PET_ROW: rp.colorize(petSlice, petColors, low, high)}
            if rgbNode is not None:         # the colored PET (score color x PET brightness), as computed for the views
                rgb = rp.sampleVolume(slicer.util.arrayFromVolume(rgbNode), _ijkToRas(rgbNode), points)
                images[nr.SCORE_ROW] = np.clip(rgb / 255.0, 0.0, 1.0)
            elif score is not None:         # the score map over the PET in grey (or black)
                values = rp.sampleVolume(slicer.util.arrayFromVolume(score), _ijkToRas(score), points, nearest=True,
                                         cval=nr.OUTSIDE)
                background = (rp.colorize(petSlice, grey, low, high)
                              if self.scoreBackgroundCombo.currentData == "pet" else np.zeros(values.shape + (3,)))
                rgba, (scoreLow, scoreHigh) = self._scoreColors()
                images[nr.SCORE_ROW] = rp.overlay(background, values, rgba, scoreLow, scoreHigh,
                                                  transparent=~np.isfinite(values) | (values <= nr.OUTSIDE / 2.0))
            else:                           # no scores: the PET in rainbow colors
                images[nr.SCORE_ROW] = rp.colorize(petSlice, self._rgbOf(NO_SCORE_COLORMAP), low, high)
            for row, image in images.items():
                path = os.path.join(work, f"slice_{row or 'pet'}_{orientation}.png")
                out[row].append((orientation, rp.writePng(image, path)))
        return out

    def _reportSlices(self, report, work):
        """The PET and the score slices at the slices' position on screen, drawn from the images, with a colour bar
        under each row."""
        report.keepWithNext()
        report.heading("Slices")
        try:
            rows = self._reportSliceImages(work)
        except Exception:
            logging.exception("Static Neuro: the report's slices could not be drawn")
            report.paragraph("The slices could not be drawn.", italic=True)
            return
        volume = self.reviewVolume()
        low, high = self.petWindowRange.minimumValue, self.petWindowRange.maximumValue
        scored = self.scoreNode is not None and slicer.mrmlScene.IsNodePresent(self.scoreNode)
        for row, title in ((nr.PET_ROW, self.REPORT_IMAGE_TITLE),
                           (nr.SCORE_ROW, f"{self._scoreKind()} score" if scored
                            else f"{self.REPORT_IMAGE_TITLE} (colors)")):
            images = [qt.QImage(path) for _, path in rows[row]]
            report.imageRow(images, captions=[o for o, _ in rows[row]], rowTitle=title, maxHeightMm=REPORT_SLICE_MM)
            if row == nr.PET_ROW:
                report.colorbar(self._withAlpha(self._rgbOf(self.petColorCombo.currentText)), low, high,
                                self._petUnits(volume), "{:.2f}")
            elif scored:
                rgba, (scoreLow, scoreHigh) = self._scoreColors()
                report.colorbar(rgba, scoreLow, scoreHigh, self._scoreKind())
            else:
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
    def _renderReport(self, path, work, patient, study, scan, comparison, imageFolder=None):
        """The report drawn to the PDF path (the page count returned), or as page images (PNG) into imageFolder
        (their paths returned)."""
        report = rp.PdfReport(path, self.REPORT_TITLE, REPORT_BANNER_PATH)
        report.heading("Patient and scan")
        report.keyValues(self._reportInfoRows(patient, study, scan, comparison))
        report.paragraph(rp.DISCLAIMER, size=9, bold=True, color=rp.DISCLAIMER_COLOR)     # first page, red
        self._reportSlices(report, work)
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
        if imageFolder:
            return report.renderImages(imageFolder)
        return report.finish()

    @staticmethod
    def _reportStudyUID(volume, dicom, sourceUIDs, patient):
        if sourceUIDs:
            if not dicom.get("StudyInstanceUID") or not dicom.get("SOPClassUID"):
                raise ValueError("The source study could not be read from the DICOM database.")
            if dicom.get("PatientID") and patient.get("id") and dicom["PatientID"] != patient["id"]:
                raise ValueError("The patient ID on the Data step differs from the source DICOM study.")
            return dicom["StudyInstanceUID"]
        # File-based inputs have no DICOM study. Keep repeated reports together in the scene.
        from pydicom.uid import generate_uid
        key = NEURO_PREFIX + "ReportStudyInstanceUID"
        uid = volume.GetAttribute(key)
        if not uid:
            uid = generate_uid()
            volume.SetAttribute(key, uid)
        return uid

    @staticmethod
    def _dicomDatabase():
        database = slicer.dicomDatabase
        if database is None or not database.isOpen:
            raise RuntimeError("Open a Slicer DICOM database first (DICOM module).")
        return database

    def _importDicom(self, paths, uids):
        """The files copied into the Slicer DICOM database (its own folder: the files given may be temporary)."""
        database = self._dicomDatabase()
        indexer = ctk.ctkDICOMIndexer()
        for path in paths:
            indexer.addFile(database, path, True)    # True: a copy kept in the database's folder
        indexer.waitForImportFinished()
        folders = {os.path.normcase(os.path.abspath(os.path.dirname(path))) for path in paths}
        for uid in uids:
            indexed = database.fileForInstance(uid)
            if (not indexed or not os.path.isfile(indexed)
                    or os.path.normcase(os.path.abspath(os.path.dirname(indexed))) in folders):
                raise IOError("The DICOM database did not keep a copy of the saved images.")

    def _dicomTarget(self):
        """(volume, patient, study, scan, comparison, dicom, studyUID, sourceUIDs) of the report's DICOM outputs."""
        volume = self._reportSourceNode()
        patient, study, scan, comparison, dicom = self._reportContent()
        sourceUIDs = (volume.GetAttribute("DICOM.instanceUIDs") or "").split() if volume is not None else []
        studyUID = self._reportStudyUID(volume, dicom, sourceUIDs, patient)
        return volume, patient, study, scan, comparison, dicom, studyUID, sourceUIDs

    def onSaveReport(self, asDicom=False):
        if asDicom:
            self.onSaveDicomReport()
            return
        if self.reviewResult is None:
            slicer.util.warningDisplay("Compare the patient first (Review step).", windowTitle="PDF report")
            return
        patient, study, scan, comparison, dicom = self._reportContent()

        def render(path, work, imageFolder=None):
            return self._renderReport(path, work, patient, study, scan, comparison, imageFolder=imageFolder)
        self._savePdfReport("PDF report", render, patient, study, self.REPORT_FILE_SUFFIX, self.reportLabel,
                            self.reportOpenCheck)

    def onSaveDicomReport(self):
        """The report's pages as images (Secondary Capture), straight into the Slicer DICOM database (no file to
        pick): a new series in the PET's study."""
        if self.reviewResult is None:
            slicer.util.warningDisplay("Compare the patient first (Review step).", windowTitle="DICOM report")
            return

        def render(path, work, patient, study, scan, comparison, imageFolder=None):
            return self._renderReport(path, work, patient, study, scan, comparison, imageFolder=imageFolder)
        self._saveDicomReport("DICOM report", render, self.REPORT_TITLE, self.reportLabel)

    def _savePdfReport(self, title, render, patient, study, suffix, label, openCheck):
        """A PDF report written where the user picks (the last folder remembered); render(path, work folder) draws
        it and returns its page count."""
        folder = self._reportSettings().get("folder") or os.path.expanduser("~")
        name = rp.reportFileName(patient.get("name"), study.get("date"), suffix)
        name = os.path.splitext(name)[0] + ".pdf"
        path = qt.QFileDialog.getSaveFileName(slicer.util.mainWindow(), "Save the " + title,
                                              os.path.join(folder, name), "PDF (*.pdf)")
        if not path:
            return None
        path = str(path)
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        folder = os.path.dirname(path)
        self._saveReportSettings(folder, openCheck)
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            # Stage the final file on the destination filesystem so a failed export preserves an existing report.
            with tempfile.TemporaryDirectory(dir=folder) as staging, tempfile.TemporaryDirectory() as work:
                pdfPath = os.path.join(staging, "report.pdf")
                pages = render(pdfPath, work)
                os.replace(pdfPath, path)
        except Exception as error:
            logging.exception("Neuro: could not write the PDF report")
            slicer.util.errorDisplay(f"Could not write the {title}:\n{error}", windowTitle=title)
            return None
        finally:
            qt.QApplication.restoreOverrideCursor()
        message = f"Saved: {path} ({pages} page{'s' if pages != 1 else ''})"
        label.text = message
        slicer.util.showStatusMessage(message, 5000)
        if openCheck is None or openCheck.checked:
            qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(path))
        return path

    def _saveDicomReport(self, title, render, reportTitle, label):
        """A report's pages as secondary capture images in the image's study, into the DICOM database;
        render(path, work, patient, study, scan, comparison, imageFolder) draws the pages as PNG files."""
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            self._dicomDatabase()
            _, patient, study, scan, comparison, dicom, studyUID, sourceUIDs = self._dicomTarget()
            with tempfile.TemporaryDirectory() as work:
                pageFolder = os.path.join(work, "pages")
                os.makedirs(pageFolder)
                images = render(None, work, patient, study, scan, comparison, imageFolder=pageFolder)
                paths = [os.path.join(work, f"page{number}.dcm") for number in range(1, len(images) + 1)]
                uids = rp.writeDicomImages(paths, images, reportTitle, patient, dicom, studyUID, sourceUIDs)
                self._importDicom(paths, uids)
        except Exception as error:
            logging.exception("Neuro: could not save the DICOM report")
            slicer.util.errorDisplay(f"Could not save the {title}:\n{error}", windowTitle=title)
            return False
        finally:
            qt.QApplication.restoreOverrideCursor()
        pages = len(uids)
        message = (f"DICOM report ({pages} page{'s' if pages != 1 else ''}) added to the DICOM database "
                   f"(series {rp.REPORT_SERIES[1]}, \"{rp.REPORT_SERIES[0]}\").")
        label.text = message
        slicer.util.showStatusMessage(message, 5000)
        return True

    # --- screenshots ---------------------------------------------------------------------------------------------
    def _screenImages(self):
        """The views on screen as QImages: the main window's views, and the Monitor 2 window when shown."""
        layoutManager = slicer.app.layoutManager()
        images = []
        if layoutManager is not None:
            images.append(ctk.ctkWidgetsUtils.grabWidget(layoutManager.viewport()))
        window = self._monitor2Window() if hasattr(self, "_monitor2Window") else None
        if window is not None and window.visible:
            content = window.widget() if window.inherits("QDockWidget") and window.widget() is not None else window
            images.append(ctk.ctkWidgetsUtils.grabWidget(content))
        return [image for image in images if image is not None and not image.isNull()]

    @staticmethod
    def _sideBySide(images):
        """The images in one, left to right on black (the main window, then Monitor 2)."""
        if len(images) == 1:
            return images[0]
        width = sum(image.width() for image in images)
        height = max(image.height() for image in images)
        joined = qt.QImage(width, height, qt.QImage.Format_RGB32)
        joined.fill(qt.QColor(0, 0, 0))
        painter = qt.QPainter(joined)
        x = 0
        for image in images:
            painter.drawImage(x, 0, image)
            x += image.width()
        painter.end()
        return joined

    def onScreenshotClipboard(self):
        images = self._screenImages()
        if not images:
            slicer.util.warningDisplay("No views to capture.", windowTitle="Screenshot")
            return
        qt.QApplication.clipboard().setImage(self._sideBySide(images))
        slicer.util.showStatusMessage("Screenshot copied to the clipboard.", 3000)

    def onScreenshotDicom(self):
        """The views on screen as Secondary Capture images in the PET's study (one image per window), straight into
        the DICOM database; the screenshots of one PET go into one series."""
        title = "DICOM screenshot"
        images = self._screenImages()
        if not images:
            slicer.util.warningDisplay("No views to capture.", windowTitle=title)
            return
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            self._dicomDatabase()
            volume, patient, _, _, _, dicom, studyUID, sourceUIDs = self._dicomTarget()
            seriesKey, countKey = NEURO_PREFIX + "ScreenshotSeriesUID", NEURO_PREFIX + "ScreenshotCount"
            seriesUID = volume.GetAttribute(seriesKey)
            if not seriesUID:
                from pydicom.uid import generate_uid
                seriesUID = generate_uid()
            first = int(volume.GetAttribute(countKey) or 0) + 1
            with tempfile.TemporaryDirectory() as work:
                pngs, paths = [], []
                for index, image in enumerate(images):
                    png = os.path.join(work, f"shot{index + 1}.png")
                    if not image.convertToFormat(qt.QImage.Format_RGB888).save(png, "PNG"):
                        raise IOError("Could not write the screenshot image.")
                    pngs.append(png)
                    paths.append(os.path.join(work, f"shot{index + 1}.dcm"))
                uids = rp.writeDicomImages(paths, pngs, "Screenshot", patient, dicom, studyUID, sourceUIDs,
                                           series=rp.SCREENSHOT_SERIES, seriesUID=seriesUID, firstNumber=first,
                                           pixelSpacingMm=None)
                self._importDicom(paths, uids)
            volume.SetAttribute(seriesKey, seriesUID)
            volume.SetAttribute(countKey, str(first + len(uids) - 1))
        except Exception as error:
            logging.exception("Static Neuro: could not save the DICOM screenshot")
            slicer.util.errorDisplay(f"Could not save the {title}:\n{error}", windowTitle=title)
            return
        finally:
            qt.QApplication.restoreOverrideCursor()
        slicer.util.showStatusMessage(f"Screenshot added to the DICOM database (series {rp.SCREENSHOT_SERIES[1]}, "
                                      f"\"{rp.SCREENSHOT_SERIES[0]}\").", 5000)
