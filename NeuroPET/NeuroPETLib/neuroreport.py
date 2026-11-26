"""
The PDF report of a Neuro review (Static Neuro PET/SPECT first): what goes in it and how it is drawn.

    studyInfo(values)            the scan's date, injected activity, uptake time and scan duration from DICOM values
    infoRows(...)                the label / value rows of the report's first table
    sortedTableRows(rows)        the region table sorted by the regions' names
    reportFileName(name, date)   the default file name
    renderSurfaceViews(...)      3D-SSP views drawn off screen (VTK) on a white background, as PNG files
    PdfReport                    the A4 PDF itself (Qt: QPdfWriter and QPainter), laid out top to bottom, a block on
                                 the next page when it does not fit

Without Slicer: the data helpers are pure Python; PdfReport needs the Qt module "qt" (Slicer's PythonQt, or a stand-in
of PyQt in the tests) and renderSurfaceViews VTK.
"""

import datetime
import math
import os
import re

import numpy as np

TITLE = "Brain PET / SPECT report"
DISCLAIMER = ("For research use only. A quantitative comparison with a normal database; it does not replace the reading "
              "of the images by a physician and is not to be used alone for a diagnosis.")
SOFTWARE = "Mergen Neuro PET (3D Slicer)"
NOT_KNOWN = "-"

# the page (mm)
PAGE_WIDTH_MM, PAGE_HEIGHT_MM = 210.0, 297.0
MARGIN_MM = 12.0
HEADER_MM = 14.0            # the title line and its rule at the top of each page
FOOTER_MM = 9.0             # the software and the page number at the bottom
GAP_MM = 3.0                # between blocks
BANNER_TOP_MM = 6.0         # the first page's banner this far from the top edge (full width)
RESOLUTION_DPI = 300
IMAGE_DPI = 200             # the DICOM report pages (A4 at 200 dpi: 1654 x 2339 pixels)

# colours (r, g, b 0..255)
RULE_COLOR = (90, 103, 112)
DISCLAIMER_COLOR = (200, 0, 0)
HEADING_COLOR = (40, 60, 80)
LABEL_COLOR = (90, 90, 90)
GRID_COLOR = (200, 200, 200)
ZEBRA_COLOR = (244, 246, 248)
HEADER_FILL = (226, 232, 238)
LOW_SCORE_COLOR = (20, 70, 200)        # a regional score at or below the threshold
HIGH_SCORE_COLOR = (190, 30, 30)       # at or above minus the threshold

SURFACE_VIEW_PX = 700       # an off screen 3D-SSP view (square)
SURFACE_BACKGROUND = (1.0, 1.0, 1.0)


# --- DICOM values: the scan --------------------------------------------------------------------------------
def parseDate(text):
    """datetime.date of a DICOM DA ('20261124', also '2026-11-24' / '2026.11.24'); None when not given / invalid."""
    digits = re.sub(r"[^0-9]", "", str(text or ""))[:8]
    if len(digits) != 8:
        return None
    try:
        return datetime.date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))
    except ValueError:
        return None


def parseTime(text):
    """datetime.time of a DICOM TM ('093015.123', '0930', '09:30:15'); None when not given / invalid."""
    text = str(text or "").strip().replace(":", "")
    match = re.match(r"^(\d{2})(\d{2})?(\d{2})?(?:\.(\d{1,6}))?", text)
    if not match:
        return None
    hour, minute, second, fraction = match.groups()
    try:
        micro = int((fraction or "0").ljust(6, "0")[:6])
        return datetime.time(int(hour), int(minute or 0), int(second or 0), micro)
    except ValueError:
        return None


def parseDateTime(text):
    """datetime.datetime of a DICOM DT ('20261124093015.000000+0300': the offset ignored); None when not given."""
    text = str(text or "").strip()
    text = re.split(r"[+-]", text, maxsplit=1)[0] if len(text) > 8 else text
    date = parseDate(text[:8])
    if date is None:
        return None
    time = parseTime(text[8:]) if len(text) > 8 else datetime.time(0, 0)
    return datetime.datetime.combine(date, time or datetime.time(0, 0))


def _combine(date, time):
    if date is None or time is None:
        return None
    return datetime.datetime.combine(date, time)


def _number(value):
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def studyInfo(values):
    """The scan from DICOM values {keyword: text} (the image's first instance; the radiopharmaceutical's keywords
    from its RadiopharmaceuticalInformationSequence item):

        Modality, StudyDate, SeriesDate, SeriesTime, AcquisitionDate, AcquisitionTime, AcquisitionDateTime,
        ActualFrameDuration (ms), NumberOfFrames, Radiopharmaceutical, RadiopharmaceuticalStartTime,
        RadiopharmaceuticalStartDateTime, RadionuclideTotalDose (Bq), RadionuclideCodeMeaning

    -> {"start": datetime of the scan's start or None, "date": date or None, "injection": datetime or None,
        "uptakeMinutes": float or None (injection to the scan's start), "durationMinutes": float or None,
        "activityMBq": float or None}. The PET's start is its series time (the time its decay correction refers to),
    else the acquisition's."""
    values = {k: v for k, v in (values or {}).items() if v not in (None, "")}
    acquisition = (parseDateTime(values.get("AcquisitionDateTime"))
                   or _combine(parseDate(values.get("AcquisitionDate")), parseTime(values.get("AcquisitionTime"))))
    series = _combine(parseDate(values.get("SeriesDate") or values.get("AcquisitionDate") or values.get("StudyDate")),
                      parseTime(values.get("SeriesTime")))
    pet = str(values.get("Modality", "")).upper() in ("PT", "PET")
    start = (series or acquisition) if pet else (acquisition or series)
    date = (start.date() if start is not None else
            parseDate(values.get("AcquisitionDate") or values.get("SeriesDate") or values.get("StudyDate")))
    injection = parseDateTime(values.get("RadiopharmaceuticalStartDateTime"))
    if injection is None and values.get("RadiopharmaceuticalStartTime") and date is not None:
        injection = _combine(date, parseTime(values.get("RadiopharmaceuticalStartTime")))
        if injection is not None and start is not None and injection > start:
            injection -= datetime.timedelta(days=1)         # injected before midnight, scanned after it
    uptake = None
    if injection is not None and start is not None:
        uptake = (start - injection).total_seconds() / 60.0
        if uptake < 0 or uptake > 24 * 60:
            uptake = None
    duration = _number(values.get("ActualFrameDuration"))
    if duration is not None:
        frames = _number(values.get("NumberOfFrames"))
        multiple = not pet and frames is not None and frames > 1     # SPECT projections: each frame's duration
        duration = duration * (frames if multiple else 1.0) / 60000.0
        duration = duration if duration > 0 else None
    dose = _number(values.get("RadionuclideTotalDose"))
    activity = dose / 1.0e6 if dose is not None and dose > 0 else None
    return {"start": start, "date": date, "injection": injection, "uptakeMinutes": uptake,
            "durationMinutes": duration, "activityMBq": activity}


def formatDate(date):
    return date.strftime("%d.%m.%Y") if date is not None else NOT_KNOWN


def formatDateTime(moment):
    return moment.strftime("%d.%m.%Y %H:%M") if moment is not None else NOT_KNOWN


def formatMinutes(minutes):
    """'62 min' ('1 h 32 min' from 90 minutes); NOT_KNOWN without."""
    if minutes is None:
        return NOT_KNOWN
    total = int(round(minutes))
    if total < 90:
        return f"{minutes:.1f} min" if minutes < 10 else f"{total} min"
    return f"{total // 60} h {total % 60:02d} min"


def formatActivity(megabecquerel):
    if megabecquerel is None:
        return NOT_KNOWN
    return f"{megabecquerel:.0f} MBq ({megabecquerel / 37.0:.2f} mCi)"


def formatValue(value, unit=""):
    if value in (None, ""):
        return NOT_KNOWN
    return f"{value}{(' ' + unit) if unit else ''}"


def formatPersonName(text):
    """A DICOM Person Name ('DOE^JOHN^^^') as plain text ('DOE JOHN')."""
    text = str(text or "").split("=")[0]
    return " ".join(p.strip() for p in text.split("^") if p.strip())


# --- the report's content -----------------------------------------------------------------------------------
def infoRows(patient, study, scan, comparison, extra=(), leaveOut=()):
    """The rows (label, value) of the report's first table, in pairs of columns (left then right):

        patient     {"name", "id", "birthDate" (date), "age" (years), "sex", "weight" (kg)}
        study       studyInfo()
        scan        {"modality", "radionuclide", "radiopharmaceutical", "scanner", "image"}
        comparison  {"database", "normals", "reference", "score", "smoothing" (mm), "regions", "normalization",
                     "units"}
        extra       more rows (label, value) at the end; leaveOut: labels of rows not shown
    Values not known are NOT_KNOWN."""
    patient, study, scan, comparison = (d or {} for d in (patient, study, scan, comparison))
    age = patient.get("age")
    ageText = f"{age:g}" if isinstance(age, (int, float)) else formatValue(age)
    sex = patient.get("sex") or NOT_KNOWN
    weight = patient.get("weight")
    smoothing = comparison.get("smoothing")
    rows = [
        ("Patient", formatValue(patient.get("name"))),
        ("Patient ID", formatValue(patient.get("id"))),
        ("Birth date", formatDate(patient.get("birthDate"))),
        ("Age / sex", f"{ageText} / {sex}"),
        ("Weight", f"{weight:g} kg" if isinstance(weight, (int, float)) and weight > 0 else NOT_KNOWN),
        ("Scan date", formatDateTime(study.get("start")) if study.get("start") else formatDate(study.get("date"))),
        ("Modality", formatValue(scan.get("modality"))),
        ("Radionuclide", formatValue(scan.get("radionuclide"))),
        ("Radiopharmaceutical", formatValue(scan.get("radiopharmaceutical"))),
        ("Injected activity", formatActivity(study.get("activityMBq"))),
        ("Injection to scan", formatMinutes(study.get("uptakeMinutes"))),
        ("Scan duration", formatMinutes(study.get("durationMinutes"))),
        ("Scanner", formatValue(scan.get("scanner"))),
        ("Normal database", formatValue(comparison.get("database"))),
        ("Normals compared", formatValue(comparison.get("normals"))),
        ("Reference region", formatValue(comparison.get("reference"))),
        ("Score", formatValue(comparison.get("score"))),
        ("Image values", formatValue(comparison.get("units"))),
        ("Smoothing", f"{smoothing:g} mm FWHM" if isinstance(smoothing, (int, float)) and smoothing > 0
         else NOT_KNOWN),
        ("Regions", formatValue(comparison.get("regions"))),
        ("Normalization", formatValue(comparison.get("normalization"))),
        ("Image", formatValue(scan.get("image"))),
    ] + [(label, formatValue(value)) for label, value in extra]
    return [row for row in rows if row[0] not in leaveOut]


def _naturalKey(text):
    """Sort key of a region's name: case-insensitive, numbers by value ('Region 2' before 'Region 10')."""
    return [(0, int(part)) if part.isdigit() else (1, part) for part in re.split(r"(\d+)", str(text).lower()) if part]


def sortedTableRows(rows):
    """The region table's rows (lists, the region's name first) sorted by the name."""
    return sorted(rows, key=lambda row: _naturalKey(row[0] if row else ""))


def scoreEmphasis(score, threshold):
    """-1: at or below the threshold (below the normals), 1: at or above minus it, 0: else or no score."""
    if score is None or threshold is None or not isinstance(score, (int, float)) or not math.isfinite(score):
        return 0
    limit = abs(float(threshold))
    if limit == 0:
        return 0
    if score <= -limit:
        return -1
    if score >= limit:
        return 1
    return 0


def reportFileName(patientName, date=None, suffix="NeuroPET"):
    """'DOE_JOHN_2026-11-24_NeuroPET.pdf' (characters a file name cannot hold left out)."""
    name = re.sub(r"[^\w\-]+", "_", str(patientName or "").strip(), flags=re.UNICODE).strip("_") or "Patient"
    parts = [name] + ([date.strftime("%Y-%m-%d")] if date is not None else []) + [suffix]
    return "_".join(parts) + ".pdf"


def plainText(html):
    """Text of a label's HTML (tags left out, <br> as a space)."""
    text = re.sub(r"<br\s*/?>", " ", str(html or ""), flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<")
                  .replace("&gt;", ">")).strip()


def colorbarTicks(low, high, count=5):
    """Tick values of a colour bar from low to high (0 among them when it is inside)."""
    if not (isinstance(low, (int, float)) and isinstance(high, (int, float))) or high <= low:
        return [low, high]
    ticks = [low + (high - low) * i / (count - 1) for i in range(count)]
    if low < 0 < high and not any(abs(t) < 1e-9 for t in ticks):
        ticks = sorted(ticks + [0.0])
    return ticks


# --- dynamic series --------------------------------------------------------------------------------------
def _seconds(value):
    value = float(value)
    if abs(value - round(value)) < 0.05:
        return f"{int(round(value))} s" if value < 120 or value % 60 else f"{int(round(value / 60))} min"
    return f"{value:.1f} s"


def frameSchemeText(durationsS):
    """'6 x 30 s, 4 x 1 min, 5 x 5 min': the frame durations (s) grouped where they repeat, in order."""
    groups = []
    for duration in durationsS:
        duration = float(duration)
        if groups and abs(groups[-1][1] - duration) < 0.5:
            groups[-1][0] += 1
        else:
            groups.append([1, duration])
    return ", ".join(f"{count} \u00d7 {_seconds(duration)}" for count, duration in groups)


def frameCaption(number, startMin, endMin):
    """'3: 1-1.5 min': a frame's number and its time after time zero."""
    return f"{number}: {startMin:.3g}\u2013{endMin:.3g} min"


def niceTicks(low, high, count=6):
    """Round tick values covering low..high (about count of them)."""
    if not (math.isfinite(low) and math.isfinite(high)) or high <= low:
        return [low]
    raw = (high - low) / max(1, count - 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    first = math.floor(low / step) * step
    ticks, value = [], first
    while value <= high + step * 1e-6:
        if value >= low - step * 1e-6:
            ticks.append(round(value, 10))
        value += step
    return ticks


def tacPlotImage(curves, units="", width=2000, height=1050, title="", tStar=None, window=None, legend=True):
    """The time-activity curves as a QImage on white, as the Modeling step draws them (tacplot): the curves with
    emphasis (the target, the reference) as lines with points and their fit dashed (darker) on top, the others faint
    behind them; the value axis fits the curves with emphasis (the others may go out of the plot, as in the module).

        curves  [{"name", "color" ('#rrggbb'), "times" (min), "values", "emphasis" (bool), "role",
                  "fitTimes", "fitValues" (optional)}]
        tStar   a vertical line at t* (min); window: the (start, end) of a ratio's time window (min), shaded."""
    qt = _qt()
    image = qt.QImage(int(width), int(height), qt.QImage.Format_RGB32)
    image.fill(qt.QColor(255, 255, 255))
    painter = qt.QPainter(image)
    try:
        painter.setRenderHint(qt.QPainter.Antialiasing, True)
        painter.setRenderHint(qt.QPainter.TextAntialiasing, True)
        unit = height / 1050.0
        font = qt.QFont("Helvetica")
        font.setPixelSize(int(26 * unit))
        painter.setFont(font)
        left, right, top, bottom = 150 * unit, 40 * unit, (70 if title else 30) * unit, 110 * unit
        plotW, plotH = width - left - right, height - top - bottom

        def finite(values):
            values = np.asarray(values, np.float64) if values is not None else np.zeros(0)
            return values[np.isfinite(values)]
        times = np.concatenate([finite(c.get("times")) for c in curves] + [finite(c.get("fitTimes")) for c in curves]
                               + [np.zeros(1)])
        values = np.concatenate([finite(c.get("values")) for c in curves]
                                + [finite(c.get("fitValues")) for c in curves] + [np.zeros(1)])
        strong = [c for c in curves if c.get("emphasis")] or curves
        topValues = np.concatenate([finite(c.get("values")) for c in strong]
                                   + [finite(c.get("fitValues")) for c in strong] + [np.zeros(1)])
        xMax = float(times.max()) if times.size else 1.0
        yMax = float(topValues.max()) if topValues.size else 1.0
        yMin = min(0.0, float(values.min())) if values.size else 0.0
        xMax = xMax if xMax > 0 else 1.0
        yMax = yMax if yMax > yMin else yMin + 1.0
        xTicks, yTicks = niceTicks(0.0, xMax), niceTicks(yMin, yMax * 1.05)
        for ticks, limit in ((xTicks, xMax), (yTicks, yMax * 1.05)):       # the axes end on a tick
            step = ticks[1] - ticks[0] if len(ticks) > 1 else 1.0
            while ticks[-1] < limit - 1e-9:
                ticks.append(round(ticks[-1] + step, 10))
        xMax, yMax = max(xMax, xTicks[-1]), max(yMax, yTicks[-1])

        def px(t):
            return left + plotW * t / xMax

        def py(v):
            return top + plotH * (1.0 - (v - yMin) / (yMax - yMin))

        if window:
            painter.fillRect(qt.QRectF(px(window[0]), top, px(window[1]) - px(window[0]), plotH),
                             qt.QColor(255, 236, 179))
        grid = qt.QPen(qt.QColor(225, 225, 225))
        grid.setWidthF(2 * unit)
        axis = qt.QPen(qt.QColor(60, 60, 60))
        axis.setWidthF(3 * unit)
        for t in xTicks:
            painter.setPen(grid)
            painter.drawLine(qt.QPointF(px(t), top), qt.QPointF(px(t), top + plotH))
            painter.setPen(axis)
            painter.drawText(qt.QRectF(px(t) - 80 * unit, top + plotH + 8 * unit, 160 * unit, 34 * unit),
                             int(qt.Qt.AlignHCenter | qt.Qt.AlignTop), f"{t:g}")
        for v in yTicks:
            painter.setPen(grid)
            painter.drawLine(qt.QPointF(left, py(v)), qt.QPointF(left + plotW, py(v)))
            painter.setPen(axis)
            painter.drawText(qt.QRectF(0, py(v) - 17 * unit, left - 14 * unit, 34 * unit),
                             int(qt.Qt.AlignRight | qt.Qt.AlignVCenter), f"{v:g}")
        painter.setPen(axis)
        painter.drawRect(qt.QRectF(left, top, plotW, plotH))
        painter.drawText(qt.QRectF(left, height - 52 * unit, plotW, 40 * unit), int(qt.Qt.AlignCenter),
                         "Time from time zero (min)")
        painter.save()
        painter.translate(30 * unit, top + plotH / 2.0)
        painter.rotate(-90)
        painter.drawText(qt.QRectF(-plotH / 2.0, -20 * unit, plotH, 40 * unit), int(qt.Qt.AlignCenter),
                         f"Activity ({units})" if units else "Activity")
        painter.restore()
        if title:
            bold = qt.QFont(font)
            bold.setBold(True)
            bold.setPixelSize(int(30 * unit))
            painter.setFont(bold)
            painter.drawText(qt.QRectF(left, 10 * unit, plotW, 50 * unit), int(qt.Qt.AlignLeft | qt.Qt.AlignVCenter),
                             title)
            painter.setFont(font)
        if tStar is not None and math.isfinite(tStar):
            dashed = qt.QPen(qt.QColor(120, 120, 120))
            dashed.setWidthF(3 * unit)
            dashed.setStyle(qt.Qt.DashLine)
            painter.setPen(dashed)
            painter.drawLine(qt.QPointF(px(tStar), top), qt.QPointF(px(tStar), top + plotH))
            painter.drawText(qt.QRectF(px(tStar) + 6 * unit, top + 4 * unit, 200 * unit, 34 * unit),
                             int(qt.Qt.AlignLeft | qt.Qt.AlignTop), "t*")
        painter.setClipRect(qt.QRectF(left, top, plotW, plotH))

        def finiteOrNan(values):
            return np.asarray(values if values is not None else [], np.float64)

        def polyline(ts, vs, pen):
            ts, vs = np.asarray(ts, np.float64), np.asarray(vs, np.float64)
            ok = np.isfinite(ts) & np.isfinite(vs)
            path = qt.QPainterPath()
            started = False
            for t, v, good in zip(ts, vs, ok):
                if not good:
                    started = False
                    continue
                if started:
                    path.lineTo(px(t), py(v))
                else:
                    path.moveTo(px(t), py(v))
                    started = True
            painter.strokePath(path, pen)

        ordered = [c for c in curves if not c.get("emphasis")] + [c for c in curves if c.get("emphasis")]
        for curve in ordered:
            color = qt.QColor(curve.get("color") or "#808080")
            strong = bool(curve.get("emphasis"))
            if not strong:
                color.setAlphaF(0.2)
            pen = qt.QPen(color)
            pen.setWidthF((4.5 if strong else 2.0) * unit)
            polyline(curve.get("times"), curve.get("values"), pen)
            painter.setPen(qt.QPen(qt.Qt.NoPen))
            painter.setBrush(qt.QBrush(color))
            radius = (6.5 if strong else 3.5) * unit
            for t, v in zip(finiteOrNan(curve.get("times")), finiteOrNan(curve.get("values"))):
                if math.isfinite(t) and math.isfinite(v):
                    painter.drawEllipse(qt.QPointF(px(t), py(v)), radius, radius)
            painter.setBrush(qt.QBrush())
            if curve.get("fitTimes") is not None and curve.get("fitValues") is not None:
                fitPen = qt.QPen(color.darker(150) if strong else color)
                fitPen.setWidthF((4.0 if strong else 2.0) * unit)
                fitPen.setStyle(qt.Qt.DashLine)
                polyline(curve["fitTimes"], curve["fitValues"], fitPen)
        painter.setClipping(False)
        if legend:
            entries = [c for c in curves if c.get("emphasis")]
            faint = len(curves) - len(entries)
            y = top + 14 * unit
            x = left + plotW - 20 * unit
            texts = []
            for c in entries:
                color = qt.QColor(c.get("color") or "#808080")
                role = c.get("role") or ""
                name = c.get("name", "")
                texts.append((name + (f" ({role})" if role and f"({role})" not in name.lower() else ""), color, False))
                if c.get("fitTimes") is not None and c.get("fitValues") is not None:
                    texts.append(("fit", color.darker(150), True))
            if faint:
                other = qt.QColor(128, 128, 128)
                other.setAlphaF(0.4)
                texts.append((f"Other regions ({faint})", other, False))
            if tStar is not None and math.isfinite(tStar):
                texts.append(("t*", qt.QColor(120, 120, 120), True))
            metrics = painter.fontMetrics()
            boxW = max([metrics.width(t) for t, _, _ in texts] + [0]) + 90 * unit
            if texts:
                painter.fillRect(qt.QRectF(x - boxW, y - 6 * unit, boxW, len(texts) * 40 * unit + 12 * unit),
                                 qt.QColor(255, 255, 255, 230))
            for text, color, dashed in texts:
                pen = qt.QPen(color)
                pen.setWidthF(5 * unit)
                if dashed:
                    pen.setStyle(qt.Qt.DashLine)
                painter.setPen(pen)
                painter.drawLine(qt.QPointF(x - boxW + 12 * unit, y + 20 * unit),
                                 qt.QPointF(x - boxW + 60 * unit, y + 20 * unit))
                painter.setPen(qt.QColor(30, 30, 30))
                painter.drawText(qt.QRectF(x - boxW + 72 * unit, y, boxW - 80 * unit, 40 * unit),
                                 int(qt.Qt.AlignLeft | qt.Qt.AlignVCenter), text)
                y += 40 * unit
    finally:
        painter.end()
    return image


def gridBox(shape, ijkToRas):
    """(RAS minimum, RAS maximum) of a whole grid (k, j, i): the slices drawn uncut."""
    k, j, i = (n - 1 for n in shape[:3])
    corners = np.array([[a, b, c, 1.0] for a in (0, i) for b in (0, j) for c in (0, k)], np.float64)
    ras = (np.asarray(ijkToRas, np.float64) @ corners.T).T[:, :3]
    return ras.min(axis=0), ras.max(axis=0)


INVERTED_GREY = np.tile(np.linspace(1.0, 0.0, 256)[:, None], (1, 3))     # white (0) to black (the window's top)


# --- slices from the images (not from the views) ----------------------------------------------------------
def contentBox(array, ijkToRas, fraction=0.02, marginMm=4.0):
    """(RAS minimum, RAS maximum) of the voxels above fraction of the array's maximum (the head; the black around
    it left out), margin mm wider; the whole grid when nothing is above it."""
    array = np.asarray(array)
    finite = np.where(np.isfinite(array), array, 0)
    top = float(finite.max()) if finite.size else 0.0
    mask = finite > fraction * top if top > 0 else np.ones(finite.shape, bool)
    if not mask.any():
        mask = np.ones(finite.shape, bool)
    index = [np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a != axis))) for axis in range(3)]
    k0, k1 = index[0][[0, -1]]
    j0, j1 = index[1][[0, -1]]
    i0, i1 = index[2][[0, -1]]
    corners = np.array([[i, j, k, 1.0] for i in (i0, i1) for j in (j0, j1) for k in (k0, k1)], dtype=np.float64)
    ras = (np.asarray(ijkToRas, np.float64) @ corners.T).T[:, :3]
    return ras.min(axis=0) - marginMm, ras.max(axis=0) + marginMm


def planePoints(box, point, xAxis, yAxis, spacingMm):
    """RAS points (rows, columns, 3) of a slice through point with screen axes xAxis (left to right) and yAxis (bottom
    to top), covering the box (RAS minimum, maximum) as seen in that plane: the first row at its top."""
    xAxis = np.asarray(xAxis, np.float64) / np.linalg.norm(xAxis)
    yAxis = np.asarray(yAxis, np.float64) / np.linalg.norm(yAxis)
    low, high = (np.asarray(v, np.float64) for v in box)
    corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    u, v = corners @ xAxis, corners @ yAxis
    point = np.asarray(point, np.float64)
    columns = max(2, int(np.ceil((u.max() - u.min()) / spacingMm)) + 1)
    rows = max(2, int(np.ceil((v.max() - v.min()) / spacingMm)) + 1)
    us = u.min() + spacingMm * np.arange(columns)
    vs = v.max() - spacingMm * np.arange(rows)          # top row first
    base = point - (point @ xAxis) * xAxis - (point @ yAxis) * yAxis
    return base + us[None, :, None] * xAxis + vs[:, None, None] * yAxis


def sampleVolume(array, ijkToRas, points, nearest=False, cval=0.0):
    """Values of an array (k, j, i[, c]) with its 4 x 4 ijkToRas at RAS points (..., 3): trilinear (nearest: the
    nearest voxel), cval outside the grid."""
    array = np.asarray(array)
    shape = points.shape[:-1]
    flat = points.reshape(-1, 3)
    ijk = (np.linalg.inv(np.asarray(ijkToRas, np.float64)) @ np.c_[flat, np.ones(len(flat))].T).T[:, :3]
    kji = ijk[:, ::-1]
    size = np.array(array.shape[:3])
    extra = array.shape[3:]
    out = np.full((len(flat),) + extra, cval, dtype=np.float32)
    if nearest:
        index = np.rint(kji).astype(np.int64)
        inside = np.all((index >= 0) & (index < size), axis=1)
        k, j, i = index[inside].T
        out[inside] = array[k, j, i]
        return out.reshape(shape + extra)
    base = np.floor(kji).astype(np.int64)
    frac = (kji - base).astype(np.float32)
    inside = np.all((kji >= 0) & (kji <= size - 1), axis=1)
    base, frac = base[inside], frac[inside]
    total = np.zeros((len(base),) + extra, np.float32)
    for dk in (0, 1):
        for dj in (0, 1):
            for di in (0, 1):
                k = np.minimum(base[:, 0] + dk, size[0] - 1)
                j = np.minimum(base[:, 1] + dj, size[1] - 1)
                i = np.minimum(base[:, 2] + di, size[2] - 1)
                weight = ((frac[:, 0] if dk else 1 - frac[:, 0]) * (frac[:, 1] if dj else 1 - frac[:, 1])
                          * (frac[:, 2] if di else 1 - frac[:, 2]))
                values = array[k, j, i].astype(np.float32)
                total += values * (weight.reshape((-1,) + (1,) * len(extra)))
    out[inside] = total
    return out.reshape(shape + extra)


def colorize(values, rgb, low, high):
    """RGB (0..1, ... x 3) of values through a colour table (n x 3 or n x 4, 0..1) over the window low..high."""
    table = np.asarray(rgb, np.float32)[:, :3]
    high = high if high > low else low + 1e-6
    index = np.clip((np.asarray(values, np.float32) - low) / (high - low), 0.0, 1.0) * (len(table) - 1)
    index = np.where(np.isfinite(index), index, 0).astype(np.int64)
    return table[index]


def overlay(background, values, rgba, low, high, transparent=None):
    """background RGB (0..1) with values coloured through rgba (n x 4) over low..high on top, each by its colour's
    alpha; transparent: a boolean array of the values not drawn."""
    table = np.asarray(rgba, np.float32)
    high = high if high > low else low + 1e-6
    index = np.clip((np.asarray(values, np.float32) - low) / (high - low), 0.0, 1.0) * (len(table) - 1)
    index = np.where(np.isfinite(index), index, 0).astype(np.int64)
    colors = table[index]
    alpha = colors[..., 3:4] if table.shape[1] > 3 else np.ones(colors.shape[:-1] + (1,), np.float32)
    if transparent is not None:
        alpha = np.where(np.asarray(transparent)[..., None], 0.0, alpha)
    return background * (1 - alpha) + colors[..., :3] * alpha


def writePng(rgb, path):
    """An RGB image (rows from the top, 0..1 floats or uint8) as a PNG file (VTK)."""
    import vtk
    from vtk.util import numpy_support
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = np.clip(np.rint(np.asarray(rgb, np.float32) * 255.0), 0, 255).astype(np.uint8)
    rows, columns = rgb.shape[:2]
    image = vtk.vtkImageData()
    image.SetDimensions(columns, rows, 1)
    data = numpy_support.numpy_to_vtk(np.ascontiguousarray(rgb[::-1]).reshape(-1, 3), deep=True)     # VTK: bottom up
    data.SetNumberOfComponents(3)
    image.GetPointData().SetScalars(data)
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(path)
    writer.SetInputData(image)
    writer.Write()
    return path


# --- 3D-SSP views off screen ----------------------------------------------------------------------------
def renderSurfaceViews(views, rgba, window, folder, prefix="ssp", size=SURFACE_VIEW_PX, lighting=True,
                       background=SURFACE_BACKGROUND):
    """3D-SSP views drawn off screen on a white background, each as a PNG file in folder:

        views   [(key, polydata with its vertex values as active scalars, camera (focal, position, view up),
                  parallel scale)]
        rgba    the colour table (n x 4, 0..1) over window (low, high)
    -> [path] in the order of views. One off screen render window for all of them."""
    import vtk
    table = vtk.vtkLookupTable()
    table.SetNumberOfTableValues(len(rgba))
    for i, (r, g, b, a) in enumerate(rgba):
        table.SetTableValue(i, float(r), float(g), float(b), 1.0)
    low, high = float(window[0]), float(window[1])
    high = high if high > low else low + 1e-6
    table.SetRange(low, high)
    table.Build()
    for i, (r, g, b, a) in enumerate(rgba):     # Build() may fill the table again: the colors written once more
        table.SetTableValue(i, float(r), float(g), float(b), 1.0)
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetOffScreenRendering(1)
    renderWindow.SetSize(int(size), int(size))
    renderWindow.SetMultiSamples(0)
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(*background)
    renderWindow.AddRenderer(renderer)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetLookupTable(table)
    mapper.SetScalarRange(low, high)
    mapper.SetColorModeToMapScalars()
    mapper.SetScalarModeToUsePointData()
    mapper.ScalarVisibilityOn()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    if lighting:
        prop.SetAmbient(0.15)
        prop.SetDiffuse(0.85)
        prop.SetSpecular(0.1)
    else:                   # flat colors at full brightness
        prop.SetAmbient(1.0)
        prop.SetDiffuse(0.0)
        prop.SetSpecular(0.0)
    prop.BackfaceCullingOff()
    renderer.AddActor(actor)
    camera = renderer.GetActiveCamera()
    camera.ParallelProjectionOn()
    os.makedirs(folder, exist_ok=True)
    paths = []
    try:
        for index, (key, poly, (focal, position, up), scale) in enumerate(views):
            mapper.SetInputData(poly)
            camera.SetFocalPoint(*focal)
            camera.SetPosition(*position)
            camera.SetViewUp(*up)
            camera.SetParallelScale(float(scale))
            renderer.ResetCameraClippingRange()
            renderWindow.Render()
            grab = vtk.vtkWindowToImageFilter()
            grab.SetInput(renderWindow)
            grab.SetInputBufferTypeToRGB()
            grab.ReadFrontBufferOff()
            grab.Update()
            path = os.path.join(folder, f"{prefix}_{index:02d}_{key}.png")
            writer = vtk.vtkPNGWriter()
            writer.SetFileName(path)
            writer.SetInputConnection(grab.GetOutputPort())
            writer.Write()
            paths.append(path)
    finally:
        try:
            renderWindow.Finalize()
        except Exception:
            pass
    return paths


# --- the PDF ---------------------------------------------------------------------------------------------
def _qt():
    import qt
    return qt


REPORT_SERIES = ("Mergen Neuro PET report", 900)            # (series description, series number)
SCREENSHOT_SERIES = ("Mergen Neuro PET screenshots", 901)


def _reportDataset(path, sopClass, patient, dicom, studyUID, sourceUIDs, seriesUID, number, series=REPORT_SERIES):
    """A DICOM dataset with the report's patient, study and series (same study as the source PET)."""
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sopClass
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    doc = FileDataset(path, {}, file_meta=meta, preamble=bytes(128))
    doc.SOPClassUID = meta.MediaStorageSOPClassUID
    doc.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    doc.SpecificCharacterSet = "ISO_IR 192"
    for key in ("StudyDate", "StudyTime", "StudyID", "AccessionNumber", "ReferringPhysicianName",
                "PatientBirthDate"):
        setattr(doc, key, dicom.get(key, ""))
    for key in ("StudyDescription", "IssuerOfPatientID"):
        if dicom.get(key):
            setattr(doc, key, dicom[key])
    doc.PatientID = patient.get("id") or dicom.get("PatientID") or ("" if sourceUIDs else studyUID)
    doc.PatientName = (dicom.get("PatientName", "")
                       if not patient.get("name") or patient["name"] == formatPersonName(dicom.get("PatientName"))
                       else patient["name"])
    doc.PatientSex = {"Female": "F", "Male": "M", "Other": "O"}.get(patient.get("sex"), dicom.get("PatientSex", ""))
    doc.StudyInstanceUID = studyUID
    doc.SeriesInstanceUID = seriesUID
    doc.SeriesDescription, doc.SeriesNumber = series
    doc.InstanceNumber = number
    doc.Manufacturer = SOFTWARE
    created = datetime.datetime.now()
    doc.SeriesDate = doc.ContentDate = created.strftime("%Y%m%d")
    doc.SeriesTime = doc.ContentTime = created.strftime("%H%M%S")
    if sourceUIDs and dicom.get("SOPClassUID"):
        doc.SourceImageSequence = []
        for uid in dict.fromkeys(sourceUIDs):
            ref = Dataset()
            ref.ReferencedSOPClassUID = dicom["SOPClassUID"]
            ref.ReferencedSOPInstanceUID = uid
            doc.SourceImageSequence.append(ref)
    return doc


def _saveDataset(doc, path):
    import pydicom
    if int(pydicom.__version_info__[0]) >= 3:
        doc.save_as(path, enforce_file_format=True)
    else:  # Slicer versions shipping pydicom 2.x
        doc.is_little_endian = True
        doc.is_implicit_VR = False
        doc.save_as(path, write_like_original=False)


def readPngRgb(path):
    """An 8-bit RGB PNG as a (rows, columns, 3) uint8 array, top row first."""
    import vtk
    from vtk.util import numpy_support
    reader = vtk.vtkPNGReader()
    reader.SetFileName(path)
    reader.Update()
    image = reader.GetOutput()
    columns, rows, _ = image.GetDimensions()
    scalars = image.GetPointData().GetScalars()
    if scalars is None or columns < 1 or rows < 1:
        raise IOError(f"Could not read {path}")
    pixels = numpy_support.vtk_to_numpy(scalars).reshape(rows, columns, -1)
    if pixels.shape[2] == 1:
        pixels = np.repeat(pixels, 3, axis=2)
    return np.ascontiguousarray(pixels[::-1, :, :3]).astype(np.uint8)     # VTK's first row is the bottom one


def writeDicomImages(paths, pngPaths, title, patient, dicom, studyUID, sourceUIDs, series=REPORT_SERIES,
                     seriesUID=None, firstNumber=1, pixelSpacingMm=25.4 / IMAGE_DPI):
    """Images (the report pages, screenshots) as Secondary Capture instances (RGB, one per image, one series in the
    source study); the SOP Instance UIDs. pixelSpacingMm None: no Pixel Spacing (a screenshot)."""
    from pydicom.uid import generate_uid
    secondaryCapture = "1.2.840.10008.5.1.4.1.1.7"
    seriesUID = seriesUID or generate_uid()
    uids = []
    for number, (path, png) in enumerate(zip(paths, pngPaths), start=firstNumber):
        pixels = readPngRgb(png)
        doc = _reportDataset(path, secondaryCapture, patient, dicom, studyUID, sourceUIDs, seriesUID, number,
                             series)
        doc.Modality = "OT"
        doc.ConversionType = "WSD"
        doc.ImageType = ["DERIVED", "SECONDARY"]
        doc.DerivationDescription = title
        doc.ImageComments = (f"{title}, page {number} of {len(pngPaths)}" if series == REPORT_SERIES
                             else f"{title} {number}")
        doc.BurnedInAnnotation = "YES"
        doc.PatientOrientation = ""
        doc.SamplesPerPixel = 3
        doc.PhotometricInterpretation = "RGB"
        doc.PlanarConfiguration = 0
        doc.Rows, doc.Columns = int(pixels.shape[0]), int(pixels.shape[1])
        doc.BitsAllocated = 8
        doc.BitsStored = 8
        doc.HighBit = 7
        doc.PixelRepresentation = 0
        if pixelSpacingMm:
            doc.PixelSpacing = [f"{pixelSpacingMm:.6f}"] * 2      # the A4 page in mm
        doc.PixelData = pixels.tobytes()
        _saveDataset(doc, path)
        uids.append(str(doc.SOPInstanceUID))
    return uids


class PdfReport:
    """An A4 portrait PDF drawn top to bottom in blocks (a block that does not fit goes on the next page):

        report = PdfReport(path, title, banner="banner.jpg")
        report.heading("Patient and scan")
        report.keyValues(rows)
        report.imageGrid(images, columns=3, captions=..., background=(0, 0, 0))
        report.colorbar(rgba, low, high, "t")
        report.table(columns, rows, ...)
        report.paragraph(text)
        report.finish()

    Each block is drawn twice: first to count the pages (the footer's "page i of n"), then for real; the blocks are
    kept as calls and replayed in finish()."""

    def __init__(self, path, title=TITLE, banner=None, created=None):
        """banner: an image file across the top of the first page (None: the title alone)."""
        self.path = path
        self.title = title
        self.banner = banner
        self.created = created or datetime.datetime.now()
        self._blocks = []

    # the blocks (recorded)
    def heading(self, text):
        self._blocks.append(("heading", (text,), {}))

    def keyValues(self, rows, pairs=2):
        self._blocks.append(("keyValues", (rows, pairs), {}))

    def imageGrid(self, images, columns, captions=None, background=(0, 0, 0), cellAspect=1.0, sideLabels=None,
                  rowTitles=None, maxCellMm=None):
        """maxCellMm: the cells at most this wide (mm; the grid centred), else as wide as the page allows."""
        self._blocks.append(("imageGrid", (images, columns), dict(captions=captions, background=background,
                                                                  cellAspect=cellAspect, sideLabels=sideLabels,
                                                                  rowTitles=rowTitles, maxCellMm=maxCellMm)))

    def imageRow(self, images, captions=None, rowTitle=None, maxHeightMm=50.0):
        """Images side by side at one height (each as wide as its aspect asks; no padding), at most maxHeightMm high
        and as wide as the page; the row centred."""
        self._blocks.append(("imageRow", (images,), dict(captions=captions, rowTitle=rowTitle,
                                                         maxHeightMm=maxHeightMm)))

    def colorbar(self, rgba, low, high, title="", labelFormat="{:.1f}"):
        self._blocks.append(("colorbar", (rgba, low, high, title, labelFormat), {}))

    def table(self, columns, rows, swatches=None, emphasis=None, alignRight=None):
        self._blocks.append(("table", (columns, rows), dict(swatches=swatches, emphasis=emphasis,
                                                            alignRight=alignRight)))

    def paragraph(self, text, size=8.5, italic=False, bold=False, color=(0, 0, 0)):
        self._blocks.append(("paragraph", (text,), dict(size=size, italic=italic, bold=bold, color=color)))

    def space(self, mm=GAP_MM):
        self._blocks.append(("space", (mm,), {}))

    def pageBreak(self):
        self._blocks.append(("pageBreak", (), {}))

    def keepWithNext(self):
        """The next two blocks on the same page (a heading with what follows it)."""
        self._blocks.append(("keep", (), {}))

    def finish(self):
        """The PDF written (two passes: the page count, then the pages); the number of pages."""
        import tempfile
        handle, scratch = tempfile.mkstemp(suffix=".pdf")
        os.close(handle)
        try:
            pages = self._draw(scratch, None)
        finally:
            try:
                os.remove(scratch)
            except OSError:
                pass
        return self._draw(self.path, pages)

    def renderImages(self, folder, prefix="page", dpi=IMAGE_DPI):
        """The pages drawn as images (PNG files in folder, white A4 at dpi; two passes like finish()); their paths."""
        pages = len(self._draw(None, None, imageFolder=folder, prefix=None, dpi=dpi))     # counted, not saved
        return self._draw(None, pages, imageFolder=folder, prefix=prefix, dpi=dpi)

    # drawing
    def _draw(self, path, total, imageFolder=None, prefix="page", dpi=RESOLUTION_DPI):
        """Draw the blocks: to the PDF path (the page count returned), or as page images into imageFolder (their
        paths returned)."""
        qt = _qt()
        self._qt, self._total = qt, total
        self._images = None
        if imageFolder is None:
            writer = self._writerFor(qt, path)
            self._scale = RESOLUTION_DPI / 25.4           # device pixels per mm
        else:
            writer = None
            self._images = dict(folder=imageFolder, prefix=prefix, dpi=dpi, paths=[])
            self._scale = dpi / 25.4
        self._writer = writer
        painter = self._painter = qt.QPainter()
        device = writer if writer is not None else self._pageImage()
        if not painter.begin(device):
            raise IOError(f"Could not write {path or imageFolder}")
        self._page = 1
        self._pageTop = MARGIN_MM + HEADER_MM
        try:
            self._renderHints()
            self._startPage()
            blocks = list(self._blocks)
            index = 0
            while index < len(blocks):
                kind, args, kwargs = blocks[index]
                if kind == "keep":
                    following = blocks[index + 1:index + 3]
                    need = sum(self._height(*block) for block in following)
                    if need <= self._bottom() - self._top() and self._y + need > self._bottom():
                        self._newPage()
                    index += 1
                    continue
                getattr(self, "_" + kind)(*args, **kwargs)
                index += 1
            self._footer()
        finally:
            painter.end()
        if self._images is not None:
            self._savePageImage()
            return list(self._images["paths"])
        return self._page

    def _renderHints(self):
        self._painter.setRenderHint(self._qt.QPainter.Antialiasing, True)
        self._painter.setRenderHint(self._qt.QPainter.SmoothPixmapTransform, True)
        self._painter.setRenderHint(self._qt.QPainter.TextAntialiasing, True)

    def _pageImage(self):
        """A new white page image (A4 at the image dpi; the dpi set so point sizes match the PDF)."""
        qt, dpi = self._qt, self._images["dpi"]
        image = qt.QImage(int(round(PAGE_WIDTH_MM / 25.4 * dpi)), int(round(PAGE_HEIGHT_MM / 25.4 * dpi)),
                          qt.QImage.Format_RGB32)
        image.setDotsPerMeterX(int(round(dpi / 0.0254)))
        image.setDotsPerMeterY(int(round(dpi / 0.0254)))
        image.fill(qt.QColor(255, 255, 255))
        self._images["image"] = image
        return image

    def _savePageImage(self):
        images = self._images
        if images["prefix"] is None:
            images["paths"].append(None)
            return
        path = os.path.join(images["folder"], f"{images['prefix']}_{len(images['paths']) + 1:02d}.png")
        if not images["image"].save(path, "PNG"):
            raise IOError(f"Could not write {path}")
        images["paths"].append(path)

    @staticmethod
    def _writerFor(qt, path):
        """The paged device writing the PDF: a QPdfWriter, else a QPrinter in PDF mode (a Qt wrapping without
        QPdfWriter); A4, no margins, RESOLUTION_DPI."""
        if hasattr(qt, "QPdfWriter"):
            writer = qt.QPdfWriter(path)
            writer.setResolution(RESOLUTION_DPI)
            try:
                writer.setTitle(TITLE)
                writer.setCreator(SOFTWARE)
            except Exception:
                pass
        else:
            writer = qt.QPrinter(qt.QPrinter.HighResolution)
            writer.setOutputFormat(qt.QPrinter.PdfFormat)
            writer.setOutputFileName(path)
            writer.setResolution(RESOLUTION_DPI)
            try:
                writer.setFullPage(True)
            except Exception:
                pass
        try:
            writer.setPageSize(qt.QPageSize(qt.QPageSize.A4))
        except Exception:
            writer.setPageSize(qt.QPagedPaintDevice.A4)
        try:
            writer.setPageMargins(qt.QMarginsF(0, 0, 0, 0))
        except Exception:
            pass
        return writer

    def _mm(self, value):
        return value * self._scale

    def _rect(self, x, y, w, h):
        return self._qt.QRectF(self._mm(x), self._mm(y), self._mm(w), self._mm(h))

    def _font(self, size, bold=False, italic=False):
        font = self._qt.QFont("Helvetica")
        font.setPointSizeF(size)
        font.setBold(bool(bold))
        font.setItalic(bool(italic))
        self._painter.setFont(font)
        return font

    def _pen(self, rgb, width_mm=0.2):
        pen = self._qt.QPen(self._qt.QColor(*rgb))
        pen.setWidthF(self._mm(width_mm))
        self._painter.setPen(pen)

    def _text(self, x, y, w, h, text, flags=None, size=9, bold=False, italic=False, color=(0, 0, 0)):
        qt = self._qt
        self._font(size, bold, italic)
        self._pen(color)
        self._painter.drawText(self._rect(x, y, w, h), int(flags if flags is not None else
                                                          (qt.Qt.AlignLeft | qt.Qt.AlignVCenter)), str(text))

    def _textHeight(self, text, width, size, bold=False, italic=False):
        """Height (mm) of text wrapped in width mm."""
        qt = self._qt
        self._font(size, bold, italic)
        bound = self._painter.boundingRect(self._rect(0, 0, width, 1000), int(qt.Qt.TextWordWrap | qt.Qt.AlignLeft),
                                           str(text))
        return bound.height() / self._scale

    def _width(self):
        return PAGE_WIDTH_MM - 2 * MARGIN_MM

    def _top(self):
        """Where the page's blocks start (below the header; lower on the first page, under the banner)."""
        return getattr(self, "_pageTop", MARGIN_MM + HEADER_MM)

    def _bottom(self):
        return PAGE_HEIGHT_MM - MARGIN_MM - FOOTER_MM

    def _banner(self):
        """The banner image (QImage) or None."""
        image = getattr(self, "_bannerImage", None)
        if image is None and self.banner and os.path.isfile(self.banner):
            image = self._qt.QImage(self.banner)
            image = image if not image.isNull() else None
            self._bannerImage = image
        return image

    def _startPage(self):
        qt = self._qt
        x, w = MARGIN_MM, self._width()
        top = MARGIN_MM
        banner = self._banner() if self._page == 1 else None
        if banner is not None:          # the banner across the whole first page (edge to edge), the title under it
            height = PAGE_WIDTH_MM * banner.height() / float(max(banner.width(), 1))
            self._painter.drawImage(self._rect(0, BANNER_TOP_MM, PAGE_WIDTH_MM, height), banner)
            top = BANNER_TOP_MM + height + 4.0
        self._text(x, top, w * 0.7, 8, self.title, size=13, bold=True, color=HEADING_COLOR)
        self._text(x + w * 0.3, top, w * 0.7, 8, f"Created {self.created.strftime('%d.%m.%Y %H:%M')}",
                   flags=qt.Qt.AlignRight | qt.Qt.AlignVCenter, size=7.5, color=LABEL_COLOR)
        self._pen(RULE_COLOR, 0.4)
        y = top + HEADER_MM - 5
        self._painter.drawLine(qt.QPointF(self._mm(x), self._mm(y)), qt.QPointF(self._mm(x + w), self._mm(y)))
        self._y = self._pageTop = y + 3

    def _footer(self):
        qt = self._qt
        x, w = MARGIN_MM, self._width()
        top = PAGE_HEIGHT_MM - MARGIN_MM - FOOTER_MM + 2
        self._pen(GRID_COLOR, 0.2)
        self._painter.drawLine(qt.QPointF(self._mm(x), self._mm(top)), qt.QPointF(self._mm(x + w), self._mm(top)))
        self._text(x, top + 0.5, w - 25, 4, SOFTWARE, flags=qt.Qt.AlignLeft | qt.Qt.AlignTop, size=7,
                   color=LABEL_COLOR)
        page = f"Page {self._page}" + (f" of {self._total}" if self._total else "")
        self._text(x + w - 25, top + 0.5, 25, 4, page, flags=qt.Qt.AlignRight | qt.Qt.AlignTop, size=7,
                   color=LABEL_COLOR)

    def _newPage(self):
        self._footer()
        if self._images is None:
            self._writer.newPage()
        else:                       # the page image saved, the painter moved to a new one
            self._painter.end()
            self._savePageImage()
            if not self._painter.begin(self._pageImage()):
                raise IOError("Could not draw the next report page.")
            self._renderHints()
        self._page += 1
        self._startPage()

    def _ensure(self, height):
        """A new page when height mm does not fit below (unless the page is still empty)."""
        if self._y + height > self._bottom() and self._y > self._top() + 0.01:
            self._newPage()

    # heights of the blocks (for keeping a heading with what follows)
    def _height(self, kind, args, kwargs):
        if kind == "heading":
            return 8.0
        if kind == "keyValues":
            rows, pairs = args
            return math.ceil(len(rows) / pairs) * self._rowHeight + GAP_MM
        if kind == "imageGrid":
            images, columns = args
            w = self._width() / columns
            w = min(w, kwargs["maxCellMm"]) if kwargs.get("maxCellMm") else w
            rows = math.ceil(len(images) / columns)
            caption = 4.0 if kwargs.get("captions") else 0.0
            return rows * (w * kwargs.get("cellAspect", 1.0) + caption) + GAP_MM
        if kind == "imageRow":
            return self._rowGeometry(args[0], kwargs.get("rowTitle"), kwargs.get("maxHeightMm", 50.0))[0] + \
                (4.0 if kwargs.get("captions") else 0.0) + GAP_MM
        if kind == "colorbar":
            return 12.0
        if kind == "table":
            return 2 * self._tableRow + GAP_MM          # its header and a first row
        if kind == "paragraph":
            return self._textHeight(args[0], self._width(), kwargs.get("size", 8.5), kwargs.get("bold", False),
                                    kwargs.get("italic", False)) + GAP_MM
        if kind == "space":
            return args[0]
        return 0.0

    _rowHeight = 4.8
    _tableRow = 4.6

    def _heading(self, text):
        self._ensure(8.0 + 10.0)
        self._text(MARGIN_MM, self._y, self._width(), 6, text, size=11, bold=True, color=HEADING_COLOR)
        self._pen(GRID_COLOR, 0.2)
        y = self._mm(self._y + 6.5)
        self._painter.drawLine(self._qt.QPointF(self._mm(MARGIN_MM), y),
                               self._qt.QPointF(self._mm(MARGIN_MM + self._width()), y))
        self._y += 8.0

    def _space(self, mm):
        self._y += mm

    def _pageBreak(self):
        if self._y > self._top() + 0.01:
            self._newPage()

    def _keyValues(self, rows, pairs=2):
        qt = self._qt
        lines = math.ceil(len(rows) / pairs)
        self._ensure(lines * self._rowHeight)
        width = self._width() / pairs
        labelWidth = 34.0
        columnFirst = [rows[c * lines:(c + 1) * lines] for c in range(pairs)]     # down the first column first
        for line in range(lines):
            if line % 2 == 0:
                self._painter.fillRect(self._rect(MARGIN_MM, self._y, self._width(), self._rowHeight),
                                       qt.QColor(*ZEBRA_COLOR))
            for c in range(pairs):
                if line >= len(columnFirst[c]):
                    continue
                label, value = columnFirst[c][line]
                x = MARGIN_MM + c * width
                self._text(x + 1.5, self._y, labelWidth, self._rowHeight, label, size=8, bold=True,
                           color=LABEL_COLOR)
                self._text(x + 1.5 + labelWidth, self._y, width - labelWidth - 3, self._rowHeight,
                           self._elided(value, width - labelWidth - 3, 8.5), size=8.5)
            self._y += self._rowHeight
        self._y += GAP_MM

    def _elided(self, text, width, size, bold=False):
        qt = self._qt
        self._font(size, bold)
        metrics = self._painter.fontMetrics()      # the PDF's resolution (not the screen's)
        return metrics.elidedText(str(text), qt.Qt.ElideRight, int(self._mm(width)))

    def _drawImageFit(self, image, x, y, w, h, background):
        qt = self._qt
        if background is not None:
            self._painter.fillRect(self._rect(x, y, w, h), qt.QColor(*background))
        if image is None or image.isNull():
            self._text(x, y, w, h, "not available", flags=qt.Qt.AlignCenter, size=7, color=LABEL_COLOR)
            return
        iw, ih = float(image.width()), float(image.height())
        scale = min(w / iw, h / ih)
        dw, dh = iw * scale, ih * scale
        self._painter.drawImage(self._rect(x + (w - dw) / 2, y + (h - dh) / 2, dw, dh), image)

    def _imageGrid(self, images, columns, captions=None, background=(0, 0, 0), cellAspect=1.0, sideLabels=None,
                   rowTitles=None, maxCellMm=None):
        qt = self._qt
        titleWidth = 6.0 if rowTitles else 0.0
        width = (self._width() - titleWidth) / columns
        left = MARGIN_MM
        if maxCellMm and width > maxCellMm:
            left += (self._width() - titleWidth - columns * maxCellMm) / 2.0      # centred
            width = maxCellMm
        cellH = width * cellAspect
        captionH = 4.0 if captions else 0.0
        rows = math.ceil(len(images) / columns)
        for r in range(rows):
            self._ensure(cellH + captionH)
            if rowTitles and r < len(rowTitles) and rowTitles[r]:
                self._painter.save()
                self._font(8, True)
                self._pen(LABEL_COLOR)
                center = qt.QPointF(self._mm(left + titleWidth / 2), self._mm(self._y + cellH / 2))
                self._painter.translate(center)
                self._painter.rotate(-90)
                self._painter.drawText(qt.QRectF(-self._mm(cellH / 2), -self._mm(titleWidth / 2), self._mm(cellH),
                                                 self._mm(titleWidth)), int(qt.Qt.AlignCenter), str(rowTitles[r]))
                self._painter.restore()
            for c in range(columns):
                i = r * columns + c
                if i >= len(images):
                    break
                x = left + titleWidth + c * width
                pad = 0.6
                self._drawImageFit(images[i], x + pad, self._y, width - 2 * pad, cellH, background)
                labels = sideLabels[i] if sideLabels and i < len(sideLabels) else None
                if labels:
                    dark = background is not None and sum(background) < 384
                    color = (255, 255, 255) if dark else (0, 0, 0)
                    self._text(x + pad + 1, self._y, 6, cellH, labels[0], flags=qt.Qt.AlignLeft | qt.Qt.AlignVCenter,
                               size=9, bold=True, color=color)
                    self._text(x + width - pad - 7, self._y, 6, cellH, labels[1],
                               flags=qt.Qt.AlignRight | qt.Qt.AlignVCenter, size=9, bold=True, color=color)
                if captions and i < len(captions) and captions[i]:
                    self._text(x, self._y + cellH, width, captionH, captions[i], flags=qt.Qt.AlignCenter, size=7.5,
                               color=LABEL_COLOR)
            self._y += cellH + captionH
        self._y += GAP_MM

    ROW_GAP_MM = 1.5

    def _rowGeometry(self, images, rowTitle, maxHeightMm):
        """(row height, [(width, height)], left) of an image row: one scale (mm per pixel) for all its images (slices
        sampled at one spacing keep their sizes against each other), the row at most maxHeightMm high and as wide as
        the page."""
        titleWidth = 6.0 if rowTitle else 0.0
        sizes = [(float(image.width()), float(image.height())) if image is not None and not image.isNull()
                 and image.width() > 0 and image.height() > 0 else (1.0, 1.0) for image in images]
        gaps = self.ROW_GAP_MM * max(len(images) - 1, 0)
        available = self._width() - titleWidth - gaps
        scale = min(available / max(sum(w for w, _ in sizes), 1e-6),
                    float(maxHeightMm) / max(max((h for _, h in sizes), default=1.0), 1e-6))
        cells = [(w * scale, h * scale) for w, h in sizes]
        height = max((h for _, h in cells), default=0.0)
        used = sum(w for w, _ in cells) + gaps
        left = MARGIN_MM + titleWidth + (self._width() - titleWidth - used) / 2.0
        return height, cells, left

    def _imageRow(self, images, captions=None, rowTitle=None, maxHeightMm=50.0):
        qt = self._qt
        height, cells, left = self._rowGeometry(images, rowTitle, maxHeightMm)
        captionH = 4.0 if captions else 0.0
        self._ensure(height + captionH)
        if rowTitle:
            self._painter.save()
            self._font(8, True)
            self._pen(LABEL_COLOR)
            center = qt.QPointF(self._mm(left - 3.5), self._mm(self._y + height / 2))
            self._painter.translate(center)
            self._painter.rotate(-90)
            self._painter.drawText(qt.QRectF(-self._mm(height / 2), -self._mm(3.0), self._mm(height), self._mm(6.0)),
                                   int(qt.Qt.AlignCenter), str(rowTitle))
            self._painter.restore()
        x = left
        for index, (image, (width, imageHeight)) in enumerate(zip(images, cells)):
            self._drawImageFit(image, x, self._y + (height - imageHeight) / 2.0, width, imageHeight, None)
            if captions and index < len(captions) and captions[index]:
                self._text(x, self._y + height, width, captionH, captions[index], flags=qt.Qt.AlignCenter, size=7.5,
                           color=LABEL_COLOR)
            x += width + self.ROW_GAP_MM
        self._y += height + captionH + GAP_MM

    def _colorbar(self, rgba, low, high, title="", labelFormat="{:.1f}"):
        qt = self._qt
        height = 12.0
        self._ensure(height)
        barW, barH = min(110.0, self._width() * 0.6), 3.5
        x = MARGIN_MM + (self._width() - barW) / 2
        y = self._y + 1.0
        bar = qt.QImage(max(len(rgba), 1), 1, qt.QImage.Format_RGB32)      # one pixel per colour, stretched
        for i, (r, g, b, a) in enumerate(rgba):
            bar.setPixel(i, 0, qt.QColor(int(round(255 * r)), int(round(255 * g)), int(round(255 * b))).rgb())
        self._painter.setRenderHint(qt.QPainter.SmoothPixmapTransform, False)
        self._painter.drawImage(self._rect(x, y, barW, barH), bar)
        self._painter.setRenderHint(qt.QPainter.SmoothPixmapTransform, True)
        self._pen((60, 60, 60), 0.15)
        self._painter.drawRect(self._rect(x, y, barW, barH))
        if title:
            self._text(x - 42, y - 1, 40, barH + 2, title, flags=qt.Qt.AlignRight | qt.Qt.AlignVCenter, size=8,
                       bold=True)
        for tick in colorbarTicks(low, high):
            if high <= low:
                break
            tx = x + barW * (tick - low) / (high - low)
            self._pen((60, 60, 60), 0.15)
            self._painter.drawLine(qt.QPointF(self._mm(tx), self._mm(y + barH)),
                                   qt.QPointF(self._mm(tx), self._mm(y + barH + 1.0)))
            self._text(tx - 10, y + barH + 1.0, 20, 4, labelFormat.format(tick), flags=qt.Qt.AlignCenter, size=7)
        self._y += height

    def _table(self, columns, rows, swatches=None, emphasis=None, alignRight=None):
        """columns: names; rows: lists of texts; swatches: (r, g, b 0..1) or None per row (a colour box before the
        first column); emphasis: per row {column index: (r, g, b 0..255)} drawn bold in that colour."""
        qt = self._qt
        size = 7.5
        swatchW = 4.0 if swatches else 0.0
        alignRight = alignRight if alignRight is not None else [c > 0 for c in range(len(columns))]
        # column widths: the widest text of each column, the first one taking what is left
        widths = []
        for c, name in enumerate(columns):
            texts = [str(name)] + [str(row[c]) for row in rows if c < len(row)]
            self._font(size, True)
            metrics = self._painter.fontMetrics()
            widths.append(max(metrics.horizontalAdvance(t) if hasattr(metrics, "horizontalAdvance")
                              else metrics.width(t) for t in texts) / self._scale + 3.0)
        available = self._width() - swatchW
        rest = sum(widths[1:])
        if rest + widths[0] > available:
            widths[0] = max(30.0, available - rest)
        else:
            extra = available - rest - widths[0]
            widths[0] += extra * 0.5
            for c in range(1, len(widths)):
                widths[c] += extra * 0.5 / max(len(widths) - 1, 1)
        rowH = self._tableRow

        def header():
            self._painter.fillRect(self._rect(MARGIN_MM, self._y, self._width(), rowH), qt.QColor(*HEADER_FILL))
            x = MARGIN_MM + swatchW
            for c, name in enumerate(columns):
                flags = (qt.Qt.AlignRight if alignRight[c] else qt.Qt.AlignLeft) | qt.Qt.AlignVCenter
                self._text(x + 1, self._y, widths[c] - 2, rowH, name, flags=flags, size=size, bold=True)
                x += widths[c]
            self._y += rowH

        self._ensure(2 * rowH)
        header()
        for r, row in enumerate(rows):
            if self._y + rowH > self._bottom():
                self._newPage()
                header()
            if r % 2 == 1:
                self._painter.fillRect(self._rect(MARGIN_MM, self._y, self._width(), rowH), qt.QColor(*ZEBRA_COLOR))
            if swatches and r < len(swatches) and swatches[r] is not None:
                rr, gg, bb = (int(round(255 * v)) for v in swatches[r][:3])
                self._painter.fillRect(self._rect(MARGIN_MM + 0.8, self._y + 0.8, swatchW - 1.6, rowH - 1.6),
                                       qt.QColor(rr, gg, bb))
            marks = (emphasis[r] if emphasis and r < len(emphasis) else None) or {}
            x = MARGIN_MM + swatchW
            for c in range(len(columns)):
                text = str(row[c]) if c < len(row) and row[c] is not None else ""
                flags = (qt.Qt.AlignRight if alignRight[c] else qt.Qt.AlignLeft) | qt.Qt.AlignVCenter
                color = marks.get(c)
                self._text(x + 1, self._y, widths[c] - 2, rowH, self._elided(text, widths[c] - 2, size, bool(color)),
                           flags=flags, size=size, bold=bool(color), color=color or (0, 0, 0))
                x += widths[c]
            self._y += rowH
        self._pen(GRID_COLOR, 0.2)
        self._painter.drawLine(qt.QPointF(self._mm(MARGIN_MM), self._mm(self._y)),
                               qt.QPointF(self._mm(MARGIN_MM + self._width()), self._mm(self._y)))
        self._y += GAP_MM

    def _paragraph(self, text, size=8.5, italic=False, bold=False, color=(0, 0, 0)):
        qt = self._qt
        height = self._textHeight(text, self._width(), size, bold, italic)
        self._ensure(min(height, self._bottom() - self._top()))
        self._text(MARGIN_MM, self._y, self._width(), height + 1, text,
                   flags=qt.Qt.AlignLeft | qt.Qt.AlignTop | qt.Qt.TextWordWrap, size=size, bold=bold, italic=italic,
                   color=color)
        self._y += height + GAP_MM
