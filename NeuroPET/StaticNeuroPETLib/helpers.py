"""
Helpers of the Static Neuro PET/SPECT module: numbers in logs, screens (second monitor), the step icons.
"""

import logging

import numpy as np
import qt
import slicer

from DynamicNeuroPET import REFERENCE_SUFFIX, STEP_DATA, STEP_NORM, STEP_REG, STEP_REGIONS, _badgePixmap

from StaticNeuroPETLib.constants import STEP_GLYPH_COLOR, STEP_ICON_SIZE


def _fmt(value, digits=2):
    """A number for a log, '' when it is not finite."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{value:.{digits}f}" if np.isfinite(value) else ""


def _sameRegion(segmentName, title):
    """A segment name ("Cerebellum Gray Matter (reference)") naming the region title ("Cerebellum Gray Matter")."""
    def clean(text):
        text = (text or "").strip().lower()
        suffix = REFERENCE_SUFFIX.strip().lower()
        return text[:-len(suffix)].strip() if text.endswith(suffix) else text
    return clean(segmentName) == clean(title)


# ---------------------------------------------------------------------------
# Screens and the Monitor 2 window (as in SlicerPETReviewCompare's dual monitor layouts)
# ---------------------------------------------------------------------------
def _screenCount():
    try:
        return len(list(qt.QGuiApplication.screens()))
    except Exception:
        return 1


def _mainViewportAttributes():
    """XML attributes of the main window viewport, copied from Slicer's own dual monitor layouts, so that the first
    <layout> of <viewports> is shown in the main window."""
    import xml.etree.ElementTree as ElementTree
    try:
        layoutNode = slicer.app.layoutManager().layoutLogic().GetLayoutNode()
        for attributeName in dir(slicer.vtkMRMLLayoutNode):
            if "DualMonitor" not in attributeName:
                continue
            layoutId = getattr(slicer.vtkMRMLLayoutNode, attributeName)
            if not isinstance(layoutId, int) or not layoutNode.IsLayoutDescription(layoutId):
                continue
            root = ElementTree.fromstring(layoutNode.GetLayoutDescription(layoutId))
            if root.tag != "viewports":
                continue
            for element in root.findall("layout"):
                if (element.get("dockable") or "").lower() != "true":
                    name = element.get("name")
                    return f' name="{name}"' if name is not None else ""
    except Exception as error:
        logging.debug(f"Static Neuro: could not read Slicer's dual monitor layouts: {error}")
    return ""


def _secondaryScreen():
    """A screen other than the one showing the main window; None with a single screen."""
    mainWindow = slicer.util.mainWindow()
    screens = list(qt.QGuiApplication.screens())
    if mainWindow is None or len(screens) < 2:
        return None
    center = mainWindow.frameGeometry.center()
    return next((screen for screen in screens if not screen.geometry.contains(center)), None)


def _showOnSecondScreen(window):
    """The floating Monitor 2 window as a normal window (minimize / maximize / close), maximized on the second
    screen; changes only what is not so already (re-applying the flags hides and shows the window again)."""
    wanted = (qt.Qt.Window | qt.Qt.CustomizeWindowHint | qt.Qt.WindowTitleHint | qt.Qt.WindowSystemMenuHint
              | qt.Qt.WindowMinMaxButtonsHint | qt.Qt.WindowCloseButtonHint)
    try:
        hasFlags = (int(window.windowFlags()) & int(wanted)) == int(wanted)
    except (TypeError, ValueError):
        hasFlags = False
    if not hasFlags:
        window.setWindowFlags(wanted)
    screen = _secondaryScreen()
    if screen is None:
        if not window.visible:
            window.show()
        window.raise_()
        return
    onSecond = screen.geometry.contains(window.frameGeometry.center())
    if not onSecond:
        window.setGeometry(screen.availableGeometry)
    if not (onSecond and window.isMaximized() and window.visible):
        window.showMaximized()
    window.raise_()


# ---------------------------------------------------------------------------
# Step icons: a pictogram of the step next to its state badge
# ---------------------------------------------------------------------------
def _stepGlyph(key, size=STEP_ICON_SIZE):
    """A small line drawing of a step (drawn at twice the size): the patient (data), two overlapping images
    (registration), the template grid (normalization), the atlas regions (regions), a bar chart (review)."""
    scale = 2
    full = size * scale
    pixmap = qt.QPixmap(full, full)
    pixmap.fill(qt.QColor(0, 0, 0, 0))
    painter = qt.QPainter(pixmap)
    try:
        painter.setRenderHint(qt.QPainter.Antialiasing)
        color = qt.QColor(STEP_GLYPH_COLOR)
        pen = qt.QPen(color)
        pen.setWidth(2 * scale)
        painter.setPen(pen)
        painter.setBrush(qt.QBrush(qt.QColor(0, 0, 0, 0)))
        m = 3 * scale                       # margin
        w = full - 2 * m
        if key == STEP_DATA:                # head and shoulders
            head = int(w * 0.42)
            painter.drawEllipse(m + (w - head) // 2, m, head, head)
            painter.drawArc(m + w // 8, m + int(w * 0.52), w - w // 4, int(w * 0.9), 0, 180 * 16)
        elif key == STEP_REG:               # two overlapping images
            side = int(w * 0.68)
            painter.drawRect(m, m, side, side)
            painter.setBrush(qt.QBrush(qt.QColor(color.red(), color.green(), color.blue(), 90)))
            painter.drawRect(m + w - side, m + w - side, side, side)
        elif key == STEP_NORM:              # the template grid
            painter.drawRoundedRect(m, m, w, w, 2 * scale, 2 * scale)
            thin = qt.QPen(color)
            thin.setWidth(scale)
            painter.setPen(thin)
            for i in (1, 2):
                painter.drawLine(m + i * w // 3, m, m + i * w // 3, m + w)
                painter.drawLine(m, m + i * w // 3, m + w, m + i * w // 3)
        elif key == STEP_REGIONS:           # a brain divided into regions
            painter.drawEllipse(m, m + w // 8, w, int(w * 0.75))
            painter.drawLine(m + w // 2, m + w // 8, m + w // 2, m + w // 8 + int(w * 0.75))
            painter.drawLine(m + w // 2, m + w // 2, m + w, m + w // 2)
            painter.drawLine(m + w // 4, m + w // 6, m + w // 4, m + int(w * 0.8))
        else:                               # review: a bar chart
            painter.setPen(qt.Qt.NoPen)
            painter.setBrush(qt.QBrush(color))
            bar = w // 5
            for i, height in enumerate((0.45, 0.85, 0.65)):
                h = int(w * height)
                painter.drawRect(m + i * (bar + bar // 2) + bar // 4, m + w - h, bar, h)
    finally:
        painter.end()
    return pixmap


def _stepIcon(key, state, number, size=STEP_ICON_SIZE):
    """The pictogram of a step and its state badge side by side (twice as wide as high)."""
    scale = 2
    full = size * scale
    pixmap = qt.QPixmap(2 * full, full)
    pixmap.fill(qt.QColor(0, 0, 0, 0))
    painter = qt.QPainter(pixmap)
    try:
        painter.drawPixmap(0, 0, _stepGlyph(key, size))
        painter.drawPixmap(full, 0, _badgePixmap(state, number, size))
    finally:
        painter.end()
    return qt.QIcon(pixmap)
