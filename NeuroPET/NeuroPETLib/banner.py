"""The Mergen banner (Qt): Resources/Images/banner.jpg, scaled to the panel's width with its aspect ratio
kept. Shown at the top of the panel on the Data step of the workflows and the Review step of Static Neuro PET/SPECT,
and at the top of the Neuro PET Database Manager."""

import os

import qt

BANNER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Resources", "Images",
                           "banner.jpg")
MAX_HEIGHT = 240        # logical pixels: the banner never takes more of the panel than this


class MergenBanner(qt.QLabel):
    """A label that shows the banner picture at the panel's width (aspect kept, at most MAX_HEIGHT high). Hidden
    when the picture is missing."""

    def __init__(self, parent=None):
        if parent is None:
            qt.QLabel.__init__(self)
        else:
            qt.QLabel.__init__(self, parent)
        self._pixmap = None
        self._shownWidth = -1
        self.setAlignment(qt.Qt.AlignCenter)
        self.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
        self.setMinimumWidth(1)
        self.toolTip = "Mergen: Neuro PET/SPECT Analysis"
        if os.path.exists(BANNER_PATH):
            pixmap = qt.QPixmap(BANNER_PATH)
            self._pixmap = None if pixmap.isNull() else pixmap
        if self._pixmap is None:
            self.hide()
        else:
            self._fit()

    def hasPicture(self):
        return self._pixmap is not None

    def _fit(self):
        if self._pixmap is None:
            return
        width = max(1, self.contentsRect().width())
        if width == self._shownWidth:
            return
        self._shownWidth = width
        ratio = self._pixmap.height() / max(1, self._pixmap.width())
        height = int(round(width * ratio))
        if height > MAX_HEIGHT:
            height = MAX_HEIGHT
            width = int(round(height / ratio))
        try:
            scale = float(self.devicePixelRatioF())
        except Exception:       # noqa: BLE001 - older Qt
            scale = 1.0
        scaled = self._pixmap.scaled(int(width * scale), int(height * scale), qt.Qt.KeepAspectRatio,
                                     qt.Qt.SmoothTransformation)
        try:
            scaled.setDevicePixelRatio(scale)
        except Exception:       # noqa: BLE001 - older Qt
            pass
        self.setFixedHeight(height)
        self.setPixmap(scaled)

    def showEvent(self, event):
        qt.QLabel.showEvent(self, event)
        self._fit()

    def resizeEvent(self, event):
        qt.QLabel.resizeEvent(self, event)
        self._fit()
