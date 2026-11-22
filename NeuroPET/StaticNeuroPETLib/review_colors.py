"""
Review: score maps and their colours, the PET, fusion and CT/MR display.

Part of the Static Neuro PET/SPECT module (StaticNeuroPET.py): the methods of ReviewColorsMixin, mixed into
StaticNeuroPETWidget there.
"""

import numpy as np
import slicer
import vtk

from DICOMGatedDynamicPETLib import colormaps, colortables
from NeuroWorkflowLib.base import _ijkToRas, _percentiles, _vtkMatrix
from NeuroWorkflowLib.scenestate import unsaved
from NeuroPETLib import neuroreview as nr

from StaticNeuroPETLib.constants import SCORE_COLOR_NODE, STEP_REVIEW


class ReviewColorsMixin:
    """The review colors methods of StaticNeuroPETWidget."""

    def _makeScoreNode(self):
        result = self.reviewResult
        scene = slicer.mrmlScene
        if result is None or result.get("z") is None:
            if self.scoreNode is not None and scene.IsNodePresent(self.scoreNode):
                scene.RemoveNode(self.scoreNode)
            self.scoreNode = None
            self.setNode("ReviewScore", None)
            return
        if self.scoreNode is None or not scene.IsNodePresent(self.scoreNode):
            self.scoreNode = scene.AddNewNodeByClass("vtkMRMLScalarVolumeNode",
                                                     scene.GenerateUniqueName(f"{result['volume'].GetName()} score"))
            self.scoreNode.CreateDefaultDisplayNodes()
        # saved with the scene and taken back when it is loaded (the comparison then made again into it)
        self.setNode("ReviewScore", self.scoreNode)
        self.scoreNode.SetIJKToRASMatrix(_vtkMatrix(_ijkToRas(result["grid"])))
        slicer.util.updateVolumeFromArray(self.scoreNode, result["z"])

    def _scoreKind(self):
        return self.scoreCombo.currentData or nr.DEFAULT_SCORE

    def _showScores(self):
        """The score map (z, or t = z / sqrt(1 + 1/n)) and the region table of the last comparison."""
        result = self.reviewResult
        if result is None:
            return
        kind = self._scoreKind()
        self._showScoreMap()
        self._updateReviewTable(kind)
        self._showSsp()

    def _showScoreMap(self):
        """The score map shown where the normals' uptake is high (the threshold of the Display box)."""
        result = self.reviewResult
        if result is None or self.scoreNode is None or result.get("z") is None:
            return
        kind = self._scoreKind()
        factor = nr.tFactor(result.get("n")) if kind == nr.SCORE_T else 1.0
        mask = result["mask"]
        masked = self._scoreStyle() == nr.STYLE_MASKED
        greyMatter = self.scoreMaskCombo.currentData == nr.MASK_GM and result.get("gm") is not None
        for widget in (self.scoreMaskCombo, self.hideBelowSpin, self.scoreBackgroundCombo):
            widget.enabled = masked
        self.gmSpin.enabled = masked and greyMatter
        self.uptakeSpin.enabled = masked and not greyMatter
        if not masked:
            pass        # the colored PET: every scored voxel (the score node for the data probe and the legend)
        elif greyMatter:
            mask = nr.greyMatterMask(result["gm"], mask, float(self.gmSpin.value))
        elif result.get("mean") is not None:
            mask = nr.uptakeMask(result["mean"], mask, self.uptakeSpin.value / 100.0)
        slicer.util.updateVolumeFromArray(self.scoreNode, nr.maskedScores(
            result["z"], mask, factor, float(self.hideBelowSpin.value) if masked else 0.0))
        self.scoreNode.SetName(f"{result['volume'].GetName()} {kind} map")
        self._applyScoreDisplay()
        self._updateLegends()

    # display
    @staticmethod
    def _rgbOf(label):
        """256 x 3 RGB (0..1) of a color map of colormaps.LABELS or a Slicer color table (by name)."""
        colors = colortables.table(label)
        if colors is None:
            colors = nr.colorTable(label)
        if colors is not None:
            return colors
        node = colormaps.colorNodeForLabel(label) if label in colormaps.LABELS else colormaps._builtInNode(label)
        lookup = node.GetScalarsToColors() if node is not None else None
        if lookup is None:
            return np.tile(np.linspace(0, 1, 256)[:, None], (1, 3))
        low, high = lookup.GetRange()
        colors = np.zeros((256, 3), dtype=float)
        rgb = [0.0, 0.0, 0.0]
        for i in range(256):        # GetColor(value, rgb): the output array form works in every VTK wrapping
            lookup.GetColor(low + (high - low) * i / 255.0, rgb)
            colors[i] = rgb
        return colors

    def _scoreColorNode(self):
        scene = slicer.mrmlScene
        node = scene.GetFirstNodeByName(SCORE_COLOR_NODE)
        if node is None or not node.IsA("vtkMRMLColorTableNode"):
            node = scene.CreateNodeByClass("vtkMRMLColorTableNode")
            node.UnRegister(None)
            node.SetName(SCORE_COLOR_NODE)
            node.SetTypeToUser()
            node.SetNumberOfColors(256)
            node.SetHideFromEditors(False)
            scene.AddNode(node)
        rgba, _ = self._scoreColors()
        wasModifying = node.StartModify()       # one Modified for the 256 colors (each redraws the score slices)
        try:
            for i, (r, g, b, a) in enumerate(rgba):
                node.SetColor(i, f"{i}", float(r), float(g), float(b), float(a))
        finally:
            node.EndModify(wasModifying)
        return node

    def _scoreSide(self):
        return self.scoreSideCombo.currentData if hasattr(self, "scoreSideCombo") else nr.SIDE_ALL

    def _scoreColors(self, label=None, maximum=None, side=None):
        """(RGBA, (low, high)) of the score colors: the score slices' (or a color map label's, a window's, a side's)."""
        label = label or self.scoreColorCombo.currentText
        return nr.scoreColors(self._rgbOf(label), side or self._scoreSide(),
                              float(self.scoreMaximumSpin.value if maximum is None else maximum),
                              float(self.thresholdSpin.value), neutralMidpoint=label in nr.NEUTRAL_MIDPOINT_COLORMAPS)

    def _applyScoreDisplay(self):
        if self.scoreNode is None or not slicer.mrmlScene.IsNodePresent(self.scoreNode):
            return
        display = self.scoreNode.GetDisplayNode()
        if display is None:
            self.scoreNode.CreateDefaultDisplayNodes()
            display = self.scoreNode.GetDisplayNode()
        _, (low, high) = self._scoreColors()
        display.SetAndObserveColorNodeID(self._scoreColorNode().GetID())
        display.SetAutoWindowLevel(False)
        display.SetWindowLevelMinMax(low, high)
        display.SetApplyThreshold(True)         # outside the brain mask (OUTSIDE) transparent: black
        display.SetLowerThreshold(nr.OUTSIDE / 2.0)
        display.SetUpperThreshold(-nr.OUTSIDE)
        self._updateScoreColors()
        self._updateCutFaces()
        self._updatePeel()

    def _scoreStyle(self):
        return self.scoreStyleCombo.currentData if hasattr(self, "scoreStyleCombo") else nr.STYLE_MASKED

    def _scoreColorsNode(self):
        """The colored PET (RGB: the score's color times the PET's brightness) when shown, else None."""
        node = getattr(self, "scoreRgbNode", None)
        if self._scoreStyle() != nr.STYLE_BRIGHTNESS or node is None or not slicer.mrmlScene.IsNodePresent(node):
            return None
        return node

    def _updateScoreColors(self):
        """The colored PET of the score row: each voxel the score's color, as bright as the patient's uptake."""
        result = self.reviewResult or {}
        if (self._scoreStyle() != nr.STYLE_BRIGHTNESS or result.get("z") is None
                or result.get("brightness") is None):
            return
        scene = slicer.mrmlScene
        node = getattr(self, "scoreRgbNode", None)
        if node is None or not scene.IsNodePresent(node):
            node = unsaved(scene.AddNewNodeByClass("vtkMRMLVectorVolumeNode", scene.GenerateUniqueName(
                f"{result['volume'].GetName()} score colors")))      # made again from the scores
            self.scoreRgbNode = node
        node.SetIJKToRASMatrix(_vtkMatrix(_ijkToRas(result["grid"])))
        factor = nr.tFactor(result.get("n")) if self._scoreKind() == nr.SCORE_T else 1.0
        rgba, (low, high) = self._scoreColors()
        # the scores quantized and the brightness as 0..255 once per comparison and score: a window change is then
        # a table look-up and an integer product (a right drag on the score row redraws it many times)
        # (the cache holds the comparison and its scores themselves and is compared by identity: an id() can be
        # given again to another comparison once this one is freed, which would color it with these values)
        cache = getattr(self, "_brightnessCache", None)
        if cache is None or cache[0][0] is not result or cache[0][1] is not result["z"] or cache[0][2] != factor:
            cache = self._brightnessCache = ((result, result["z"], factor), nr.quantizedBrightness(
                np.asarray(result["z"], np.float32) / np.float32(factor), result["brightness"]))
        current = slicer.util.arrayFromVolume(node) if node.GetImageData() is not None else None
        if (current is not None and current.shape == cache[1]["flat"].shape + (3,) and current.dtype == np.uint8
                and current.flags["C_CONTIGUOUS"]):
            nr.fastBrightnessColors(cache[1], rgba, high, low, out=current)       # in place: no new image
            slicer.util.arrayFromVolumeModified(node)
        else:
            slicer.util.updateVolumeFromArray(node, nr.fastBrightnessColors(cache[1], rgba, high, low))
        if node.GetDisplayNode() is None:
            node.CreateDefaultDisplayNodes()
        display = node.GetDisplayNode()
        if display is not None:     # the RGB as it is
            display.SetAutoWindowLevel(False)
            display.SetWindowLevelMinMax(0.0, 255.0)
        self._updateCutFaces()
        self._updatePeel()

    def _initPetWindow(self, percentile=False):
        """The PET window: 0 to the maximum (default), or 0 to the 99.5th percentile."""
        volume = self.volume()
        if volume is None:
            return
        low, high = _percentiles(volume, 0.0, 99.5)
        top = float(np.nanmax(slicer.util.arrayFromVolume(volume)))
        self.petWindowRange.blockSignals(True)
        self.petWindowRange.minimum = min(0.0, low)
        self.petWindowRange.maximum = max(top, high)
        self.petWindowRange.singleStep = max(high / 200.0, 0.001)
        self.petWindowRange.setValues(0.0, max(high if percentile else top, 1e-6))
        self.petWindowRange.blockSignals(False)
        self._applyPetDisplay()

    def _applyPetDisplay(self):
        volume = self.volume()
        if volume is None or self.current != STEP_REVIEW and self.reviewResult is None:
            return
        display = volume.GetDisplayNode()
        if display is None:
            volume.CreateDefaultDisplayNodes()
            display = volume.GetDisplayNode()
        display.SetAndObserveColorNodeID(colormaps.colorNodeForLabel(self.petColorCombo.currentText).GetID())
        display.SetAutoWindowLevel(False)
        low, high = self.petWindowRange.minimumValue, self.petWindowRange.maximumValue
        display.SetWindowLevelMinMax(low, max(high, low + 1e-6))
        grey = getattr(self, "greyPetNode", None)
        if grey is not None and slicer.mrmlScene.IsNodePresent(grey) and grey.GetDisplayNode() is not None:
            grey.GetDisplayNode().SetWindowLevelMinMax(low, max(high, low + 1e-6))
        fused = self._fusionVolume() is not None
        display.SetApplyThreshold(fused)
        if fused:
            display.SetLowerThreshold(low + 0.02 * (high - low))
            display.SetUpperThreshold(float(self.petWindowRange.maximum) * 10.0 + 1.0)
        self._applyPeelWindows()

    def _greyPet(self):
        """The PET in grey under the score slices: a second volume node sharing the PET's voxels (no copy), with the
        PET window."""
        pet = self.volume()
        if pet is None:
            return None
        scene = slicer.mrmlScene
        node = getattr(self, "greyPetNode", None)
        if node is None or not scene.IsNodePresent(node):
            node = scene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", scene.GenerateUniqueName("Neuro review PET (grey)"))
            node.SetHideFromEditors(True)
            node.CreateDefaultDisplayNodes()
            unsaved(node)       # shares the PET's voxels: not written again with the scene
            self.greyPetNode = node
        if node.GetImageData() is not pet.GetImageData():
            node.SetAndObserveImageData(pet.GetImageData())
        matrix = vtk.vtkMatrix4x4()
        pet.GetIJKToRASMatrix(matrix)
        node.SetIJKToRASMatrix(matrix)
        display = node.GetDisplayNode()
        display.SetAndObserveColorNodeID("vtkMRMLColorTableNodeGrey")
        display.SetAutoWindowLevel(False)
        low, high = self.petWindowRange.minimumValue, self.petWindowRange.maximumValue
        display.SetWindowLevelMinMax(low, max(high, low + 1e-6))
        return node

    def _onFusionChanged(self):
        anatomy = self._fusionVolume()
        if anatomy is not None and getattr(self, "_anatomyWindowFor", None) is not anatomy:
            self._anatomyWindowFor = anatomy
            low, high = _percentiles(anatomy, 1.0, 99.5)
            self.anatomyWindowRange.blockSignals(True)
            self.anatomyWindowRange.minimum = min(low, 0.0)
            self.anatomyWindowRange.maximum = high * 1.5 if high > 0 else high + 1.0
            self.anatomyWindowRange.setValues(low, high)
            self.anatomyWindowRange.blockSignals(False)
            self._applyAnatomyDisplay()
        self._applyPetDisplay()
        self._assignReviewViews()

    def _applyAnatomyDisplay(self):
        anatomy = self._fusionVolume()
        if anatomy is None:
            return
        display = anatomy.GetDisplayNode()
        if display is None:
            anatomy.CreateDefaultDisplayNodes()
            display = anatomy.GetDisplayNode()
        display.SetAndObserveColorNodeID("vtkMRMLColorTableNodeGrey")
        display.SetAutoWindowLevel(False)
        display.SetWindowLevelMinMax(self.anatomyWindowRange.minimumValue, self.anatomyWindowRange.maximumValue)
        self._applyPeelWindows()
