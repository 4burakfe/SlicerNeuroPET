"""
Review: the region table, a click on a region (flash and slice jump).

Part of the Static Neuro PET/SPECT module (StaticNeuroPET.py): the methods of ReviewTableMixin, mixed into
StaticNeuroPETWidget there.
"""

import logging

import numpy as np
import qt
import slicer
import vtk

from NeuroWorkflowLib.base import _ijkToRas
from NeuroPETLib import normaldb, neuroreview as nr

from StaticNeuroPETLib.constants import REVIEW_FLASH_MS


class ReviewTableMixin:
    """The review table methods of StaticNeuroPETWidget."""

    def _updateReviewTable(self, kind):
        result = self.reviewResult
        scene = slicer.mrmlScene
        n = result.get("n")
        reference = {k: [v[0], v[1], v[2] if len(v) > 2 and v[2] is not None else n]
                     for k, v in (result.get("reference") or {}).items()}
        rows = nr.tableRows(result["values"], result["suv"], reference, kind)
        referenceKey = (result.get("header") or {}).get("referenceRegion")
        referenceStructure = (result.get("header") or {}).get("referenceStructure")
        # name: (SUVmean, score) at the mouse; the reference regions without a score (left out of the table too)
        self._hoverValues = {row[0]: (row[1], None if nr.isReferenceRegion(row[0], referenceKey, referenceStructure)
                                      else row[6]) for row in rows}
        self._hoverKind = kind
        rows = [row for row in rows                                                       # the reference left out
                if not nr.isReferenceRegion(row[0], referenceKey, referenceStructure)]
        units = result["header"].get("mapType") if result.get("header") else "SUVR"
        compared = result.get("ratio") and units in (normaldb.MAP_SUVR, normaldb.MAP_RATIO, "SUVR")
        scored = result.get("header") is not None
        # the score (the most important) right after the region's name
        columns = ["Region"] + ([kind] if scored else []) + ["SUVmean"] + ([units or "SUVR"] if compared else [])
        if scored:
            columns += ["Normal (mean ± SD)"]
        table = vtk.vtkTable()
        arrays = []
        for name in columns:
            array = vtk.vtkStringArray()
            array.SetName(name)
            table.AddColumn(array)
            arrays.append(array)

        def text(value, digits=3):
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                return ""
            return f"{value:.{digits}g}" if isinstance(value, float) else str(value)

        def rounded(value, digits=3):     # the number shown, kept a number (numeric sorting)
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                return None
            return float(f"{value:.{digits}g}") if isinstance(value, float) else value

        cellRows = []
        for name, suv, value, mean, sd, _, score in rows:
            cells, numbers = [name], [name]
            if scored:
                cells.append("" if score is None else f"{score:.2f}")
                numbers.append(None if score is None else round(float(score), 2))
            cells += [text(suv)] + ([text(value)] if compared else [])
            numbers += [rounded(suv)] + ([rounded(value)] if compared else [])
            if scored:
                normal = text(mean)
                if normal and text(sd):         # mean and SD with the same decimals (2; none from 100)
                    decimals = 2 if abs(float(mean)) < 100 else 0
                    normal = f"{float(mean):.{decimals}f} ± {float(sd):.{decimals}f}"
                cells.append(normal)
                numbers.append(normal or None)
            for array, cell in zip(arrays, cells):
                array.InsertNextValue(cell)
            cellRows.append(numbers)
        if self.reviewTable is None or not scene.IsNodePresent(self.reviewTable):
            self.reviewTable = scene.AddNewNodeByClass("vtkMRMLTableNode", scene.GenerateUniqueName("Neuro review"))
        self.reviewTable.SetAndObserveTable(table)
        if hasattr(self.reviewTable, "SetUseColumnNameAsColumnHeader"):
            self.reviewTable.SetUseColumnNameAsColumnHeader(True)
        self.reviewTable.SetName(f"{result['volume'].GetName()} regions")
        self._regionTableData = (columns, cellRows)
        self._showTable()

    def _showTable(self):
        """The region table in the layout's table view: a sortable table with the region colors (clicking a region
        shows it alone for a few seconds); Slicer's own table view of the table node when it cannot be embedded."""
        if self.reviewTable is None:
            return
        if self._fillRegionTable():
            return
        selection = slicer.app.applicationLogic().GetSelectionNode()
        selection.SetActiveTableID(self.reviewTable.GetID())
        slicer.app.applicationLogic().PropagateTableSelection()

    @staticmethod
    def _tableViewWidget():
        """The qMRMLTableWidget of the review layout's table view, None when the layout has none."""
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None or not hasattr(layoutManager, "tableWidget"):
            return None
        for index in range(getattr(layoutManager, "tableViewCount", 0)):
            widget = layoutManager.tableWidget(index)
            viewNode = widget.mrmlTableViewNode() if widget is not None else None
            if viewNode is not None and viewNode.GetSingletonTag() == nr.TABLE_VIEW:
                return widget
        return None

    def _regionTableWidget(self):
        """The QTableWidget inside the table view (made there the first time); None when it cannot be embedded."""
        host = self._tableViewWidget()
        if host is None:
            return None
        table = self.regionTable
        try:
            alive = table is not None and table.parent() is host
        except Exception:       # the C++ widget deleted with its view
            alive = False
        views = [host.tableView()] if hasattr(host, "tableView") else slicer.util.findChildren(
            host, className="qMRMLTableView")
        for child in views:
            child.setVisible(False)
        controller = host.tableController() if hasattr(host, "tableController") else None
        if controller is not None:
            controller.setVisible(False)
        if alive:
            table.setVisible(True)
            return table
        hostLayout = host.layout()
        if hostLayout is None:
            return None
        table = qt.QTableWidget(host)
        table.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        table.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.toolTip = ("Click a column header to sort by it; click a region to show it alone on the slices for "
                         f"{REVIEW_FLASH_MS // 1000} s.")
        hostLayout.addWidget(table)
        table.connect("cellClicked(int,int)", self._onRegionClicked)
        self.regionTable = table
        self._regionTableHost = host
        return table

    def _fillRegionTable(self):
        data = getattr(self, "_regionTableData", None)
        table = self._regionTableWidget() if data is not None else None
        if table is None:
            return False
        columns, rows = data
        colors, ids = self._segmentColors()
        sortColumn = table.horizontalHeader().sortIndicatorSection() if table.sortingEnabled else -1
        sortOrder = table.horizontalHeader().sortIndicatorOrder()
        table.setSortingEnabled(False)
        table.clear()
        table.setColumnCount(len(columns) + 1)
        table.setRowCount(len(rows))
        table.setHorizontalHeaderLabels([""] + list(columns))
        for r, cells in enumerate(rows):
            name = cells[0]
            swatch = qt.QTableWidgetItem()
            color = colors.get(name)
            if color is not None:
                swatch.setBackground(qt.QBrush(qt.QColor(*(int(round(255 * c)) for c in color))))
            swatch.setData(qt.Qt.UserRole, ids.get(name, ""))
            table.setItem(r, 0, swatch)
            for c, value in enumerate(cells):
                item = qt.QTableWidgetItem()
                if value is None:
                    item.setText("")
                elif isinstance(value, str):
                    item.setText(value)
                else:
                    item.setData(qt.Qt.DisplayRole, value)      # a number: sorted by size
                if c > 0:
                    item.setTextAlignment(qt.Qt.AlignRight | qt.Qt.AlignVCenter)
                item.setData(qt.Qt.UserRole, ids.get(name, ""))
                table.setItem(r, c + 1, item)
        table.setSortingEnabled(True)
        if sortColumn >= 0:
            table.sortItems(sortColumn, sortOrder)
        table.resizeColumnsToContents()
        table.setColumnWidth(0, 14)
        qt.QTimer.singleShot(100, self._fitTableWidth)
        return True

    def _segmentColors(self):
        """({segment name: (r, g, b)}, {segment name: segment ID}) of the regions."""
        segmentation = self.node("Segmentation")
        colors, ids = {}, {}
        if segmentation is None:
            return colors, ids
        seg = segmentation.GetSegmentation()
        for segmentId in self.logic.segmentIds(segmentation):
            segment = seg.GetSegment(segmentId)
            colors[segment.GetName()] = tuple(segment.GetColor())
            ids[segment.GetName()] = segmentId
        return colors, ids

    def _fitTableWidth(self):
        """The table view as wide as the table (the slices take the rest)."""
        table = self.regionTable
        try:
            if table is None or not table.visible:
                return
            width = sum(table.columnWidth(c) for c in range(table.columnCount))
            width += 2 * table.frameWidth + table.verticalScrollBar().sizeHint.width() + 8
            child = self._regionTableHost
            splitter = child.parent() if child is not None else None
            while splitter is not None and not splitter.inherits("QSplitter"):
                child, splitter = splitter, splitter.parent()
            if splitter is None:
                return
            index = splitter.indexOf(child)
            sizes = list(splitter.sizes())
            if index < 0 or len(sizes) < 2:
                return
            total = sum(sizes)
            width = min(width, int(total * 0.6))
            others = [i for i in range(len(sizes)) if i != index]
            rest = total - width
            spread = sum(sizes[i] for i in others) or 1
            for i in others:
                sizes[i] = int(rest * sizes[i] / spread)
            sizes[index] = width
            splitter.setSizes(sizes)
        except Exception:
            logging.debug("Static Neuro: could not resize the region table", exc_info=True)

    def _onRegionClicked(self, row, column):
        """The region of the row alone on the slices for REVIEW_FLASH_MS, then the regions shown as before."""
        item = self.regionTable.item(row, 0) if self.regionTable is not None else None
        segmentId = item.data(qt.Qt.UserRole) if item is not None else ""
        segmentation = self.node("Segmentation")
        display = segmentation.GetDisplayNode() if segmentation is not None else None
        if not segmentId or display is None:
            return
        self._endFlash()
        ids = self.logic.segmentIds(segmentation)
        saved = {i: bool(display.GetSegmentVisibility(i)) for i in ids}
        token = object()
        self._flash = (token, display, saved)
        wasModifying = display.StartModify()
        for i in ids:
            display.SetSegmentVisibility(i, i == segmentId)
        self._showOutlines(display, True)       # also with the outlines hidden
        display.EndModify(wasModifying)
        qt.QTimer.singleShot(REVIEW_FLASH_MS, lambda: self._endFlash(token))
        self._jumpToRegion(segmentation, segmentId)
        try:        # its outline on the 3D-SSP too
            names = ((self.reviewResult or {}).get("labels") or (None, []))[1]
            name = segmentation.GetSegmentation().GetSegment(segmentId).GetName()
            if name in names:
                self._sspOutline(names.index(name) + 1)
        except Exception:
            logging.debug("Static Neuro: no 3D-SSP outline of the region", exc_info=True)

    def _regionPoint(self, segmentation, segmentId):
        """RAS of a point inside a region: the voxel of the compared regions nearest to its centroid, else the
        segment's center."""
        result = self.reviewResult or {}
        name = segmentation.GetSegmentation().GetSegment(segmentId).GetName()
        if result.get("labels") is not None and result.get("segmentation") is segmentation:
            labels, names = result["labels"]
            if name in names:
                point = nr.insidePoint(labels, names.index(name) + 1)
                if point is not None:
                    ijk = np.array([point[2], point[1], point[0], 1.0])
                    return (_ijkToRas(result["grid"]) @ ijk)[:3]
        if hasattr(segmentation, "GetSegmentCenterRAS"):
            return np.asarray(segmentation.GetSegmentCenterRAS(segmentId), dtype=float)
        return None

    def _jumpToRegion(self, segmentation, segmentId):
        """The review's slice views moved through a point inside the region (zoom and pan kept)."""
        try:
            ras = self._regionPoint(segmentation, segmentId)
            if ras is None:
                return
            for _, widget in self._reviewSliceWidgets():
                widget.mrmlSliceNode().JumpSliceByOffsetting(*(float(v) for v in ras))
        except Exception:
            logging.debug("Static Neuro: could not move the slices to the region", exc_info=True)

    def _endFlash(self, token=None):
        flash = self._flash
        if flash is None or (token is not None and flash[0] is not token):
            return
        self._flash = None
        _, display, saved = flash
        self._sspOutline(None)
        if not slicer.mrmlScene.IsNodePresent(display):
            return
        wasModifying = display.StartModify()
        for segmentId, visible in saved.items():
            display.SetSegmentVisibility(segmentId, visible)
        if self._segmentationDisplay is not None and self._segmentationDisplay[0] is display:
            self._showOutlines(display, bool(self.outlineCheck.checked))
        display.EndModify(wasModifying)
