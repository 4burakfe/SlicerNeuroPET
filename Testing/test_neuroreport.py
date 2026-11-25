"""Tests of the PDF report helpers without Slicer: python -m pytest Testing."""
import datetime
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "NeuroPET"))
from NeuroPETLib import neuroreport as rp  # noqa: E402


def test_dicom_dates_and_times():
    assert rp.parseDate("20261124") == datetime.date(2026, 11, 24)
    assert rp.parseDate("2026-11-24") == datetime.date(2026, 11, 24)
    assert rp.parseDate("") is None and rp.parseDate("20261340") is None
    assert rp.parseTime("093015.5") == datetime.time(9, 30, 15, 500000)
    assert rp.parseTime("0930") == datetime.time(9, 30)
    assert rp.parseTime("09:30:15") == datetime.time(9, 30, 15)
    assert rp.parseTime("x") is None
    assert rp.parseDateTime("20261124093015.000000+0300") == datetime.datetime(2026, 11, 24, 9, 30, 15)


def test_study_info_pet():
    info = rp.studyInfo({"Modality": "PT", "SeriesDate": "20261124", "SeriesTime": "101500",
                         "AcquisitionDate": "20261124", "AcquisitionTime": "101530",
                         "RadiopharmaceuticalStartTime": "091000", "RadionuclideTotalDose": "185000000",
                         "ActualFrameDuration": "600000"})
    assert info["start"] == datetime.datetime(2026, 11, 24, 10, 15)       # the series time (decay correction)
    assert info["uptakeMinutes"] == pytest.approx(65.0)
    assert info["durationMinutes"] == pytest.approx(10.0)
    assert info["activityMBq"] == pytest.approx(185.0)


def test_study_info_midnight_and_spect():
    info = rp.studyInfo({"Modality": "PT", "SeriesDate": "20261125", "SeriesTime": "001000",
                         "RadiopharmaceuticalStartTime": "231000"})
    assert info["uptakeMinutes"] == pytest.approx(60.0)       # injected the day before
    spect = rp.studyInfo({"Modality": "NM", "AcquisitionDate": "20261124", "AcquisitionTime": "100000",
                          "ActualFrameDuration": "20000", "NumberOfFrames": "60"})
    assert spect["durationMinutes"] == pytest.approx(20.0)     # 60 projections of 20 s
    empty = rp.studyInfo({})
    assert empty["start"] is None and empty["uptakeMinutes"] is None and empty["activityMBq"] is None


def test_formats():
    assert rp.formatMinutes(None) == rp.NOT_KNOWN
    assert rp.formatMinutes(62.4) == "62 min"
    assert rp.formatMinutes(95) == "1 h 35 min"
    assert rp.formatActivity(185.0) == "185 MBq (5.00 mCi)"
    assert rp.formatPersonName("DOE^JOHN^^^") == "DOE JOHN"
    assert rp.plainText("a<br>b <b>c</b> &amp; d") == "a b c & d"


def test_info_rows():
    rows = rp.infoRows({"name": "DOE JOHN", "age": 67, "sex": "Male"}, {"activityMBq": 200.0}, {"modality": "PET"},
                       {"database": "DB", "smoothing": 8.0})
    values = dict(rows)
    assert values["Patient"] == "DOE JOHN" and values["Age / sex"] == "67 / Male"
    assert values["Injected activity"].startswith("200 MBq") and values["Smoothing"] == "8 mm FWHM"
    assert values["Birth date"] == rp.NOT_KNOWN and values["Scan date"] == rp.NOT_KNOWN
    assert len(rows) % 2 == 0


def test_table_sorted_by_name_and_emphasis():
    rows = [["Region 10 L", -1.0], ["region 2 R", 0.5], ["Amygdala L", -3.1], ["Region 2 L", 2.4]]
    assert [r[0] for r in rp.sortedTableRows(rows)] == ["Amygdala L", "Region 2 L", "region 2 R", "Region 10 L"]
    assert rp.scoreEmphasis(-3.1, -2.0) == -1 and rp.scoreEmphasis(2.4, -2.0) == 1
    assert rp.scoreEmphasis(-1.0, -2.0) == 0 and rp.scoreEmphasis(None, -2.0) == 0


def test_file_name_and_ticks():
    assert rp.reportFileName("DOE JOHN", datetime.date(2026, 11, 24)) == "DOE_JOHN_2026-11-24_NeuroPET.pdf"
    assert rp.reportFileName("", None) == "Patient_NeuroPET.pdf"
    assert rp.reportFileName("A/B:C") == "A_B_C_NeuroPET.pdf"
    assert 0.0 in rp.colorbarTicks(-7.0, 7.0)
    assert rp.colorbarTicks(0.0, 4.0) == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_slices_from_the_image():
    import numpy as np
    matrix = np.array([[2.0, 0, 0, -10], [0, 2.0, 0, -20], [0, 0, 2.0, -30], [0, 0, 0, 1]])
    array = np.zeros((10, 20, 30), np.float32)          # (k, j, i)
    array[2:8, 5:15, 5:25] = 1.0
    array[4, 10, 20] = 5.0                               # i = 20: x = +30 mm (the patient's right)
    low, high = rp.contentBox(array, matrix, marginMm=0.0)
    assert np.allclose(low, [0.0, -10.0, -26.0]) and np.allclose(high, [38.0, 8.0, -16.0])
    # axial as in Slicer: screen right = the patient's left (-x), screen up = anterior
    points = rp.planePoints((low, high), (30.0, 0.0, -22.0), (-1, 0, 0), (0, 1, 0), 2.0)
    assert points.shape == (10, 20, 3)
    assert np.allclose(points[0, 0], [38.0, 8.0, -22.0])      # top left: right, anterior
    values = rp.sampleVolume(array, matrix, points)
    assert values[4, 4] == 5.0                                 # x = 30, y = 0
    assert rp.sampleVolume(array, matrix, np.array([[1000.0, 0, 0]]), cval=-1.0)[0] == -1.0
    assert rp.sampleVolume(array, matrix, np.array([[29.0, 0.0, -22.0]]))[0] == 3.0     # halfway 5 -> 1
    assert rp.sampleVolume(array, matrix, np.array([[29.2, 0.0, -22.0]]), nearest=True)[0] == 5.0
    rgb = rp.colorize(np.array([0.0, 5.0, 10.0]), np.tile(np.linspace(0, 1, 256)[:, None], (1, 3)), 0.0, 10.0)
    assert np.allclose(rgb[:, 0], [0.0, 128 / 255.0, 1.0], atol=0.01)
    rgba = np.array([[1.0, 0, 0, 1.0], [0, 0, 1.0, 0.0]])
    out = rp.overlay(np.full((2, 3), 0.5), np.array([0.0, 1.0]), rgba, 0.0, 1.0)
    assert np.allclose(out, [[1.0, 0, 0], [0.5, 0.5, 0.5]])
