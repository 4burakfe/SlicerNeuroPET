"""Tests of the Dynamic Neuro Receptor PET helpers without Slicer: python -m pytest Testing."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "NeuroPET"))
from NeuroPETLib import neuro  # noqa: E402

CSV = """Mindboggle ID,Label Name,RH Label,LH Labels,Notes,Dice Kappa
2002,Caudal Anterior Cingulate,30,81,,0.79
16,Brainstem,11,62,"Completed filling, removed labelled voxels",0.65
45,Cerebellum Gray Matter,46,97,"Completed filling using threshold for CGM, removed cerebellum labels",0.83
51,Putamen,21,72,Corrected uniformity using threshold,0.87
"""


@pytest.mark.parametrize("args, expected", [
    (dict(modality="CT"), "CT"),
    (dict(modality="MR", texts=["Sag T1 MPRAGE"]), "T1"),
    (dict(modality="MR", texts=["T1W_3D_TFE"]), "T1"),
    (dict(modality="MR", texts=["AX T2 FLAIR"]), "T2"),
    (dict(modality="MR", texts=["Ax T1 FLAIR post"]), "T1"),
    (dict(modality="MR", texts=["t2_tse_tra"]), "T2"),
    (dict(modality="MR", texts=["PD axial"]), "PD"),
    (dict(modality="MR", texts=["3D BRAVO"]), "T1"),
    (dict(modality="MR", texts=["series 5"], repetitionTime="4000", echoTime="100"), "T2"),
    (dict(modality="MR", texts=["series 5"], repetitionTime="500", echoTime="10"), "T1"),
    (dict(modality="MR", texts=["series 5"], repetitionTime="2300", echoTime="2.9", inversionTime="900"), "T1"),
    (dict(modality="MR", texts=["series 5"], repetitionTime="3000", echoTime="15"), "PD"),
    (dict(modality="MR", texts=["series 5"]), None),
    (dict(texts=["patient CT 2.5mm"]), "CT"),
    (dict(texts=["ACT scan"]), None),
])
def test_classify(args, expected):
    contrast, reason = neuro.classifyAnatomy(**args)
    assert contrast == expected, reason
    assert reason


def test_template_for_contrast():
    assert neuro.templateForContrast("T2") == "T2"
    assert neuro.templateForContrast("PD") == "PD"
    for c in ("T1", None):
        assert neuro.templateForContrast(c) == "T1"
    assert neuro.templateForContrast("CT") == neuro.CT_TEMPLATE
    assert neuro.templateTitle(neuro.CT_TEMPLATE) == neuro.CT_TEMPLATE_TITLE


def test_read_labels_text_and_file(tmp_path):
    labels = neuro.readCerebraLabels(CSV)
    names = ["Caudal Anterior Cingulate", "Brainstem", "Cerebellum Gray Matter", "Putamen"]
    assert [s["name"] for s in labels] == names
    assert labels[3]["right"] == 21 and labels[3]["left"] == 72
    path = tmp_path / "labels.csv"
    path.write_text("﻿" + CSV, encoding="utf-8")
    assert neuro.readCerebraLabels(str(path)) == labels
    assert neuro.structureValues(labels, "cerebellum gray matter") == (46, 97)
    assert neuro.structureValues(labels, "nope") == ()
    with pytest.raises(ValueError):
        neuro.readCerebraLabels("a,b\n1,2\n")


def test_real_label_file():
    """The CerebrA file shipped with the atlas: 51 structures, labels 1 - 102 each used once."""
    path = os.environ.get("CEREBRA_CSV")
    if not path or not os.path.isfile(path):
        pytest.skip("set CEREBRA_CSV to CerebrA_LabelDetails.csv")
    labels = neuro.readCerebraLabels(path)
    assert len(labels) == 51
    values = sorted(v for s in labels for v in (s["right"], s["left"]))
    assert values == list(range(1, 103))


def test_label_table_and_remap():
    labels = neuro.readCerebraLabels(CSV)
    entries, mapping = neuro.labelTable(labels)
    names = {v: n for v, n, _ in entries}
    assert names[21] == "Putamen R" and names[72] == "Putamen L" and len(entries) == 8
    assert all(mapping[v] == v for v in names)
    assert [v for v, _, _ in entries] == sorted(names)
    merged, mergedMap = neuro.labelTable(labels, mergeSides=True)
    assert [n for _, n, _ in merged] == [s["name"] for s in labels]
    assert mergedMap[21] == mergedMap[72] == 4
    array = np.array([[0, 21, 72], [46, 97, 5]], dtype=np.float32)
    np.testing.assert_array_equal(neuro.remapLabels(array, mergedMap), [[0, 4, 4], [3, 3, 0]])
    np.testing.assert_array_equal(neuro.remapLabels(array.astype(np.int16), mapping), [[0, 21, 72], [46, 97, 0]])
    for _, _, color in entries:
        assert all(0.0 <= c <= 1.0 for c in color)


def test_cerebra_semicolon_csv_and_pons(tmp_path):
    text = ("Mindboggle ID;Label Name;RH Label;LH Labels;Notes;Dice Kappa\n"
            "16;Brainstem;11;62;Completed filling, removed labelled voxels;0,65\n"
            "444;Pons;103;104;;0\n")
    structures = neuro.readCerebraLabels(text)
    assert [s["name"] for s in structures] == ["Brainstem", "Pons"]
    assert structures[0]["notes"] == "Completed filling, removed labelled voxels"
    assert neuro.structureValues(structures, "pons") == (103, 104)
    assert neuro.segmentNamesOfLabels((103, 104), "Pons") == {"segment_103", "segment_104", "pons", "pons r",
                                                              "pons l"}


def _structures(names):
    return [{"id": "", "name": n, "right": 2 * i + 1, "left": 2 * i + 2, "notes": ""} for i, n in enumerate(names)]


CEREBRA_NAMES = [
    "Caudal Anterior Cingulate",
    "Caudal Middle Frontal",
    "Cuneus",
    "Entorhinal",
    "Fusiform",
    "Inferior Parietal",
    "Inferior temporal",
    "Isthmus Cingulate",
    "Lateral Occipital",
    "Lateral Orbitofrontal",
    "Lingual",
    "Medial Orbitofrontal",
    "Middle Temporal",
    "Parahippocampal",
    "Paracentral",
    "Pars Opercularis",
    "Pars Orbitalis",
    "Pars Triangularis",
    "Pericalcarine",
    "Postcentral",
    "Posterior Cingulate",
    "Precentral",
    "Precuneus",
    "Rostral Anterior Cingulate",
    "Rostral Middle Frontal",
    "Superior Frontal",
    "Superior Parietal",
    "Superior Temporal",
    "Supramarginal",
    "Transverse Temporal",
    "Insula",
    "Brainstem",
    "Third Ventricle",
    "Fourth Ventricle",
    "Optic Chiasm",
    "Lateral Ventricle",
    "Inferior Lateral Ventricle",
    "Cerebellum Gray Matter",
    "Cerebellum White Matter",
    "Thalamus",
    "Caudate",
    "Putamen",
    "Pallidum",
    "Hippocampus",
    "Amygdala",
    "Accumbens Area",
    "Ventral Diencephalon",
    "Basal Forebrain",
    "Vermal lobules I-V",
    "Vermal lobules VI-VII",
    "Vermal lobules VIII-X",
    "Pons",
]


BUNDLED_CEREBRA = os.path.join(neuro.BUNDLED_FOLDER, neuro.SEGMENTATION_FOLDER, neuro.DEFAULT_SEGMENTATION)


def test_default_groups_cover_cerebra():
    groups = neuro.readGroups(os.path.join(BUNDLED_CEREBRA, neuro.GROUPS_CSV_NAME))
    structures = _structures(CEREBRA_NAMES)
    entries, mapping, unknown = neuro.groupedTable(structures, groups)
    assert unknown == ["Brainstem (except pons)"]     # the name in the CerebrA with pons
    grouped = {m.lower() for _, members in groups for m in members}
    left_out = sorted(n for n in CEREBRA_NAMES if n.lower() not in grouped)
    assert left_out == ["Fourth Ventricle", "Inferior Lateral Ventricle", "Lateral Ventricle", "Optic Chiasm",
                        "Third Ventricle"]
    names = [n for _, n, _ in entries]
    assert "Prefrontal R" in names and "Lateral Occipital L" in names and "Cerebellum Gray Matter R" in names
    bilateral, _, _ = neuro.groupedTable(structures, groups, mergeSides=True)
    assert "Cerebellum Gray Matter" in [n for _, n, _ in bilateral]


def test_groups_table_and_grouped_csv():
    text = ("Group;Structures;Notes\nFrontal;Superior Frontal, Rostral Middle Frontal;x\n"
            "Striatum;Caudate, Putamen, Nope;\n")
    groups = neuro.readGroups(text)
    assert groups == [("Frontal", ["Superior Frontal", "Rostral Middle Frontal"]),
                      ("Striatum", ["Caudate", "Putamen", "Nope"])]
    structures = _structures(["Superior Frontal", "Rostral Middle Frontal", "Caudate", "Putamen", "Brainstem"])
    entries, mapping, unknown = neuro.groupedTable(structures, groups)
    assert unknown == ["Nope"]
    assert [(v, n) for v, n, _ in entries] == [(1, "Frontal R"), (2, "Frontal L"), (3, "Striatum R"),
                                              (4, "Striatum L")]
    assert mapping == {1: 1, 3: 1, 2: 2, 4: 2, 5: 3, 7: 3, 6: 4, 8: 4}
    merged, mergedMap, _ = neuro.groupedTable(structures, groups, mergeSides=True)
    assert [n for _, n, _ in merged] == ["Frontal", "Striatum"] and mergedMap[8] == 2
    csvText = neuro.groupedCsvText(entries, groups)
    assert [n for _, n, _ in neuro.readBilateralLabels(csvText)] == ["Frontal R", "Frontal L", "Striatum R",
                                                                      "Striatum L"]
    assert "Superior Frontal, Rostral Middle Frontal" in csvText
    with pytest.raises(ValueError, match="in the groups"):
        neuro.readGroups("Group;Structures\nA;Putamen\nB;Putamen\n")
    with pytest.raises(ValueError, match="columns"):
        neuro.readGroups("Name;Members\nA;Putamen\n")
    assert neuro.readGroups("Group,Structures\nA,Caudate|Putamen\n") == [("A", ["Caudate", "Putamen"])]


def test_compatible_groups():
    """Broad (Compatible): the GE CortexID regions, sides separate but for the cerebellum and the pons."""
    path = os.path.join(BUNDLED_CEREBRA, neuro.COMPATIBLE_GROUPS_CSV_NAME)
    groups = neuro.readGroups(path)
    merged = neuro.readMergedGroups(path)
    assert merged == {"cerebellum whole", "pons"}
    structures = _structures(CEREBRA_NAMES + ["Pons"])
    entries, mapping, unknown = neuro.groupedTable(structures, groups, mergedGroups=merged)
    assert unknown == []
    names = [n for _, n, _ in entries]
    cortexId = ["Prefrontal Lateral", "Prefrontal Medial", "Sensorimotor", "Anterior Cingulate", "Posterior Cingulate",
                "Isthmus Cingulate", "Precuneus", "Parietal Superior", "Parietal Inferior", "Occipital Lateral", "Primary Visual",
                "Temporal Lateral", "Temporal Mesial", "Insula", "Basal Ganglia", "Thalamus"]
    assert names == [f"{g} {s}" for g in cortexId for s in ("R", "L")] + ["Cerebellum Whole", "Pons"]
    values = [v for v, _, _ in entries]
    assert len(set(values)) == len(values)
    byName = {s["name"]: s for s in structures}
    whole = dict((n, v) for v, n, _ in entries)["Cerebellum Whole"]
    for name in ("Cerebellum Gray Matter", "Cerebellum White Matter", "Vermal lobules VI-VII"):
        assert mapping[byName[name]["right"]] == mapping[byName[name]["left"]] == whole
    assert byName["Pallidum"]["right"] in mapping and byName["Brainstem"]["left"] not in mapping
    assert neuro.readMergedGroups(os.path.join(BUNDLED_CEREBRA, neuro.GROUPS_CSV_NAME)) == set()


def test_grow_labels_nearest_region_without_overlap():
    labels = np.zeros((1, 1, 11), dtype=np.int16)
    labels[0, 0, 0] = 1
    labels[0, 0, 10] = 2
    grown = neuro.growLabels(labels, 3.0)
    assert grown[0, 0].tolist() == [1, 1, 1, 1, 0, 0, 0, 2, 2, 2, 2]
    grown = neuro.growLabels(labels, 10.0)
    assert grown[0, 0].tolist() == [1] * 5 + [1 if grown[0, 0, 5] == 1 else 2] + [2] * 5
    assert (grown > 0).all()
    allowed = np.ones_like(labels, dtype=bool)
    allowed[0, 0, 2] = False
    assert neuro.growLabels(labels, 3.0, allowed=allowed)[0, 0, :4].tolist() == [1, 1, 0, 1]
    labels[0, 0, 5] = 9        # a ventricle: neither grows nor is overwritten
    grown = neuro.growLabels(labels, 10.0, fixedValues=[9])
    assert grown[0, 0, 5] == 9 and grown[0, 0, 4] == 1 and grown[0, 0, 6] == 2
    assert neuro.growLabels(labels, 2.0, spacing=(1.0, 1.0, 2.0))[0, 0, :3].tolist() == [1, 1, 0]
    assert (neuro.growLabels(labels, 0) == labels).all()
    assert neuro.fixedValuesOf([(1, "Lateral Ventricle R", None), (2, "Putamen", None),
                                (3, "Optic Chiasm", None)]) == [1, 3]


def test_erode_mask():
    mask = np.zeros((9, 9, 9), dtype=bool)
    mask[1:8, 1:8, 1:8] = True
    assert neuro.erodeMask(mask, 1.0).sum() == 5 ** 3
    assert neuro.erodeMask(mask, 2.0).sum() == 3 ** 3
    assert neuro.erodeMask(mask, 0).sum() == 7 ** 3
    full = np.ones((5, 5, 5), dtype=bool)          # the edge of the array counts as outside
    assert neuro.erodeMask(full, 1.0).sum() == 3 ** 3
    assert not neuro.erodeMask(mask, 10.0).any()


def test_pseudo_pet(tmp_path):
    gm = np.zeros((20, 20, 20), dtype=np.float32)
    wm = np.zeros_like(gm)
    gm[5:8] = 100.0           # stored in percent
    wm[10:14] = 1.0
    image = neuro.pseudoPetArray(gm, wm, (1.0, 1.0, 1.0), fwhmMm=0)
    assert image.max() == 1.0 and np.isclose(image[12, 0, 0], 0.3) and image[0, 0, 0] == 0
    smooth = neuro.pseudoPetArray(gm, wm, (2.0, 2.0, 2.0))
    assert smooth.dtype == np.float32 and np.isclose(smooth.max(), 1.0) and smooth[9, 10, 10] > 0
    with pytest.raises(ValueError):
        neuro.pseudoPetArray(gm, wm[:-1])
    found = {"GM": str(tmp_path / "GM.nrrd"), "WM": str(tmp_path / "WM.nrrd")}
    assert neuro.pseudoPetPath(found) == str(tmp_path / "pseudoPET.nrrd")      # outside the atlas roots: next to GM
    assert neuro.canMakeTemplate(found, neuro.PSEUDO_PET) and not neuro.canMakeTemplate({"GM": "x"}, neuro.PSEUDO_PET)
    assert neuro.pseudoPetPath({}) is None


def _head_ct(shape=(60, 120, 110), spacing=(2.0, 2.0, 2.0)):
    """A ball head: scalp, skull, brain with ventricles, a neck under a foramen magnum, orbits, a table (HU)."""
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    Z, Y, X = (z - 38) * spacing[0], (y - 60) * spacing[1], (x - 55) * spacing[2]
    r = np.sqrt(X ** 2 + (Y / 1.15) ** 2 + (Z / 0.95) ** 2)
    hu = np.full(shape, -1000.0, dtype=np.float32)
    hu[r < 88], hu[r < 82], hu[r < 76], hu[r < 20] = 40, 900, 35, 8
    hu[(np.sqrt(X ** 2 + (Y - 10) ** 2) < 45) & (Z < -40)] = 40
    hu[(np.sqrt(X ** 2 + (Y - 10) ** 2) < 16) & (Z < -60) & (r < 84)] = 35
    for side in (-30, 30):
        hu[np.sqrt((X - side) ** 2 + (Y - 95) ** 2 + (Z + 10) ** 2) < 13] = 20
    hu[(Y < -105) & (np.abs(X) < 90) & (Z < -50)] = 300
    return hu, r < 76


def test_ct_brain():
    pytest.importorskip("scipy")
    hu, brain = _head_ct()
    assert neuro.looksLikeCt(hu) and not neuro.looksLikeCt(np.zeros((4, 4, 4)))
    mask = neuro.ctBrainMask(hu, (2.0, 2.0, 2.0))
    assert 2 * (mask & brain).sum() / (mask.sum() + brain.sum()) > 0.9      # the brain, not the scalp or the skull
    assert (mask & ~brain).sum() < 0.02 * brain.sum()
    image, count = neuro.ctBrainImage(hu, (2.0, 2.0, 2.0))
    assert count == mask.sum() and image.max() <= neuro.CT_BRAIN_WINDOW[1] and not image[~mask].any()
    assert not neuro.ctBrainMask(np.full((20, 20, 20), -1000.0)).any()


def test_ct_template(tmp_path):
    stretched = neuro.ctStretch(np.array([-2000, -1000, -100, 0, 100, 3000]))
    assert list(stretched) == [0, 0, 100, 600, 1100, 1200] and np.all(np.diff(stretched) >= 0)
    found = {neuro.CT_TEMPLATE: str(tmp_path / "CT.nrrd"), neuro.CT_TEMPLATE_GM: str(tmp_path / "CT_GM.nrrd")}
    assert neuro.ctCorrectionPath(found) == str(tmp_path / ("CT" + neuro.CT_CORRECTION_SUFFIX))
    assert neuro.ctCorrectedPath(found).endswith("CT_to_space.nrrd") and neuro.ctCorrectionPath({}) is None


def test_file_stamp_changes_when_the_file_is_written_again(tmp_path):
    path = tmp_path / "mean.nii"
    assert neuro.fileStamp(str(path)) == "" and neuro.fileStamp(None) == ""
    path.write_bytes(b"12345")
    first = neuro.fileStamp(str(path))
    assert first and neuro.fileStamp(str(path)) == first
    path.write_bytes(b"123456")                                   # another size
    second = neuro.fileStamp(str(path))
    assert second != first
    os.utime(str(path), ns=(1, 1))                                # same size, another time
    assert neuro.fileStamp(str(path)) not in ("", first, second)


# --- CerebrA PET: segmentation files read without Slicer, regions grown by their own distance ----------------------
def _writeSegNrrd(path, layers, segments, lps=True):
    import gzip
    layers = np.asarray(layers, dtype=np.uint8)
    count, k, j, i = layers.shape
    head = ["NRRD0004", "type: unsigned char"]
    if count > 1:
        head += ["dimension: 4", f"sizes: {count} {i} {j} {k}", "kinds: list domain domain domain",
                 "space directions: none (-1,0,0) (0,-1,0) (0,0,1)"]
        data = np.moveaxis(layers, 0, 3)
    else:
        head += ["dimension: 3", f"sizes: {i} {j} {k}", "kinds: domain domain domain",
                 "space directions: (-1,0,0) (0,-1,0) (0,0,1)"]
        data = layers[0]
    head += ["space: left-posterior-superior", "encoding: gzip", "space origin: (5,6,-7)"]
    for n, (name, layer, value) in enumerate(segments):
        head += [f"Segment{n}_Name:={name}", f"Segment{n}_Layer:={layer}", f"Segment{n}_LabelValue:={value}",
                 f"Segment{n}_Color:=0.5 0.25 1"]
    path.write_bytes(("\n".join(head) + "\n\n").encode("latin-1") + gzip.compress(data.tobytes()))


def test_readSegmentationFile_and_labels(tmp_path):
    layer0 = np.zeros((3, 4, 5), np.uint8)
    layer0[0, 0, 0] = 1     # Caudate R
    layer0[1, 2, 3] = 2     # Caudate L
    layer0[2, 3, 4] = 7     # a segment not in the table
    layer1 = np.zeros_like(layer0)
    layer1[0, 0, 0] = 1     # Putamen R over the caudate: the caudate keeps it
    layer1[0, 0, 1] = 1
    path = tmp_path / "CerebrA_PET.seg.nrrd"
    _writeSegNrrd(path, [layer0, layer1], [("Caudate R", 0, 1), ("Caudate L", 0, 2), ("Nope R", 0, 7),
                                           ("Putamen R", 1, 1)])
    layers, matrix, segments = neuro.readSegmentationFile(str(path))
    assert layers.shape == (2, 3, 4, 5)
    assert [s["name"] for s in segments] == ["Caudate R", "Caudate L", "Nope R", "Putamen R"]
    assert segments[0]["color"] == (0.5, 0.25, 1.0)
    # LPS directions (-1, -1, 1), origin (5, 6, -7) -> RAS identity, origin (-5, -6, -7)
    assert np.allclose(matrix, [[1, 0, 0, -5], [0, 1, 0, -6], [0, 0, 1, -7], [0, 0, 0, 1]])
    structures = neuro.readCerebraLabels("Mindboggle ID;Label Name;RH Label;LH Labels\n50;Caudate;49;100\n"
                                         "51;Putamen;21;72\n")
    labels, unknown = neuro.segmentationLabels(layers, segments, structures)
    assert unknown == ["Nope R"]
    assert labels[0, 0, 0] == 49 and labels[1, 2, 3] == 100 and labels[0, 0, 1] == 21
    assert int(np.count_nonzero(labels)) == 3
    # one layer (3D file)
    _writeSegNrrd(path, [layer0], [("Segment_49", 0, 1)])
    layers, _, segments = neuro.readSegmentationFile(str(path))
    labels, unknown = neuro.segmentationLabels(layers, segments, structures)
    assert unknown == [] and labels[0, 0, 0] == 49 and int(np.count_nonzero(labels)) == 1


def test_drawn_white_matter_reference(tmp_path):
    """A 'Centrum Semiovale' segment of a segmentation is its reference, not a region of its label map."""
    layer0 = np.zeros((3, 4, 5), np.uint8)
    layer0[0, 0, 0] = 1     # Caudate R
    layer0[2, 1:3, 1:4] = 105
    path = tmp_path / "labels.seg.nrrd"
    _writeSegNrrd(path, [layer0], [("Caudate R", 0, 1), ("Centrum Semiovale", 0, 105)])
    mask, matrix = neuro.drawnWhiteMatterReference({"labels": str(path)})
    assert mask.shape == (3, 4, 5) and int(mask.sum()) == 6 and mask[2, 1, 1] and not mask[0, 0, 0]
    assert np.allclose(matrix[:3, 3], [-5, -6, -7])
    layers, matrix, segments = neuro.readSegmentationFile(str(path))
    structures = neuro.readCerebraLabels("Mindboggle ID;Label Name;RH Label;LH Labels\n50;Caudate;49;100\n")
    labels, unknown = neuro.segmentationLabels(layers, segments, structures)
    assert unknown == ["Centrum Semiovale"] and int(np.count_nonzero(labels)) == 1
    # a region of the table (both sides): split at the midline, RAS x >= 0 right (x = i - 5 here)
    structures = neuro.readCerebraLabels("Mindboggle ID;Label Name;RH Label;LH Labels\n50;Caudate;49;100\n"
                                         ";Centrum Semiovale;105;106\n")
    labels, unknown = neuro.segmentationLabels(layers, segments, structures, matrix)
    assert unknown == [] and labels[0, 0, 0] == 49
    assert labels[2, 1, 1] == 106 and labels[2, 1, 3] == 106 and int(np.count_nonzero(labels == 105)) == 0
    shifted = np.array(matrix, dtype=float)
    shifted[0, 3] = -2.0                                # x = i - 2: i 1 left, i 2 and 3 right
    labels, _ = neuro.segmentationLabels(layers, segments, structures, shifted)
    assert int(np.count_nonzero(labels == 106)) == 2 and int(np.count_nonzero(labels == 105)) == 4
    assert neuro.fixedValuesOf(neuro.labelTable(structures)[0]) == [105, 106]
    _writeSegNrrd(path, [layer0], [("Caudate R", 0, 1)])
    assert neuro.drawnWhiteMatterReference({"labels": str(path)}) is None
    assert neuro.drawnWhiteMatterReference({"labels": str(tmp_path / "labels.nrrd")}) is None
    assert neuro.isWhiteMatterSegment(" centrum semiovale ") and neuro.isWhiteMatterSegment("Centrum Semiovale L")
    assert not neuro.isWhiteMatterSegment("Caudate R")


def test_growLabelsByRegion():
    labels = np.zeros((1, 1, 20), np.int32)
    labels[0, 0, 0] = 1     # grows 3
    labels[0, 0, 19] = 2    # grows 2
    labels[0, 0, 10] = 9    # fixed (a ventricle), not grown
    grown = neuro.growLabelsByRegion(labels, {1: 3, 2: 2, 9: 0})
    assert grown[0, 0].tolist() == [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 2, 2, 2]
    allowed = np.ones_like(labels, bool)
    allowed[0, 0, 2] = False
    assert neuro.growLabelsByRegion(labels, {1: 3}, allowed=allowed)[0, 0, :5].tolist() == [1, 1, 0, 1, 0]
    # two regions reaching a voxel: the nearer one takes it
    labels = np.zeros((1, 1, 7), np.int32)
    labels[0, 0, 0], labels[0, 0, 6] = 1, 2
    assert neuro.growLabelsByRegion(labels, {1: 3, 2: 2})[0, 0].tolist() == [1, 1, 1, 1, 2, 2, 2]


def test_regionValues_and_mergedRegionNames():
    entries = [(49, "Caudate R", None), (100, "Caudate L", None), (103, "Pons", None),
               (11, "Brainstem (except pons) R", None), (62, "Brainstem (except pons) L", None)]
    assert neuro.regionValues(entries, "caudate") == [49, 100]
    assert neuro.regionValues(entries, "Pons") == [103]
    assert neuro.regionValues(entries, "Brainstem (except pons)") == [11, 62]
    assert neuro.regionValues(entries, "Putamen") == []
    assert neuro.mergedRegionNames(entries) == ["Caudate", "Pons", "Brainstem (except pons)"]


def test_reference_only_groups():
    """Cerebellum Whole of groups.csv: a reference region made of the groups GM, WM and Vermis, not a region."""
    path = os.path.join(BUNDLED_CEREBRA, neuro.GROUPS_CSV_NAME)
    assert "Cerebellum Whole" not in [n for n, _ in neuro.readGroups(path)]
    assert neuro.readReferenceGroups(path) == [("Cerebellum Whole", ["Cerebellum Gray Matter",
                                                                     "Cerebellum White Matter", "Vermis"])]
    references = neuro.readReferenceGroups(path)
    structures = _structures(CEREBRA_NAMES)
    merged, mapping, _ = neuro.groupedTable(structures, neuro.readGroups(path), mergeSides=True)
    names = neuro.referenceRegionNames(merged, references)
    assert names[-1] == "Cerebellum Whole" and "Cerebellum Gray Matter" in names
    values = neuro.referenceRegionValues(merged, "cerebellum whole", references)
    byName = {n: v for v, n, _ in merged}
    assert values == sorted(byName[n] for n in ("Cerebellum Gray Matter", "Cerebellum White Matter", "Vermis"))
    assert neuro.referenceRegionValues(merged, "Pons", references) == [byName["Pons"]]
    # a member missing: not listed and no values (never a part taken for the whole)
    partial = [e for e in merged if e[1] != "Vermis"]
    assert "Cerebellum Whole" not in neuro.referenceRegionNames(partial, references)
    assert neuro.referenceRegionValues(partial, "Cerebellum Whole", references) == []
    # the segmentation's own regions: the member groups as their structures
    regions, _ = neuro.labelTable(structures, mergeSides=True)
    info = {"groups": path}
    expanded = neuro.referenceGroups(neuro.regionAtlasKey("CerebrA", neuro.REGIONS, merged=True), info)
    assert expanded == [("Cerebellum Whole", ["Cerebellum Gray Matter", "Cerebellum White Matter",
                                              "Vermal lobules I-V", "Vermal lobules VI-VII", "Vermal lobules VIII-X"])]
    assert "Cerebellum Whole" in neuro.referenceRegionNames(regions, expanded)
    byRegion = {n: v for v, n, _ in regions}
    assert neuro.referenceRegionValues(regions, "Cerebellum Whole", expanded) == sorted(
        byRegion[n] for n in expanded[0][1])
    # a member in a reference row may be in a group too; a table of reference rows only has no groups
    with pytest.raises(ValueError, match="No groups"):
        neuro.readGroups("Group;Structures;Sides\nWhole;A, B;reference\n")


# ---------------------------------------------------------------------------
# Atlas library: spaces and segmentations
# ---------------------------------------------------------------------------
@pytest.fixture
def library(tmp_path, monkeypatch):
    """An extension folder and a user folder of their own, a cache folder."""
    bundled, user, cache = tmp_path / "bundled", tmp_path / "user", tmp_path / "cache"
    for folder in (bundled, user, cache):
        folder.mkdir()
    monkeypatch.setattr(neuro, "BUNDLED_FOLDER", str(bundled))
    monkeypatch.setenv(neuro.USER_FOLDER_VARIABLE, str(user))
    monkeypatch.setenv(neuro.GENERATED_FOLDER_VARIABLE, str(cache))
    space = bundled / "Space" / "MNI"
    space.mkdir(parents=True)
    (space / "space.json").write_text('{"name": "MNI test", "voxelSizeMm": [1, 1, 1], "sources": ["icbm152"]}')
    for name in ("T1.nrrd", "GM.nrrd", "WM.nrrd", "mask.nrrd", "notes.txt"):
        (space / name).write_bytes(b"x")
    segmentation = bundled / "Segmentation" / "Regions"
    segmentation.mkdir(parents=True)
    (segmentation / "segmentation.json").write_text('{"name": "Test regions", "space": "MNI"}')
    (segmentation / "labels.seg.nrrd").write_bytes(b"x")
    (segmentation / "labels.csv").write_text(CSV)
    (segmentation / "groups.csv").write_text("Group;Structures\nStriatum;Putamen\nHindbrain;Brainstem, "
                                              "Cerebellum Gray Matter\n")
    return bundled, user, cache


def test_spaces_of_both_roots(library):
    bundled, user, cache = library
    spaces = neuro.listSpaces()
    assert [(s["id"], s["name"]) for s in spaces] == [("MNI", "MNI test")]
    found = neuro.atlasFiles("MNI")
    assert sorted(found) == ["GM", "T1", "WM", "mask"]
    assert found["T1"] == str(bundled / "Space" / "MNI" / "T1.nrrd")
    # the synthetic PET of the extension's maps: in the cache, under the same path
    assert neuro.pseudoPetPath(found) == str(cache / "Space" / "MNI" / "pseudoPET.nrrd")
    assert neuro.normalizationTemplates(found, "pet") == ["T1", "GM", neuro.PSEUDO_PET]
    assert neuro.normalizationTemplates(found, "anatomy", "T2") == ["T1"]
    assert neuro.normalizationTemplates(found, "anatomy", "CT") == ["T1"]       # no CT template: the T1
    # the user adds T2 and a CT template to the same space (no manifest of its own), and a space of their own
    mine = user / "Space" / "MNI"
    mine.mkdir(parents=True)
    for name in ("T2.nii.gz", "CT.nrrd", "CT_GM.nrrd", "T1.nii"):
        (mine / name).write_bytes(b"u")
    (user / "Space" / "Other").mkdir()
    (user / "Space" / "Other" / "T1.nrrd").write_bytes(b"u")
    (user / "Space" / "Empty").mkdir()
    spaces = neuro.listSpaces()
    assert [s["id"] for s in spaces] == ["MNI", "Other"] and spaces[0]["name"] == "MNI test"
    found = neuro.atlasFiles("MNI")
    assert found["T1"] == str(mine / "T1.nii") and found["T2"] == str(mine / "T2.nii.gz")
    assert found["GM"].startswith(str(bundled))
    assert neuro.normalizationTemplates(found, "anatomy", "CT") == [neuro.CT_TEMPLATE]
    assert neuro.normalizationTemplates(found, "anatomy", "T2") == ["T1", "T2"]
    assert neuro.ctCorrectionPath(found) == str(cache / "user" / "Space" / "MNI" / ("CT" + neuro.CT_CORRECTION_SUFFIX))
    assert neuro.spaceTitle("Other") == "Other" and neuro.atlasFiles("Nope") == {}
    # .nrrd first when both are there
    (mine / "T1.nrrd").write_bytes(b"u")
    assert neuro.atlasFiles("MNI")["T1"] == str(mine / "T1.nrrd")
    assert neuro.ensureUserAtlasFolder() == str(user) and (user / "Segmentation").is_dir()


def test_segmentations_and_region_atlas_keys(library):
    bundled, user, cache = library
    segmentations = neuro.listSegmentations("MNI")
    assert [(s["id"], s["name"], s["space"]) for s in segmentations] == [("Regions", "Test regions", "MNI")]
    info = segmentations[0]
    assert neuro.segmentationGroupings(info) == [neuro.REGIONS, neuro.BROAD]
    assert neuro.listSegmentations("Other") == []
    key = neuro.regionAtlasKey("Regions", neuro.BROAD)
    assert key == "Regions:broad" and neuro.splitRegionAtlas(key) == ("Regions", "broad", False)
    assert neuro.mergedRegionAtlas(key) == "Regions:broadMerged"
    assert neuro.splitRegionAtlas("Regions:broadMerged") == ("Regions", "broad", True)
    assert neuro.splitRegionAtlas("My atlas:regions")[0] == "My atlas"
    assert neuro.isRegionAtlasKey(key) and not neuro.isRegionAtlasKey("lateral")
    assert neuro.regionAtlasTitle(key) == "Test regions, broad regions"
    entries, mapping, groups, _ = neuro.regionTable("Regions:regions")
    assert [n for _, n, _ in entries][:2] == ["Brainstem R", "Putamen R"] and mapping[21] == 21
    entries, mapping, groups, _ = neuro.regionTable(key)
    assert [n for _, n, _ in entries] == ["Striatum R", "Striatum L", "Hindbrain R", "Hindbrain L"]
    assert mapping[11] == mapping[46] == 3 and groups[0] == ("Striatum", ["Putamen"])
    merged = neuro.regionTable(neuro.mergedRegionAtlas(key))[0]
    assert [n for _, n, _ in merged] == ["Striatum", "Hindbrain"]
    with pytest.raises(ValueError, match="no Broad"):
        neuro.regionTable("Regions:compatible")
    with pytest.raises(ValueError, match="not in the atlas"):
        neuro.regionTable("Nope:regions")
    # the label maps are made in the cache (a .seg.nrrd: its label map first)
    folder = cache / "Segmentation" / "Regions"
    assert neuro.segmentationLabelMapPath(info) == str(folder / "labels_map.nrrd")
    assert neuro.regionMapPaths(key) == (str(folder / "broad.nrrd"), str(folder / "broad_LabelDetails.csv"))
    assert neuro.regionMapPaths("Regions:regionsMerged")[0] == str(folder / "regionsMerged.nrrd")
    # a segmentation of the user's with the same id replaces the extension's; a label volume is used as it is
    mine = user / "Segmentation" / "Regions"
    mine.mkdir(parents=True)
    (mine / "labels.nrrd").write_bytes(b"u")
    (mine / "labels.csv").write_text(CSV)
    info = neuro.segmentationInfo("Regions")
    assert info["folder"] == str(mine) and info["name"] == "Regions" and info["groups"] is None
    assert neuro.segmentationLabelMapPath(info) == str(mine / "labels.nrrd")
    assert neuro.regionMapPaths("Regions:regionsMerged")[0] == str(cache / "user" / "Segmentation" / "Regions" /
                                                                   "regionsMerged.nrrd")
    (user / "Segmentation" / "Incomplete").mkdir()
    assert [s["id"] for s in neuro.listSegmentations()] == ["Regions"]


def test_bundled_library():
    """The extension's own space and segmentations."""
    spaces = {s["id"]: s for s in neuro.listSpaces()}
    assert neuro.DEFAULT_SPACE in spaces
    assert {"mask"} <= set(spaces[neuro.DEFAULT_SPACE]["files"])
    segmentations = {s["id"]: s for s in neuro.listSegmentations(neuro.DEFAULT_SPACE)}
    assert {"CerebrA", "CerebrA_PET"} <= set(segmentations)
    for info in segmentations.values():
        assert neuro.segmentationGroupings(info) == [neuro.REGIONS, neuro.BROAD, neuro.COMPATIBLE]
        assert info["labels"].endswith("labels.seg.nrrd")
    names = [n for _, n, _ in neuro.regionTable("CerebrA:regions")[0]]
    assert "Pons R" in names and "Brainstem (except pons) L" in names and "Optic Chiasm R" not in names
    assert neuro.regionAtlasTitle("CerebrA_PET:compatible") == "CerebrA PET, Broad (Compatible)"
    for info in segmentations.values():                 # the centrum semiovale drawn in both
        drawn = neuro.drawnWhiteMatterReference(info)
        assert drawn is not None and 80000 < int(drawn[0].sum()) < 130000
    names = [n for _, n, _ in neuro.regionTable("CerebrA_PET:regions")[0]]
    assert "Centrum Semiovale R" in names and "Centrum Semiovale L" in names
    assert not any("Centrum" in n for _, n, _ in neuro.regionTable("CerebrA_PET:broad")[0])



def test_white_matter_reference():
    """The centrum semiovale: white matter (probability >= 0.9) above the lateral ventricles, in no atlas region."""
    shape = (20, 8, 8)                                  # k (superior), j, i
    wm = np.zeros(shape, dtype=np.float32)
    wm[:, 2:6, 2:6] = 100.0                             # stored 0-100
    wm[:, 2, 2] = 80.0                                  # 0.8: not the core
    labels = np.zeros(shape, dtype=np.int32)
    labels[5:8, 3:5, 3:5] = 41                          # lateral ventricle up to k = 7
    labels[15:, 2:6, 2:4] = 7                           # cortex
    entries = [(41, "Lateral Ventricle R", (0, 0, 0)), (5, "Inferior Lateral Ventricle R", (0, 0, 0)),
               (7, "Precuneus R", (0, 0, 0))]
    assert neuro.lateralVentricleValues(entries) == [41]
    ijkToRas = np.eye(4)
    ijkToRas[2, 3] = -10.0
    mask = neuro.whiteMatterReferenceMask(wm, labels, [41], ijkToRas)
    rows = np.where(mask.any(axis=(1, 2)))[0]
    assert rows.min() == 8 and rows.max() == 19          # above the ventricles only
    assert not mask[:, 2, 2].any() and not (mask & (labels > 0)).any()
    brain = np.ones(shape, dtype=np.uint8)
    brain[12:] = 0
    assert np.where(neuro.whiteMatterReferenceMask(wm, labels, [41], ijkToRas, brain).any(axis=(1, 2)))[0].max() == 11
    flipped = np.diag([1.0, 1.0, -1.0, 1.0])            # superior is k down: the voxels under the ventricles
    rows = np.where(neuro.whiteMatterReferenceMask(wm, labels, [41], flipped).any(axis=(1, 2)))[0]
    assert rows.max() == 4
    with pytest.raises(ValueError):
        neuro.whiteMatterReferenceMask(wm, labels, [99], ijkToRas)
    assert neuro.isWhiteMatterReference(" centrum semiovale (white matter) ")
