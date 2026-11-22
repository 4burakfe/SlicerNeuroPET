"""Tests of the Dynamic Neuro Compare helpers without Slicer: python -m pytest Testing."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "NeuroPET"))
from NeuroPETLib import dynamicneuro as dn, normaldb  # noqa: E402


def state(**changes):
    """A DynamicPETModeling fitState() of SRTM2 with the automatic k2'."""
    out = {"kind": "maps", "model": "SRTM2", "input": "segment", "timeZero": "first", "injectionMin": 0.0,
           "weighting": "duration", "parameters": None, "tStar": 20.0, "autoTStar": False, "maxError": 10.0,
           "k2ref": 0.0, "window": [40.0, 70.0], "target": "seg1", "reference": "seg9", "mapThreshold": 0.0,
           "inside": False, "clip": True}
    out.update(changes)
    return out


def test_map_settings_of_a_fit_state():
    settings = dn.mapSettings(state(), "Cerebellum Gray Matter (reference)")
    assert settings["model"] == "SRTM2" and settings["k2ref"] is None      # 0: automatic
    assert settings["window"] == {"startMinutes": 40.0, "endMinutes": 70.0}
    assert settings["reference"] == "Cerebellum Gray Matter (reference)"
    assert "injectionMin" not in settings
    assert dn.mapSettings(json.dumps(state(k2ref=0.12)))["k2ref"] == 0.12
    assert dn.mapSettings(None) is None and dn.mapSettings("") is None and dn.mapSettings("not json") is None
    assert dn.mapSettings(state(timeZero="manual", injectionMin=30.0))["injectionMin"] == 30.0
    assert dn.mapSettings(state(window=[50.0, 50.0]))["window"] == {"startMinutes": None, "endMinutes": None}


def test_same_settings_only_what_changes_the_maps():
    a = dn.mapSettings(state(), "Cerebellum Gray Matter (reference)")
    # SRTM2: the window and t* do not change its maps; the reference's mark and case do not count
    assert dn.sameSettings(a, dn.mapSettings(state(window=[10.0, 20.0], tStar=35.0), "cerebellum gray matter"))
    assert not dn.sameSettings(a, dn.mapSettings(state(k2ref=0.1), "Cerebellum Gray Matter"))
    assert not dn.sameSettings(a, dn.mapSettings(state(weighting="uniform"), "Cerebellum Gray Matter"))
    assert not dn.sameSettings(a, dn.mapSettings(state(clip=False), "Cerebellum Gray Matter"))
    assert not dn.sameSettings(a, dn.mapSettings(state(), "Pons"))
    differences = dn.settingsDifferences(a, dn.mapSettings(state(model="SRTM", weighting="uniform"), "Pons"))
    assert differences == ["model: SRTM (expected SRTM2)"]         # the rest means other things then
    # SUVR: the window does; no window = the last 30 min of the series
    suvr = dn.mapSettings(state(model="SUVR", window=[40.0, 70.0]), "Pons")
    assert not dn.sameSettings(suvr, dn.mapSettings(state(model="SUVR", window=[30.0, 70.0]), "Pons"))
    open_window = dict(suvr, window={"startMinutes": None, "endMinutes": None})
    assert dn.sameSettings(suvr, open_window, lastEndMinutes=70.0)
    assert not dn.sameSettings(suvr, open_window, lastEndMinutes=90.0)
    # LoganRef: t*
    logan = dn.mapSettings(state(model="LoganRef", tStar=20.0), "Pons")
    assert not dn.sameSettings(logan, dn.mapSettings(state(model="LoganRef", tStar=25.0), "Pons"))


def test_database_settings_compare_the_same_way():
    database = normaldb.defaultKinetics("SRTM2")
    patient = dn.mapSettings(state(), "Cerebellum Gray Matter")
    assert dn.settingsDifferences(database, patient, skip=("reference",)) == []
    differences = dn.settingsDifferences(database, dn.mapSettings(state(k2ref=0.1, weighting="uniform")),
                                         skip=("reference",))
    assert len(differences) == 2 and any("k2'" in d and "automatic" in d for d in differences)
    assert dn.settingsDifferences(None, patient) == []


def test_maps_acceptable():
    assert dn.mapsAcceptable(state())[0]
    assert not dn.mapsAcceptable(state(mapThreshold=5.0))[0]
    assert not dn.mapsAcceptable(json.dumps(state(inside=True)))[0]
    assert dn.mapsAcceptable(None)[0]


def test_region_fit_values_and_warnings():
    results = {"s1": {"values": {"BPnd": 2.5, "R1": 1.1}, "warnings": []},
               "s2": {"values": {"BPnd": None, "R1": 0.9}, "warnings": ["at a bound: BPnd"]},
               "s3": {"error": "empty segment", "values": {}, "warnings": ["empty segment"]},
               "s4": {"values": {"BPnd": float("nan")}}}
    names = {"s1": "Putamen L", "s2": "Caudate L", "s3": "Pons L", "s4": "Insula L"}
    assert dn.regionFitValues(results, names, "BPnd") == {"Putamen L": 2.5}
    assert dn.regionFitValues(results, names, "R1") == {"Putamen L": 1.1, "Caudate L": 0.9}
    warnings = dn.fitWarnings(results, names, "BPnd")
    assert warnings == ["Caudate L (at a bound: BPnd)", "Pons L (not fitted: empty segment)"]


def test_default_map_and_titles():
    assert dn.defaultMap("SRTM2") == "BPnd"
    assert dn.defaultMap("SUVR") == "SUVR"
    assert dn.defaultMap("LoganRef", ["DVR"]) == "DVR"
    assert dn.defaultMap("unknown") is None
    assert dn.regionValueTitle("BPnd", dn.REGION_TAC_FIT) == "BPnd (TAC fit)"
    assert dn.regionValueTitle("R1", dn.REGION_VOXEL_MEAN) == "R1 mean"
    assert set(dn.COMPARE_MODELS) == set(normaldb.KINETIC_MAPS)


def dynamicHeader(**statistics):
    header = normaldb.newHeader("D", normaldb.DYNAMIC)
    header["statistics"].update(statistics)
    return header


def test_dynamic_subject_files_and_keys():
    header = dynamicHeader()
    assert normaldb.isDynamic(header) and not normaldb.isRatioMap(header)
    assert normaldb.subjectMapKey(header) == "map_BPnd"           # the primary map of SRTM2
    subject = {"id": "s1", "mapFiles": json.dumps({"BPnd": "subjects/s1/s1_BPnd.nii.gz"}), "suvrFile": "x"}
    assert normaldb.subjectMapFile(subject, "map_BPnd") == "subjects/s1/s1_BPnd.nii.gz"
    assert normaldb.subjectMapFile(subject, "map_R1") == "" and normaldb.subjectMapFile(subject, "suvrFile") == "x"
    assert normaldb.subjectMapFiles({"mapFiles": "not json"}) == {}
    assert normaldb.mapFileName("s1", "BPnd") == "subjects/s1/s1_BPnd.nii.gz"
    static = normaldb.newHeader("S")
    assert normaldb.isRatioMap(static) and normaldb.regionValuesKey(static) == "suvrFile"
    assert "fitsFile" in normaldb.SUBJECT_COLUMNS and "mapFiles" in normaldb.SUBJECT_COLUMNS


def test_map_header_is_one_map_of_the_database():
    groups = {"master": {"name": "Master", "n": 3, "mean": "stats/master_BPnd_mean.nii.gz",
                         "sd": "stats/master_BPnd_sd.nii.gz",
                         "maps": {m: {"mean": f"stats/master_{m}_mean.nii.gz", "sd": f"stats/master_{m}_sd.nii.gz"}
                                  for m in ("BPnd", "R1", "k2")}},
              "old": {"name": "Old", "n": 2, "maps": {"BPnd": {"mean": "a", "sd": "b"}}}}
    perMap = {"R1": {"regions": {"tac": "stats/regions_R1_tac.json", "voxel": "stats/regions_R1_voxel.json"},
                     "ssp": "stats/ssp_R1.npz"}}
    header = dynamicHeader(groups=groups, mapStatistics=perMap)
    assert normaldb.statisticsMaps(header) == ["BPnd", "R1", "k2"]
    view = normaldb.mapHeader(header, "R1", normaldb.REGION_VOXEL_MEAN)
    assert view["statistics"]["groups"]["master"]["mean"] == "stats/master_R1_mean.nii.gz"
    assert "old" not in view["statistics"]["groups"]                  # no R1 maps there
    assert view["statistics"]["regions"] == "stats/regions_R1_voxel.json"
    assert view["statistics"]["ssp"] == "stats/ssp_R1.npz"
    assert view["mapType"] == "R1" and not normaldb.isRatioMap(view)
    assert normaldb.subjectMapKey(view) == "map_R1" and normaldb.regionValuesKey(view) == "map_R1_voxel"
    assert header["statistics"]["groups"]["master"]["mean"].endswith("BPnd_mean.nii.gz")     # not changed
    suvr = normaldb.mapHeader(dynamicHeader(), "SUVR")
    assert not normaldb.isRatioMap(suvr)                                # never divided by the reference again


def test_fit_values(tmp_path):
    folder = str(tmp_path)
    os.makedirs(os.path.join(folder, "subjects", "s1"))
    with open(os.path.join(folder, "subjects", "s1", "s1_fits.json"), "w") as stream:
        json.dump({"regions": {"Putamen L": {"BPnd": 2.5, "R1": None}, "Caudate L": {"BPnd": "nan"}}}, stream)
    document = normaldb.readFits(folder, {"fitsFile": "subjects/s1/s1_fits.json"})
    assert normaldb.fitValues(document, "BPnd") == {"Putamen L": 2.5}
    assert normaldb.fitValues(document, "R1") == {}
    assert normaldb.readFits(folder, {"fitsFile": "subjects/s1/none.json"}) is None
