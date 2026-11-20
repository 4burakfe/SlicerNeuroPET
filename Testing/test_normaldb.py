"""Tests of the Neuro PET normal database helpers without Slicer: python -m pytest Testing."""
import json
import os
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "NeuroPET"))
from NeuroPETLib import normaldb  # noqa: E402

GROUPS = [{"name": "20-39", "min": 20, "max": 39}, {"name": "40-59", "min": 40, "max": 59},
          {"name": "60+", "min": 60, "max": None}]


def header(name="FDG normals", **fields):
    fields.setdefault("nuclide", "F-18")
    fields.setdefault("tracer", "FDG")
    return normaldb.newHeader(name, normaldb.STATIC, normaldb.USER_DEFINED, **fields)


def test_age_groups():
    assert normaldb.validateAgeGroups(GROUPS) == []
    assert normaldb.validateAgeGroups([]) == ["Define at least one age group"]
    errors = normaldb.validateAgeGroups([{"name": "a", "min": 20, "max": 50}, {"name": "b", "min": 50, "max": 60}])
    assert any("overlap" in e for e in errors)
    assert any("twice" in e for e in normaldb.validateAgeGroups([{"name": "a", "min": 1, "max": 2},
                                                                  {"name": "a", "min": 3, "max": 4}]))
    assert any("not valid" in e for e in normaldb.validateAgeGroups([{"name": "a", "min": 50, "max": 20}]))
    assert normaldb.ageGroupOf(39, GROUPS) == "20-39"
    assert normaldb.ageGroupOf(85, GROUPS) == "60+"
    assert normaldb.ageGroupOf(10, GROUPS) is None
    assert normaldb.ageGroupText(GROUPS[2]) == "60+ (60+)"
    assert normaldb.ageSummary([30, 50, None, ""]) == {"n": 2, "mean": 40.0, "sd": 14.1, "min": 30.0, "max": 50.0}


def test_header_validation():
    assert normaldb.validateHeader(header()) == []
    errors = normaldb.validateHeader(header(name=" ", tracer="", nuclide=""))
    assert len(errors) == 3
    assert normaldb.validateHeader(header(ageStratified=True, ageGroups=[])) == ["Define at least one age group"]
    with pytest.raises(KeyError):
        normaldb.newHeader("x", nope=1)


def test_create_list_delete(tmp_path):
    root = str(tmp_path)
    folder = normaldb.createDatabase(root, header(name="FDG / normals"))
    assert os.path.basename(folder) == "FDG _ normals"
    assert os.path.isdir(os.path.join(folder, normaldb.SUBJECTS_DIR))
    assert normaldb.readSubjects(folder) == []
    with pytest.raises(ValueError):
        normaldb.createDatabase(root, header(name="FDG / normals"))
    normaldb.writeSubjects(folder, [{"id": "S1", "age": 44}, {"id": "S2", "age": 61}])
    (tmp_path / "not a db").mkdir()
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / normaldb.HEADER_NAME).write_text("{")
    found, errors = normaldb.listDatabases(root)
    assert [h["name"] for _, h in found] == ["FDG / normals"]
    assert len(errors) == 1 and errors[0][0].endswith("broken")
    assert normaldb.listDatabases(root, normaldb.DYNAMIC)[0] == []
    row = normaldb.summary(*found[0])
    assert row["subjects"] == 2 and row["statistics"] == "Not computed" and row["ageGroups"] == "No"
    assert row["reference"] == "Cerebellum Gray Matter" and row["version"] == 1
    with pytest.raises(ValueError):
        normaldb.deleteDatabase(str(tmp_path / "not a db"), root)
    with pytest.raises(ValueError):
        normaldb.deleteDatabase(folder, str(tmp_path / "broken"))
    normaldb.deleteDatabase(folder, root)
    assert not os.path.exists(folder)


def _maps(tmp_path, names):
    maps = {}
    for name in names:
        entry = {"n": 10}
        for which in ("mean", "sd"):
            path = tmp_path / "in" / f"{normaldb.safeName(name)}_{which}.nii.gz"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(which.encode())
            entry[which] = str(path)
        maps[name] = entry
    return maps


def test_predefined_and_package(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    h = header(name="cnpet", ageStratified=True, ageGroups=GROUPS[:2], mapType=normaldb.MAP_SUV)
    maps = _maps(tmp_path, [normaldb.MASTER, "20-39"])
    with pytest.raises(ValueError, match="40-59"):
        normaldb.createPredefined(str(root), h, maps)
    assert not (root / "cnpet").exists()
    maps = _maps(tmp_path, [normaldb.MASTER, "20-39", "40-59"])
    maps["40-59"]["n"] = 3
    folder = normaldb.createPredefined(str(root), h, maps)
    saved = normaldb.readHeader(folder)
    assert saved["kind"] == normaldb.PREDEFINED and saved["mapType"] == "SUV"
    assert saved["statistics"]["groups"]["age_40-59"] == {"name": "40-59", "n": 3,
                                                          "mean": "stats/age_40-59_mean.nii.gz",
                                                          "sd": "stats/age_40-59_sd.nii.gz"}
    assert (root / "cnpet" / "stats" / "master_sd.nii.gz").read_bytes() == b"sd"
    assert normaldb.subjectCount(folder, saved) == 10
    assert not os.path.exists(os.path.join(folder, normaldb.SUBJECTS_DIR))
    with pytest.raises(ValueError, match="user defined"):
        normaldb.exportPredefined(folder, str(tmp_path / "x.zip"))


def test_export_import(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    folder = normaldb.createDatabase(str(root), header(name="mine"))
    with pytest.raises(ValueError, match="Compute"):
        normaldb.exportPredefined(folder, str(tmp_path / "x.zip"))
    normaldb.writeSubjects(folder, [{"id": "S1", "age": 40, "suvFile": "subjects/S1_SUV.nii.gz"},
                                    {"id": "S2", "age": 60}])
    (root / "mine" / "subjects" / "S1_SUV.nii.gz").write_bytes(b"patient")
    for which in ("mean", "sd"):
        (root / "mine" / "stats" / f"master_{which}.nii.gz").write_bytes(which.encode())
    (root / "mine" / normaldb.REFERENCE_MASK_NAME).write_bytes(b"mask")
    h = normaldb.readHeader(folder)
    h["referenceMask"] = normaldb.REFERENCE_MASK_NAME
    h["statistics"] = {"computed": "2026-11-07T10:00:00", "stale": True, "groups": {
        normaldb.MASTER: {"name": normaldb.MASTER, "n": 2, "mean": "stats/master_mean.nii.gz",
                          "sd": "stats/master_sd.nii.gz"}}}
    normaldb.writeHeader(folder, h)
    package = str(tmp_path / "mine.zip")
    with pytest.raises(ValueError, match="out of date"):       # stale statistics are not exported
        normaldb.exportPredefined(folder, package)
    h["statistics"]["stale"] = False
    normaldb.writeHeader(folder, h)
    errors, warnings, contents = normaldb.exportCheck(folder)
    assert not errors and any("3D-SSP" in w for w in warnings) and any("regional" in w for w in warnings)
    assert any("n = 2" in c for c in contents)
    normaldb.exportPredefined(folder, package)
    with zipfile.ZipFile(package) as archive:
        names = sorted(archive.namelist())
    assert names == ["mine/database.json", "mine/reference_mask.nii.gz", "mine/stats/master_mean.nii.gz",
                     "mine/stats/master_sd.nii.gz"]
    packed = normaldb.readPackage(package)
    assert packed["kind"] == normaldb.PREDEFINED and packed["subjectAges"]["n"] == 2
    with pytest.raises(ValueError, match="already exists"):
        normaldb.importPackage(str(root), package)
    imported = normaldb.importPackage(str(root), package, name="mine copy")
    ih = normaldb.readHeader(imported)
    assert ih["name"] == "mine copy" and ih["kind"] == normaldb.PREDEFINED
    assert normaldb.subjectCount(imported, ih) == 2
    assert open(os.path.join(imported, "stats", "master_mean.nii.gz"), "rb").read() == b"mean"
    # the regional statistics go with the maps; a missing file is dropped from the header
    h["statistics"]["regions"] = "stats/regions.json"
    normaldb.writeHeader(folder, h)
    normaldb.exportPredefined(folder, str(tmp_path / "noregions.zip"))
    assert "regions" not in normaldb.readPackage(str(tmp_path / "noregions.zip"))["statistics"]
    settings = {"regionAtlas": h["regionAtlas"], "growMm": 0.0, "growInto": normaldb.growInto(h),
                "referenceRegion": h["referenceRegion"], "erodeReferenceMm": 0.0, "mapType": h["mapType"]}
    document = {"maps": normaldb.subjectMapKey(h), "settings": settings,
                "groups": {normaldb.MASTER: {"regions": {"Pons": [1.0, 0.1, 2]}}}}
    (root / "mine" / "stats" / "regions.json").write_text(json.dumps(document))
    normaldb.exportPredefined(folder, str(tmp_path / "regions.zip"))
    with zipfile.ZipFile(str(tmp_path / "regions.zip")) as archive:
        assert "mine/stats/regions.json" in archive.namelist()
    # regional statistics of another region atlas: not exported
    document["settings"]["regionAtlas"] = "other"
    (root / "mine" / "stats" / "regions.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="other settings"):
        normaldb.exportPredefined(folder, str(tmp_path / "wrong.zip"))
    document["settings"]["regionAtlas"] = h["regionAtlas"]
    (root / "mine" / "stats" / "regions.json").write_text(json.dumps(document))
    # the 3D-SSP projections: those of the statistics go in, older ones stop the export
    from NeuroPETLib import ssp
    surface = {"points": np.zeros((3, 3)), "normals": np.ones((3, 3)), "triangles": np.array([[0, 1, 2]])}
    groups = {normaldb.MASTER: {"mean": np.zeros(3), "sd": np.ones(3), "n": 2, "name": normaldb.MASTER}}
    meta = {"depthMm": ssp.DEPTH_MM, "stepMm": ssp.STEP_MM, "clipToBrain": True, "surface": ssp.SURFACE_VERSION,
            "fwhmMm": float(h.get("smoothingFwhmMm") or 0), "maps": normaldb.subjectMapKey(h),
            "version": h.get("version", 1)}
    ssp.write(str(root / "mine" / "stats" / "ssp.npz"), surface, groups, meta)
    h["statistics"]["ssp"] = ssp.SSP_FILE
    normaldb.writeHeader(folder, h)
    errors, warnings, contents = normaldb.exportCheck(folder)
    assert not errors and not any("3D-SSP" in w for w in warnings) and any("3D-SSP" in c for c in contents)
    normaldb.exportPredefined(folder, str(tmp_path / "ssp.zip"))
    with zipfile.ZipFile(str(tmp_path / "ssp.zip")) as archive:
        assert "mine/stats/ssp.npz" in archive.namelist()
    # the regions themselves (a segmentation) go in when stored; without them the export warns
    assert any("regions themselves" in w for w in warnings)
    (root / "mine" / "stats" / "regions.seg.nrrd").write_bytes(b"seg")
    h["statistics"].update(regionsSegmentation="stats/regions.seg.nrrd", regionsReference="Pons (reference)")
    normaldb.writeHeader(folder, h)
    errors, warnings, contents = normaldb.exportCheck(folder)
    assert not errors and not warnings and any("segmentation" in c for c in contents)
    normaldb.exportPredefined(folder, str(tmp_path / "seg.zip"))
    with zipfile.ZipFile(str(tmp_path / "seg.zip")) as archive:
        assert "mine/stats/regions.seg.nrrd" in archive.namelist()
    packed = normaldb.readPackage(str(tmp_path / "seg.zip"))
    assert packed["statistics"]["regionsReference"] == "Pons (reference)"
    from NeuroPETLib import staticneuro as sn
    assert sn.storedRegions(folder, h) == (os.path.join(folder, "stats/regions.seg.nrrd"), "Pons (reference)")
    assert sn.storedRegions(folder, dict(h, statistics=dict(h["statistics"], stale=True))) is None
    ssp.write(str(root / "mine" / "stats" / "ssp.npz"), surface, groups, dict(meta, surface=1))
    with pytest.raises(ValueError, match="older surface"):
        normaldb.exportPredefined(folder, str(tmp_path / "old.zip"))
    # a region setting changed: the statistics of a user defined database become out of date
    edited, changes = normaldb.editedHeader(normaldb.readHeader(folder), dict(h, regionGrowMm=2.0))
    assert changes["statistics"] and edited["statistics"]["stale"]


def test_import_refuses_unsafe_paths(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    package = tmp_path / "evil.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("db/database.json", json.dumps(header(name="evil")))
        archive.writestr("db/../../outside.txt", "x")
    with pytest.raises(ValueError, match="unsafe"):
        normaldb.importPackage(str(root), str(package))
    assert not (root / "evil").exists() and not (tmp_path / "outside.txt").exists()
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr("readme.txt", "x")
    with pytest.raises(ValueError, match="not a packaged"):
        normaldb.readPackage(str(bad))


def test_mean_and_sd():
    arrays = [np.full((2, 2), v, dtype=np.float32) for v in (1.0, 2.0, 3.0)]
    mean, sd = normaldb.meanAndSd(arrays)
    assert np.allclose(mean, 2.0) and np.allclose(sd, 1.0) and mean.dtype == np.float32
    mean, sd = normaldb.meanAndSd(arrays[:1])
    assert np.allclose(sd, 0.0)
    with pytest.raises(ValueError):
        normaldb.meanAndSd([np.zeros(2), np.zeros(3)])
    with pytest.raises(ValueError):
        normaldb.meanAndSd([])


def test_modality_nuclides_tracers():
    assert normaldb.nuclides(normaldb.SPECT) == ["Tc-99m", "I-123", "Tl-201"]
    assert "F-18" in normaldb.nuclides() and "Tc-99m" not in normaldb.nuclides()
    assert normaldb.tracers("F-18", normaldb.PET)[:2] == ["F-18", "FDG"]
    assert "PiB" not in normaldb.tracers("F-18") and "PiB" in normaldb.tracers("c-11")
    assert normaldb.tracers("Tc-99m", normaldb.SPECT) == ["Tc-99m", "HMPAO", "ECD", "TRODAT-1"]
    assert normaldb.tracers("Tc-99m", normaldb.PET) == ["Tc-99m"]
    assert normaldb.tracers("Lu-177") == ["Lu-177"]
    assert normaldb.tracers("") == []
    spect = normaldb.newHeader("perfusion", modality=normaldb.SPECT, nuclide="Tc-99m", tracer="HMPAO")
    assert spect["mapType"] == normaldb.MAP_RATIO and normaldb.validateHeader(spect) == []
    spect["mapType"] = normaldb.MAP_SUVR
    assert any("Ratio or Counts" in e for e in normaldb.validateHeader(spect))
    assert any("modality" in e for e in normaldb.validateHeader(header(modality="CT")))
    assert normaldb.newHeader("x")["mapType"] == normaldb.MAP_SUVR


def test_old_header_defaults_to_pet(tmp_path):
    folder = normaldb.createDatabase(str(tmp_path), header(name="old"))
    data = json.load(open(normaldb.headerPath(folder)))
    del data["modality"]
    json.dump(data, open(normaldb.headerPath(folder), "w"))
    loaded = normaldb.readHeader(folder)
    assert loaded["modality"] == normaldb.PET
    assert normaldb.summary(folder, loaded)["modality"] == "PET"


def test_region_settings_in_header():
    h = header(regionAtlas="CerebrA:broad", regionGrowMm=2.0, growInto=normaldb.GROW_ONLY_WM)
    assert normaldb.validateHeader(h) == []
    assert header()["growInto"] == normaldb.GROW_NO_WM
    assert any("growInto" in e for e in normaldb.validateHeader(header(growInto="nope")))
    assert normaldb.growInto({"growIntoWhiteMatter": True}) == normaldb.GROW_WITH_WM
    assert normaldb.growInto({}) == normaldb.GROW_NO_WM
    assert normaldb.regionAtlasTitle("CerebrA:broad") == "CerebrA, broad regions"
    assert "Broad (Compatible)" in normaldb.regionAtlasTitle("CerebrA_PET:compatible")
    assert not any("region atlas" in e for e in normaldb.validateHeader(header(regionAtlas="My atlas:compatible")))
    assert any("region atlas" in e for e in normaldb.validateHeader(header(regionAtlas="CerebrA:regionsMerged")))
    assert any("growth" in e for e in normaldb.validateHeader(header(regionGrowMm=-1)))
    assert any("region atlas" in e for e in normaldb.validateHeader(header(regionAtlas="nope")))
    assert normaldb.validateHeader(header(space=""))
    assert header()["regionAtlas"] == normaldb.DEFAULT_REGION_ATLAS and header()["regionGrowMm"] == 0.0
    assert header()["space"] == "MNI152NLin2009cSym" and "segmentation" not in header()


def test_normalization_settings(tmp_path):
    assert normaldb.normalizationText(None) == "Not specified"
    db = {"method": "pet", "template": "GM", "engine": "iterative", "linear": "Affine", "deformable": False,
          "brainMask": True}
    text = normaldb.normalizationText(db)
    assert "Grey matter" in text and "Iterative" in text and "no deformable" in text and "skull-stripped" in text
    assert normaldb.normalizationDifferences(db, db) == []
    used = dict(db, template="T1", engine="brainsfit", deformable=True)
    diffs = normaldb.normalizationDifferences(db, used)
    assert len(diffs) == 3 and diffs[0].startswith("template T1 (database: Grey matter")
    assert any("deformable stage on" in d for d in diffs)
    assert normaldb.normalizationDifferences({"method": "anatomy", "template": "T1"},
                                             {"method": "anatomy", "template": "T2"}) == []
    assert normaldb.normalizationDifferences({}, used) == []
    assert any("normalization" in e for e in normaldb.validateHeader(header(normalization={"method": "x"})))
    folder = normaldb.createDatabase(str(tmp_path), header(name="old"))
    data = json.load(open(normaldb.headerPath(folder)))
    del data["normalization"]
    del data["growInto"]
    data["growIntoWhiteMatter"] = True          # before the three growth targets
    json.dump(data, open(normaldb.headerPath(folder), "w"))
    old = normaldb.readHeader(folder)
    assert old["normalization"] == normaldb.emptyNormalization()
    assert old["growInto"] == normaldb.GROW_WITH_WM and "growIntoWhiteMatter" not in old


def test_add_subject(tmp_path):
    folder = normaldb.createDatabase(str(tmp_path), header(name="db"))
    assert normaldb.subjectFileNames("sub/1") == ("subjects/sub_1/sub_1_SUV.nii.gz",
                                                    "subjects/sub_1/sub_1_SUVR.nii.gz")
    norm = {"method": "pet", "template": "GM"}
    h = normaldb.addSubject(folder, {"id": "S1", "age": 52, "sex": "M", "referenceMean": "7.1",
                                     "reference": "Cerebellum Gray Matter (reference)"}, norm)
    assert h["normalization"]["template"] == "GM" and h["statistics"]["stale"] is False
    assert normaldb.readSubjects(folder)[0]["reference"] == "Cerebellum Gray Matter (reference)"
    with pytest.raises(ValueError, match="already"):
        normaldb.addSubject(folder, {"id": "s1"})
    assert normaldb.hasSubject(folder, "s1")
    h["statistics"]["computed"] = "2026-11-08T10:00:00"
    normaldb.writeHeader(folder, h)
    h = normaldb.addSubject(folder, {"id": "S1", "age": 53}, {"method": "anatomy"}, replace=True)
    assert h["statistics"]["stale"] is True and h["normalization"]["method"] == "pet"
    # a batch plan's other keys (the template pass) stay out of the database's settings
    other = normaldb.createDatabase(str(tmp_path), header(name="other"))
    h = normaldb.addSubject(other, {"id": "S1"}, {"method": "pet", "template": "pseudoPET", "templatePass": 0})
    assert set(h["normalization"]) == set(normaldb.NORM_KEYS) and h["normalization"]["template"] == "pseudoPET"
    normaldb.addSubject(folder, {"id": "S2", "age": 60})
    rows = normaldb.readSubjects(folder)
    assert [r["id"] for r in rows] == ["S1", "S2"] and rows[0]["age"] == "53" and rows[0]["added"]
    with pytest.raises(ValueError, match="ID"):
        normaldb.addSubject(folder, {"id": " "})
    predefined = normaldb.createPredefined(str(tmp_path), header(name="pre"), _maps(tmp_path, [normaldb.MASTER]))
    with pytest.raises(ValueError, match="user defined"):
        normaldb.addSubject(predefined, {"id": "S1"})


def test_remove_subjects(tmp_path):
    folder = normaldb.createDatabase(str(tmp_path), header(name="db"))
    for subjectId in ("S1", "S2"):
        suv, suvr = normaldb.subjectFileNames(subjectId)
        os.makedirs(os.path.join(folder, normaldb.subjectDir(subjectId)), exist_ok=True)
        for relative in (suv, suvr):
            open(os.path.join(folder, relative), "wb").close()
        normaldb.addSubject(folder, {"id": subjectId, "suvFile": suv, "suvrFile": suvr})
    h = normaldb.readHeader(folder)
    h["statistics"]["computed"] = "2026-11-08T10:00:00"
    normaldb.writeHeader(folder, h)
    assert normaldb.removeSubjects(folder, ["s1", "nope"]) == ["S1"]
    assert [r["id"] for r in normaldb.readSubjects(folder)] == ["S2"]
    assert not os.path.exists(os.path.join(folder, "subjects", "S1"))
    assert os.path.exists(os.path.join(folder, "subjects", "S2", "S2_SUVR.nii.gz"))
    assert normaldb.readHeader(folder)["statistics"]["stale"] is True
    assert normaldb.removeSubjects(folder, ["S1"]) == []


def test_edit_header(tmp_path):
    old = header(name="FDG normals", smoothingFwhmMm=8.0)
    old["statistics"]["computed"] = "2026-11-08T10:00:00"
    old["referenceMask"] = normaldb.REFERENCE_MASK_NAME
    new = dict(old, description="cnpet", normalization=dict(normaldb.emptyNormalization(), template="GM"),
               kind="predefined", referenceMask="x")
    h, changes = normaldb.editedHeader(old, new)
    assert h["description"] == "cnpet" and h["normalization"]["template"] == "GM"
    assert h["kind"] == normaldb.USER_DEFINED and h["referenceMask"] == normaldb.REFERENCE_MASK_NAME
    assert changes["normalization"] and not changes["reference"] and not changes["statistics"]
    assert h["statistics"]["stale"] is False
    h, changes = normaldb.editedHeader(old, dict(old, referenceStructure="Pons", name="FDG 2"))
    assert changes["reference"] and changes["statistics"] and changes["name"] and h["statistics"]["stale"] is True
    with pytest.raises(ValueError):
        normaldb.editedHeader(old, dict(old, regionGrowMm=-1))
    predefined = header(name="pre", smoothingFwhmMm=8.0)
    predefined["kind"] = normaldb.PREDEFINED
    h, changes = normaldb.editedHeader(predefined, dict(predefined, smoothingFwhmMm=4.0, scanner="Biograph"))
    assert h["smoothingFwhmMm"] == 8.0 and h["scanner"] == "Biograph" and changes["keys"] == ["scanner"]
    # its regions: editable without stored regions, fixed with them (the Review uses the stored ones)
    assert "regionAtlas" not in normaldb.fixedKeys(predefined) and normaldb.fixedKeys(old) == ()
    h, changes = normaldb.editedHeader(predefined, dict(predefined, regionGrowMm=2.0))
    assert h["regionGrowMm"] == 2.0
    predefined["statistics"]["regionsSegmentation"] = "stats/regions.seg.nrrd"
    h, changes = normaldb.editedHeader(predefined, dict(predefined, regionGrowMm=2.0, regionAtlas="CerebrA:broad",
                                                        growInto=normaldb.GROW_WITH_WM))
    assert h["regionGrowMm"] == 0.0 and h["regionAtlas"] == normaldb.DEFAULT_REGION_ATLAS and not changes["keys"]
    folder = normaldb.createDatabase(str(tmp_path), header(name="db"))
    moved = normaldb.renameDatabase(folder, str(tmp_path), "db two")
    assert os.path.isdir(moved) and not os.path.exists(folder) and normaldb.isDatabase(moved)
    assert normaldb.renameDatabase(moved, str(tmp_path), "db two") == moved
    other = normaldb.createDatabase(str(tmp_path), header(name="other"))
    with pytest.raises(ValueError, match="already exists"):
        normaldb.renameDatabase(other, str(tmp_path), "db two")


def test_statistics_helpers(tmp_path):
    h = header(ageStratified=True, ageGroups=GROUPS)
    subjects = [{"id": "a", "age": "25"}, {"id": "b", "age": "45"}, {"id": "c", "age": ""}, {"id": "d", "age": "70"}]
    groups = normaldb.statisticsGroups(h, subjects)
    assert groups[normaldb.MASTER]["ids"] == ["a", "b", "c", "d"]
    assert sum(len(g["ids"]) for k, g in groups.items() if k != normaldb.MASTER) == 3
    assert list(normaldb.statisticsGroups(header(), subjects)) == [normaldb.MASTER]
    assert normaldb.subjectMapKey(header()) == "suvrFile"
    assert normaldb.subjectMapKey(header(mapType=normaldb.MAP_SUV)) == "suvFile"
    rng = np.random.default_rng(1)
    maps = [rng.random((4, 5, 6)).astype(np.float32) for _ in range(5)]
    stats = normaldb.RunningStatistics()
    for m in maps:
        stats.add(m)
    assert np.allclose(stats.mean(), np.mean(maps, axis=0), atol=1e-6)
    assert np.allclose(stats.sd(), np.std(maps, axis=0, ddof=1), atol=1e-5)
    with pytest.raises(ValueError):
        stats.add(np.zeros((2, 2, 2)))
    one = normaldb.RunningStatistics()
    one.add(maps[0])
    with pytest.raises(ValueError):
        one.sd()
    sigma = normaldb.gaussianSigmas(8.0, (2.0, 1.0, 1.0))
    assert abs(sigma[1] - 8.0 / 2.3548) < 1e-3 and abs(sigma[0] - sigma[1] / 2) < 1e-6
    point = np.zeros((41, 41, 41), dtype=np.float32)
    point[20, 20, 20] = 1.0
    smooth = normaldb.smoothMap(point, 8.0, (1.0, 1.0, 1.0))
    assert abs(float(smooth.sum()) - 1.0) < 1e-3 and smooth[20, 20, 20] < 0.01
    assert normaldb.smoothMap(point, 0, (1, 1, 1))[20, 20, 20] == 1
    assert normaldb.templatePath(str(tmp_path), header()) is None
    (tmp_path / normaldb.TEMPLATE_NAME).write_bytes(b"t")
    assert normaldb.templatePath(str(tmp_path), header(normalizationTemplate={"file": normaldb.TEMPLATE_NAME}))


def test_freeze_template(tmp_path):
    folder = normaldb.createDatabase(str(tmp_path), header(name="db", normalization={"method": "pet",
                                                                                     "template": "pseudoPET"}))
    with pytest.raises(ValueError, match="mean"):
        normaldb.freezeTemplate(folder)
    h = normaldb.readHeader(folder)
    relative = normaldb.statsFileName(normaldb.MASTER, "mean")
    with open(os.path.join(folder, relative), "wb") as f:
        f.write(b"mean1")
    h["statistics"] = {"computed": "2026-11-08T10:00:00", "stale": False,
                       "groups": {normaldb.MASTER: {"name": "Master", "n": 100, "mean": relative, "sd": ""}}}
    normaldb.writeHeader(folder, h)
    h = normaldb.freezeTemplate(folder)
    assert h["normalization"]["method"] == "petTemplate" and h["normalization"]["template"] == ""
    assert h["normalizationTemplate"]["iteration"] == 1 and h["normalizationTemplate"]["n"] == 100
    path = normaldb.templatePath(folder, h)
    assert open(path, "rb").read() == b"mean1"
    with open(os.path.join(folder, relative), "wb") as f:
        f.write(b"mean2")
    assert open(path, "rb").read() == b"mean1"            # frozen
    assert normaldb.freezeTemplate(folder)["normalizationTemplate"]["iteration"] == 2
    assert open(path, "rb").read() == b"mean2"


def test_migrate_layout(tmp_path):
    folder = normaldb.createDatabase(str(tmp_path), header(name="db"))
    h = normaldb.readHeader(folder)
    h["formatVersion"] = 1
    h.pop("storage", None)
    normaldb.writeHeader(folder, h, touch=False)
    os.makedirs(os.path.join(folder, "subjects"), exist_ok=True)
    os.makedirs(os.path.join(folder, "transforms"), exist_ok=True)
    for name in ("S1_SUV.nii.gz", "S1_SUVR.nii.gz"):
        open(os.path.join(folder, "subjects", name), "wb").close()
    json.dump({"pass": {}}, open(os.path.join(folder, "transforms", "S1.json"), "w"))
    normaldb.writeSubjects(folder, [{"id": "S1", "suvFile": "subjects/S1_SUV.nii.gz",
                                     "suvrFile": "subjects/S1_SUVR.nii.gz"}])
    assert normaldb.migrateLayout(folder) == 3
    row = normaldb.readSubjects(folder)[0]
    assert row["suvFile"] == "subjects/S1/S1_SUV.nii.gz" and row["suvrFile"] == "subjects/S1/S1_SUVR.nii.gz"
    assert os.path.isfile(os.path.join(folder, row["suvFile"]))
    assert os.path.isfile(os.path.join(folder, normaldb.transformName("S1")))
    assert not os.path.exists(os.path.join(folder, "transforms"))
    h = normaldb.readHeader(folder)
    assert h["formatVersion"] == normaldb.FORMAT_VERSION and h["storage"]["interpolation"] == "linear"
    assert normaldb.migrateLayout(folder) == 0
    assert normaldb.nativeCount(folder) == (0, 1)
    native = normaldb.nativeFileName("S1")
    open(os.path.join(folder, native), "wb").close()
    row["nativeFile"] = native
    normaldb.writeSubjects(folder, [row])
    assert normaldb.nativeCount(folder) == (1, 1)
    labels = [label for label, _ in normaldb.pipelineRows(folder, normaldb.readHeader(folder))]
    assert "Native images" in labels and "Mean / SD" in labels


def test_subject_table(tmp_path):
    path = os.path.join(str(tmp_path), "subjects.tsv")
    with open(path, "w") as f:
        f.write("participant_id\tAge\tSex\nsub-01\t54\tM\nsub-02\t61\tF\n")
    table = normaldb.readSubjectTable(path)
    assert table["sub-01"]["age"] == "54" and table["sub-02"]["sex"] == "F"
    match = normaldb.matchSubject("sub-02_ses-SUV_pet.nii", table)
    assert match and match["id"] == "sub-02"
    assert not normaldb.matchSubject("sub-03_pet", table)


def test_normalization_target():
    h = header()
    assert normaldb.normalizationTarget(h) == "pseudoPET"
    h["normalization"] = dict(normaldb.emptyNormalization(), method=normaldb.NORM_PET, template="GM")
    assert normaldb.normalizationTarget(h) == "GM"
    h["normalization"]["method"] = normaldb.NORM_PET_TEMPLATE
    h["normalizationTemplate"] = {"target": "GM"}
    assert normaldb.normalizationTarget(h) == "GM"


def test_subject_id_checks_and_ages(tmp_path):
    root = str(tmp_path)
    folder = normaldb.createDatabase(root, header(name="Ages"))
    normaldb.addSubject(folder, {"id": "A/1", "age": "45Y"})
    # another ID giving the same folder (subjects/A_1), also in another letter case, is refused before writing
    for other in ("A_1", "a_1", "a/1 "):
        with pytest.raises(ValueError):
            normaldb.checkSubjectId(folder, other)
    assert normaldb.checkSubjectId(folder, "a/1", replace=True) == "a/1"
    with pytest.raises(ValueError):
        normaldb.addSubject(folder, {"id": "A_1"})
    normaldb.addSubject(folder, {"id": "B", "age": 50})
    # an age that is not a number counts as unknown in the summary (as in the age groups), not an error
    assert normaldb.ageSummary(s.get("age") for s in normaldb.readSubjects(folder)) == \
        {"n": 1, "mean": 50.0, "sd": 0.0, "min": 50.0, "max": 50.0}
    assert normaldb.subjectAge("45Y") is None and normaldb.subjectAge(" 61 ") == 61.0
    # the subject table is written whole (no temporary file left)
    assert not [n for n in os.listdir(folder) if n.endswith(".tmp")]


def test_header_numbers_and_old_template(tmp_path):
    assert normaldb.headerNumber("8") == 8.0 and normaldb.headerNumber(None) == 0.0
    assert normaldb.headerNumber("eight", 6.0) == 6.0 and normaldb.headerNumber("", 1.0) == 1.0
    folder = normaldb.createDatabase(str(tmp_path), header(name="db"))
    old = tmp_path / "old.nrrd"
    old.write_bytes(b"old")
    h = normaldb.useTemplate(folder, str(old), n=3, iteration=1)
    assert h["normalizationTemplate"]["file"] == "template.nrrd"
    new = tmp_path / "new.nii.gz"
    new.write_bytes(b"new")
    h = normaldb.useTemplate(folder, str(new), n=3, iteration=2)
    assert h["normalizationTemplate"]["file"] == "template.nii.gz"
    assert not os.path.exists(os.path.join(folder, "template.nrrd"))       # the previous one is not left behind
    assert open(normaldb.templatePath(folder, h), "rb").read() == b"new"
    h = normaldb.useTemplate(folder, str(new), n=3, iteration=3)           # same name: kept
    assert os.path.isfile(os.path.join(folder, "template.nii.gz"))


def _nifti(path, shape):
    import gzip
    raw = bytearray(352)
    raw[0:4] = np.int32(348).tobytes()
    raw[40:56] = np.array([3, *shape, 1, 1, 1, 1], dtype="<i2").tobytes()
    with gzip.open(path, "wb") as f:
        f.write(bytes(raw))


def _segmentation(path, names):
    lines = ["NRRD0004", "type: unsigned char", "dimension: 3", "sizes: 4 5 6", "encoding: raw"]
    lines += [f"Segment{i}_Name:={n}" for i, n in enumerate(names)]
    with open(path, "wb") as f:
        f.write(("\n".join(lines) + "\n\n").encode() + b"\0" * 120)


def test_export_checks_what_the_review_needs(tmp_path):
    """The package must hold what the Review of a predefined database reads: maps on one grid, regional statistics
    of every group and of the stored regions, medial 3D-SSP surfaces for regions with sides."""
    from NeuroPETLib import ssp
    root = tmp_path / "root"
    root.mkdir()
    h = header(name="db", ageStratified=True, ageGroups=GROUPS[:1])
    folder = normaldb.createDatabase(str(root), h)
    normaldb.writeSubjects(folder, [{"id": "S1", "age": 30}, {"id": "S2", "age": 31}])
    stats = root / "db" / "stats"
    groups = {}
    for key, name in ((normaldb.MASTER, normaldb.MASTER), ("age_20-39", "20-39")):
        for which in ("mean", "sd"):
            _nifti(stats / f"{key}_{which}.nii.gz", (8, 9, 10))
        groups[key] = {"name": name, "n": 2, "mean": f"stats/{key}_mean.nii.gz", "sd": f"stats/{key}_sd.nii.gz"}
    assert normaldb.imageShape(str(stats / "master_mean.nii.gz")) == (8, 9, 10)
    _nifti(root / "db" / normaldb.REFERENCE_MASK_NAME, (8, 9, 10))
    h = normaldb.readHeader(folder)
    h["referenceMask"] = normaldb.REFERENCE_MASK_NAME
    h["statistics"] = {"computed": "2026-11-16T10:00:00", "stale": False, "groups": groups}
    settings = {"regionAtlas": h["regionAtlas"], "growMm": 0.0, "growInto": normaldb.growInto(h),
                "referenceRegion": h["referenceRegion"], "erodeReferenceMm": 0.0, "mapType": h["mapType"]}
    regions = {"Precuneus R": [1, 0.1, 2], "Precuneus L": [1, 0.1, 2]}
    document = {"maps": normaldb.subjectMapKey(h), "version": h.get("version", 1), "settings": settings,
                "groups": {normaldb.MASTER: {"regions": regions}}}
    (stats / "regions.json").write_text(json.dumps(document))
    _segmentation(stats / "regions.seg.nrrd", ["Precuneus R", "Precuneus L", "Pons (reference)"])
    assert normaldb.segmentNames(str(stats / "regions.seg.nrrd"))[-1] == "Pons (reference)"
    assert normaldb.imageShape(str(stats / "regions.seg.nrrd")) == (4, 5, 6)
    surface = {"points": np.zeros((3, 3)), "normals": np.ones((3, 3)), "triangles": np.array([[0, 1, 2]])}
    meta = {"depthMm": ssp.DEPTH_MM, "stepMm": ssp.STEP_MM, "clipToBrain": True, "surface": ssp.SURFACE_VERSION,
            "fwhmMm": float(h.get("smoothingFwhmMm") or 0), "maps": normaldb.subjectMapKey(h),
            "version": h.get("version", 1)}
    sspGroups = {k: {"mean": np.zeros(3), "sd": np.ones(3), "n": 2, "name": g["name"]} for k, g in groups.items()}
    ssp.write(str(stats / "ssp.npz"), surface, sspGroups, meta)
    h["statistics"].update(regions="stats/regions.json", ssp=ssp.SSP_FILE, regionsSegmentation="stats/regions.seg.nrrd",
                           regionsReference="Pons (reference)")
    normaldb.writeHeader(folder, h)
    errors = normaldb.exportCheck(folder)[0]
    assert any("'20-39'" in e for e in errors)                      # no regional values of the age group
    assert any("medial" in e for e in errors)                       # R / L regions, surface without hemispheres
    document["groups"]["age_20-39"] = {"regions": regions}
    (stats / "regions.json").write_text(json.dumps(document))
    parts = dict(surface, points=np.zeros((9, 3)), normals=np.ones((9, 3)), parts=np.repeat([0, 1, 2], 3))
    ssp.write(str(stats / "ssp.npz"), parts, {k: dict(g, mean=np.zeros(9), sd=np.ones(9)) for k, g in
                                              sspGroups.items()}, meta)
    assert normaldb.exportCheck(folder)[0] == []
    # stored regions other than those of the statistics
    _segmentation(stats / "regions.seg.nrrd", ["Putamen R", "Putamen L"])
    assert any("do not match" in e for e in normaldb.exportCheck(folder)[0])
    _segmentation(stats / "regions.seg.nrrd", ["Precuneus R", "Precuneus L", "Pons (reference)"])
    # a map on another grid
    _nifti(stats / "age_20-39_sd.nii.gz", (8, 9, 11))
    assert any("not on the grid" in e for e in normaldb.exportCheck(folder)[0])
    _nifti(stats / "age_20-39_sd.nii.gz", (8, 9, 10))
    package = str(tmp_path / "db.zip")
    normaldb.exportPredefined(folder, package)
    imported = normaldb.importPackage(str(root), package, name="db copy")
    ih = normaldb.readHeader(imported)
    assert all(os.path.isfile(os.path.join(imported, r)) for r in normaldb.packageFiles(ih))
    assert len(normaldb.packageFiles(ih)) == 8


def test_reference_region_of_the_atlas():
    header = normaldb.newHeader("db", nuclide="F-18", tracer="FDG", referenceRegion=normaldb.REF_ATLAS,
                                referenceStructure="Cerebellum Gray Matter", regionAtlas="CerebrA_PET:regions")
    assert normaldb.validateHeader(header) == []
    assert normaldb.referenceTitle(header) == "Cerebellum Gray Matter"
    assert normaldb.referenceStructures(normaldb.REF_ATLAS, "Pons") == ("Pons",)
    assert normaldb.referenceStructures(normaldb.REF_GLOBAL) is None
    default = normaldb.newHeader("db2", nuclide="F-18", tracer="FDG")
    assert default["referenceRegion"] == normaldb.REF_ATLAS
    assert default["referenceStructure"] == normaldb.DEFAULT_REFERENCE_STRUCTURE
    missing = dict(header, referenceStructure="")
    assert any("reference" in e for e in normaldb.validateHeader(missing))
    # another grouping of the segmentation: the reference mask (a region of it) is made again
    _, changes = normaldb.editedHeader(header, dict(header, regionAtlas="CerebrA_PET:broad"))
    assert changes["reference"]
    _, changes = normaldb.editedHeader(header, dict(header, referenceStructure="Pons"))
    assert changes["reference"] and changes["keys"] == ["referenceStructure"]
    glob = dict(header, referenceRegion=normaldb.REF_GLOBAL, referenceStructure="")
    _, changes = normaldb.editedHeader(glob, dict(glob, regionAtlas="CerebrA_PET:broad"))
    assert not changes["reference"]
    h, changes = normaldb.editedHeader(header, dict(header, space="Other"))
    assert h["space"] == header["space"] and "space" not in changes["keys"]


def test_database_volumes_and_anatomy(tmp_path):
    import numpy as np
    folder = normaldb.createDatabase(str(tmp_path), header(name="ct db"))
    h = normaldb.readHeader(folder)
    assert normaldb.databaseVolumes(folder, h) == {}
    (tmp_path / "ct db" / "stats").mkdir(exist_ok=True)
    for relative in ("stats/master_mean.nii.gz", "stats/anatomy_CT_mean.nii.gz", "template_T1.nii.gz"):
        (tmp_path / "ct db" / relative).write_bytes(b"x")
    h["statistics"] = {"computed": "x", "stale": False, "groups": {normaldb.MASTER: {"mean": "stats/master_mean.nii.gz"}},
                       "anatomy": {"CT": {"mean": "stats/anatomy_CT_mean.nii.gz", "n": 3}}}
    h["normalizationTemplate"] = {"anatomy": {"T1": {"file": "template_T1.nii.gz", "n": 2}}}
    volumes = normaldb.databaseVolumes(folder, h)
    assert set(volumes) == {"db:PET", "db:CT", "db:T1"} and volumes["db:PET"].endswith("master_mean.nii.gz")
    assert normaldb.databaseTemplateKeys(volumes, normaldb.NORM_PET_TEMPLATE) == ["db:PET"]
    assert normaldb.databaseTemplateKeys(volumes, normaldb.NORM_PET) == ["db:T1"]
    assert normaldb.databaseTemplateKeys(volumes, normaldb.NORM_ANATOMY, "CT") == ["db:CT"]
    assert normaldb.databaseTemplateKeys(volumes, normaldb.NORM_ANATOMY) == ["db:CT", "db:T1"]
    assert normaldb.templateTitle("db:CT", "FDG") == "From DB: FDG - CT mean"
    assert normaldb.templateTitle("db:PET") == "From DB: PET template"
    assert normaldb.templateTitle("ctTemplate") == "CT template (CT to CT)"
    assert "From DB: CT mean" in normaldb.normalizationText({"method": "pet", "template": "db:CT"})
    files = normaldb.packageFiles(h)
    assert "stats/anatomy_CT_mean.nii.gz" in files and "template_T1.nii.gz" in files
    # file names, subject columns, counts
    assert normaldb.anatomyFileName("S/1", "CT") == "subjects/S_1/S_1_CT.nii.gz"
    assert normaldb.anatomyNativeFileName("S1") == "subjects/S1/S1_anat_native.nii.gz"
    assert normaldb.anatomyStatsFileName("T1") == "stats/anatomy_T1_mean.nii.gz"
    assert {"anatomy", "anatomyFile", "anatomyNativeFile"} <= set(normaldb.SUBJECT_COLUMNS)
    subject = {"id": "S1", "anatomy": "CT", "anatomyFile": "subjects/S1/S1_CT.nii.gz",
               "anatomyNativeFile": "subjects/S1/S1_anat_native.nii.gz"}
    (tmp_path / "ct db" / "subjects" / "S1").mkdir(parents=True)
    (tmp_path / "ct db" / "subjects" / "S1" / "S1_CT.nii.gz").write_bytes(b"x")
    assert normaldb.anatomyPath(folder, subject) and normaldb.anatomyNativePath(folder, subject) is None
    assert normaldb.anatomyCounts(folder, [subject, {"id": "S2"}]) == {"CT": 1}
    normaldb.writeAnatomyInfo(folder, "S1", {"contrast": "CT"})
    assert normaldb.readAnatomyInfo(folder, "S1")["contrast"] == "CT"
    # scaling: a CT as it is, an MR to its brain median
    ct = np.full((4, 4, 4), -1000.0)
    assert np.array_equal(normaldb.anatomyScale(ct, "CT"), ct.astype(np.float32))
    mr = np.zeros((4, 4, 4))
    mr[1:3, 1:3, 1:3] = 50.0
    mask = mr > 0
    scaled = normaldb.anatomyScale(mr, "T1", mask)
    assert float(np.median(scaled[mask])) == normaldb.ANATOMY_REFERENCE_VALUE
    assert float(np.median(normaldb.anatomyScale(mr, "T1")[mask])) == normaldb.ANATOMY_REFERENCE_VALUE


def test_anatomy_templates_kept_with_the_pet_template(tmp_path):
    folder = normaldb.createDatabase(str(tmp_path), header(name="tdb"))
    source = tmp_path / "pass_1.nii.gz"
    source.write_bytes(b"t")
    ct = tmp_path / "pass_1_CT.nii.gz"
    ct.write_bytes(b"c")
    h = normaldb.useTemplate(folder, str(source), n=5, iteration=1, anatomy={"CT": {"file": str(ct), "n": 4}})
    assert h["normalizationTemplate"]["anatomy"] == {"CT": {"file": "template_CT.nii.gz", "n": 4}}
    assert os.path.isfile(os.path.join(folder, "template_CT.nii.gz"))
    h = normaldb.useTemplate(folder, str(source), n=5, iteration=2)
    assert "anatomy" not in h["normalizationTemplate"]
    assert not os.path.isfile(os.path.join(folder, "template_CT.nii.gz"))       # the old one removed


def test_subject_table_anatomy_column(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("id;age;ct\nA1;50;ct/A1_ct.nii.gz\nA2;60;\n")
    table = normaldb.readSubjectTable(str(path))
    assert table["a1"]["anatomy"] == os.path.normpath(str(tmp_path / "ct" / "A1_ct.nii.gz"))
    assert table["a2"]["anatomy"] == ""


def test_subject_folder_backup_and_restore(tmp_path):
    """A batch add that fails half way: the subject's files as they were (replace), no folder (new subject)."""
    folder = str(tmp_path)
    own = os.path.join(folder, normaldb.subjectDir("s1"))
    os.makedirs(own)
    with open(os.path.join(own, "s1_native.nii.gz"), "w") as f:
        f.write("old")
    state = normaldb.backupSubjectFolder(folder, "s1")
    assert state["existed"] and os.path.isdir(state["backup"])
    with open(os.path.join(own, "s1_native.nii.gz"), "w") as f:
        f.write("new")
    with open(os.path.join(own, "s1_extra.json"), "w") as f:
        f.write("{}")
    normaldb.restoreSubjectFolder(folder, "s1", state)
    assert sorted(os.listdir(own)) == ["s1_native.nii.gz"]
    with open(os.path.join(own, "s1_native.nii.gz")) as f:
        assert f.read() == "old"
    assert not os.path.exists(os.path.join(folder, normaldb.BACKUP_DIR))
    state = normaldb.backupSubjectFolder(folder, "s2")           # a new subject
    assert state == {"existed": False, "backup": None}
    os.makedirs(os.path.join(folder, normaldb.subjectDir("s2")))
    normaldb.restoreSubjectFolder(folder, "s2", state)
    assert not os.path.exists(os.path.join(folder, normaldb.subjectDir("s2")))
    state = normaldb.backupSubjectFolder(folder, "s1")           # added: the copy dropped, the new files kept
    normaldb.dropSubjectBackup(folder, state)
    assert os.path.isdir(own) and not os.path.exists(os.path.join(folder, normaldb.BACKUP_DIR))
