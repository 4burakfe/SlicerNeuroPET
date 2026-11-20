"""Tests of the predefined database download helpers (GitHub Releases) without Slicer and without network:
python -m pytest Testing."""
import hashlib
import io
import json
import os
import sys
import urllib.error
import zipfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "NeuroPET"))
from NeuroPETLib import dbdownload, normaldb  # noqa: E402


class _Response(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _opener(data=None, error=None):
    def open_(request, timeout=None):
        if error:
            raise error
        return _Response(data)
    return open_


def _package(path, name="CN FDG PET - CerebrA (Broad) - Pons", description="first"):
    header = normaldb.newHeader(name, normaldb.STATIC, normaldb.PREDEFINED, nuclide="F-18", tracer="FDG")
    header["description"] = description
    base = normaldb.safeName(name)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{base}/{normaldb.HEADER_NAME}", json.dumps(header))
        archive.writestr(f"{base}/stats/master_mean.nii.gz", b"mean")
    return path


def _asset(path, name="CN.FDG.PET.-.CerebrA.Broad.-.Pons.zip", updated="2026-11-16T21:00:00Z", digest=True):
    data = open(path, "rb").read()
    return {"name": name, "title": dbdownload.assetTitle({"name": name}), "size": len(data), "url": "https://x/y",
            "updated": updated, "key": dbdownload.matchKey(name),
            "digest": ("sha256:" + hashlib.sha256(data).hexdigest()) if digest else ""}, data


def test_names():
    assert dbdownload.matchKey("CN FDG PET - CerebrA (Broad) - Pons") == dbdownload.matchKey(
        "CN.FDG.PET.-.CerebrA.Broad.-.Pons.zip") == dbdownload.matchKey("CN FDG PET - CerebrA _Broad_ - Pons")
    assert dbdownload.matchKey("CerebrA - Pons") != dbdownload.matchKey("CerebrA (Broad) - Pons")
    assert dbdownload.assetTitle({"name": "CN.FDG.PET.-.CerebrA.-.Pons.zip"}) == "CN FDG PET - CerebrA - Pons"
    assert dbdownload.assetTitle({"name": "x.zip", "label": "CN FDG PET - CerebrA (Broad) - Pons.zip"}) == \
        "CN FDG PET - CerebrA (Broad) - Pons"
    assert dbdownload.sizeText(250 * 1024 * 1024) == "250.0 MB"


def test_release_assets():
    release = {"assets": [
        {"name": "b.zip", "size": 3, "browser_download_url": "u", "updated_at": "t", "digest": "sha256:aa"},
        {"name": "LICENSE-CNPET.txt", "size": 1},
        {"name": "A.zip", "size": 2}]}
    assets = dbdownload.releaseAssets(release)
    assert [a["name"] for a in assets] == ["A.zip", "b.zip"]
    assert assets[1]["url"] == "u" and assets[1]["digest"] == "sha256:aa"
    assert dbdownload.releaseUrl("o/r", "t") == "https://api.github.com/repos/o/r/releases/tags/t"


def test_fetch_release_errors():
    assert dbdownload.fetchRelease(opener=_opener(b'{"assets": []}')) == {"assets": []}
    for code, text in ((404, "not found"), (403, "too many")):
        error = urllib.error.HTTPError("u", code, "x", {}, None)
        with pytest.raises(RuntimeError, match=text):
            dbdownload.fetchRelease(opener=_opener(error=error))
    with pytest.raises(RuntimeError, match="internet"):
        dbdownload.fetchRelease(opener=_opener(error=urllib.error.URLError("offline")))


def test_fetch_releases():
    def release(tag, published, draft=False):
        return {"tag_name": tag, "name": "", "published_at": published, "draft": draft, "body": "notes",
                "html_url": "page/" + tag, "assets": [{"name": "a.zip", "size": 1}]}
    batch = [release("v1.0", "2026-11-01"), release("static-fdg-normal-databases-v1", "2026-11-16"),
             release("static-amyloid-v1", "2026-12-01"), release("static-draft", "2026-12-02", draft=True),
             release("dynamic-bpnd-v1", "2026-12-03")]
    found = dbdownload.fetchReleases("static", opener=_opener(json.dumps(batch).encode()))
    assert [r["tag"] for r in found] == ["static-amyloid-v1", "static-fdg-normal-databases-v1"]
    assert found[1]["name"] == "static-fdg-normal-databases-v1" and found[1]["body"] == "notes"
    assert found[1]["assets"][0]["name"] == "a.zip" and found[1]["page"] == "page/static-fdg-normal-databases-v1"
    dynamic = dbdownload.fetchReleases("dynamic", opener=_opener(json.dumps(batch).encode()))
    assert [r["tag"] for r in dynamic] == ["dynamic-bpnd-v1"]
    with pytest.raises(RuntimeError, match="repository"):
        dbdownload.fetchReleases(opener=_opener(error=urllib.error.HTTPError("u", 404, "x", {}, None)))


def test_download_install_update(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    package = _package(str(tmp_path / "p.zip"))
    asset, data = _asset(package)
    index = dbdownload.installedIndex(normaldb.listDatabases(str(root))[0])
    assert dbdownload.status(asset, index) == (dbdownload.NEW, None)

    target = str(tmp_path / "d.zip")
    seen = []
    dbdownload.download(asset, target, lambda done, total: seen.append((done, total)), opener=_opener(data))
    assert open(target, "rb").read() == data and seen[-1] == (len(data), len(data))
    folder = normaldb.importPackage(str(root), target, origin=dbdownload.sourceInfo(asset, "static-x-v1"))
    header = normaldb.readHeader(folder)
    assert header["source"]["asset"] == asset["name"] and header["kind"] == normaldb.PREDEFINED
    index = dbdownload.installedIndex(normaldb.listDatabases(str(root))[0])
    assert dbdownload.status(asset, index) == (dbdownload.INSTALLED, folder)

    # uploaded again with a new description: an update, installed in place of the old one
    package2 = _package(str(tmp_path / "p2.zip"), description="second")
    asset2, data2 = _asset(package2, updated="2026-11-17T09:00:00Z")
    assert dbdownload.status(asset2, index) == (dbdownload.UPDATE, folder)
    dbdownload.download(asset2, target, opener=_opener(data2))
    newFolder = normaldb.replacePredefined(str(root), target, folder, dbdownload.sourceInfo(asset2, "static-x-v2"))
    assert newFolder == folder and normaldb.readHeader(folder)["description"] == "second"
    assert not os.path.exists(folder + ".old")
    # without digests the upload time decides
    a, b = dict(asset2, digest=""), {"source": {"asset": asset2["name"], "updated": "2026-11-17T09:00:00Z"}}
    assert dbdownload.status(a, {a["key"]: (folder, dict(b, kind="predefined"))})[0] == dbdownload.INSTALLED


def test_installed_by_hand_and_conflict(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    asset, _ = _asset(_package(str(tmp_path / "p.zip")))
    # imported by hand (Add predefined database): found by its name, no source -> installed, no update offered
    folder = normaldb.importPackage(str(root), str(tmp_path / "p.zip"))
    index = dbdownload.installedIndex(normaldb.listDatabases(str(root))[0])
    assert dbdownload.status(asset, index) == (dbdownload.INSTALLED, folder)
    # a user defined database of the same name is never replaced
    h = normaldb.readHeader(folder)
    index = {asset["key"]: (folder, dict(h, kind=normaldb.USER_DEFINED))}
    assert dbdownload.status(asset, index)[0] == dbdownload.CONFLICT
    user = normaldb.createDatabase(str(root), normaldb.newHeader("mine", normaldb.STATIC, normaldb.USER_DEFINED,
                                                                 nuclide="F-18", tracer="FDG"))
    with pytest.raises(ValueError, match="not a predefined"):
        normaldb.replacePredefined(str(root), str(tmp_path / "p.zip"), user)


def test_replace_keeps_old_on_error(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    folder = normaldb.importPackage(str(root), _package(str(tmp_path / "p.zip")))
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr("nothing.txt", "x")
    with pytest.raises(ValueError):
        normaldb.replacePredefined(str(root), str(bad), folder)
    assert normaldb.readHeader(folder)["description"] == "first"
    assert not os.path.exists(folder + ".old")


def test_download_checks(tmp_path):
    asset, data = _asset(_package(str(tmp_path / "p.zip")))
    target = str(tmp_path / "d.zip")
    with pytest.raises(RuntimeError, match="checksum"):
        dbdownload.download(dict(asset, size=0), target, opener=_opener(data[:-1] + b"x"))
    with pytest.raises(RuntimeError, match="stopped"):
        dbdownload.download(asset, target, opener=_opener(data[:-5]))
    with pytest.raises(dbdownload.Cancelled):
        dbdownload.download(asset, target, lambda d, t: False, opener=_opener(data))
    assert not os.path.exists(target) and not os.path.exists(target + ".part")


def test_default_root(tmp_path):
    assert normaldb.defaultRootFolder(str(tmp_path)) == os.path.join(str(tmp_path), "NeuroPET", "Databases")
