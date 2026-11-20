"""Predefined normal databases published on GitHub Releases: the releases of databases (their tag tells the kind:
static-... for static databases, dynamic-... later for dynamic ones), the packages (.zip) of a release, which of
them are already in the database folder (or have a newer version), and the download itself. Standard library only
(no Slicer, no Qt), so it is tested without Slicer; the Neuro PET Database Manager shows the list and installs the
picked packages with normaldb.importPackage / normaldb.replacePredefined.

A database downloaded here remembers where it came from in its header (database.json "source": repository, release,
asset, updated, digest), so a package uploaded again to the release later (e.g. with a new description) shows as an
update.
"""

import datetime
import hashlib
import json
import os
import re
import urllib.error
import urllib.request

REPOSITORY = "4burakfe/SlicerNeuroPET"
# releases of databases are found by the start of their tag, by database mode (normaldb.STATIC / DYNAMIC); the other
# releases of the repository (the extension itself) are left out
TAG_PREFIXES = {"static": "static-", "dynamic": "dynamic-"}
RELEASE_TAG = "static-fdg-normal-databases-v1"      # the first one, for the tests and the messages
API_ROOT = "https://api.github.com"
USER_AGENT = "NeuroPET-3D-Slicer-extension"
TIMEOUT_S = 60
CHUNK = 1 << 20

NEW = "new"
INSTALLED = "installed"
UPDATE = "update"
CONFLICT = "conflict"       # a user defined database of the same name is in the folder: never overwritten
STATUS_TEXT = {NEW: "Not installed", INSTALLED: "Installed", UPDATE: "Update available",
               CONFLICT: "Name used by a user defined database"}


def releaseUrl(repository=REPOSITORY, tag=RELEASE_TAG):
    """GitHub API address of a release (by tag)."""
    return f"{API_ROOT}/repos/{repository}/releases/tags/{tag}"


def releasePage(repository=REPOSITORY, tag=RELEASE_TAG):
    """The release's web page (for the user); the list of all releases without a tag."""
    if not tag:
        return f"https://github.com/{repository}/releases"
    return f"https://github.com/{repository}/releases/tag/{tag}"


def releasesUrl(repository=REPOSITORY, page=1):
    return f"{API_ROOT}/repos/{repository}/releases?per_page=100&page={page}"


def _request(url, accept="application/vnd.github+json"):
    return urllib.request.Request(url, headers={"Accept": accept, "User-Agent": USER_AGENT})


def fetchRelease(repository=REPOSITORY, tag=RELEASE_TAG, timeout=TIMEOUT_S, opener=None):
    """The release document of the GitHub API. Raises RuntimeError with a readable message."""
    return _fetch(releaseUrl(repository, tag), repository, tag, timeout, opener)


def fetchReleases(mode="static", repository=REPOSITORY, timeout=TIMEOUT_S, opener=None, pages=5):
    """The published releases of databases of a mode (tag starting with TAG_PREFIXES[mode]), newest first, as
    releaseInfo dicts. Raises RuntimeError with a readable message."""
    prefix = TAG_PREFIXES[mode]
    found = []
    for page in range(1, pages + 1):
        batch = _fetch(releasesUrl(repository, page), repository, "", timeout, opener)
        if not isinstance(batch, list):
            raise RuntimeError("GitHub answered with something else than a list of releases")
        for release in batch:
            tag = str(release.get("tag_name") or "")
            if tag.lower().startswith(prefix) and not release.get("draft"):
                found.append(releaseInfo(release))
        if len(batch) < 100:
            break
    found.sort(key=lambda r: r["published"], reverse=True)
    return found


def releaseInfo(release):
    """What the Database Manager shows of a release: {tag, name, published, prerelease, body (Markdown), page,
    assets (releaseAssets)}."""
    tag = str(release.get("tag_name") or "")
    return {"tag": tag, "name": str(release.get("name") or "").strip() or tag,
            "published": str(release.get("published_at") or release.get("created_at") or ""),
            "prerelease": bool(release.get("prerelease")), "body": str(release.get("body") or ""),
            "page": release.get("html_url") or releasePage(tag=tag), "assets": releaseAssets(release)}


def _fetch(url, repository, tag, timeout, opener):
    try:
        with (opener or urllib.request.urlopen)(_request(url), timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            what = f"The release '{tag}' of {repository}" if tag else f"The repository {repository}"
            raise RuntimeError(f"{what} was not found ({releasePage(repository, tag)})")
        if e.code in (403, 429):
            raise RuntimeError("GitHub refused the request (too many requests from this network for now). "
                               "Try again later, or download the .zip files from "
                               f"{releasePage(repository, tag)} and use Add predefined database.")
        raise RuntimeError(f"GitHub answered {e.code} for {url}")
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(f"Could not reach GitHub ({getattr(e, 'reason', e)}). Check the internet connection, or "
                           f"download the .zip files from {releasePage(repository, tag)} and use Add predefined "
                           "database.")


def matchKey(text):
    """A name reduced to its letters and digits, lower case, without .zip: GitHub renames uploaded files (spaces to
    dots, brackets dropped), the Database Manager makes folder names with '_' for brackets; the key is the same for
    all of them ('CN FDG PET - CerebrA (Broad) - Pons', 'CN.FDG.PET.-.CerebrA.Broad.-.Pons.zip')."""
    text = str(text or "")
    if text.lower().endswith(".zip"):
        text = text[:-4]
    return re.sub(r"[^0-9a-z]", "", text.lower())


def assetTitle(asset):
    """The name shown for a package: its label when the uploader gave one, else the file name made readable."""
    label = str(asset.get("label") or "").strip()
    if label:
        return label[:-4] if label.lower().endswith(".zip") else label
    name = str(asset.get("name") or "")
    if name.lower().endswith(".zip"):
        name = name[:-4]
    if " " not in name:
        name = name.replace(".", " ")
    return re.sub(r"\s+", " ", name).strip()


def releaseAssets(release):
    """The database packages (.zip) of a release document: [{name, title, size, url, updated, digest, key}],
    sorted by title."""
    assets = []
    for asset in (release or {}).get("assets") or []:
        name = str(asset.get("name") or "")
        if not name.lower().endswith(".zip"):
            continue
        assets.append({"name": name, "title": assetTitle(asset), "size": int(asset.get("size") or 0),
                       "url": asset.get("browser_download_url") or "", "updated": asset.get("updated_at") or "",
                       "digest": asset.get("digest") or "", "key": matchKey(asset.get("label") or name)})
    assets.sort(key=lambda a: a["title"].lower())
    return assets


def installedIndex(databases):
    """{key: (folder, header)} of the databases of a folder ([(folder, header)] of normaldb.listDatabases), under the
    keys of the asset it was downloaded from, its folder name and its name."""
    index = {}
    for folder, header in databases:
        source = header.get("source") or {}
        keys = [matchKey(source.get("asset")), matchKey(os.path.basename(folder)), matchKey(header.get("name"))]
        for key in keys:
            if key:
                index.setdefault(key, (folder, header))
    return index


def _sourceKey(asset):
    return [matchKey(asset.get("name")), asset.get("key")]


def status(asset, index, predefinedKind="predefined"):
    """(status, folder) of a package against the installed databases (installedIndex)."""
    found = None
    for key in _sourceKey(asset):
        if key and key in index:
            found = index[key]
            break
    if not found:
        return NEW, None
    folder, header = found
    if header.get("kind") != predefinedKind:
        return CONFLICT, folder
    source = header.get("source") or {}
    if source.get("asset") and matchKey(source.get("asset")) == matchKey(asset.get("name")):
        if asset.get("digest") and source.get("digest"):
            newer = asset["digest"] != source["digest"]
        else:
            newer = bool(asset.get("updated")) and str(asset["updated"]) > str(source.get("updated") or "")
        if newer:
            return UPDATE, folder
    return INSTALLED, folder


def sourceInfo(asset, tag, repository=REPOSITORY):
    """What a downloaded database keeps in its header ("source")."""
    return {"repository": repository, "release": tag, "asset": asset.get("name"), "url": asset.get("url"),
            "updated": asset.get("updated"), "digest": asset.get("digest"),
            "downloaded": datetime.datetime.now().isoformat(timespec="seconds")}


def checkDigest(path, digest):
    """True when the file fits the 'sha256:<hex>' digest GitHub gives for the asset (or no digest is given)."""
    if not digest:
        return True
    algorithm, _, expected = str(digest).partition(":")
    if algorithm.lower() != "sha256" or not expected:
        return True
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest().lower() == expected.lower()


class Cancelled(Exception):
    pass


def download(asset, path, progress=None, timeout=TIMEOUT_S, opener=None):
    """Download a package to path (through path + '.part'), checking its size and digest. progress(done, total)
    may return False to cancel (Cancelled). Returns path."""
    part = path + ".part"
    total = int(asset.get("size") or 0)
    done = 0
    try:
        with (opener or urllib.request.urlopen)(_request(asset["url"], "application/octet-stream"),
                                                timeout=timeout) as response, open(part, "wb") as out:
            total = total or int(response.headers.get("Content-Length") or 0)
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                out.write(block)
                done += len(block)
                if progress and progress(done, total) is False:
                    raise Cancelled()
        if total and done != total:
            raise RuntimeError(f"{asset.get('name')}: the download stopped at {done} of {total} bytes")
        if not checkDigest(part, asset.get("digest")):
            raise RuntimeError(f"{asset.get('name')}: the downloaded file does not match its checksum")
        os.replace(part, path)
    except (urllib.error.URLError, OSError) as e:
        _remove(part)
        raise RuntimeError(f"Could not download {asset.get('name')}: {getattr(e, 'reason', e)}")
    except BaseException:
        _remove(part)
        raise
    return path


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


def sizeText(size):
    size = float(size or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit in ("B", "KB") else f"{size:.1f} {unit}"
        size /= 1024
