"""Pure helpers of the Neuro workflows (numpy only, no Slicer imports): the atlas library (template spaces and region
segmentations), the MR contrast (T1 / T2 / PD) of the patient's anatomy, the template that fits it, and the region
label tables (names, left / right, colours, broad groups).

Atlas library: two root folders, the user's (userAtlasFolder(): ~/NeuroPET/Atlas) and the extension's own
(Resources/Atlas, BUNDLED_FOLDER), each with

    Space/<id>/             a template space: space.json (name, voxel size, dimensions, description, url, references,
                            license) and its volumes under fixed names (SPACE_VOLUMES; .nrrd, or .nii.gz / .nii):
                            T1, T2, PD (MR templates), GM, WM, CSF (tissue probability maps), mask (brain mask),
                            CT (a CT template in this space) and CT_GM (the grey matter map of the CT template, when
                            its space is not quite this one: puts the CT template onto it)
    Segmentation/<id>/      a region atlas on the grid of a space: segmentation.json (name, space, voxel size, ...),
                            labels.seg.nrrd (segments named after the structures, 'Putamen R') or labels.nrrd (a
                            label volume), labels.csv (CerebrA_LabelDetails format: Mindboggle ID;Label Name;RH Label;LH
                            Labels;Notes;Dice Kappa), and optionally groups.csv / groups_compatible.csv (broad regions)

The volumes are NRRD files (.nrrd, the standard of the extension; .nii.gz / .nii are read too). A space in both roots
is read volume by volume (the user's file first, e.g. the user adds T2 and CT to the extension's MNI space); a
segmentation in both roots is the user's whole. What is made from the files (the synthetic PET, the label maps of the
region groupings) is saved in generatedFolder() under the same Space/<id> or Segmentation/<id> path (user/... for the
user's files), not in the atlas folders.

A region atlas is a segmentation and a grouping, as one key 'CerebrA:regions' (regionAtlasKey): the segmentation's
regions, its broad regions (groups.csv) or the CortexID-like ones (groups_compatible.csv), sides separate; the sides
merged ('CerebrA:regionsMerged') for the reference regions.
"""

import colorsys
import csv
import io
import json
import os
import re

import numpy as np

# ---------------------------------------------------------------------------
# Atlas library: spaces
# ---------------------------------------------------------------------------
SPACE_FOLDER, SEGMENTATION_FOLDER = "Space", "Segmentation"
SPACE_MANIFEST, SEGMENTATION_MANIFEST = "space.json", "segmentation.json"
DEFAULT_SPACE = "MNI152NLin2009cSym"         # MNI ICBM 152 2009c nonlinear symmetric, 1 mm (the extension's)
DEFAULT_SEGMENTATION = "CerebrA"
USER_FOLDER_VARIABLE = "NEUROPET_ATLAS_FOLDER"
TEMPLATES = (   # key, title, file name stem in a space folder
    ("T1", "T1 (default)", "T1"),
    ("T2", "T2", "T2"),
    ("PD", "PD", "PD"),
    ("GM", "Grey matter probability", "GM"),
    ("WM", "White matter probability", "WM"),
    ("CSF", "CSF probability", "CSF"),
    # made on first use from the grey and white matter maps (pseudoPetArray), saved next to them
    ("pseudoPET", "Synthetic PET (GM + 0.3 WM, smoothed)", "pseudoPET"),
)
PSEUDO_PET = "pseudoPET"
PSEUDO_PET_WM_WEIGHT = 0.3      # FDG: white matter uptake about a third of the grey matter's
PSEUDO_PET_FWHM_MM = 6.0        # smoothed to a PET resolution
DEFAULT_TEMPLATE = "T1"
BRAIN_MASK = "mask"
CT_TEMPLATE = "ctTemplate"      # a CT template in the space: a patient's CT is registered to it (CT to CT)
CT_TEMPLATE_GM = "ctTemplateGm" # the grey matter map in the CT template's space (its MNI is not quite the space's)
CT_CORRECTION_SUFFIX = "_to_space.json"     # the CT template to space matrix, cached next to the CT template
SPACE_VOLUMES = tuple((key, stem) for key, _, stem in TEMPLATES if key != PSEUDO_PET) + (
    (BRAIN_MASK, "mask"), (CT_TEMPLATE, "CT"), (CT_TEMPLATE_GM, "CT_GM"))
GENERATED = (PSEUDO_PET,)       # made by the workflows
IMAGE_EXTENSIONS = (".seg.nrrd", ".nrrd", ".nii.gz", ".nii", ".mnc")     # read; the files made are .nrrd (the standard)
REQUIRED = (DEFAULT_TEMPLATE,)  # without these the normalization cannot run
# Volumes of a space a normalization method registers to (normalizationTemplates)
TEMPLATES_PET = ("T1", "T2", "PD", "GM", PSEUDO_PET)       # the PET to an MR template / a synthetic PET
TEMPLATES_MR = ("T1", "T2", "PD")                          # an MR to an MR template
TEMPLATES_CT = (CT_TEMPLATE,)                              # a CT to a CT template


def _stem(fileName):
    lower = fileName.lower()
    for ext in IMAGE_EXTENSIONS:
        if lower.endswith(ext):
            return fileName[:-len(ext)]
    return None


# The extension's atlas files (Resources/Atlas: the MNI ICBM 152 2009c T1, brain mask, grey and white matter maps;
# the CerebrA and CerebrA PET segmentations; their licenses in LICENSE.txt there). What the workflows make from them
# (GENERATED, region label maps) is written into generatedFolder(), not into the extension.
BUNDLED_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Resources", "Atlas")
GENERATED_FOLDER_VARIABLE = "PET4D_NEURO_ATLAS_CACHE"


def userAtlasFolder():
    """The user's atlas folder (spaces and segmentations of their own): $NEUROPET_ATLAS_FOLDER, else
    ~/NeuroPET/Atlas."""
    return os.environ.get(USER_FOLDER_VARIABLE) or os.path.join(os.path.expanduser("~"), "NeuroPET", "Atlas")


def ensureUserAtlasFolder():
    """The user's atlas folder with its Space and Segmentation folders, made when missing; '' when it cannot be."""
    folder = userAtlasFolder()
    try:
        for sub in (SPACE_FOLDER, SEGMENTATION_FOLDER):
            os.makedirs(os.path.join(folder, sub), exist_ok=True)
    except OSError:
        return ""
    return folder


def atlasRoots():
    """The atlas root folders, the user's first."""
    roots = []
    for folder in (userAtlasFolder(), BUNDLED_FOLDER):
        if folder and os.path.isdir(folder) and not any(_sameFolder(folder, r) for r in roots):
            roots.append(folder)
    return roots


def generatedFolder():
    """Where the files made from the extension's atlas files are kept: $PET4D_NEURO_ATLAS_CACHE, else
    ~/.PET4D/NeuroAtlas."""
    return os.environ.get(GENERATED_FOLDER_VARIABLE) or os.path.join(os.path.expanduser("~"), ".PET4D", "NeuroAtlas")


def fileStamp(path):
    """Modification time and size of a file as a text, '' when it cannot be read: another text once the file is
    written again (a copy of the file loaded before is then out of date)."""
    try:
        status = os.stat(path)
    except (OSError, TypeError, ValueError):
        return ""
    return f"{status.st_mtime_ns}:{status.st_size}"


def _sameFolder(a, b):
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def _inside(path, folder):
    path, folder = os.path.normcase(os.path.realpath(path)), os.path.normcase(os.path.realpath(folder))
    return path == folder or path.startswith(folder.rstrip(os.sep) + os.sep)


def outputFolder(source):
    """The folder the files made from the atlas file source are written to: for a file of an atlas root the same
    subfolder of generatedFolder() (under user/ for the user's root), made when missing; next to it otherwise."""
    folder = os.path.dirname(source)
    for root, base in ((BUNDLED_FOLDER, generatedFolder()), (userAtlasFolder(), os.path.join(generatedFolder(), "user"))):
        if root and _inside(folder, root):
            break
    else:
        return folder
    relative = os.path.relpath(os.path.realpath(folder), os.path.realpath(root))
    target = base if relative in (".", "") else os.path.join(base, relative)
    try:
        os.makedirs(target, exist_ok=True)
    except OSError:
        pass        # the workflow then saves in Slicer's temporary folder
    return target


def readManifest(path):
    """A manifest (space.json / segmentation.json) as a dict; {} when missing or not readable."""
    try:
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _volumeFiles(folder):
    """{key: path} of the volumes of a space folder (SPACE_VOLUMES, file names without regard to case)."""
    wanted = {stem.lower(): key for key, stem in SPACE_VOLUMES}
    found = {}
    try:
        names = os.listdir(folder)
    except OSError:
        return found
    order = {ext: i for i, ext in enumerate(IMAGE_EXTENSIONS)}
    names.sort(key=lambda n: (order.get(next((e for e in IMAGE_EXTENSIONS if n.lower().endswith(e)), ""), 99), n))
    for name in names:
        stem = _stem(name)
        key = wanted.get(stem.lower()) if stem else None
        path = os.path.join(folder, name)
        if key is not None and key not in found and os.path.isfile(path):
            found[key] = path
    return found


def _subfolders(kind):
    """[(id, folder)] of the Space or Segmentation folders of the roots, the user's first."""
    out = []
    for root in atlasRoots():
        base = os.path.join(root, kind)
        try:
            names = sorted(os.listdir(base))
        except OSError:
            continue
        for name in names:
            folder = os.path.join(base, name)
            if os.path.isdir(folder) and not name.startswith("."):
                out.append((name, folder))
    return out


def _ordered(items, first):
    return sorted(items, key=lambda item: (item["id"] != first, item["name"].lower()))


def listSpaces():
    """The template spaces of the atlas roots: [{"id" (folder name), "name", "manifest", "folders", "files" {key:
    path}}], the default space first. A space in both roots: its volumes the user's first, the manifest the user's
    when it has one. A folder without a manifest and without any volume is not a space."""
    spaces = {}
    for spaceId, folder in _subfolders(SPACE_FOLDER):
        manifest = readManifest(os.path.join(folder, SPACE_MANIFEST))
        files = _volumeFiles(folder)
        if not manifest and not files:
            continue
        space = spaces.setdefault(spaceId, {"id": spaceId, "manifest": {}, "folders": [], "files": {}})
        space["folders"].append(folder)
        if manifest and not space["manifest"]:
            space["manifest"] = manifest
        for key, path in files.items():
            space["files"].setdefault(key, path)
    for space in spaces.values():
        space["name"] = str(space["manifest"].get("name") or space["id"])
    return _ordered(spaces.values(), DEFAULT_SPACE)


def spaceInfo(spaceId=None):
    """The space spaceId (listSpaces; the default space for None / ''), None when there is no such space."""
    spaceId = spaceId or DEFAULT_SPACE
    return next((s for s in listSpaces() if s["id"] == spaceId), None)


def spaceTitle(spaceId=None):
    info = spaceInfo(spaceId)
    return info["name"] if info else str(spaceId or DEFAULT_SPACE)


def atlasFiles(space=None):
    """{key: path} of the volumes of a space (an id, or a listSpaces() item; the default space for None): T1, T2,
    PD, GM, WM, CSF, mask, ctTemplate, ctTemplateGm as found, and the synthetic PET when it was made. {} when the
    space is not there."""
    info = space if isinstance(space, dict) else spaceInfo(space)
    if not info:
        return {}
    found = dict(info["files"])
    path = pseudoPetPath(found)
    if path and os.path.isfile(path):
        found[PSEUDO_PET] = path
    return found


def spaceVolumes(found):
    """The template keys of a space's files (atlasFiles) that can be registered to, in TEMPLATES order, then the CT
    template."""
    keys = [k for k, _, _ in TEMPLATES if canMakeTemplate(found, k) and k not in ("WM", "CSF")]
    return keys + ([CT_TEMPLATE] if CT_TEMPLATE in found else [])


def normalizationTemplates(found, method, contrast=None):
    """Template keys a method registers to, of those of the space (found: atlasFiles): the PET (method 'pet') to an
    MR template or the synthetic PET; a CT / MR ('anatomy', contrast 'CT' or an MR contrast) to the CT template or
    the MR templates. A CT in a space without a CT template: the T1 (the CT's brain to the skull-stripped T1)."""
    if method == "anatomy":
        if contrast == "CT":
            keys = [k for k in TEMPLATES_CT if k in found]
            return keys or [k for k in ("T1",) if k in found]
        keys = TEMPLATES_MR
    else:
        keys = TEMPLATES_PET
    return [k for k in keys if canMakeTemplate(found, k)]


def ctCorrectionPath(found):
    """Where the CT template to space matrix is kept (next to the CT template; generatedFolder() for the
    extension's), None without one."""
    path = found.get(CT_TEMPLATE)
    if not path:
        return None
    stem = _stem(os.path.basename(path)) or os.path.basename(path)
    return os.path.join(outputFolder(path), stem + CT_CORRECTION_SUFFIX)


def ctCorrectedPath(found):
    """Where the CT template moved onto the space is saved (to look at; its geometry carries the correction), None
    without a CT template."""
    path = ctCorrectionPath(found)
    return path[:-len(".json")] + ".nrrd" if path else None


def pseudoPetPath(found):
    """Where the synthetic PET template is: the file found, else next to the grey matter map; None without the grey
    and white matter maps."""
    if PSEUDO_PET in found:
        return found[PSEUDO_PET]
    if "GM" not in found or "WM" not in found:
        return None
    stem = next(s for k, _, s in TEMPLATES if k == PSEUDO_PET)
    return os.path.join(outputFolder(found["GM"]), stem + ".nrrd")


def canMakeTemplate(found, key):
    """Whether the template key can be used: its file is there, or (the synthetic PET) it can be made."""
    return key in found or (key == PSEUDO_PET and pseudoPetPath(found) is not None)


def _probability(array):
    array = np.asarray(array, dtype=np.float32)
    top = float(array.max()) if array.size else 0.0
    return array / top if top > 1.5 else array        # stored 0-100 or 0-255


def pseudoPetArray(gm, wm, spacing=(1.0, 1.0, 1.0), wmWeight=PSEUDO_PET_WM_WEIGHT, fwhmMm=PSEUDO_PET_FWHM_MM):
    """A synthetic FDG-like brain image from the grey and white matter probability maps (same grid; spacing of the
    numpy axes in mm): GM + wmWeight x WM, smoothed with a Gaussian of fwhmMm, scaled to a maximum of 1."""
    gm, wm = _probability(gm), _probability(wm)
    if gm.shape != wm.shape:
        raise ValueError("The grey and white matter maps are not on the same grid.")
    image = gm + float(wmWeight) * wm
    if fwhmMm > 0:
        from scipy import ndimage
        sigma = [fwhmMm / 2.3548 / float(s) for s in spacing]
        image = ndimage.gaussian_filter(image, sigma)
    top = float(image.max()) if image.size else 0.0
    return (image / top if top > 0 else image).astype(np.float32)


CT_TEMPLATE_TITLE = "CT template (CT to CT)"


def templateTitle(key):
    if key == CT_TEMPLATE:
        return CT_TEMPLATE_TITLE
    return next((title for k, title, _ in TEMPLATES if k == key), key)


# ---------------------------------------------------------------------------
# Atlas library: segmentations and region atlases
# ---------------------------------------------------------------------------
SEGMENTATION_LABELS_STEM = "labels"
SEGMENTATION_CSV_NAME = "labels.csv"
GROUPS_CSV_NAME = "groups.csv"
COMPATIBLE_GROUPS_CSV_NAME = "groups_compatible.csv"
REGIONS, BROAD, COMPATIBLE = "regions", "broad", "compatible"
GROUPINGS = ((REGIONS, "Regions, left and right (Putamen R / L)"),
             (BROAD, "Broad regions, left and right (Prefrontal R / L)"),
             (COMPATIBLE, "Broad (Compatible), CortexID-like (Prefrontal Lateral R / L, Pons)"))
GROUPING_SHORT = {REGIONS: "regions", BROAD: "broad regions", COMPATIBLE: "Broad (Compatible)"}
MERGED = "Merged"           # grouping suffix: the sides merged ('regionsMerged'), for the reference regions
DEFAULT_REGION_ATLAS = DEFAULT_SEGMENTATION + ":" + REGIONS


def _segmentationOf(segmentationId, folder):
    manifest = readManifest(os.path.join(folder, SEGMENTATION_MANIFEST))
    labels = None
    for ext in (".seg.nrrd", ".nrrd", ".nii.gz", ".nii"):
        for name in os.listdir(folder):
            if name.lower() == SEGMENTATION_LABELS_STEM + ext and os.path.isfile(os.path.join(folder, name)):
                labels = os.path.join(folder, name)
                break
        if labels:
            break
    files = {n.lower(): os.path.join(folder, n) for n in os.listdir(folder)}
    table = files.get(SEGMENTATION_CSV_NAME)
    if not labels or not table:
        return None
    return {"id": segmentationId, "name": str(manifest.get("name") or segmentationId), "folder": folder,
            "manifest": manifest, "space": str(manifest.get("space") or DEFAULT_SPACE), "labels": labels,
            "csv": table, "groups": files.get(GROUPS_CSV_NAME), "groupsCompatible": files.get(COMPATIBLE_GROUPS_CSV_NAME)}


def listSegmentations(space=None):
    """The segmentations of the atlas roots (of the space space when given): [{"id" (folder name), "name", "folder",
    "manifest", "space", "labels" (labels.seg.nrrd / labels.nrrd), "csv", "groups", "groupsCompatible" (paths or
    None)}], the default first. A segmentation in both roots: the user's. Folders without labels and labels.csv are
    left out."""
    found = {}
    for segmentationId, folder in _subfolders(SEGMENTATION_FOLDER):
        if segmentationId in found:
            continue
        try:
            info = _segmentationOf(segmentationId, folder)
        except OSError:
            info = None
        if info:
            found[segmentationId] = info
    items = [s for s in found.values() if not space or s["space"] == space]
    return _ordered(items, DEFAULT_SEGMENTATION)


def segmentationInfo(segmentationId=None):
    segmentationId = segmentationId or DEFAULT_SEGMENTATION
    return next((s for s in listSegmentations() if s["id"] == segmentationId), None)


def segmentationGroupings(info):
    """The groupings (GROUPINGS keys) a segmentation offers: its regions, and the broad ones it has a table for."""
    if not info:
        return []
    return [REGIONS] + ([BROAD] if info.get("groups") else []) + ([COMPATIBLE] if info.get("groupsCompatible") else [])


def regionAtlasKey(segmentation=None, grouping=REGIONS, merged=False):
    return f"{segmentation or DEFAULT_SEGMENTATION}:{grouping or REGIONS}{MERGED if merged else ''}"


def splitRegionAtlas(key):
    """(segmentation id, grouping, sides merged) of a region atlas key ('CerebrA:broad', 'CerebrA PET:regionsMerged';
    a key without ':' is a segmentation's regions)."""
    key = str(key or DEFAULT_REGION_ATLAS)
    segmentation, _, grouping = key.rpartition(":") if ":" in key else (key, ":", REGIONS)
    merged = grouping.endswith(MERGED)
    if merged:
        grouping = grouping[:-len(MERGED)]
    return segmentation or DEFAULT_SEGMENTATION, grouping or REGIONS, merged


def mergedRegionAtlas(key):
    """The same region atlas with the sides merged (the reference regions)."""
    segmentation, grouping, _ = splitRegionAtlas(key)
    return regionAtlasKey(segmentation, grouping, merged=True)


def isRegionAtlasKey(key):
    segmentation, grouping, _ = splitRegionAtlas(key)
    return ":" in str(key or "") and bool(segmentation) and grouping in dict(GROUPINGS)


def regionAtlasTitle(key, info=None):
    """'CerebrA PET, broad regions' of a region atlas key (the segmentation's name from its manifest)."""
    if not key:
        return ""
    segmentation, grouping, merged = splitRegionAtlas(key)
    info = info or segmentationInfo(segmentation)
    name = info["name"] if info else segmentation
    text = f"{name}, {GROUPING_SHORT.get(grouping, grouping)}"
    return text + (", sides merged" if merged else "")


def regionTable(key, info=None):
    """The regions of a region atlas key: (entries [(value, name, color)], mapping {label value of the segmentation:
    region value}, groups [(name, structures)] or None, segmentation info). ValueError when the segmentation or its
    groups table is not there. Group members not in the segmentation are left out (a shared table lists
    'Brainstem' and 'Brainstem (except pons)')."""
    segmentation, grouping, merged = splitRegionAtlas(key)
    info = info or segmentationInfo(segmentation)
    if not info:
        raise ValueError(f"The segmentation '{segmentation}' is not in the atlas folders.")
    structures = readCerebraLabels(info["csv"])
    if grouping == REGIONS:
        entries, mapping = labelTable(structures, mergeSides=merged)
        return entries, mapping, None, info
    table = info.get("groupsCompatible" if grouping == COMPATIBLE else "groups")
    if grouping not in (BROAD, COMPATIBLE) or not table:
        raise ValueError(f"The segmentation '{info['name']}' has no {GROUPING_SHORT.get(grouping, grouping)} table "
                         f"({COMPATIBLE_GROUPS_CSV_NAME if grouping == COMPATIBLE else GROUPS_CSV_NAME}).")
    groups = readGroups(table)
    entries, mapping, _ = groupedTable(structures, groups, merged, readMergedGroups(table))
    return entries, mapping, groups, info


def referenceGroups(key, info=None):
    """Reference-only regions of a region atlas key (readReferenceGroups of its groups table): [(name, members)].
    The segmentation's own regions take those of its groups.csv too, their member groups as the structures in them
    (Cerebellum Whole: Cerebellum Gray Matter, Cerebellum White Matter, Vermal lobules I-V, ...). [] without such
    rows."""
    segmentation, grouping, _ = splitRegionAtlas(key)
    if grouping not in (REGIONS, BROAD, COMPATIBLE):
        return []
    info = info or segmentationInfo(segmentation)
    table = (info or {}).get("groupsCompatible" if grouping == COMPATIBLE else "groups")
    if not table:
        return []
    references = readReferenceGroups(table)
    if grouping != REGIONS or not references:
        return references
    groups = {name.lower(): members for name, members in readGroups(table)}
    return [(name, [s for m in members for s in groups.get(m.lower(), [m])]) for name, members in references]


def referenceRegionValues(entries, name, references=()):
    """Label values of a reference region in an atlas's entries: the region (sides merged, regionValues), or a
    reference-only region of references [(name, members)]: the values of all its members ([] when one is missing,
    so that a part of it is never taken for the whole)."""
    values = regionValues(entries, name)
    if values:
        return values
    base = str(name or "").strip().lower()
    for refName, members in references or ():
        if refName.strip().lower() == base:
            parts = [regionValues(entries, m) for m in members]
            return sorted({v for p in parts for v in p}) if parts and all(parts) else []
    return []


def referenceRegionNames(entries, references=()):
    """The reference regions of an atlas's entries: its regions (sides merged) and the reference-only regions whose
    members are all there."""
    names = mergedRegionNames(entries)
    lower = {n.lower() for n in names}
    for name, _ in references or ():
        if name.lower() not in lower and referenceRegionValues(entries, name, references):
            names.append(name)
            lower.add(name.lower())
    return names


def regionMapPaths(key, info=None):
    """(label map path, CSV path) where the label map of a region atlas is made (next to the segmentation's files;
    generatedFolder() for the extension's): the segmentation's own label map file when it is a label volume of the
    regions as they are (no CSV then)."""
    segmentation, grouping, merged = splitRegionAtlas(key)
    info = info or segmentationInfo(segmentation)
    if not info:
        return None, None
    labels = info["labels"]
    if grouping == REGIONS and not merged and not labels.lower().endswith(".seg.nrrd"):
        return labels, None
    folder = outputFolder(labels)
    stem = "labels_map" if grouping == REGIONS and not merged else grouping + (MERGED if merged else "")
    return os.path.join(folder, stem + ".nrrd"), os.path.join(folder, stem + "_LabelDetails.csv")


def segmentationLabelMapPath(info):
    """The label volume of a segmentation (its regions as they are): the labels file itself, or the label map made
    from labels.seg.nrrd."""
    return regionMapPaths(regionAtlasKey(info["id"]), info)[0]


# ---------------------------------------------------------------------------
# Contrast of the patient's anatomy
# ---------------------------------------------------------------------------
# words in the series description / protocol / sequence name, checked in this order (word boundaries, any case)
CONTRAST_WORDS = (
    ("T2", ("flair", "t2", "t2w", "tse_t2", "space_t2", "cube_t2", "stir", "dark_fluid", "darkfluid")),
    ("PD", ("pd", "pdw", "proton density")),
    ("T1", ("t1", "t1w", "mprage", "mp-rage", "mp2rage", "spgr", "fspgr", "bravo", "tfe", "t1_tfe", "vibe", "flash",
            "ir-fspgr", "irfspgr", "3d_t1", "t1_mpr", "mpr")),
)
# echo / repetition time rules (ms) when the names say nothing
T2_MIN_TE, T2_MIN_TR = 60.0, 1500.0
T1_MAX_TE, T1_MAX_TR = 30.0, 1000.0
PD_MIN_TR = 1500.0


def _hasWord(text, word):
    return re.search(r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])", text) is not None


def _number(value):
    try:
        text = str(value).split("\\")[0].strip()
        return float(text) if text else None
    except (TypeError, ValueError):
        return None


def classifyAnatomy(modality=None, texts=(), repetitionTime=None, echoTime=None, inversionTime=None):
    """(contrast, reason) of the patient's CT/MR: contrast "CT", "T1", "T2", "PD" or None (unknown). texts: series
    description, protocol name, sequence name, volume name...; times in ms (DICOM strings are fine)."""
    modality = (modality or "").strip().upper()
    if modality == "CT":
        return "CT", "modality CT"
    joined = " ".join(str(t) for t in texts if t).lower()
    if _hasWord(joined, "t1") and _hasWord(joined, "flair"):     # T1 FLAIR is T1 weighted
        return "T1", "'T1' in the series description / protocol"
    for contrast, words in CONTRAST_WORDS:
        for word in words:
            if _hasWord(joined, word):
                return contrast, f"'{word.upper()}' in the series description / protocol"
    tr, te, ti = _number(repetitionTime), _number(echoTime), _number(inversionTime)
    if te is not None and tr is not None:
        if te >= T2_MIN_TE and tr >= T2_MIN_TR:
            return "T2", f"long TE ({te:g} ms) and TR ({tr:g} ms)"
        if te <= T1_MAX_TE and (tr <= T1_MAX_TR or (ti is not None and ti > 0)):
            detail = f" and TR ({tr:g} ms)" if tr <= T1_MAX_TR else f" with inversion ({ti:g} ms)"
            return "T1", f"short TE ({te:g} ms)" + detail
        if te <= T1_MAX_TE and tr >= PD_MIN_TR:
            return "PD", f"short TE ({te:g} ms) and long TR ({tr:g} ms)"
    if modality == "MR":
        return None, "MR, contrast not recognised"
    if _hasWord(joined, "ct"):
        return "CT", "'CT' in the name"
    return None, "not recognised"


def templateForContrast(contrast):
    """Template key that fits a contrast: T2 and PD their own template, a CT the CT template (CT_TEMPLATE; the T1
    when it is not in the atlas folder), T1 and unknown the T1."""
    if contrast == "CT":
        return CT_TEMPLATE
    return contrast if contrast in ("T2", "PD") else DEFAULT_TEMPLATE


# ---------------------------------------------------------------------------
# A CT for the MR template: the brain only (numpy + scipy)
# ---------------------------------------------------------------------------
# The MR shows no skull (cortical bone has no signal) and the T1's scalp is bright fat, while the CT's bone is its
# brightest and its brain lies in a few HU; on the whole CT range a mutual information histogram holds the brain in
# one bin. So a CT is normalized as its brain only, in a brain window, to the skull-stripped template.
CT_BONE_HU = 200.0              # bone at and above this
CT_BRAIN_HU = (-10.0, 100.0)    # brain, CSF and blood (a haemorrhage too) between these
CT_BRAIN_WINDOW = (0.0, 80.0)   # the brain image clipped to this window
CT_SKULL_CLOSE_MM = 2.0         # the bone grown by this much first (thin bone, sutures, partial volume)
CT_BRAIN_ERODE_MM = 4.0         # the brain eroded by this much to cut it from the orbits and the neck, then grown back


def looksLikeCt(voxels):
    """Whether an image is in Hounsfield units (air at about -1000)."""
    array = np.asarray(voxels)
    return bool(array.size) and float(np.nanmin(array[::2, ::2, ::2])) <= -500.0


def _ellipsoid(radiusMm, spacing):
    """A boolean structuring element of an ellipsoid of radiusMm (numpy axes' spacing in mm)."""
    half = [max(0, int(np.floor(radiusMm / float(s) + 1e-6))) for s in spacing]
    grids = np.ogrid[tuple(slice(-h, h + 1) for h in half)]
    distance = sum((g * float(s) / radiusMm) ** 2 for g, s in zip(grids, spacing))
    return distance <= 1.0 + 1e-6


def _enclosedPerSlice(mask, axis):
    """The voxels enclosed by mask in the slices across one axis (holes of each slice, mask not included)."""
    from scipy import ndimage
    structure = np.zeros((3, 3, 3), dtype=bool)
    index = [slice(None)] * 3
    index[axis] = 1
    structure[tuple(index)] = ndimage.generate_binary_structure(2, 1)
    labels, _ = ndimage.label(~mask, structure=structure)
    border = set()
    for other in (a for a in range(3) if a != axis):
        for end in (0, -1):
            border.update(np.unique(np.take(labels, end, axis=other)).tolist())
    border.discard(0)
    return (labels > 0) & ~np.isin(labels, list(border))


def ctBrainMask(hu, spacing=(1.0, 1.0, 1.0)):
    """The brain of a head CT (Hounsfield units; numpy axes' spacing in mm): soft tissue enclosed by the skull in the
    slices of any of the three orientations, eroded, its largest piece, grown back. Empty when nothing is found."""
    from scipy import ndimage
    hu = np.asarray(hu, dtype=np.float32)
    skull = ndimage.binary_dilation(hu >= CT_BONE_HU, structure=_ellipsoid(CT_SKULL_CLOSE_MM, spacing))
    inside = np.zeros(hu.shape, dtype=bool)
    for axis in range(3):
        inside |= _enclosedPerSlice(skull, axis)
    candidate = inside & (hu >= CT_BRAIN_HU[0]) & (hu <= CT_BRAIN_HU[1])
    ball = _ellipsoid(CT_BRAIN_ERODE_MM, spacing)
    core = ndimage.binary_erosion(candidate, structure=ball)
    labels, count = ndimage.label(core)
    if count == 0:
        return np.zeros(hu.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    brain = ndimage.binary_dilation(labels == int(np.argmax(sizes)), structure=ball) & candidate
    return ndimage.binary_fill_holes(brain)


# CT intensities stretched for a mutual information histogram (as the Cormack units of the Clinical Toolbox): the soft
# tissue range (CT_STRETCH_HU) gets most of the scale, air and bone a little at either end
CT_STRETCH_HU = (-100.0, 100.0)
CT_STRETCH_POINTS = ((-1000.0, 0.0), (CT_STRETCH_HU[0], 100.0), (CT_STRETCH_HU[1], 1100.0), (1500.0, 1200.0))


def ctStretch(hu):
    """A CT in Hounsfield units mapped (monotonic, piecewise linear) so that -100 ... 100 HU spans 1000 of 1200 units:
    the brain is spread over the histogram instead of a bin or two of the whole HU range."""
    xs, ys = zip(*CT_STRETCH_POINTS)
    return np.interp(np.asarray(hu, dtype=np.float32), xs, ys).astype(np.float32)


def ctBrainImage(hu, spacing=(1.0, 1.0, 1.0)):
    """(image, brain voxel count): the CT's brain in the brain window (CT_BRAIN_WINDOW), 0 outside the brain."""
    mask = ctBrainMask(hu, spacing)
    image = np.clip(np.asarray(hu, dtype=np.float32), *CT_BRAIN_WINDOW)
    image[~mask] = 0.0
    return image, int(mask.sum())


# ---------------------------------------------------------------------------
# CerebrA labels
# ---------------------------------------------------------------------------
RIGHT, LEFT, BOTH = "R", "L", "R+L"
REFERENCE_STRUCTURE = "Cerebellum Gray Matter"    # default reference region of the reference tissue models


def readCerebraLabels(source):
    """Structures of CerebrA_LabelDetails.csv (path or text): [{"id" (Mindboggle ID), "name", "right" (label value),
    "left" (label value), "notes"}], in the file's order. Columns: Mindboggle ID, Label Name, RH Label, LH Labels,
    Notes, Dice Kappa (header names matched loosely)."""
    if isinstance(source, str) and ("\n" in source or "," in source) and not os.path.isfile(source):
        text = source
    else:
        with open(source, newline="", encoding="utf-8-sig") as f:
            text = f.read()
    text = text.lstrip("﻿")
    firstLine = text.split("\n", 1)[0]
    delimiter = ";" if firstLine.count(";") > firstLine.count(",") else ","     # Excel saves with ';' in some locales
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = [r for r in reader if any(c.strip() for c in r)]
    if not rows:
        raise ValueError("The CerebrA label file is empty")
    header = [c.strip().lower() for c in rows[0]]

    def column(*words):
        for i, name in enumerate(header):
            if all(w in name for w in words):
                return i
        raise ValueError(f"The CerebrA label file has no '{' '.join(words)}' column")

    nameCol, rightCol, leftCol = column("name"), column("rh"), column("lh")
    try:
        idCol = column("id")
    except ValueError:
        idCol = None
    try:
        notesCol = column("note")
    except ValueError:
        notesCol = None
    out = []
    for row in rows[1:]:
        def cell(i):
            return row[i].strip() if i is not None and i < len(row) else ""
        name = cell(nameCol)
        right, left = _number(cell(rightCol)), _number(cell(leftCol))
        if not name or (right is None and left is None):
            continue
        out.append({"id": cell(idCol), "name": name, "right": int(right) if right is not None else None,
                    "left": int(left) if left is not None else None, "notes": cell(notesCol)})
    if not out:
        raise ValueError("No labels in the CerebrA label file")
    return out


def structureColor(index, count, side=RIGHT):
    """(r, g, b) 0-1 of a structure: hues spread by the golden ratio, the left side a little darker."""
    hue = (index * 0.618033988749895) % 1.0
    value = 0.92 if side != LEFT else 0.70
    saturation = 0.65 if index % 2 == 0 else 0.85
    return colorsys.hsv_to_rgb(hue, saturation, value)


def labelTable(structures, mergeSides=False):
    """Segments to make from the CerebrA labels. Returns (entries, mapping): entries [(new label value, name,
    (r, g, b))], mapping {CerebrA label value: new label value}. Separate sides: one segment per side, 'Putamen R' /
    'Putamen L' (the CerebrA values are kept); merged: one segment per structure, numbered 1, 2, ..."""
    entries, mapping = [], {}
    count = len(structures)
    if mergeSides:
        for index, s in enumerate(structures):
            value = index + 1
            entries.append((value, s["name"], structureColor(index, count, BOTH)))
            for side in ("right", "left"):
                if s[side] is not None:
                    mapping[s[side]] = value
        return entries, mapping
    for index, s in enumerate(structures):
        for side, suffix in (("right", RIGHT), ("left", LEFT)):
            if s[side] is not None:
                value = s[side]
                entries.append((value, f"{s['name']} {suffix}", structureColor(index, count, suffix)))
                mapping[value] = value
    entries.sort(key=lambda e: e[0])
    return entries, mapping


BILATERAL_HEADER = ("Label", "Label Name", "RH Label", "LH Label")


def bilateralCsvText(structures):
    """CSV of the bilateral CerebrA: one row per structure, its new label value (1, 2, ...) and the CerebrA values of
    the sides it merges."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(BILATERAL_HEADER)
    for index, s in enumerate(structures):
        writer.writerow((index + 1, s["name"], "" if s["right"] is None else s["right"],
                         "" if s["left"] is None else s["left"]))
    return out.getvalue()


def readBilateralLabels(source):
    """Segments of CerebrA_bilateral_LabelDetails.csv (path or text): [(label value, name, (r, g, b))]."""
    if isinstance(source, str) and "\n" in source and not os.path.isfile(source):
        text = source
    else:
        with open(source, newline="", encoding="utf-8-sig") as f:
            text = f.read()
    rows = [r for r in csv.reader(io.StringIO(text.lstrip("\ufeff"))) if any(c.strip() for c in r)]
    if not rows or rows[0][0].strip().lower() != "label":
        raise ValueError("Not a bilateral CerebrA label file")
    entries = []
    body = rows[1:]
    for index, row in enumerate(body):
        value = _number(row[0]) if row else None
        if value is None or len(row) < 2 or not row[1].strip():
            continue
        entries.append((int(value), row[1].strip(), structureColor(index, len(body), BOTH)))
    if not entries:
        raise ValueError("No labels in the bilateral CerebrA label file")
    return entries


def remapLabels(array, mapping):
    """Label array with its values replaced through mapping (values not in it become 0)."""
    array = np.asarray(array)
    values = np.rint(array).astype(np.int64) if array.dtype.kind == "f" else array.astype(np.int64)
    top = max(int(values.max(initial=0)), max(mapping, default=0)) + 1
    lookup = np.zeros(top, dtype=np.int32)
    for source, target in mapping.items():
        if 0 <= source < top:
            lookup[source] = target
    values = np.clip(values, 0, top - 1)
    return lookup[values]


def regionValues(entries, name):
    """Label values of the region name in an atlas's entries [(value, name, color)], sides merged: the entries
    called name, 'name R' or 'name L' (any case)."""
    base = str(name or "").strip().lower()
    return [v for v, n, _ in entries if n.strip().lower() in (base, base + " r", base + " l")]


def mergedRegionNames(entries):
    """The region names of an atlas's entries with the sides merged ('Putamen R', 'Putamen L' -> 'Putamen'), in
    their order."""
    out = []
    for _, name, _ in entries:
        base = name.strip()
        if base[-2:] in (" R", " L"):
            base = base[:-2].rstrip()
        if base not in out:
            out.append(base)
    return out


def structureValues(structures, name):
    """CerebrA label values (right, left) of the structure called name (any case), or ()."""
    for s in structures:
        if s["name"].strip().lower() == name.strip().lower():
            return tuple(v for v in (s["right"], s["left"]) if v is not None)
    return ()


def segmentNamesOfLabels(values, name=None):
    """Segment names (lower case) that stand for these CerebrA label values in a segmentation file: 'Segment_<value>'
    (Slicer's names of a label map imported as segments), and name / 'name R' / 'name L' when given."""
    names = {f"segment_{int(v)}" for v in values}
    if name:
        base = name.strip().lower()
        names.update({base, f"{base} r", f"{base} l"})
    return names



# ---------------------------------------------------------------------------
# Grouped CerebrA (broad regions: prefrontal, lateral occipital, ...)
# ---------------------------------------------------------------------------
MERGED_SIDES = ("merged", "both", "r+l", "bilateral")     # Sides column values: one segment for both sides
REFERENCE_SIDES = ("reference",)    # Sides column value: a reference region only (made of other groups, no segment)


def readGroups(source):
    """Broad regions of a groups table (path or text): [(group name, [CerebrA structure names])]. Columns Group and
    Structures (';' or ',' separated file; the structures of a group separated by ',' in a ';' file, by '|' in a ','
    file); other columns (Sides, Notes) are ignored here (readMergedGroups reads Sides). Raises ValueError for a
    structure in two groups. The reference-only rows (readReferenceGroups) are not groups."""
    return [(name, members) for name, members, sides in _readGroupRows(source) if sides.lower() not in REFERENCE_SIDES]


def readReferenceGroups(source):
    """Reference-only regions of a groups table (Sides 'reference'): [(name, [group or structure names])], e.g.
    'Cerebellum Whole' = Cerebellum Gray Matter + Cerebellum White Matter + Vermis. They are not segments of the
    region atlas (their members are); they are listed and made as reference regions only (referenceRegionNames,
    referenceRegionValues)."""
    return [(name, members) for name, members, sides in _readGroupRows(source) if sides.lower() in REFERENCE_SIDES]


def readMergedGroups(source):
    """Lower-case names of the groups whose Sides column says merged (MERGED_SIDES): one segment for both sides
    ('Pons', 'Cerebellum Whole') even when the other groups are 'R' / 'L'."""
    return {name.lower() for name, _, sides in _readGroupRows(source) if sides.lower() in MERGED_SIDES}


def _readGroupRows(source):
    if isinstance(source, str) and "\n" in source and not os.path.isfile(source):
        text = source
    else:
        with open(source, newline="", encoding="utf-8-sig") as f:
            text = f.read()
    text = text.lstrip("\ufeff")
    firstLine = text.split("\n", 1)[0]
    delimiter = ";" if firstLine.count(";") >= firstLine.count(",") else ","
    memberSeparator = "," if delimiter == ";" else "|"
    rows = [r for r in csv.reader(io.StringIO(text), delimiter=delimiter) if any(c.strip() for c in r)]
    if not rows:
        raise ValueError("The groups table is empty")
    header = [c.strip().lower() for c in rows[0]]
    if "group" not in header or "structures" not in header:
        raise ValueError("The groups table needs the columns Group and Structures")
    groupCol, membersCol = header.index("group"), header.index("structures")
    sidesCol = header.index("sides") if "sides" in header else None
    groups, seen = [], {}
    for row in rows[1:]:
        name = row[groupCol].strip() if groupCol < len(row) else ""
        members = [m.strip() for m in (row[membersCol] if membersCol < len(row) else "").split(memberSeparator)]
        members = [m for m in members if m]
        if not name or not members:
            continue
        sides = row[sidesCol].strip() if sidesCol is not None and sidesCol < len(row) else ""
        if sides.lower() in REFERENCE_SIDES:       # a union of other groups: its members are in them
            groups.append((name, members, sides))
            continue
        for member in members:
            if member.lower() in seen:
                raise ValueError(f"'{member}' is in the groups '{seen[member.lower()]}' and '{name}'")
            seen[member.lower()] = name
        groups.append((name, members, sides))
    if not any(s.lower() not in REFERENCE_SIDES for _, _, s in groups):
        raise ValueError("No groups in the groups table")
    return groups


def groupedTable(structures, groups, mergeSides=False, mergedGroups=()):
    """Segments of the grouped CerebrA. Returns (entries [(value, name, color)], mapping {CerebrA label value: group
    value}, unknown structure names). Sides separate: 'Prefrontal R' (value 2i+1) and 'Prefrontal L' (2i+2), except
    the groups named in mergedGroups (lower case; readMergedGroups): 'Pons' (2i+1); merged: 'Prefrontal' (i+1).
    CerebrA structures in no group are left out (ventricles, optic chiasm)."""
    mergedGroups = {g.lower() for g in mergedGroups}
    byName = {s["name"].strip().lower(): s for s in structures}
    entries, mapping, unknown = [], {}, []
    count = len(groups)
    for index, (name, members) in enumerate(groups):
        found = [byName[m.lower()] for m in members if m.lower() in byName]
        unknown.extend(m for m in members if m.lower() not in byName)
        if not found:
            continue
        if mergeSides or name.lower() in mergedGroups:
            value = index + 1 if mergeSides else 2 * index + 1
            entries.append((value, name, structureColor(index, count, BOTH)))
            for s in found:
                for side in ("right", "left"):
                    if s[side] is not None:
                        mapping[s[side]] = value
            continue
        for side, suffix, value in (("right", RIGHT, 2 * index + 1), ("left", LEFT, 2 * index + 2)):
            values = [s[side] for s in found if s[side] is not None]
            if values:
                entries.append((value, f"{name} {suffix}", structureColor(index, count, suffix)))
                mapping.update({v: value for v in values})
    return entries, mapping, unknown


GROUPED_HEADER = ("Label", "Label Name", "Structures")


def groupedCsvText(entries, groups):
    """CSV of a grouped CerebrA (read with readBilateralLabels): label value, segment name, its structures."""
    members = {name.lower(): ", ".join(m) for name, m in groups}
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(GROUPED_HEADER)
    for value, name, _ in entries:
        base = name[:-2] if name[-2:] in (" " + RIGHT, " " + LEFT) and name[:-2].lower() in members else name
        writer.writerow((value, name, members.get(base.lower(), "")))
    return out.getvalue()


# ---------------------------------------------------------------------------
# Growing the regions, eroding a reference (numpy + scipy)
# ---------------------------------------------------------------------------
def growLabels(labels, distanceMm, spacing=(1.0, 1.0, 1.0), allowed=None, fixedValues=()):
    """Regions grown by distanceMm into the unlabelled voxels: each such voxel within distanceMm of a region (and in
    allowed, when given) goes to the nearest region, so regions never overlap and the order does not matter. Labelled
    voxels are kept; fixedValues (ventricles, ...) neither grow nor are grown into. spacing: of the array axes (mm).
    """
    from scipy import ndimage
    labels = np.asarray(labels)
    if distanceMm <= 0:
        return labels.copy()
    seeds = labels > 0
    if len(fixedValues):
        seeds &= ~np.isin(labels, list(fixedValues))
    if not seeds.any():
        return labels.copy()
    distance, indices = ndimage.distance_transform_edt(~seeds, sampling=spacing, return_indices=True)
    grow = (labels == 0) & (distance <= distanceMm + 1e-6)
    if allowed is not None:
        grow &= np.asarray(allowed, dtype=bool)
    out = labels.copy()
    nearest = labels[tuple(indices)]
    out[grow] = nearest[grow]
    return out


# ---------------------------------------------------------------------------
# Segmentation files (.seg.nrrd) read without Slicer: the CerebrA PET atlas
# ---------------------------------------------------------------------------
_NRRD_TYPES = {"unsigned char": np.uint8, "uchar": np.uint8, "uint8": np.uint8, "uint8_t": np.uint8,
               "signed char": np.int8, "int8": np.int8, "short": np.int16, "int16": np.int16,
               "unsigned short": np.uint16, "ushort": np.uint16, "uint16": np.uint16, "int": np.int32,
               "int32": np.int32, "unsigned int": np.uint32, "uint32": np.uint32}


def readSegmentationFile(path):
    """A Slicer segmentation file (.seg.nrrd; gzip or raw; one label layer or more) without Slicer: (layers array
    (layer, k, j, i), IJK to RAS 4 x 4 matrix, segments [{"name", "layer", "value", "color"}] in the file's order)."""
    import gzip
    with open(path, "rb") as f:
        data = f.read()
    end = data.find(b"\n\n")
    if not data.startswith(b"NRRD") or end < 0:
        raise ValueError(f"Not a NRRD file: {path}")
    fields, segments = {}, {}
    for line in data[:end].decode("latin-1").splitlines()[1:]:
        if line.startswith("#") or not line.strip():
            continue
        match = re.match(r"Segment(\d+)_(\w+):=(.*)$", line)
        if match:
            segments.setdefault(int(match.group(1)), {})[match.group(2)] = match.group(3).strip()
            continue
        if ":=" in line:
            continue
        key, _, value = line.partition(":")
        fields[key.strip().lower()] = value.strip()
    sizes = [int(v) for v in fields["sizes"].split()]
    dtype = np.dtype(_NRRD_TYPES[fields["type"].lower()])
    if fields.get("endian", "little").lower() == "big":
        dtype = dtype.newbyteorder(">")
    payload = data[end + 2:]
    encoding = fields.get("encoding", "raw").lower()
    if encoding in ("gzip", "gz"):
        payload = gzip.decompress(payload)
    elif encoding != "raw":
        raise ValueError(f"NRRD encoding '{encoding}' is not supported")
    count = int(np.prod(sizes))
    array = np.frombuffer(payload[:count * dtype.itemsize], dtype=dtype).reshape(sizes[::-1])
    if len(sizes) == 4:     # a list axis (the layers): the first NRRD axis is the last numpy axis
        kinds = [k.lower() for k in fields.get("kinds", "").split()]
        listAxis = next((i for i, k in enumerate(kinds) if k != "domain"), 0)
        layers = np.moveaxis(array, 3 - listAxis, 0)
    elif len(sizes) == 3:
        layers = array[np.newaxis]
    else:
        raise ValueError(f"A segmentation of {len(sizes)} dimensions is not supported")
    matrix = np.eye(4)
    for column, text in enumerate(re.findall(r"\(([^)]*)\)", fields["space directions"])[:3]):
        matrix[:3, column] = [float(v) for v in text.split(",")]
    origin = re.findall(r"\(([^)]*)\)", fields.get("space origin", "(0,0,0)"))
    if origin:
        matrix[:3, 3] = [float(v) for v in origin[0].split(",")]
    if fields.get("space", "").lower() in ("left-posterior-superior", "lps"):
        matrix = np.diag([-1.0, -1.0, 1.0, 1.0]) @ matrix
    out = []
    for index in sorted(segments):
        s = segments[index]
        try:
            color = tuple(float(v) for v in s.get("Color", "").split())[:3] or None
        except ValueError:
            color = None
        out.append({"name": s.get("Name", ""), "layer": int(s.get("Layer") or 0),
                    "value": int(s.get("LabelValue") or index + 1), "color": color})
    return np.asarray(layers), matrix, out


def segmentationLabels(layers, segments, structures):
    """Label array (k, j, i) of a segmentation file's layers (readSegmentationFile), numbered by the label table
    structures: each segment by its name ('Caudate R' is the right Caudate value of the table; 'Segment_<value>' that
    value). Where segments overlap, the earlier one keeps the voxel. Returns (labels, names of the segments not in the
    table)."""
    values = {name.strip().lower(): value for value, name, _ in labelTable(structures)[0]}
    known = set(values.values())
    labels = np.zeros(np.shape(layers)[1:], dtype=np.int32)
    unknown = []
    for s in segments:
        name = s["name"].strip().lower()
        value = values.get(name)
        if value is None:
            match = re.fullmatch(r"segment_(\d+)", name)
            value = int(match.group(1)) if match and int(match.group(1)) in known else None
        if value is None or not 0 <= s["layer"] < len(layers):
            unknown.append(s["name"])
            continue
        labels[(layers[s["layer"]] == s["value"]) & (labels == 0)] = value
    return labels, unknown


def growLabelsByRegion(labels, limitsMm, spacing=(1.0, 1.0, 1.0), allowed=None):
    """Regions grown into the unlabelled voxels, each by its own distance (limitsMm {label value: mm}; the others not
    grown): a voxel (in allowed, when given) goes to the nearest region among those that reach it, so regions never
    overlap. Labelled voxels (ventricles, other structures) are kept. How the CerebrA PET atlas was made."""
    from scipy import ndimage
    labels = np.asarray(labels)
    out = labels.copy()
    free = labels == 0
    if allowed is not None:
        free &= np.asarray(allowed, dtype=bool)
    best = np.full(labels.shape, np.inf)
    for mm in sorted({float(m) for m in limitsMm.values() if m and float(m) > 0}):
        seeds = np.isin(labels, [v for v, m in limitsMm.items() if m and float(m) == mm])
        if not seeds.any():
            continue
        distance, indices = ndimage.distance_transform_edt(~seeds, sampling=spacing, return_indices=True)
        take = free & (distance <= mm + 1e-6) & (distance < best)
        out[take] = labels[tuple(indices)][take]
        best[take] = distance[take]
    return out


def erodeMask(mask, distanceMm, spacing=(1.0, 1.0, 1.0)):
    """The mask without its voxels within distanceMm of the outside (mm, any voxel size). Nothing removed for 0."""
    from scipy import ndimage
    mask = np.asarray(mask, dtype=bool)
    if distanceMm <= 0 or not mask.any():
        return mask.copy()
    padded = np.pad(mask, 1)        # the edge of the array counts as outside
    distance = ndimage.distance_transform_edt(padded, sampling=spacing)[1:-1, 1:-1, 1:-1]
    return mask & (distance > distanceMm + 1e-6)


# ---------------------------------------------------------------------------
# A reference region of the template space, not of a region atlas: the centrum semiovale
# ---------------------------------------------------------------------------
# For tracers with specific binding in all grey matter ([11C]UCB-J, synaptic density: the cerebellum is no reference).
# Listed among the reference regions of every region atlas when the space has its white matter map (WM).
WHITE_MATTER_REFERENCE = "Centrum Semiovale (white matter)"
WM_REFERENCE_PROBABILITY = 0.9      # the core of the white matter (MNI white matter probability)


def isWhiteMatterReference(name):
    return str(name or "").strip().lower() == WHITE_MATTER_REFERENCE.lower()


def lateralVentricleValues(entries):
    """Label values of the lateral ventricles (not the inferior horns) in an atlas's entries."""
    return [v for v, name, _ in entries
            if "lateral ventricle" in name.lower() and "inferior" not in name.lower()]


def whiteMatterReferenceMask(wm, labels, ventricleValues, ijkToRas, brainMask=None,
                             probability=WM_REFERENCE_PROBABILITY):
    """The centrum semiovale (boolean, k j i): the cerebral white matter above the top of the lateral ventricles
    (ventricleValues of labels; superior in RAS through ijkToRas), where the white matter probability wm (stored 0-1,
    0-100 or 0-255) is at least probability, in no region of the atlas labels (cortex, deep grey matter, cerebellum,
    brainstem) and in brainMask when given; all on one grid. About 113 ml in MNI 2009c with CerebrA (both sides
    alike, at least 1 mm and 3.3 mm in the median from the grey matter); erode it for the PET resolution."""
    wm = np.asarray(wm, dtype=np.float32)
    labels = np.asarray(labels)
    if wm.shape != labels.shape:
        raise ValueError("The white matter map is not on the grid of the region atlas.")
    top = float(wm.max()) if wm.size else 0.0
    if top > 1.5:
        wm = wm / top
    ventricles = np.isin(labels, list(ventricleValues))
    if not ventricles.any():
        raise ValueError("The region atlas has no lateral ventricles (the centrum semiovale lies above them).")
    matrix = np.asarray(ijkToRas, dtype=float)
    k, j, i = np.ogrid[0:labels.shape[0], 0:labels.shape[1], 0:labels.shape[2]]
    superior = (matrix[2, 0] * i + matrix[2, 1] * j + matrix[2, 2] * k + matrix[2, 3]).astype(np.float32)
    superior = np.broadcast_to(superior, labels.shape)
    mask = (wm >= probability) & (labels == 0) & (superior > float(superior[ventricles].max()))
    if brainMask is not None and np.shape(brainMask) == labels.shape:
        mask &= np.asarray(brainMask) > 0
    if not mask.any():
        raise ValueError("The centrum semiovale is empty (no white matter above the lateral ventricles).")
    return mask


NO_GROW_WORDS = ("ventricle", "optic chiasm")       # CSF and the chiasm: never grown


def fixedValuesOf(entries):
    """Label values of the entries that must not grow (ventricles, optic chiasm)."""
    return [v for v, name, _ in entries if any(w in name.lower() for w in NO_GROW_WORDS)]
