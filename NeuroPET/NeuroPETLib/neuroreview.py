"""Static Neuro PET/SPECT Review: the patient compared with a normal database (numpy only, no Slicer).

    voxel scores    z = (patient - mean) / SD and the Crawford-Howell t = (patient - mean) / (SD * sqrt(1 + 1/n))
                    (df n - 1), both negative where the patient is below the normals (hypometabolism / hypoperfusion);
                    computed in a mask (the brain mask and the database mean above a fraction of its maximum) with
                    an SD floor (a fraction of the median SD in the mask), so the edge of the brain and the
                    near-constant voxels do not blow up
    regional scores the regions' means of the normals (their mean, SD, n per age group) computed with the mean / SD
                    maps by the Neuro PET Database Manager (stats/regions.json), or on the fly from the subjects'
                    maps for another segmentation (cached in the database's cache/ folder); for a predefined
                    database without them, an approximation from the mean / SD maps (the mean of the voxel SDs
                    over-estimates the SD of the regional mean, so that score is conservative)
    display         the colours of the score maps (a diverging colour map over a window symmetric about 0; the
                    side above the normals grey when only the side below is wanted) and the
                    Review layout (4 x 4: PET, score map, 3D-SSP views and the region table)
"""

import hashlib
import json
import math
import os

import numpy as np

from . import neuro, normaldb

SCORE_Z, SCORE_T = "z", "t"
SCORES = ((SCORE_T, "t (Crawford-Howell)"), (SCORE_Z, "z"))
DEFAULT_SCORE = SCORE_T
MASK_FRACTION = 0.05            # voxels scored: the database mean above 5 % of its maximum (and in the brain mask);
                                # the slices show a part of them (grey matter, or high normal uptake)
UPTAKE_FRACTION = 0.5           # score map shown: the database mean above 50 % of its maximum (cortex, deep grey
                                # matter; white matter and CSF, low in the normals, hidden); adjustable >= MASK_FRACTION
GM_MINIMUM = 0.5                # score slices: where the MNI grey matter probability is at least this (tracer
                                # independent: the cortex also with a striatal tracer such as fallypride)
HIDE_BELOW = 2.0                # score slices: |score| below this transparent (the PET / MR under it)
MASK_GM, MASK_UPTAKE = "gm", "uptake"
STYLE_BRIGHTNESS, STYLE_MASKED = "brightness", "masked"
SCORE_STYLES = ((STYLE_BRIGHTNESS, "Score as color, PET as brightness"),
                (STYLE_MASKED, "Score map (masked, thresholded)"))
BRIGHTNESS_PERCENTILE = 99.5    # the PET brightness: full at this percentile of the patient's map
BRIGHTNESS_GAMMA = 0.8
SCORE_MASKS = ((MASK_GM, "Grey matter (MNI probability)"), (MASK_UPTAKE, "High normal uptake"))
SD_FLOOR_FRACTION = 0.2         # SD floor: 20 % of the median SD in the mask
DEFAULT_MAXIMUM = 7.0           # score window: -7 ... +7 SD, 0 in the middle
# the score colour maps made here: (position 0..255, R, G, B 0..1) points, linear between them. From the CAT12
# colours: SSP_Rainbow, SSP-PET-DICOM, SSP_HotIron, SSP_Inferno, SSP_Viridis; SSP_Rainbow10 has ten flat
# steps (_STEPS). Diverging, for scores (0 in the middle): SSP_Turbo (a smooth rainbow), SSP_IceFire (Burak's
# procedural map: black at 0, blue to cyan on the cold side, red - yellow - light yellow on the warm side),
# SSP_IceFire2 (white - blue, dark grey at 0, red - yellow), SSP_CoolWarm (blue - light grey - red), SSP_GreyZero
# (the rainbow with a grey band around 0, about +/- 0.5 of a +/- 4 window), SSP_PurpleOrange
_POINTS = {
    "SSP_Rainbow": ((0, 0, 0, 0), (41, 0, 0, 0.5), (84, 0, 0.5, 1), (126, 0, 1, 0), (169, 1, 1, 0), (211, 1, 0, 0),
                    (255, 1, 1, 1)),
    "SSP-PET-DICOM": ((0, 0, 0, 0), (31, 0, 0.4, 0.4), (63, 0, 0, 0.8), (95, 0.4, 0, 0.8), (127, 0.8, 0, 0.8),
                      (159, 0.9, 0.3, 0.3), (191, 1, 0.5, 0), (223, 1, 0.8, 0.5), (255, 1, 1, 1)),
    "SSP_HotIron": ((0, 0, 0, 0), (63, 0.5, 0, 0), (127, 1, 0, 0), (191, 1, 1, 0), (255, 1, 1, 1)),
    "SSP_Inferno": ((0, 0, 0, 0.02), (51, 0.259, 0.039, 0.408), (102, 0.576, 0.149, 0.404),
                    (153, 0.867, 0.318, 0.227), (204, 0.988, 0.647, 0.039), (255, 0.988, 1.0, 0.643)),
    "SSP_Viridis": ((0, 0.267, 0.004, 0.329), (51, 0.255, 0.267, 0.529), (102, 0.165, 0.471, 0.557),
                    (153, 0.133, 0.659, 0.518), (204, 0.478, 0.820, 0.318), (255, 0.992, 0.906, 0.145)),
    "SSP_Turbo": ((0, 0.190, 0.072, 0.232), (32, 0.268, 0.416, 0.936), (64, 0.100, 0.620, 0.950),
                  (96, 0.253, 0.954, 0.569), (128, 0.595, 0.981, 0.310), (160, 0.936, 0.809, 0.175),
                  (192, 1.0, 0.494, 0.113), (224, 0.777, 0.168, 0.045), (255, 0.480, 0.016, 0.011)),
    # Burak's procedural colour file (icefire.txt, 9 points)
    "SSP_IceFire": ((0, 0, 1, 1), (32, 0, 0.501961, 0.752941), (64, 0, 0.12549, 0.752941), (96, 0, 0, 0.501961),
                    (127, 0, 0, 0), (160, 0.501961, 0, 0), (192, 0.752941, 0.25098, 0),
                    (224, 0.878431, 0.878431, 0), (255, 1, 1, 0.752941)),
    # Burak's second procedural file (icefireCopy.txt, 9 points) turned around: white - blue at the low end, dark grey
    # at 0, red to yellow at the high end
    "SSP_IceFire2": ((0, 1, 1, 1), (31, 0, 0.568627, 1), (63, 0, 0.0431373, 0.870588),
                     (95, 0.152941, 0.152941, 0.752941), (128, 0.25098, 0.25098, 0.25098), (163, 0.752941, 0, 0),
                     (191, 1, 0.25098, 0), (223, 1, 0.501961, 0), (255, 1, 1, 0)),
    "SSP_CoolWarm": ((0, 0.230, 0.299, 0.754), (64, 0.552, 0.690, 0.996), (128, 0.865, 0.865, 0.865),
                     (192, 0.958, 0.604, 0.483), (255, 0.706, 0.016, 0.150)),
    "SSP_GreyZero": ((0, 0, 0, 0), (40, 0, 0, 0.8), (80, 0, 0.8, 1), (112, 0.55, 0.55, 0.55),
                     (144, 0.55, 0.55, 0.55), (176, 1, 0.9, 0), (216, 1, 0.2, 0), (255, 1, 1, 1)),
    "SSP_PurpleOrange": ((0, 0.176, 0, 0.294), (51, 0.502, 0.451, 0.675), (102, 0.847, 0.855, 0.922),
                         (128, 0.969, 0.969, 0.969), (153, 0.996, 0.878, 0.714), (204, 0.878, 0.510, 0.078),
                         (255, 0.498, 0.231, 0.031)),
}
_STEPS = {
    "SSP_Rainbow10": ((0, 0, 0), (0.3, 0, 0.5), (0, 0, 1), (0, 0.7, 1), (0, 1, 0), (0.6, 1, 0), (1, 1, 0),
                      (1, 0.5, 0), (1, 0, 0), (1, 1, 1)),
}
SCORE_COLORMAPS = ("SSP_Turbo", "SSP_IceFire", "SSP_IceFire2", "SSP_Rainbow", "SSP_Rainbow10", "SSP-PET-DICOM",
                   "SSP_HotIron", "SSP_Inferno", "SSP_Viridis", "SSP_CoolWarm", "SSP_GreyZero", "SSP_PurpleOrange",
                   "ColdToHotRainbow", "DivergingBlueRed", "Plasma", "Viridis")     # the first is the default
# the diverging maps with a neutral colour at 0: shown below the normals / the threshold they are neither reversed nor
# rescaled (the window stays -maximum .. maximum, the other scores grey)
NEUTRAL_MIDPOINT_COLORMAPS = ("SSP_IceFire", "SSP_IceFire2", "SSP_CoolWarm", "SSP_GreyZero", "SSP_PurpleOrange",
                              "DivergingBlueRed")
# Plasma and Viridis (matplotlib, CC0): 17 anchors, interpolated to 256 (the others are Slicer's colour tables)
_ANCHORS = {
    "Plasma": "0d08873105974c02a16600a77e03a89511a1aa2395bc3587cc4778da5a6ae66c5cf0804ef89540fdac33fdc527f8df25f0f921",
    "Viridis": "44015448186a472d7b4240863b528b33638d2c728e26828e21918c1fa08828ae803fbc735ec96284d44baddc30d8e219fde725",
}
GREY = (0.62, 0.62, 0.62)       # the scores not shown (above the normals, or above the threshold)
# which scores are colored: all (the colour table across -maximum .. maximum, 0 in the middle); or only those below
# the normals / below a threshold (the colour table reversed across -maximum .. 0, the lowest scores at its top end
# as in other 3D-SSP software; the rest grey)
SIDE_ALL, SIDE_BELOW, SIDE_THRESHOLD = "all", "below", "threshold"
SCORE_SIDES = ((SIDE_ALL, "All scores"), (SIDE_BELOW, "Below the normals only (colors reversed)"),
               (SIDE_THRESHOLD, "Below the threshold only (colors reversed)"))
DEFAULT_THRESHOLD = -2.0
OUTSIDE = -1.0e6                # score map voxels outside the scored mask (shown black: below the display threshold)
REGIONS_FILE = normaldb.STATS_DIR + "/regions.json"
# the regions the statistics were made on (a segmentation, made with the mean / SD maps): the Regions step loads them
# as they are (the same regions for everyone, also without the atlas files); PENDING until the statistics are written
REGIONS_SEGMENTATION_FILE = normaldb.STATS_DIR + "/regions.seg.nrrd"
REGIONS_SEGMENTATION_PENDING = normaldb.STATS_DIR + "/regions_pending.seg.nrrd"
CACHE_DIR = "cache"
DROPPED_SEGMENT_WORDS = ("optic chiasm",)       # left out of the review (overlay, surface, table)


# ---------------------------------------------------------------------------
# Voxel scores
# ---------------------------------------------------------------------------
def scoreMask(mean, brainMask=None, fraction=MASK_FRACTION):
    """Voxels scored: finite database mean above fraction x its maximum, inside brainMask when given (same shape).
    fraction None: no threshold on the mean (the brain mask alone; without one, the mean above 0)."""
    mean = np.asarray(mean, dtype=np.float32)
    finite = np.isfinite(mean)
    if not finite.any():
        return np.zeros(mean.shape, dtype=bool)
    if fraction is None and brainMask is not None:
        mask = finite.copy()
    else:
        top = float(np.max(mean[finite]))
        mask = finite & (mean > (fraction or 0.0) * top) if top > 0 else np.zeros(mean.shape, dtype=bool)
    if brainMask is not None:
        brainMask = np.asarray(brainMask)
        if brainMask.shape != mean.shape:
            raise ValueError("The brain mask is not on the grid of the database maps.")
        mask &= brainMask > 0
    return mask


def uptakeMask(mean, scored, fraction=UPTAKE_FRACTION):
    """The score map voxels shown: scored (the scoreMask) where the database mean is above fraction x its maximum,
    the regions of high normal uptake (grey matter); white matter and CSF, low in the normals anyway, hidden."""
    return np.asarray(scored, dtype=bool) & scoreMask(mean, None, max(float(fraction), 0.0))


def greyMatterMask(greyMatter, scored, minimum=GM_MINIMUM):
    """The score map voxels shown: scored where the grey matter probability (on the same grid) is at least
    minimum."""
    greyMatter = np.nan_to_num(np.asarray(greyMatter, dtype=np.float32))
    return np.asarray(scored, dtype=bool) & (greyMatter >= np.float32(minimum))


def sdFloor(sd, mask, fraction=SD_FLOOR_FRACTION):
    """The SD with a floor of fraction x its median inside the mask (and never 0)."""
    sd = np.nan_to_num(np.asarray(sd, dtype=np.float32))
    inside = sd[np.asarray(mask, dtype=bool)]
    inside = inside[inside > 0]
    floor = float(np.median(inside)) * fraction if inside.size else 0.0
    floor = max(floor, 1e-6)
    return np.maximum(sd, floor).astype(np.float32)


def tFactor(n):
    """sqrt(1 + 1/n) of the Crawford-Howell t (1 without a known n)."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return 1.0
    return math.sqrt(1.0 + 1.0 / n) if n > 0 else 1.0


def scoreMap(patient, mean, sd, mask, kind=DEFAULT_SCORE, n=None, floorFraction=SD_FLOOR_FRACTION):
    """Voxel z or t map (negative: the patient below the mean), 0 outside the mask."""
    patient = np.asarray(patient, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    if patient.shape != mean.shape or np.shape(sd) != mean.shape:
        raise ValueError(f"The patient's map {patient.shape} is not on the grid of the database maps {mean.shape}.")
    mask = np.asarray(mask, dtype=bool) & np.isfinite(patient)
    denominator = sdFloor(sd, mask, floorFraction)
    if kind == SCORE_T:
        denominator = denominator * np.float32(tFactor(n))
    out = np.zeros(mean.shape, dtype=np.float32)
    out[mask] = (patient[mask] - mean[mask]) / denominator[mask]
    return out


SD_RELATIVE_FLOOR = 1e-6      # an SD below this part of the mean is rounding noise (a region that is the reference)


def regionScore(value, mean, sd, n=None, kind=DEFAULT_SCORE):
    """z or t of one regional value (negative: below the mean); None when it cannot be computed, also when the SD is
    only rounding noise (below SD_RELATIVE_FLOOR of the mean: e.g. a region made of the reference voxels, whose ratio
    is 1 in every normal, so the score would be noise divided by noise)."""
    try:
        value, mean, sd = float(value), float(mean), float(sd)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(value) and math.isfinite(mean) and math.isfinite(sd)) or sd <= 0:
        return None
    if sd < SD_RELATIVE_FLOOR * max(abs(mean), 1e-12):
        return None
    return (value - mean) / (sd * (tFactor(n) if kind == SCORE_T else 1.0))


# ---------------------------------------------------------------------------
# Age group
# ---------------------------------------------------------------------------
def statisticsGroups(header):
    """[(key, name, n)] of the computed mean / SD maps of a database: the master first."""
    groups = ((header or {}).get("statistics") or {}).get("groups") or {}
    out = []
    for key, entry in groups.items():
        if entry.get("mean") and entry.get("sd"):
            out.append((key, entry.get("name") or key, entry.get("n")))
    out.sort(key=lambda g: (g[0] != normaldb.MASTER, str(g[1]).lower()))
    return out


def patientGroup(header, age):
    """(group key, why) of the maps a patient is compared with: the age group of the patient's age when the
    database is age stratified and that group has maps, else the master."""
    available = {key for key, _, _ in statisticsGroups(header)}
    if not header or not header.get("ageStratified"):
        return normaldb.MASTER, "the database has no age groups"
    if age is None:
        return normaldb.MASTER, "the age is not known"
    name = normaldb.ageGroupOf(age, header.get("ageGroups") or [])
    if name is None:
        return normaldb.MASTER, f"no age group holds {age:g} years"
    key = normaldb.groupKey(name)
    if key not in available:
        return normaldb.MASTER, f"the age group '{name}' has no mean / SD maps"
    return key, f"age group '{name}'"


# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------
def isDroppedSegment(name):
    text = str(name or "").strip().lower()
    return any(word in text for word in DROPPED_SEGMENT_WORDS)


REFERENCE_MARK = "(reference)"      # the end of a reference segment's name (DynamicNeuroPET.REFERENCE_SUFFIX)


def isReferenceRegion(name, referenceKey=None, referenceStructure=None):
    """Whether a region (segment name) is the normalization's reference region, left out of the region table: a
    "... (reference)" segment, or the region of the region atlas the database's reference is (referenceStructure
    of the key REF_ATLAS), of both sides ("Pons R", "Pons L")."""
    text = str(name or "").strip().lower()
    if text.endswith(REFERENCE_MARK):
        return True
    if text[-2:] in (" r", " l"):
        text = text[:-2].rstrip()
    structures = normaldb.referenceStructures(referenceKey, referenceStructure) or ()
    return text in {s.lower() for s in structures}


class RegionMask:
    """A region's boolean mask kept as its bounding box (a whole-grid mask per region of a 100-region atlas takes about
    a GB). Works wherever a boolean array does through np.asarray(mask, dtype=bool) (the whole grid, made on demand);
    the functions of this module use the box directly."""

    __slots__ = ("shape", "box", "crop")

    def __init__(self, shape, box, crop):
        self.shape = tuple(int(v) for v in shape)
        self.box = box              # tuple of slices of the grid, or None (an empty region)
        self.crop = crop            # boolean array of the box, or None

    @classmethod
    def fromArray(cls, mask):
        mask = np.asarray(mask, dtype=bool)
        where = [np.flatnonzero(mask.any(axis=tuple(a for a in range(mask.ndim) if a != axis)))
                 for axis in range(mask.ndim)]
        if any(w.size == 0 for w in where):
            return cls(mask.shape, None, None)
        box = tuple(slice(int(w[0]), int(w[-1]) + 1) for w in where)
        return cls(mask.shape, box, np.ascontiguousarray(mask[box]))

    @classmethod
    def empty(cls, shape):
        return cls(shape, None, None)

    @property
    def ndim(self):
        return len(self.shape)

    def any(self):
        return self.box is not None and bool(self.crop.any())

    def __array__(self, dtype=None, copy=None):
        full = np.zeros(self.shape, dtype=bool)
        if self.box is not None:
            full[self.box] = self.crop
        return full if dtype is None else full.astype(dtype, copy=False)


def _boxOf(mask, shape):
    """(box slices, cropped boolean) of a mask (a RegionMask or an array) on a grid of shape; (None, None) when the
    mask is empty."""
    if isinstance(mask, RegionMask):
        if mask.shape != tuple(shape):
            raise ValueError("not on the grid")
        return mask.box, mask.crop
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != tuple(shape):
        raise ValueError("not on the grid")
    return (tuple(slice(None) for _ in shape), mask) if mask.ndim else ((), mask)


def regionMeans(values, masks):
    """{name: mean of the finite values in the mask} (NaN when the mask holds none)."""
    values = np.asarray(values, dtype=np.float32)
    out = {}
    for name, mask in masks.items():
        try:
            box, crop = _boxOf(mask, values.shape)
        except ValueError:
            raise ValueError(f"The region '{name}' is not on the grid of the map.") from None
        if box is None:
            out[name] = float("nan")
            continue
        inside = values[box][crop]
        inside = inside[np.isfinite(inside)]
        out[name] = float(inside.mean()) if inside.size else float("nan")
    return out


def regionStatistics(subjectValues, groups):
    """{group key: {"name", "n", "regions": {region: [mean, SD, n]}}} from subjectValues {subject ID: {region:
    value}} and groups {key: {"name", "ids"}} (normaldb.statisticsGroups); groups with fewer than two subjects are
    left out, and per region the subjects with a value (SD with n - 1)."""
    out = {}
    for key, group in groups.items():
        ids = [i for i in group.get("ids") or [] if i in subjectValues]
        if len(ids) < 2:
            continue
        names = []
        for i in ids:
            for name in subjectValues[i]:
                if name not in names:
                    names.append(name)
        regions = {}
        for name in names:
            column = np.array([subjectValues[i].get(name, np.nan) for i in ids], dtype=np.float64)
            column = column[np.isfinite(column)]
            if column.size >= 2:
                regions[name] = [float(column.mean()), float(column.std(ddof=1)), int(column.size)]
        out[key] = {"name": group.get("name") or key, "n": len(ids), "regions": regions}
    return out


def unionMask(masks, shape):
    """All the regions together (the brain the 3D-SSP surface is made of), the centrum semiovale left out (deep white
    matter: the 3D-SSP rays stop where they leave the grey matter regions, as before it was a region)."""
    union = np.zeros(shape, dtype=bool)
    for name, mask in masks.items():
        if neuro.isWhiteMatterSegment(name):
            continue
        if isinstance(mask, RegionMask):
            if mask.box is not None:
                union[mask.box] |= mask.crop
        else:
            union |= np.asarray(mask, dtype=bool)
    return union


def labelMap(masks, shape):
    """(int16 labels, names) of the regions: label i + 1 is names[i] (where regions overlap, the first)."""
    labels = np.zeros(shape, dtype=np.int16)
    names = []
    for name, mask in masks.items():
        names.append(name)
        if isinstance(mask, RegionMask):
            if mask.box is not None:
                part = labels[mask.box]
                part[mask.crop & (part == 0)] = len(names)
        else:
            labels[np.asarray(mask, dtype=bool) & (labels == 0)] = len(names)
    return labels, names


def labelAt(labels, ijkToRas, ras):
    """The label (0: none) of the voxel at a RAS point (nearest voxel of the grid ijkToRas; 0 outside it)."""
    inverse = np.linalg.inv(np.asarray(ijkToRas, dtype=np.float64))
    ijk = inverse[:3, :3] @ np.asarray(ras, dtype=np.float64) + inverse[:3, 3]
    k, j, i = (int(np.floor(v + 0.5)) for v in ijk[::-1])
    shape = np.shape(labels)
    if not (0 <= k < shape[0] and 0 <= j < shape[1] and 0 <= i < shape[2]):
        return 0
    return int(labels[k, j, i])


def insidePoint(labels, label):
    """(k, j, i) of the voxel of a label nearest to its centroid (a point inside the region, also for a curved one);
    None when the label has no voxel."""
    where = np.argwhere(np.asarray(labels) == label)
    if not len(where):
        return None
    centroid = where.mean(axis=0)
    return tuple(int(v) for v in where[np.argmin(((where - centroid) ** 2).sum(axis=1))])


def approximateRegions(meanMap, sdMap, masks):
    """{region: [mean of the mean map, mean of the SD map, None]}: a predefined database without regional
    statistics. The mean voxel SD is larger than the SD of the regional mean, so scores from it are conservative."""
    means = regionMeans(meanMap, masks)
    sds = regionMeans(sdMap, masks)
    return {name: [means[name], sds[name], None] for name in masks}


def masksSignature(masks, extra=()):
    """Short hash of the regions (names and voxels) and of extra values, the key of the cached regional
    statistics."""
    digest = hashlib.sha1()
    for name in sorted(masks):
        mask = np.asarray(masks[name], dtype=bool)
        digest.update(name.encode("utf-8"))
        digest.update(str(mask.shape).encode())
        digest.update(np.packbits(mask.ravel()).tobytes())
    for value in extra:
        digest.update(str(value).encode("utf-8"))
    return digest.hexdigest()[:16]


def regionsDocument(statistics, mapKey, version, settings=None, signature=""):
    return {"maps": mapKey, "version": version, "settings": settings or {}, "signature": signature,
            "groups": statistics}


def writeJson(path, document):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=1, ensure_ascii=False)
    os.replace(temporary, path)


def readJson(path):
    try:
        with open(path, encoding="utf-8") as stream:
            document = json.load(stream)
        return document if isinstance(document, dict) else None
    except (OSError, ValueError):
        return None


def regionsPath(folder, header):
    """The regional statistics of a database (stats/regions.json), None when missing."""
    relative = ((header or {}).get("statistics") or {}).get("regions")
    path = os.path.join(folder, relative) if relative else ""
    return path if path and os.path.isfile(path) else None


def cachePath(folder, signature, mapKey, version):
    return os.path.join(folder, CACHE_DIR, f"regions_{mapKey}_v{version}_{signature}.json")


def matchRegions(statistics, groupKey, names):
    """{name: [mean, SD, n]} of the group for the names found (case and spaces ignored), and the names missing."""
    regions = ((statistics or {}).get(groupKey) or {}).get("regions") or {}
    byClean = {" ".join(k.lower().split()): v for k, v in regions.items()}
    found, missing = {}, []
    for name in names:
        entry = byClean.get(" ".join(name.lower().split()))
        if entry is None:
            missing.append(name)
        else:
            found[name] = entry
    return found, missing


def tableRows(patientValues, patientSuv, reference, kind=DEFAULT_SCORE):
    """Rows of the region table: (name, SUVmean, value compared, DB mean, DB SD, n, score) per region of
    patientValues {name: value in the database's map type}; patientSuv {name: SUV mean}; reference {name: [mean,
    SD, n]} (None entries: no normal values). The scores use their own n (the region's) when given."""
    rows = []
    for name, value in patientValues.items():
        entry = reference.get(name)
        mean = sd = n = None
        if entry:
            mean, sd, n = entry[0], entry[1], entry[2] if len(entry) > 2 else None
        score = regionScore(value, mean, sd, n, kind) if entry else None
        rows.append((name, patientSuv.get(name), value, mean, sd, n, score))
    return rows


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def colorTable(label, count=256):
    """count x 3 RGB (0..1) of the SSP_ maps, Plasma and Viridis; None for the other colour maps (Slicer's)."""
    steps = _STEPS.get(label)
    if steps is not None:
        steps = np.asarray(steps, dtype=float)
        return steps[np.minimum((np.arange(count) * len(steps)) // count, len(steps) - 1)]
    points = _POINTS.get(label)
    if points is not None:
        points = np.asarray(points, dtype=float)
        at = np.linspace(0.0, 255.0, count)
        return np.stack([np.interp(at, points[:, 0], points[:, c]) for c in (1, 2, 3)], axis=1)
    anchors = _ANCHORS.get(label)
    if anchors is None:
        return None
    rgb = np.array([[int(anchors[i + c:i + c + 2], 16) for c in (0, 2, 4)] for i in range(0, len(anchors), 6)],
                   dtype=float) / 255.0
    where = np.linspace(0.0, 1.0, len(rgb))
    at = np.linspace(0.0, 1.0, count)
    return np.stack([np.interp(at, where, rgb[:, c]) for c in range(3)], axis=1)


def divergingColors(table, hypoOnly=False, count=256, grey=GREY):
    """count x 4 RGBA (0..1) of a score map shown over a window symmetric about 0 ([-maximum, maximum]): the colour
    table (n x 3 RGB 0..1) across it, 0 in the middle; with hypoOnly the side above 0 (above the normals) grey."""
    table = np.asarray(table, dtype=float)
    index = np.clip(np.rint(np.linspace(0, len(table) - 1, count)).astype(int), 0, len(table) - 1)
    rgba = np.ones((count, 4), dtype=float)
    rgba[:, :3] = table[index]
    if hypoOnly:
        rgba[np.linspace(-1.0, 1.0, count) > 0, :3] = grey
    return rgba


def scoreColors(table, side=SIDE_ALL, maximum=DEFAULT_MAXIMUM, threshold=DEFAULT_THRESHOLD, count=256, grey=GREY,
                neutralMidpoint=False):
    """(count x 4 RGBA, (low, high) window) of a score map by its side (SCORE_SIDES): all: the colour table across
    -maximum .. maximum; below / threshold: the table reversed across -maximum .. 0 (-maximum its top end), grey at
    0 (the scores above it clamped there) and, for threshold, above the threshold (a negative score). A table with
    a neutral midpoint (NEUTRAL_MIDPOINT_COLORMAPS) stays as it is across -maximum .. maximum, the scores above 0 /
    the threshold grey."""
    maximum = float(maximum)
    if side not in (SIDE_BELOW, SIDE_THRESHOLD) or neutralMidpoint:
        rgba = divergingColors(table, count=count, grey=grey)
        if side in (SIDE_BELOW, SIDE_THRESHOLD):
            limit = min(float(threshold), 0.0) if side == SIDE_THRESHOLD else 0.0
            rgba[np.linspace(-maximum, maximum, count) > limit + 1e-9 * max(maximum, 1.0), :3] = grey
        return rgba, (-maximum, maximum)
    table = np.asarray(table, dtype=float)
    index = np.clip(np.rint(np.linspace(len(table) - 1, 0, count)).astype(int), 0, len(table) - 1)
    rgba = np.ones((count, 4), dtype=float)
    rgba[:, :3] = table[index]
    values = np.linspace(-maximum, 0.0, count)
    rgba[-1, :3] = grey
    if side == SIDE_THRESHOLD:
        rgba[values > min(float(threshold), 0.0) + 1e-9, :3] = grey
    return rgba, (-maximum, 0.0)


def petBrightness(values, percentile=BRIGHTNESS_PERCENTILE, gamma=BRIGHTNESS_GAMMA):
    """0..1 brightness of the patient's map: its values over the percentile of the positive ones, to a gamma."""
    values = np.nan_to_num(np.asarray(values, dtype=np.float32))
    positive = values[values > 0]
    top = float(np.percentile(positive, percentile)) if positive.size else 1.0
    out = np.clip(values / np.float32(top if top > 0 else 1.0), 0.0, 1.0)
    return out ** np.float32(gamma) if gamma != 1.0 else out


def brightnessColors(scores, brightness, rgba, maximum, low=None):
    """uint8 RGB (shape + (3,)) of a score map in color (the colour table rgba over low .. maximum; low -maximum
    when not given, 0 in the middle) times the brightness (0..1): the PET's own pattern lit in the score's colors, no
    threshold or mask."""
    rgba = np.asarray(rgba, dtype=np.float32)
    count = len(rgba)
    low = -maximum if low is None else low
    span = (maximum - low) or 1.0
    scores = np.nan_to_num(np.asarray(scores, dtype=np.float32))
    index = np.clip(np.rint((scores - low) / span * (count - 1)), 0, count - 1).astype(np.int32)
    rgb = rgba[index, :3] * np.asarray(brightness, dtype=np.float32)[..., None]
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


SCORE_QUANTUM = 0.01       # the scores' step in quantizedBrightness


def quantizedBrightness(scores, brightness, quantum=SCORE_QUANTUM):
    """{"flat", "first", "count", "quantum"} for fastBrightnessColors: each voxel's (score step, brightness level)
    as one index into a (score steps x 256 levels) colour table: the score quantized to first, first + quantum, ...
    and the brightness to 0..255. Made once; a window change then needs one look-up per voxel."""
    scores = np.nan_to_num(np.asarray(scores, dtype=np.float32))
    steps = np.rint(scores / np.float32(quantum)).astype(np.int32)
    first = int(steps.min()) if steps.size else 0
    last = int(steps.max()) if steps.size else 0
    if last - first > 8000:             # beyond +-40 with 0.01: clipped (shown at the window's ends anyway)
        first, last = max(first, -4000), min(last, 4000)
        np.clip(steps, first, last, out=steps)
    levels = np.clip(np.rint(np.asarray(brightness, dtype=np.float32) * 255.0), 0, 255).astype(np.int32)
    steps -= first
    steps *= 256
    steps += levels
    return {"flat": steps, "first": first, "count": last - first + 1, "quantum": float(quantum)}


def fastBrightnessColors(quantized, rgba, maximum, low=None, out=None):
    """brightnessColors() from quantizedBrightness(): the (score step x brightness level) colours made once for the
    window, then looked up per voxel (into out, uint8 shape + (3,), when given)."""
    rgba = np.asarray(rgba, dtype=np.float32)
    count = len(rgba)
    low = -maximum if low is None else low
    span = (maximum - low) or 1.0
    values = (quantized["first"] + np.arange(quantized["count"], dtype=np.float64)) * quantized["quantum"]
    index = np.clip(np.rint((values - low) / span * (count - 1)), 0, count - 1).astype(np.int32)
    colors = rgba[index, :3].astype(np.float32)                                            # (steps, 3)
    levels = np.arange(256, dtype=np.float32) / np.float32(255.0)
    table = np.clip(np.rint(colors[:, None, :] * levels[None, :, None] * 255.0), 0, 255).astype(np.uint8)
    if out is not None:
        return np.take(table.reshape(-1, 3), quantized["flat"], axis=0, out=out)
    return np.take(table.reshape(-1, 3), quantized["flat"], axis=0)


def maskedScores(scores, mask, factor=1.0, hideBelow=0.0):
    """The score map divided by factor (t from z) with OUTSIDE (transparent) where the mask is not and where
    |score| < hideBelow."""
    out = np.full(np.shape(scores), OUTSIDE, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    out[mask] = np.asarray(scores, dtype=np.float32)[mask] / np.float32(factor)
    if hideBelow > 0:
        out[mask & (np.abs(out) < np.float32(hideBelow))] = OUTSIDE
    return out


# Layouts (the views of equal, fixed size: no splitters between them, so a maximized view restores to its size;
# only the table's width can be dragged): rows 1-2 slice views (PET with the region outlines; the score map), rows 3-4 the 3D-SSP views, the
# region table on the right. Single monitor: all of it; dual monitor: the slices and the table in the main window,
# the 3D-SSP views in a floating window placed on the second screen; and the slices or the 3D-SSP views, each with
# the table. The "+ low scores" variants add a second set of the 3D-SSP views below the first: the scores below
# the threshold only (colors reversed), 16 views in all. "Slices + cut 3D-SSP": 3 x 4, a row per orientation (the
# PET, the score slice, the 3D-SSP cut at that slice seen from the left and the right, anterior superior) and a last
# column of the whole 3D-SSP from the left and the right anterior superior (2 views), no table; its "low scores"
# variant the same with the low scores' 3D-SSP (below the threshold, colors reversed, the rest grey) and cut faces.
# More dual monitor layouts (the region table always in the main window): the slices in the main window and the cut
# 3D-SSP (or its low scores variant) on the second screen; the single monitor layout in the main window and the
# unfolded brain (its slices and maps) on the second screen; the single monitor layout in the main window and the
# low scores (their 3D-SSP views and their unfolded map) on the second screen.
LAYOUT_ID = LAYOUT_SINGLE = 7810
LAYOUT_DUAL = 7811
LAYOUT_SLICES = 7812
LAYOUT_SSP = 7813
LAYOUT_DUAL_LOW = 7814
LAYOUT_SSP_LOW = 7815
LAYOUT_CUTS = 7816
LAYOUT_PEEL = 7817
LAYOUT_CUTS_LOW = 7850
LAYOUT_DUAL_CUTS = 7851
LAYOUT_DUAL_CUTS_LOW = 7852
LAYOUT_DUAL_PEEL = 7853
LAYOUT_DUAL_LOWMAP = 7854
# the order of the menus (toolbar Single / Dual Layouts) and of the Display box's combo: the single monitor ones,
# then the dual monitor ones, each with its default first
LAYOUTS = ((LAYOUT_SINGLE, "Single monitor (slices, 3D-SSP, table)"),
           (LAYOUT_SSP_LOW, "3D-SSP + low scores and table (16 views)"),
           (LAYOUT_SLICES, "Slices and table"),
           (LAYOUT_SSP, "3D-SSP and table"),
           (LAYOUT_CUTS, "Slices + cut 3D-SSP (3 x 4)"),
           (LAYOUT_CUTS_LOW, "Slices + cut 3D-SSP, low scores (3 x 4)"),
           (LAYOUT_PEEL, "Unfolded brain (curvilinear, slices + 4 maps) and table"),
           (LAYOUT_DUAL_LOW, "Dual monitor, 3D-SSP + low scores (16 views)"),
           (LAYOUT_DUAL, "Dual monitor (3D-SSP on the second screen)"),
           (LAYOUT_DUAL_CUTS, "Dual monitor, cut 3D-SSP on the second screen"),
           (LAYOUT_DUAL_CUTS_LOW, "Dual monitor, cut 3D-SSP low scores on the second screen"),
           (LAYOUT_DUAL_PEEL, "Dual monitor, unfolded brain on the second screen"),
           (LAYOUT_DUAL_LOWMAP, "Dual monitor, low scores (3D-SSP + unfolded map) on the second screen"))
DUAL_LAYOUTS = (LAYOUT_DUAL, LAYOUT_DUAL_LOW, LAYOUT_DUAL_CUTS, LAYOUT_DUAL_CUTS_LOW, LAYOUT_DUAL_PEEL,
                LAYOUT_DUAL_LOWMAP)
PEEL_LAYOUTS = (LAYOUT_PEEL, LAYOUT_DUAL_PEEL)          # the whole unfolded brain layout (its slices and 4 maps)
LAYOUT_IDS = tuple(k for k, _ in LAYOUTS)
MONITOR2_TITLE = "Static Neuro (Monitor 2)"
ORIENTATIONS = (("Axial", "#F34A33"), ("Coronal", "#6EB04B"), ("Sagittal", "#EDD54C"))
PET_ROW, SCORE_ROW = "PET", "Score"
VIEW_PREFIX = "NeuroRev"
VIEW_GROUP = 93
# 2 x 4: each medial view beside the lateral one seen from the same side (top row from the left: the left lateral,
# the right hemisphere's medial face; bottom row from the right: the right lateral, the left hemisphere's medial face)
SSP_VIEWS = (("Top", "Superior"), ("Ant", "Anterior"), ("Left", "Left"), ("RightMedial", "Right medial"),
             ("Bottom", "Inferior"), ("Post", "Posterior"), ("Right", "Right"), ("LeftMedial", "Left medial"))
SSP_COLUMNS = 4
MEDIAL_KEYS = ("LeftMedial", "RightMedial")     # only with regions with sides (R / L): 2 x 3 views without them
# the sets of 3D-SSP views: the main one (as set in the 3D-SSP box) and the low scores' (below the threshold)
SSP_MAIN, SSP_LOW = "", "Low"
TABLE_VIEW = VIEW_PREFIX + "Table"
# the cut 3D-SSP layout (3 rows: axial, coronal, sagittal; 4 columns: the PET, the scores, the 3D-SSP cut at that
# row's slice seen from the left-anterior-superior, the same from the right-anterior-superior): (key, title,
# direction from the brain centre to the camera, RAS)
CUT_SIDES = (("LAS", "Left ant sup", (-1.0, 1.0, 1.0)), ("RAS", "Right ant sup", (1.0, 1.0, 1.0)))
CUT_VIEW_GROUP = VIEW_GROUP + 2
# its last column: the whole (uncut) 3D-SSP from the left / right anterior superior (main 3D-SSP views: region
# names and scores at the mouse)
SSP_OBLIQUE_VIEWS = (("LeftAntSup", "Left ant sup"), ("RightAntSup", "Right ant sup"))
# the unfolded brain layout (curvilinear reformatting): the whole brain's surface peeled at one depth and unfolded in
# one map (2D slice views of 1-slice volumes). Top row (a third of the bottom's height): the anatomy (patient's MR or
# the MNI T1) coronal, sagittal, axial with the shell the maps are taken from, the superior 3D-SSP; bottom row the
# maps: the PET, the PET over the anatomy with the region outlines, the scores, the scores below the
# threshold (colors reversed) over the anatomy
ANATOMY_ROW = "Anat"        # the anatomy map (no view of its own: under the fusion and the low scores)
FUSION_ROW = "Fusion"
LOW_ROW = "Low"
PEEL_ROWS = ((PET_ROW, "PET"), (FUSION_ROW, "PET / MR"), (SCORE_ROW, "Score"), (LOW_ROW, "Low scores"))
PEEL_VIEWS = (("Map", "unfolded"),)
PEEL_SLICES = ("Coronal", "Sagittal", "Axial")
TABLE_HIDDEN_LAYOUTS = (LAYOUT_PEEL, LAYOUT_CUTS, LAYOUT_CUTS_LOW)     # no region table at start (the toolbar's Table button shows it)
PEEL_VIEW_GROUP = VIEW_GROUP + 3


def sliceViewName(row, orientation):
    return f"{VIEW_PREFIX}{row}{orientation}"


def sspViewName(key, viewSet=SSP_MAIN):
    return f"{VIEW_PREFIX}SSP{viewSet}{key}"


def cutViewName(orientation, side, viewSet=SSP_MAIN):
    """viewSet: the cut views of the main 3D-SSP (SSP_MAIN) or of the low scores' (SSP_LOW)."""
    return f"{VIEW_PREFIX}Cut{viewSet}{orientation}{side}"


def cutViewTags():
    """{singleton tag: (orientation, side key, view set)} of every cut 3D-SSP view."""
    return {cutViewName(o, side, viewSet): (o, side, viewSet) for viewSet in (SSP_MAIN, SSP_LOW)
            for o, _ in ORIENTATIONS for side, _, _ in CUT_SIDES}


def cutDirection(side):
    for key, _, direction in CUT_SIDES:
        if key == side:
            return direction
    raise KeyError(side)


def peelViewName(row, key):
    return f"{VIEW_PREFIX}Peel{row}{key}"


def peelViewTags():
    """{singleton tag: (row, view key)} of every peel view."""
    return {peelViewName(row, key): (row, key) for row, _ in PEEL_ROWS for key, _ in PEEL_VIEWS}


def peelSliceName(orientation):
    return f"{VIEW_PREFIX}PeelSlice{orientation}"


def peelSliceTags():
    """{singleton tag: orientation} of the slice views of the unfolded brain layout."""
    return {peelSliceName(o): o for o in PEEL_SLICES}


def _peelView(row, rowTitle, key):
    return (f'<item><view class="vtkMRMLSliceNode" singletontag="{peelViewName(row, key)}">'
            '<property name="orientation" action="default">Axial</property>'
            f'<property name="viewlabel" action="default">{rowTitle}</property>'
            '<property name="viewcolor" action="default">#808080</property>'
            f'<property name="viewgroup" action="default">{PEEL_VIEW_GROUP}</property>'
            '</view></item>')


def _peelLayout(table=True, grid=False, superior=True):
    """The unfolded brain layout: on the top the slices and (superior) the superior 3D-SSP view (equal columns),
    below the maps in a row of 4 or (grid: the wide front-back poles maps) 2 x 2; the region table on the right."""
    colors = dict(ORIENTATIONS)

    def slice(orientation):
        return (f'<item><view class="vtkMRMLSliceNode" singletontag="{peelSliceName(orientation)}">'
                f'<property name="orientation" action="default">{orientation}</property>'
                f'<property name="viewlabel" action="default">{orientation[0]}</property>'
                f'<property name="viewcolor" action="default">{colors.get(orientation, "#808080")}</property>'
                f'<property name="viewgroup" action="default">{PEEL_VIEW_GROUP}</property>'
                '</view></item>')
    def equal(items):
        """A row of views of equal widths (a splitter with equal sizes, whatever the views' own minimum sizes)."""
        return ('<layout type="horizontal" split="true">'
                + "".join(item.replace("<item>", '<item splitSize="250">', 1) for item in items) + '</layout>')
    top = equal([slice(o) for o in PEEL_SLICES]
                + ([f'<item><view class="vtkMRMLViewNode" singletontag="{sspViewName("Top")}">'
                    '<property name="viewlabel" action="default">Superior</property>'
                    f'<property name="viewgroup" action="default">{VIEW_GROUP + 1}</property></view></item>']
                   if superior else []))
    maps = [_peelView(row, rowTitle, key) for row, rowTitle in PEEL_ROWS for key, _ in PEEL_VIEWS]
    if grid:
        bottom = ('<layout type="vertical" split="true">'
                  + "".join(f'<item splitSize="500">{equal(maps[start:start + 2])}</item>'
                            for start in range(0, len(maps), 2)) + '</layout>')
    else:
        bottom = equal(maps)
    left = (f'<layout type="vertical" split="true"><item splitSize="250">{top}</item>'
            f'<item splitSize="750">{bottom}</item></layout>')
    return _tableRight(left, table, 800)


def sspViewTags():
    """{singleton tag: (view set, view key)} of every 3D-SSP view (the oblique ones of the main set too)."""
    tags = {sspViewName(key, viewSet): (viewSet, key) for viewSet in (SSP_MAIN, SSP_LOW) for key, _ in SSP_VIEWS}
    tags.update({sspViewName(key, viewSet): (viewSet, key) for viewSet in (SSP_MAIN, SSP_LOW)
                 for key, _ in SSP_OBLIQUE_VIEWS})
    return tags


def defaultLayout(screens):
    """Dual monitor with the 16 3D-SSP views (with the low scores) with two screens or more, else the single monitor
    layout."""
    return LAYOUT_DUAL_LOW if screens >= 2 else LAYOUT_SINGLE


def _sliceView(row, orientation, color):
    label = ("" if row == PET_ROW else "S") + orientation[0]
    return (f'<item><view class="vtkMRMLSliceNode" singletontag="{sliceViewName(row, orientation)}">'
            f'<property name="orientation" action="default">{orientation}</property>'
            f'<property name="viewlabel" action="default">{label}</property>'
            f'<property name="viewcolor" action="default">{color}</property>'
            f'<property name="viewgroup" action="default">{VIEW_GROUP}</property>'
            '</view></item>')


def _sliceRows():
    return ['<item><layout type="horizontal" split="false">' + "".join(
        _sliceView(row, o, c) for o, c in ORIENTATIONS) + '</layout></item>' for row in (PET_ROW, SCORE_ROW)]


def _cutRows(viewSet=SSP_MAIN, slices=True):
    """The cut 3D-SSP layout's columns: the PET slices, the score slices (axial, coronal, sagittal; not without
    slices), the 3D-SSP (viewSet: the main one or the low scores') cut at each slice from the left and from the right
    anterior superior, the whole 3D-SSP from both."""
    low = " (low)" if viewSet else ""

    def threeD(orientation, side, title, color):
        return (f'<item><view class="vtkMRMLViewNode" singletontag="{cutViewName(orientation, side, viewSet)}">'
                f'<property name="viewlabel" action="default">{orientation[0]} {title}{low}</property>'
                f'<property name="viewcolor" action="default">{color}</property>'
                f'<property name="viewgroup" action="default">{CUT_VIEW_GROUP}</property>'
                '</view></item>')
    def whole(key, title):
        return (f'<item><view class="vtkMRMLViewNode" singletontag="{sspViewName(key, viewSet)}">'
                f'<property name="viewlabel" action="default">{title}{low}</property>'
                f'<property name="viewgroup" action="default">{VIEW_GROUP + 1}</property>'
                '</view></item>')

    def column(items):
        return '<item><layout type="vertical" split="false">' + "".join(items) + '</layout></item>'
    # columns (not rows): the last one has two views, all five equally wide
    return ([column(_sliceView(PET_ROW, o, c) for o, c in ORIENTATIONS),
             column(_sliceView(SCORE_ROW, o, c) for o, c in ORIENTATIONS)] if slices else []) + [
        column(threeD(o, side, title, c) for o, c in ORIENTATIONS) for side, title, _ in CUT_SIDES] + [
        column(whole(key, title) for key, title in SSP_OBLIQUE_VIEWS)]


def sspViews(medial=True):
    """(the 3D-SSP views (key, title), columns): 2 x 4 with the medial views, else 2 x 3."""
    if medial:
        return SSP_VIEWS, SSP_COLUMNS
    return tuple(v for v in SSP_VIEWS if v[0] not in MEDIAL_KEYS), SSP_COLUMNS - 1


def _sspRows(viewSet=SSP_MAIN, medial=True):
    def threeD(key, title):
        return (f'<item><view class="vtkMRMLViewNode" singletontag="{sspViewName(key, viewSet)}">'
                f'<property name="viewlabel" action="default">{title}{" (low)" if viewSet else ""}</property>'
                f'<property name="viewgroup" action="default">{VIEW_GROUP + 1}</property>'
                '</view></item>')
    views, columns = sspViews(medial)
    return ['<item><layout type="horizontal" split="false">' + "".join(
        threeD(k, t) for k, t in views[start:start + columns]) + '</layout></item>'
        for start in range(0, len(views), columns)]


def _tableRight(left, table=True, leftSize=750):
    """left (a layout) with the region table on its right (always the rightmost view), or left alone."""
    if not table:
        return left
    view = (f'<view class="vtkMRMLTableViewNode" singletontag="{TABLE_VIEW}">'
            '<property name="viewlabel" action="default">R</property></view>')
    return (f'<layout type="horizontal" split="true"><item splitSize="{leftSize}">{left}</item>'
            f'<item splitSize="{1000 - leftSize}">{view}</item></layout>')


def _withTable(rows, table=True):
    return _tableRight(f'<layout type="vertical" split="false">{"".join(rows)}</layout>', table)


def tableByDefault(layoutId, default=True):
    """Whether a layout shows the region table until the toolbar's Table button says otherwise: the setting
    (default), but never the unfolded brain's (its maps need the room)."""
    return False if layoutId in TABLE_HIDDEN_LAYOUTS else bool(default)


def layoutXml(layoutId=LAYOUT_SINGLE, mainViewportAttributes="", medial=True, table=True, peelGrid=False):
    """The layout description of a review layout. mainViewportAttributes: XML attributes of the main window viewport
    of the dual monitor layout (as in Slicer's own dual monitor layouts); medial: the medial 3D-SSP views (regions
    with sides); table: the region table on the right (in the main window of the dual monitor layouts); peelGrid:
    the unfolded brain's maps 2 x 2 (the wide maps)."""
    if layoutId == LAYOUT_SLICES:
        return _withTable(_sliceRows(), table)
    if layoutId == LAYOUT_PEEL:
        return _peelLayout(table, peelGrid)
    if layoutId in (LAYOUT_CUTS, LAYOUT_CUTS_LOW):
        rows = _cutRows(SSP_LOW if layoutId == LAYOUT_CUTS_LOW else SSP_MAIN)
        return _tableRight(f'<layout type="horizontal" split="false">{"".join(rows)}</layout>', table, 800)
    sspRows = _sspRows(SSP_MAIN, medial) + (_sspRows(SSP_LOW, medial) if layoutId in (LAYOUT_DUAL_LOW, LAYOUT_SSP_LOW)
                                            else [])
    if layoutId in (LAYOUT_SSP, LAYOUT_SSP_LOW):
        return _withTable(sspRows, table)
    if layoutId in DUAL_LAYOUTS:
        # the main window: the slices (and the 3D-SSP: the single monitor layout) and the table
        if layoutId in (LAYOUT_DUAL_PEEL, LAYOUT_DUAL_LOWMAP):
            main = _withTable(_sliceRows() + _sspRows(SSP_MAIN, medial), table)
        else:
            main = _withTable(_sliceRows(), table)
        main = main.replace('<layout ', f'<layout{mainViewportAttributes} ', 1)
        # the second screen
        if layoutId in (LAYOUT_DUAL_CUTS, LAYOUT_DUAL_CUTS_LOW):
            rows = _cutRows(SSP_LOW if layoutId == LAYOUT_DUAL_CUTS_LOW else SSP_MAIN, slices=False)
            second = f'<item><layout type="horizontal" split="false">{"".join(rows)}</layout></item>'
        elif layoutId == LAYOUT_DUAL_PEEL:
            second = f'<item>{_peelLayout(False, peelGrid, superior=False)}</item>'
        elif layoutId == LAYOUT_DUAL_LOWMAP:     # the low scores' 3D-SSP over their unfolded map
            second = (f'<item><layout type="vertical" split="false">{"".join(_sspRows(SSP_LOW, medial))}</layout>'
                      f'</item>{_peelView(LOW_ROW, dict(PEEL_ROWS)[LOW_ROW], PEEL_VIEWS[0][0])}')
        else:
            second = "".join(sspRows)
        return ('<viewports>' + main
                + f'<layout name="StaticNeuroMonitor2" type="vertical" split="false" label="{MONITOR2_TITLE}" '
                  'dockable="true" dockPosition="floating">' + second + '</layout></viewports>')
    return _withTable(_sliceRows() + _sspRows(SSP_MAIN, medial), table)
