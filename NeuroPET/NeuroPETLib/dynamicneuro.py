"""
Dynamic Neuro Compare: the kinetic settings of parametric maps, compared with each other and with a dynamic normal
database, and the regional values of the region (TAC) fits.

Pure Python (no Slicer): the settings are those of the PET4D Dynamic PET Modeling panel (its fitState(), kept on the
maps it makes as the DynamicPET.Modeling.FitState attribute) put in the shape of a dynamic database's kinetic
settings (normaldb.kinetics), so that both compare the same way.
"""

import json

from NeuroPETLib import normaldb

# the models the Compare step makes maps of: reference tissue models (a database's models, normaldb.KINETIC_MAPS)
COMPARE_MODELS = tuple(normaldb.KINETIC_MAPS)
TIME_MANUAL = "manual"              # DynamicPETModeling's typed injection time
# the regional value of a region: its TAC fitted with the model (default), or the mean of the voxel map in it
REGION_TAC_FIT, REGION_VOXEL_MEAN = "tac", "voxel"
REGION_VALUES = ((REGION_TAC_FIT, "TAC fit of the region (model fitted to its mean curve)"),
                 (REGION_VOXEL_MEAN, "Mean of the voxel map in the region"))
SETTING_TITLES = {"model": "model", "reference": "reference region", "weighting": "frame weighting",
                  "timeZero": "time zero", "injectionMin": "injection time", "tStarMin": "t*",
                  "k2ref": "reference k2'", "window": "SUVR window", "clipNegative": "negative values set to 0"}


def mapSettings(state, reference=None):
    """The kinetic settings of a Modeling fitState() (a dict, or its JSON) as a database's (normaldb.kinetics shape):
    {"model", "weighting", "timeZero", "tStarMin", "k2ref" (None: automatic), "window" {"startMinutes",
    "endMinutes"} (None: none given), "clipNegative", "reference" (the reference segment's name; given, or the
    state's), "injectionMin" (a typed injection time only)}. None for no state."""
    if isinstance(state, str):
        try:
            state = json.loads(state) if state else None
        except ValueError:
            state = None
    if not state:
        return None
    window = list(state.get("window") or [])
    start, end = (float(window[0]), float(window[1])) if len(window) == 2 else (None, None)
    if start is None or end is None or end <= start:
        start = end = None
    k2ref = state.get("k2ref")
    try:
        k2ref = float(k2ref) if k2ref not in (None, "") and float(k2ref) > 0 else None
    except (TypeError, ValueError):
        k2ref = None
    out = {"model": state.get("model"), "weighting": state.get("weighting") or normaldb.KINETIC_WEIGHTINGS[0][0],
           "timeZero": state.get("timeZero") or normaldb.TIME_FIRST_FRAME,
           "tStarMin": float(state.get("tStar") if state.get("tStar") is not None else normaldb.DEFAULT_TSTAR_MIN),
           "k2ref": k2ref, "window": {"startMinutes": start, "endMinutes": end},
           "clipNegative": bool(state.get("clip", True)), "reference": reference or state.get("referenceName")}
    if out["timeZero"] == TIME_MANUAL:
        out["injectionMin"] = float(state.get("injectionMin") or 0.0)
    return out


def relevantKeys(model):
    """The settings that change the maps of a model."""
    keys = ["model", "reference", "weighting", "timeZero", "clipNegative"]
    if model == "LoganRef":
        keys += ["tStarMin", "k2ref"]
    elif model == "SRTM2":
        keys.append("k2ref")
    elif model == "SUVR":
        keys.append("window")
    return keys


def _window(settings, lastEndMinutes=None):
    """(start, end) minutes of the SUVR window; no window: the last SUVR_WINDOW_MINUTES of the series when its end is
    known (kinetics.defaultWindow), else None."""
    window = settings.get("window") or {}
    start, end = window.get("startMinutes"), window.get("endMinutes")
    if start is not None and end is not None:
        return round(float(start), 3), round(float(end), 3)
    if lastEndMinutes is not None:
        end = float(lastEndMinutes)
        return round(max(0.0, end - normaldb.SUVR_WINDOW_MINUTES), 3), round(end, 3)
    return None


def _same(key, a, b, lastEndMinutes=None):
    if key == "window":
        return _window(a, lastEndMinutes) == _window(b, lastEndMinutes)
    x, y = a.get(key), b.get(key)
    if key == "reference":
        return _sameName(x, y)
    if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
        return abs(float(x) - float(y)) <= 1e-6 * max(1.0, abs(float(x)), abs(float(y)))
    return x == y


def _sameName(a, b):
    """Whether two region names are the same region: case and the ' (reference)' mark not counted."""
    def clean(name):
        text = str(name or "").strip().lower()
        for mark in (" (reference)", "(reference)"):
            if text.endswith(mark):
                text = text[:-len(mark)].strip()
        return text
    return clean(a) == clean(b)


def _text(key, settings, lastEndMinutes=None):
    value = settings.get(key)
    if key == "window":
        window = _window(settings, lastEndMinutes)
        return f"{window[0]:g}-{window[1]:g} min" if window else "the last 30 min"
    if key == "k2ref":
        return "automatic" if value is None else f"{float(value):g}/min"
    if key == "tStarMin":
        return f"{float(value):g} min"
    if key == "clipNegative":
        return "yes" if value else "no"
    if key == "timeZero":
        return dict(normaldb.KINETIC_TIME_ZEROS).get(value, value or "?").lower()
    return str(value if value not in (None, "") else "none")


def settingsDifferences(expected, actual, lastEndMinutes=None, skip=()):
    """The differences of two kinetic settings (mapSettings / normaldb.kinetics shape) that change the maps of the
    expected model: ["model: SRTM2 here, SRTM in the database", ...] as "<what>: <actual> vs <expected>" texts. The
    reference is compared only when both name one; skip: keys left out."""
    if not expected or not actual:
        return []
    out = []
    model = expected.get("model")
    for key in relevantKeys(model):
        if key in skip:
            continue
        if key == "reference" and (not expected.get(key) or not actual.get(key)):
            continue
        if not _same(key, expected, actual, lastEndMinutes):
            out.append(f"{SETTING_TITLES.get(key, key)}: {_text(key, actual, lastEndMinutes)} "
                       f"(expected {_text(key, expected, lastEndMinutes)})")
        if key == "model" and expected.get("model") != actual.get("model"):
            break           # the other settings mean other things then
    return out


def sameSettings(a, b, lastEndMinutes=None):
    return not settingsDifferences(a, b, lastEndMinutes)


def mapsAcceptable(state):
    """Whether maps made with a Modeling fitState cover the brain (no threshold, not only inside the segments): (ok,
    reason)."""
    if isinstance(state, str):
        try:
            state = json.loads(state) if state else {}
        except ValueError:
            state = {}
    state = state or {}
    if float(state.get("mapThreshold") or 0.0) > 0.0:
        return False, f"made with an activity threshold ({float(state['mapThreshold']):g} %)"
    if state.get("inside"):
        return False, "made only inside the segments"
    return True, ""


def regionFitValues(results, names, mapName):
    """{region name: the value mapName of its fit} of Modeling fit results ({segment ID: kinetics.fit result}) and
    {segment ID: region name}; a region whose fit failed or has no such value is left out."""
    out = {}
    for segmentId, result in (results or {}).items():
        name = names.get(segmentId)
        if not name or not result or result.get("error"):
            continue
        value = (result.get("values") or {}).get(mapName)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if value == value:          # not NaN
            out[name] = value
    return out


def fitWarnings(results, names, mapName):
    """'Cerebellum L (at a bound: BPnd)' texts of the fits with warnings or errors."""
    out = []
    for segmentId, result in (results or {}).items():
        name = names.get(segmentId, segmentId)
        if not result:
            continue
        if result.get("error"):
            out.append(f"{name} (not fitted: {result['error']})")
        elif result.get("warnings"):
            out.append(f"{name} ({'; '.join(str(w) for w in result['warnings'])})")
    return out


def defaultMap(model, available=None):
    """The map compared first: the model's primary map (BPnd of SRTM / SRTM2 / LoganRef, SUVR), else the first."""
    primary = normaldb.PRIMARY_MAPS.get(model)
    maps = list(available or normaldb.KINETIC_MAPS.get(model, ()))
    if primary in maps:
        return primary
    return maps[0] if maps else None


def regionValueTitle(mapName, how):
    """The region table's value column: 'BPnd (TAC fit)' or 'BPnd mean'."""
    return f"{mapName} (TAC fit)" if how == REGION_TAC_FIT else f"{mapName} mean"
