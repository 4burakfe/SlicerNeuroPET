"""Where the atlases, templates and normal databases come from: description, references to cite and license, shown in
the read-only info boxes next to the places they are picked (Static / Dynamic Neuro, Neuro PET Database Manager).

The texts of the extension's atlas files follow Resources/Atlas/LICENSE.txt: a space's or segmentation's manifest
(space.json / segmentation.json) names them in "sources"; a manifest of the user's without "sources" is shown by its
own name, description, url, cite and license. Pure Python apart from the atlas library (neuro, read lazily); infoBox()
alone needs Qt.
"""

import html
import re

ICBM152 = "icbm152"
CEREBRA = "cerebra"
CEREBRA_PONS = "cerebraPons"
CEREBRA_GROUPS = "cerebraGroups"
CEREBRA_PET = "cerebraPet"
CNPET = "cnpet"
CT_TEMPLATE = "ctTemplate"

MIT_TEXT = (
    "Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated "
    "documentation files (the \"Software\"), to deal in the Software without restriction, including without "
    "limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the "
    "Software, and to permit persons to whom the Software is furnished to do so, subject to the following "
    "conditions:\n\n"
    "The above copyright notice and this permission notice shall be included in all copies or substantial portions "
    "of the Software.\n\n"
    "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED "
    "TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL "
    "THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF "
    "CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER "
    "DEALINGS IN THE SOFTWARE.")

CNPET_CITATION = ("Wang H, Tian Y, Liu Y, et al. Population-specific brain [18F]-FDG PET templates of Chinese "
                  "subjects for statistical parametric mapping. Scientific Data 2021;8(1):305. "
                  "https://doi.org/10.1038/s41597-021-01089-1")

SOURCES = {
    ICBM152: {
        "title": "MNI ICBM 152 2009c nonlinear symmetric atlas",
        "about": ("The default template space of the Neuro workflows: T1 / T2 / PD templates, brain mask, grey and white "
                  "matter probability maps (1 mm). The synthetic PET template (GM + 0.3 WM, smoothed) is made from "
                  "its tissue maps."),
        "url": "https://nist.mni.mcgill.ca/icbm-152-nonlinear-atlases-2009/",
        "cite": [
            "Fonov VS, Evans AC, Botteron K, Almli CR, McKinstry RC, Collins DL, BDCG. Unbiased average "
            "age-appropriate atlases for pediatric studies. NeuroImage 2011;54(1):313-327. "
            "https://doi.org/10.1016/j.neuroimage.2010.07.033",
            "Fonov VS, Evans AC, McKinstry RC, Almli CR, Collins DL. Unbiased nonlinear average age-appropriate "
            "brain templates from birth to adulthood. NeuroImage 2009;47(Suppl 1):S102. "
            "https://doi.org/10.1016/S1053-8119(09)70884-5",
        ],
        "license": "Permissive license of the McConnell Brain Imaging Centre (copyright notice to be kept)",
        "licenseText": (
            "Copyright (C) 1993-2004 Louis Collins, McConnell Brain Imaging Centre, Montreal Neurological Institute, "
            "McGill University. Permission to use, copy, modify, and distribute this software and its documentation "
            "for any purpose and without fee is hereby granted, provided that the above copyright notice appear in "
            "all copies. The authors and McGill University make no representations about the suitability of this "
            "software for any purpose. It is provided \"as is\" without express or implied warranty. The authors "
            "are not responsible for any data loss, equipment damage, property loss, or injury to subjects or "
            "patients resulting from the use or misuse of this software package."),
    },
    CEREBRA: {
        "title": "CerebrA atlas",
        "about": ("102 cortical and subcortical regions (51 per side) of the Mindboggle-101 atlas, registered to the "
                  "ICBM 152 2009c template and corrected by hand. The bilateral (sides merged) version is made from "
                  "it by the extension."),
        "url": "https://nist.mni.mcgill.ca/cerebra/ (data: https://doi.org/10.12751/g-node.be5e62)",
        "cite": [
            "Manera AL, Dadar M, Fonov V, Collins DL. CerebrA, registration and manual label correction of "
            "Mindboggle-101 atlas for MNI-ICBM152 template. Scientific Data 2020;7:237.",
        ],
        "license": "Creative Commons CC0 1.0 (public domain dedication)",
        "licenseText": "CC0 1.0 Universal: https://creativecommons.org/publicdomain/zero/1.0/",
    },
    CEREBRA_PONS: {
        "title": "CerebrA with pons (made for this extension)",
        "about": ("The CerebrA labels with the pons drawn as a structure of its own (Pons R / L) and the rest of the "
                  "brainstem as Brainstem (except pons), no optic chiasm, by the extension's author (the CerebrA "
                  "segmentation of the extension). A derivative of CerebrA: cite CerebrA."),
        "url": "",
        "cite": [],
        "license": "CC0 1.0, as CerebrA",
        "licenseText": "",
    },
    CEREBRA_GROUPS: {
        "title": "CerebrA broad regions (part of the extension)",
        "about": ("CerebrA regions put together into broad lobar regions (Prefrontal, Lateral Occipital, ...) by "
                  "the segmentation's table groups.csv, or, for Broad (Compatible), into regions like those of GE "
                  "CortexID by groups_compatible.csv. A derivative of CerebrA: cite CerebrA."),
        "url": "",
        "cite": [],
        "license": "MIT (the extension's license)",
        "licenseText": "",
    },
    CEREBRA_PET: {
        "title": "CerebrA PET (made for this extension)",
        "about": ("Regions for PET by the extension's author: the CerebrA with pons, the brainstem without the pons, "
                  "the caudate edited by hand (the left one mirrored), no optic chiasm; cortical regions grown 3 mm, "
                  "cingulate, insula, basal ganglia, thalamus, amygdala and hippocampus 2 mm into the white matter "
                  "and the unlabelled brain (brainstem, pons, cerebellum, diencephalon not grown). Used with 0 mm "
                  "growth. A derivative of CerebrA: cite CerebrA."),
        "url": "",
        "cite": [],
        "license": "CC0 1.0, as CerebrA",
        "licenseText": "",
    },
    CT_TEMPLATE: {
        "title": "CT template of the SPM Clinical Toolbox (scct)",
        "about": ("Not bundled: used only when the user puts it into a space folder (CT, with its grey matter map "
                  "CT_GM, which puts its MNI space onto the space's)."),
        "url": "https://github.com/neurolabusc/Clinical",
        "cite": [
            "Rorden C, Bonilha L, Fridriksson J, Bender B, Karnath HO. Age-specific CT and MRI templates for "
            "spatial normalization. NeuroImage 2012;61(4):957-965.",
        ],
        "license": "BSD 2-Clause (see the toolbox)",
        "licenseText": "",
    },
    CNPET: {
        "title": "Chinese Brain PET Template (CNPET) dataset",
        "about": ("116 normal [18F]FDG brain PET images (Siemens Biograph 64 PET/CT, static 3 min acquisition 60 "
                  "min after injection, SUV images), shared by the Dalian University of Technology group on NITRC."),
        "url": "https://www.nitrc.org/projects/cnpet (data: https://doi.org/10.25790/bml0cm.95)",
        "cite": [CNPET_CITATION],
        "license": "MIT License, Copyright (c) 2021 DlutMedimgGroup",
        "licenseText": "MIT License\n\nCopyright (c) 2021 DlutMedimgGroup\n\n" + MIT_TEXT,
    },
}

_TEMPLATE_NOTES = {
    "T1": "T1 template", "T2": "T2 template", "PD": "PD template", "GM": "grey matter probability map",
    "WM": "white matter probability map", "CSF": "CSF probability map", "mask": "brain mask",
    "pseudoPET": "synthetic PET template, made by the extension from the grey and white matter maps",
}


# ---------------------------------------------------------------------------
# Which sources a choice uses
# ---------------------------------------------------------------------------
def _unique(keys):
    """The known source keys and the sources given as dicts (a user's manifest), each once."""
    out = []
    for key in keys:
        if isinstance(key, dict):
            if key.get("title") and key not in out:
                out.append(key)
        elif key in SOURCES and key not in out:
            out.append(key)
    return out


def manifestSources(manifest):
    """Sources of a space / segmentation manifest: its "sources" keys, else the manifest itself as a source (its
    name, description, url, cite, license)."""
    manifest = manifest or {}
    keys = [k for k in manifest.get("sources") or () if k in SOURCES]
    if keys:
        return keys
    cite = manifest.get("cite") or []
    own = {"title": str(manifest.get("name") or ""), "about": str(manifest.get("description") or ""),
           "url": str(manifest.get("url") or ""), "cite": [cite] if isinstance(cite, str) else list(cite),
           "license": str(manifest.get("license") or ""), "licenseText": ""}
    return [own] if own["title"] else []


def spaceSources(spaceId=None):
    from . import neuro
    info = neuro.spaceInfo(spaceId)
    return manifestSources(info["manifest"] if info else {"sources": [ICBM152]})


def templateSources(key, spaceId=None):
    """Sources of a normalization template (neuro.TEMPLATES keys, the CT template) of a space."""
    if not key:
        return []
    keys = spaceSources(spaceId)
    if key in (CT_TEMPLATE, "ctTemplateGm"):
        from . import neuro
        info = neuro.spaceInfo(spaceId)
        if info is None or not (info["manifest"].get("ctSources") or ()):
            return [CT_TEMPLATE] + keys
        return [k for k in info["manifest"]["ctSources"] if k in SOURCES] + keys
    return keys


def regionAtlasSources(kind, pons=False):
    """Sources of a region atlas key (neuro.regionAtlasKey): those of its segmentation's manifest, and the groups
    tables for the broad groupings."""
    if not kind:
        return []
    from . import neuro
    segmentation, grouping, _ = neuro.splitRegionAtlas(kind)
    info = neuro.segmentationInfo(segmentation)
    keys = manifestSources(info["manifest"]) if info else []
    if grouping != neuro.REGIONS and info and segmentation in (neuro.DEFAULT_SEGMENTATION, "CerebrA_PET"):
        keys.append(CEREBRA_GROUPS)
    return _unique(keys)


def referenceSources(referenceKey, name=""):
    """Sources of a reference region (a normal database key): a region of the region atlas, the whole brain or a
    mask file have no sources of their own (the atlas's are listed)."""
    return []


def databaseSources(header):
    """Atlases a normal database is built on: its template space and its region atlas. The data source of the
    subjects is in the database's own description."""
    header = header or {}
    keys = spaceSources(header.get("space")) + regionAtlasSources(header.get("regionAtlas"))
    norm = header.get("normalization") or {}
    if norm.get("template") in (CT_TEMPLATE,):
        keys = templateSources(CT_TEMPLATE, header.get("space")) + keys
    return _unique(keys)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
_URL = re.compile(r"(https?://[^\s<>()\"']+[^\s<>()\"'.,;:])")


def _text(text):
    """Plain text as HTML: escaped, links clickable, line breaks kept."""
    escaped = html.escape(str(text or ""), quote=False)
    linked = _URL.sub(lambda m: f"<a href=\"{m.group(1)}\">{m.group(1)}</a>", escaped)
    return linked.replace("\n", "<br>")


def sourceHtml(key, licenseText=True):
    source = key if isinstance(key, dict) else SOURCES[key]
    parts = [f"<b>{html.escape(source['title'])}</b>"]
    if source.get("about"):
        parts.append(_text(source["about"]))
    if source.get("url"):
        parts.append("Source: " + _text(source["url"]))
    if source.get("cite"):
        parts.append("<i>Please cite:</i><br>" + "<br>".join("&bull; " + _text(c) for c in source["cite"]))
    if source.get("license"):
        parts.append("<i>License:</i> " + _text(source["license"]))
    if licenseText and source.get("licenseText"):
        parts.append(f"<span style='font-size:small'>{_text(source['licenseText'])}</span>")
    return "<p>" + "<br>".join(parts) + "</p>"


def sourcesHtml(keys, lead="", licenseText=True):
    keys = _unique(keys)
    body = "".join(sourceHtml(k, licenseText) for k in keys)
    return (f"<p>{_text(lead)}</p>" if lead else "") + body


def templateHtml(key, title="", spaceId=None):
    note = _TEMPLATE_NOTES.get(key, "")
    lead = f"Template: {title or key}" + (f" ({note})." if note else ".") if key else ""
    return sourcesHtml(templateSources(key, spaceId), lead)


def regionsHtml(kind, referenceName="", title=""):
    lead = f"Regions: {title or kind}" + (f"; reference region: {referenceName}" if referenceName else "") + "."
    return sourcesHtml(regionAtlasSources(kind), lead)


def databaseHtml(header, emptyText="No normal database picked."):
    """The database's own description (data source, how it was made, citation, license of the data), then the
    atlases it is built on."""
    if not header:
        return f"<p>{_text(emptyText)}</p>"
    out = [f"<p><b>{html.escape(str(header.get('name') or ''))}</b></p>"]
    description = str(header.get("description") or "").strip()
    out.append(f"<p>{_text(description)}</p>" if description else
               "<p><i>No description in the database (Edit settings: Notes).</i></p>")
    out.append(sourcesHtml(databaseSources(header), "Atlases and templates the database is built on:"))
    return "".join(out)


def infoBox(height=110):
    """A read-only text box for these texts (Qt; links open in the browser)."""
    import qt
    box = qt.QTextBrowser()
    box.openExternalLinks = True
    box.readOnly = True
    box.maximumHeight = height
    box.toolTip = "Description, references to cite and license of what is picked"
    return box
