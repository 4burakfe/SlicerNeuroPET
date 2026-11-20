"""Tests of the source / reference / license texts without Slicer: python -m pytest Testing."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "NeuroPET"))
from NeuroPETLib import credits  # noqa: E402


def test_sources_of_choices():
    assert credits.templateSources("T1") == [credits.ICBM152]
    assert credits.templateSources("pseudoPET") == [credits.ICBM152]
    assert credits.templateSources("ctTemplate")[0] == credits.CT_TEMPLATE
    assert credits.templateSources("") == []
    assert credits.regionAtlasSources("CerebrA:regions") == [credits.ICBM152, credits.CEREBRA, credits.CEREBRA_PONS]
    assert credits.CEREBRA_GROUPS in credits.regionAtlasSources("CerebrA:broad")
    assert credits.CEREBRA_GROUPS in credits.regionAtlasSources("CerebrA:compatible")
    assert credits.regionAtlasSources("CerebrA_PET:regions") == [credits.ICBM152, credits.CEREBRA,
                                                                 credits.CEREBRA_PET]
    assert credits.CEREBRA_GROUPS in credits.regionAtlasSources("CerebrA_PET:compatible")
    assert credits.regionAtlasSources("Nope:regions") == [] and credits.regionAtlasSources("") == []
    assert credits.referenceSources("custom") == [] and credits.referenceSources("atlasRegion", "Pons") == []


def test_manifest_sources():
    """A manifest of the user's without source keys: shown by its own fields."""
    own = credits.manifestSources({"name": "My atlas", "description": "drawn by hand", "cite": "Me 2026",
                                   "license": "CC BY 4.0"})
    assert len(own) == 1 and own[0]["title"] == "My atlas" and own[0]["cite"] == ["Me 2026"]
    text = credits.sourcesHtml(own)
    assert "My atlas" in text and "drawn by hand" in text and "CC BY 4.0" in text and "Me 2026" in text
    assert credits.manifestSources({"sources": ["cerebra", "nope"]}) == [credits.CEREBRA]
    assert credits.manifestSources({}) == []


def test_database_html():
    header = {"name": "CN FDG <test>", "space": "MNI152NLin2009cSym", "regionAtlas": "CerebrA:broad",
              "referenceRegion": "atlasRegion", "referenceStructure": "Pons",
              "description": "Rebuilt.\nPlease cite: https://doi.org/10.1038/s41597-021-01089-1."}
    text = credits.databaseHtml(header)
    assert "CN FDG &lt;test&gt;" in text
    assert '<a href="https://doi.org/10.1038/s41597-021-01089-1">' in text      # the final dot left out
    assert "Rebuilt.<br>Please cite" in text
    for key in (credits.ICBM152, credits.CEREBRA, credits.CEREBRA_GROUPS, credits.CEREBRA_PONS):
        assert credits.SOURCES[key]["title"].split(" (")[0] in text
    assert "No description" in credits.databaseHtml({"name": "x"})
    assert "nothing" in credits.databaseHtml(None, "nothing")


def test_licenses_kept():
    icbm = credits.sourcesHtml([credits.ICBM152])
    assert "Louis Collins" in icbm and "Fonov" in icbm
    assert "CC0" in credits.sourcesHtml([credits.CEREBRA]) and "Manera" in credits.sourcesHtml([credits.CEREBRA])
    cnpet = credits.SOURCES[credits.CNPET]
    assert "Copyright (c) 2021 DlutMedimgGroup" in cnpet["licenseText"]
    assert "Wang H" in cnpet["cite"][0] and "8(1):305" in cnpet["cite"][0]
    assert "Permission is hereby granted" in credits.MIT_TEXT
