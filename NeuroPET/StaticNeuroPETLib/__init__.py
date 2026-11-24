"""
StaticNeuroPETLib: the parts of the Static Neuro PET/SPECT module (StaticNeuroPET.py), one file per step / topic.

    constants            steps, settings keys, node attributes, layouts, review / 3D-SSP defaults
    helpers              numbers in logs, screens (second monitor), step icons
    logic_image          the image, hardening, CT/MR registration, normalization
    logic_database       adding a patient to a database, native images, regions and their values
    logic_review         the comparison with the database, 3D-SSP
    logic_batch          batch add, iterative template, QC pictures
    toolbar              the workflow toolbar and the review tool menus
    page_data            step 1: data
    page_registration    step 2: CT/MR registration
    page_normalization   step 3: normalization and the normal database
    page_regions         step 4: regions and the regions views
    review_page          step 5: review page, display settings, Compare, add to database
    review_colors        score maps and colours, PET / fusion display
    review_table         the region table
    review_ssp           3D-SSP views
    review_cut           cut 3D-SSP
    review_peel          unfolded brain
    review_layout        layouts, second monitor, review views, outlines
    review_mouse         hover names, right drag window, 3D-SSP outlines
    report               the PDF report (its layout and drawing: NeuroPETLib.neuroreport)

The logic and widget files hold mixin classes (methods only); StaticNeuroPET.py makes StaticNeuroPETLogic and
StaticNeuroPETWidget from them. Pure numpy code that can be tested without Slicer stays in NeuroPETLib.
"""

from NeuroPETLib import requireBaseExtension

requireBaseExtension()     # PET4D (DICOMGatedDynamicPETLib) importable before any file here is imported

# in reload order (a file before the files that import it)
MODULES = ("constants", "helpers", "logic_image", "logic_database", "logic_review", "logic_batch", "toolbar",
           "page_data", "page_registration", "page_normalization", "page_regions", "review_page", "review_colors",
           "review_table", "review_ssp", "review_cut", "review_peel", "review_layout", "review_mouse",
           "report")
