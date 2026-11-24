"""
Constants of the Static Neuro PET/SPECT module: the steps, settings keys, node attributes, layouts and the
review / 3D-SSP display defaults.
"""

import os

from DICOMGatedDynamicPETLib import registration as reg
from NeuroWorkflowLib.base import (METHOD_ANATOMY, METHOD_PET, METHOD_PET_TEMPLATE, NEURO_PREFIX, NORMALIZATION_GRIDS,
                                   STEP_DATA, STEP_NORM, STEP_REG, STEP_REGIONS)
from NeuroPETLib import neuroreview as nr
# the normal database's (shared with the Dynamic Neuro module): kept importable from here
from NeuroWorkflowLib.database import (DATABASE_PARAMETER, NORMALIZATION_ATTRIBUTE, REGIONS_COLUMNS,  # noqa: F401
                                       REGIONS_LAYOUT_ID, REGIONS_ORIENTATION_COLORS, REGIONS_ORIENTATIONS,
                                       REGIONS_VIEW_GROUP, SETTINGS_DB_ROOT, WARNING_STYLE)

_MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # the StaticNeuroPET module folder

STEP_REVIEW = "review"
STEPS = (
    (STEP_DATA, None, "Data", "Static PET / SPECT of the brain and the patient's CT / MR"),
    (STEP_REG, None, "CT/MR registration", "Optional: the image registered to the CT / MR"),
    (STEP_NORM, None, "Normalization", "Normal database and the MNI ICBM 152 2009c space"),
    (STEP_REGIONS, None, "Regions", "CerebrA regions (as in the normal database)"),
    (STEP_REVIEW, None, "Review", "Comparison with the normal database"),
)
METHODS = ((METHOD_PET, "PET/SPECT to MR template (T1)"),
           (METHOD_ANATOMY, "CT/MR to MR template, same transform to the PET/SPECT (a CT: to the CT template if any)"),
           (METHOD_PET_TEMPLATE, "PET/SPECT to PET template (normal database mean)"))
# application settings keys: kept from the PET4D extension (the neuro modules came from it), so the folders stay picked
SETTINGS_IMPORT_FOLDER = "DICOMGatedDynamicPET/StaticNeuro/ImportFolder"
SETTINGS_DISPLAY = "DICOMGatedDynamicPET/StaticNeuro/DisplaySettings"     # JSON of the saved review display settings
SETTINGS_REPORT = "DICOMGatedDynamicPET/StaticNeuro/Report"     # JSON: the PDF report's folder, checks
AGE_ATTRIBUTE, SEX_ATTRIBUTE = NEURO_PREFIX + "PatientAge", NEURO_PREFIX + "PatientSex"
NAME_ATTRIBUTE, ID_ATTRIBUTE = NEURO_PREFIX + "PatientName", NEURO_PREFIX + "PatientID"
SOURCE_ATTRIBUTE = NEURO_PREFIX + "Source"      # a CT / MR normalized with the image (the volume it was made from)
# Harden of the normalization: the template grid and Overwrite by default
NORM_HARDEN_GRIDS = tuple(sorted(NORMALIZATION_GRIDS, key=lambda option: option[1] != reg.OUTPUT_FIXED_GRID))
NORM_HARDEN_ACTIONS = (("Overwrite", reg.OVERWRITE), ("Save separately", reg.SAVE_SEPARATELY))
REG_LINEAR = "Rigid"                    # CT/MR registration: the patient's own CT / MR, rigid by default (affine selectable)
STEP_ICON_SIZE = 22                     # px, the height of a step's icon (its pictogram and its state badge)
STEP_GLYPH_COLOR = "#5b6770"
REVIEW_SCORE_WINDOWS = (2.0, 3.0, 4.0, 5.0, 7.0, 10.0)     # the score windows of the review toolbar (+/- SD)
FUSION_TEMPLATES = ("T1", "T2", "PD")   # the MNI templates offered for the fusion of the review
TEMPLATE_KEY_ATTRIBUTE = NEURO_PREFIX + "TemplateKey"       # the template key of a loaded template ("T1", "db:CT")
# A batch into a database normalized CT/MR: what to do with a subject without a CT / MR
BATCH_PET_FALLBACK, BATCH_SKIP = "pet", "skip"
REGISTERED_SUFFIX = " registered"
ICON_PATH = os.path.join(_MODULE_DIR, "Resources", "Icons", "StaticNeuroPET.png")
REPORT_BANNER_PATH = os.path.join(_MODULE_DIR, "Resources", "Images", "report_banner.jpg")    # top of the PDF report
REVIEW_PET_COLORMAP = "Inverted grey"                  # the top row (the normalized PET / SPECT)
REVIEW_SSP_PATIENT_COLORMAP = "PET-DICOM"       # the 3D-SSP of the patient's projected values
REVIEW_PEEL_PET_COLORMAP = "Hel"          # the PET of the unfolded brain maps (and its fusion)
REVIEW_SCORE_COLORMAP = nr.SCORE_COLORMAPS[0]     # diverging: below the normals cold, above hot
SCORE_COLOR_NODE = "Neuro review score colors"
# no scores (no database compared): the image in these colors in the score slices, with NO_SCORE_NOTE in a corner
NO_SCORE_COLORMAP = "SSP_Rainbow"
NO_SCORE_COLOR_NODE = "Neuro review no-score colors"
NO_SCORE_NOTE = "No database compared: the image in rainbow colors, no scores"
REVIEW_FUSION_OPACITY = 0.6
REVIEW_SCORE_OPACITY = 0.5              # the score map over the MR (fusion): semi-transparent
REVIEW_OUTLINE_THICKNESS = 1            # px, the region outlines on the slices
REVIEW_FLASH_MS = 5000                  # a region clicked in the table: shown alone for 5 s
MONITOR2_PLACE_DELAY_MS = 300           # the Monitor 2 window placed once the layout manager made its views
SSP_TARGET_POINTS = 60000               # vertices of the 3D-SSP surface
SSP_MAX_EDGE_MM = 4.0                   # its triangles' longest edge: the surface is contoured on a coarser grid (not
#                                         decimated: long triangles across the hemispheres' flat midline faces; nor
#                                         split again after it: vtkAdaptiveSubdivisionFilter leaves T-junctions, the
#                                         surface in thousands of pieces whose normals point in or out at random)
SSP_MODEL_NAME = "Neuro 3D-SSP surface"
SSP_OUTLINE_NAME = "Neuro 3D-SSP region outline"
SSP_OUTLINE_ATTRIBUTE = NEURO_PREFIX + "SSPOutline"
SSP_OUTLINE_COLOR = (1.0, 1.0, 1.0)     # white
SSP_OUTLINE_WIDTH = 2.0                 # px
SSP_OUTLINE_LIFT_MM = 0.6               # the outline this far outside the surface: drawn over it
SSP_PICK_MM = 4.0                       # the mouse on the surface: a vertex this near the picked point
MOUSE_PRIORITY = 100.0                  # the review's mouse observers before Slicer's interaction
CLICK_TOLERANCE_PX = 3                  # a right press moved less than this: a click (Slicer's menu), not a drag
WINDOW_DRAG_MS = 60                     # a right drag's window applied at most this often
SSP_MODEL_ATTRIBUTE = NEURO_PREFIX + "SSPModel"     # (view set)/(surface part) of the review's 3D-SSP models
SSP_COLOR_NODE = "Neuro 3D-SSP colors"
CUT_MODEL_NAME = "Neuro cut 3D-SSP"
PEEL_NODE_PREFIX = "Neuro peel"
PEEL_ATTRIBUTE = NEURO_PREFIX + "Peel"          # the peel layout's 1-slice volumes (and its label colors)
PEEL_DEPTHS = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 13.0, 15.0, 20.0, 25.0)   # the toolbar's choices, mm
CUT_MODEL_ATTRIBUTE = NEURO_PREFIX + "CutModel"     # the cut 3D-SSP layout's models: surface / face of a view / row
CUT_UPDATE_DELAY_MS = 30                # a slice moved: its cut face made again this long after (not at every event)
SSP_LOW_COLOR_NODE = "Neuro 3D-SSP low score colors"
SSP_CAMERA_DISTANCE = 600.0             # mm from the brain centre (parallel projection)
SSP_SIDE_MARKER_X = (0.17, 0.92)        # the R / L markers of the 3D-SSP views (normalized x; the left one right of
#                                         the color legend), at mid height
SSP_SETUP_DELAY_MS = 500                # the 3D-SSP cameras set again once the new layout's views are drawn
