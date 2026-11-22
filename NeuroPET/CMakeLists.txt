#-----------------------------------------------------------------------------
# Mergen: Neuro PET/SPECT Analysis. The modules use the PET4D extension (SlicerPET4D; module DICOMGatedDynamicPET, DICOMGatedDynamicPETLib
# and its modules), a dependency of this extension (EXTENSION_DEPENDS).

#-----------------------------------------------------------------------------
# Dynamic Neuro Receptor PET (visible, Nuclear Medicine): workflow app (data, motion correction, CT/MR
# registration, normalization to MNI ICBM 152 2009c, CerebrA regions, modeling). Carries the helper package and the
# atlas files of the extension.
set(NEURO_MODULE_NAME DynamicNeuroPET)

set(NEURO_LIB_SCRIPTS
  NeuroPETLib/__init__.py
  NeuroPETLib/banner.py
  NeuroPETLib/credits.py
  NeuroPETLib/dbdownload.py
  NeuroPETLib/dynamicneuro.py
  NeuroPETLib/neuro.py
  NeuroPETLib/neuroreview.py
  NeuroPETLib/normaldb.py
  NeuroPETLib/peel.py
  NeuroPETLib/ssp.py
  NeuroPETLib/staticneuro.py
  NeuroPETLib/templatebuild.py
  )

# the workflow shared by the Dynamic and Static Neuro modules (logic and widget base classes)
set(NEURO_WORKFLOW_LIB_SCRIPTS
  NeuroWorkflowLib/__init__.py
  NeuroWorkflowLib/base.py
  NeuroWorkflowLib/compare.py
  NeuroWorkflowLib/database.py
  NeuroWorkflowLib/scenestate.py
  )

slicerMacroBuildScriptedModule(
  NAME ${NEURO_MODULE_NAME}
  SCRIPTS
    ${NEURO_MODULE_NAME}.py
    ${NEURO_LIB_SCRIPTS}
    ${NEURO_WORKFLOW_LIB_SCRIPTS}
  RESOURCES
    Resources/Icons/${NEURO_MODULE_NAME}.png
    Resources/Images/banner.jpg
    Resources/Images/logo.png
    # the atlas files that come with the extension (their licenses: Resources/Atlas/LICENSE.txt): the template space
    # (Space/<id>) and the region segmentations (Segmentation/<id>), see NeuroPETLib/neuro.py
    Resources/Atlas/LICENSE.txt
    Resources/Atlas/Space/MNI152NLin2009cSym/space.json
    Resources/Atlas/Space/MNI152NLin2009cSym/T1.nrrd
    Resources/Atlas/Space/MNI152NLin2009cSym/mask.nrrd
    Resources/Atlas/Space/MNI152NLin2009cSym/GM.nrrd
    Resources/Atlas/Space/MNI152NLin2009cSym/WM.nrrd
    Resources/Atlas/Segmentation/CerebrA/segmentation.json
    Resources/Atlas/Segmentation/CerebrA/labels.seg.nrrd
    Resources/Atlas/Segmentation/CerebrA/labels.csv
    Resources/Atlas/Segmentation/CerebrA/groups.csv
    Resources/Atlas/Segmentation/CerebrA/groups_compatible.csv
    Resources/Atlas/Segmentation/CerebrA_PET/segmentation.json
    Resources/Atlas/Segmentation/CerebrA_PET/labels.seg.nrrd
    Resources/Atlas/Segmentation/CerebrA_PET/labels.csv
    Resources/Atlas/Segmentation/CerebrA_PET/groups.csv
    Resources/Atlas/Segmentation/CerebrA_PET/groups_compatible.csv
  WITH_GENERIC_TESTS
  )

if(BUILD_TESTING)
  slicer_add_python_unittest(SCRIPT ${NEURO_MODULE_NAME}.py)
endif()

#-----------------------------------------------------------------------------
# Neuro PET Database Manager (visible, Nuclear Medicine): normal databases of brain PET and SPECT (static
# first; predefined from mean / SD maps or a package, user defined filled from the Neuro workflows)
set(NDB_MODULE_NAME NeuroPETDatabaseManager)

slicerMacroBuildScriptedModule(
  NAME ${NDB_MODULE_NAME}
  SCRIPTS ${NDB_MODULE_NAME}.py
  RESOURCES Resources/Icons/${NDB_MODULE_NAME}.png
  WITH_GENERIC_TESTS
  )

if(BUILD_TESTING)
  slicer_add_python_unittest(SCRIPT ${NDB_MODULE_NAME}.py)
endif()

#-----------------------------------------------------------------------------
# Static Neuro PET/SPECT (visible, Nuclear Medicine): workflow app for a static brain PET / SPECT (data,
# optional CT/MR registration, normalization, regions with the normal database, review)
set(STATIC_NEURO_MODULE_NAME StaticNeuroPET)

# its logic and widget methods, one file per step / topic (mixins put together in StaticNeuroPET.py)
set(STATIC_NEURO_LIB_SCRIPTS
  StaticNeuroPETLib/__init__.py
  StaticNeuroPETLib/constants.py
  StaticNeuroPETLib/helpers.py
  StaticNeuroPETLib/logic_image.py
  StaticNeuroPETLib/logic_database.py
  StaticNeuroPETLib/logic_review.py
  StaticNeuroPETLib/logic_batch.py
  StaticNeuroPETLib/toolbar.py
  StaticNeuroPETLib/page_data.py
  StaticNeuroPETLib/page_registration.py
  StaticNeuroPETLib/page_normalization.py
  StaticNeuroPETLib/page_regions.py
  StaticNeuroPETLib/review_page.py
  StaticNeuroPETLib/review_colors.py
  StaticNeuroPETLib/review_table.py
  StaticNeuroPETLib/review_ssp.py
  StaticNeuroPETLib/review_cut.py
  StaticNeuroPETLib/review_peel.py
  StaticNeuroPETLib/review_layout.py
  StaticNeuroPETLib/review_mouse.py
  )

slicerMacroBuildScriptedModule(
  NAME ${STATIC_NEURO_MODULE_NAME}
  SCRIPTS
    ${STATIC_NEURO_MODULE_NAME}.py
    ${STATIC_NEURO_LIB_SCRIPTS}
  RESOURCES Resources/Icons/${STATIC_NEURO_MODULE_NAME}.png
  WITH_GENERIC_TESTS
  )

if(BUILD_TESTING)
  slicer_add_python_unittest(SCRIPT ${STATIC_NEURO_MODULE_NAME}.py)
endif()
