cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(NeuroPET)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/4burakfe/SlicerNeuroPET")
set(EXTENSION_CATEGORY "Nuclear Medicine")
set(EXTENSION_CONTRIBUTORS "Burak Demir")
set(EXTENSION_DESCRIPTION "Mergen: Neuro PET/SPECT Analysis. Brain PET and SPECT workflows built on the PET4D extension (SlicerPET4D): Dynamic Neuro Receptor PET (motion correction, CT/MR registration, normalization to MNI ICBM 152 2009c, CerebrA regions, kinetic modeling), Static Neuro PET/SPECT (normalization, regions, voxel z / t maps, 3D-SSP, regional scores against a normal database) and the Neuro PET Database Manager (user defined and predefined normal databases, iterative template building).")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/4burakfe/SlicerNeuroPET/main/Branding/Mergen_logo.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/4burakfe/SlicerNeuroPET/main/screenshot2.jpg https://raw.githubusercontent.com/4burakfe/SlicerNeuroPET/main/screenshot1.jpg https://raw.githubusercontent.com/4burakfe/SlicerNeuroPET/main/screenshot3.jpg")
set(EXTENSION_DEPENDS "PET4D") # the PET4D extension: its DICOM plugin, Motion Correction, Registration, Modeling

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(NeuroPET)

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
