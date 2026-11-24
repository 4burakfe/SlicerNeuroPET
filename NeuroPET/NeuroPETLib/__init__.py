"""Helper package of Mergen: Neuro PET/SPECT Analysis (the SlicerNeuroPET extension).

    neuro       Dynamic Neuro Receptor PET helpers: MNI ICBM 152 2009c atlas files, MR contrast (T1 / T2 / PD),
                CerebrA labels (left / right and bilateral) (numpy only)
    normaldb    Neuro PET Database Manager: normal database folders (database.json, subjects, mean / SD maps), age
                groups, predefined databases and their .zip packages (numpy only)
    neuroreview Static Neuro Review: voxel z / Crawford-Howell t maps in a mask with an SD floor, regional normal
                statistics (database, on the fly with a cache, approximate), score colours, the review layout
                (numpy only)
    neuroreport the PDF report of a review: scan values from DICOM (uptake time, activity), its rows, the 3D-SSP drawn
                off screen (VTK) and the A4 PDF (Qt: QPdfWriter / QPainter)
    ssp         3D-SSP (Minoshima): projection along the inward surface normals, vertex scores, the normals'
                projections (.npz), surface colours (numpy + scipy)
    staticneuro Static Neuro PET/SPECT helpers: what a static brain image is (PET / SPECT, units, nuclide, tracer
                from the loaders' tags) and which normal databases fit it (no Slicer)
    templatebuild   iterative study-specific template of a normal database (passes, drift correction, stop rule)
                (numpy only)

The modules of this extension also use the PET4D extension (SlicerPET4D, its DICOMGatedDynamicPETLib
package and modules), which is a dependency of this one: requireBaseExtension() makes sure it can be imported. These
helpers themselves do not need it (the tests in Testing/ run without Slicer and without PET4D).
"""

import os
import sys

BASE_EXTENSION = "PET4D"                        # the extension name (EXTENSION_DEPENDS)
OLD_BASE_EXTENSION = "DICOMGatedDynamicPET"     # its name before the rename to PET4D (older installs)
BASE_REPOSITORIES = ("SlicerPET4D", "SlicerDICOMGatedDynamicPET")   # clone folder names, new and old
BASE_MODULE = "DICOMGatedDynamicPET"            # the PET4D extension's DICOM plugin module (hidden, name kept)
BASE_PACKAGE = "DICOMGatedDynamicPETLib"        # its helper package, next to that module


def _importable(name):
    try:
        __import__(name)
        return True
    except ImportError:
        return False


MODULE_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # this extension's modules


def baseModuleFolders():
    """Folders that may hold the PET4D extension's modules, most likely first: that of its DICOM plugin module when
    Slicer has loaded it, the additional module paths of the application settings (extensions and folders added by
    hand), the installed PET4D next to this installed extension (<extensions>/NeuroPET/lib/Slicer-X.Y/
    qt-scripted-modules -> <extensions>/PET4D/lib/Slicer-X.Y/qt-scripted-modules, or DICOMGatedDynamicPET, its old
    name) and the source folder of a SlicerPET4D (or SlicerDICOMGatedDynamicPET) clone next to the SlicerNeuroPET
    one."""
    folders = []
    try:
        import slicer
    except ImportError:
        slicer = None
    if slicer is not None:
        try:
            module = getattr(slicer.modules, BASE_MODULE.lower(), None)
            if module is not None and module.path:
                folders.append(os.path.dirname(str(module.path)))
        except Exception:       # noqa: BLE001 - only a lookup
            pass
        for settings in ("revisionUserSettings", "userSettings"):
            try:
                paths = getattr(slicer.app, settings)().value("Modules/AdditionalPaths") or []
                if isinstance(paths, str):
                    paths = [paths]
                folders.extend(str(p) for p in paths)
            except Exception:   # noqa: BLE001 - only a lookup
                pass
    parts = os.path.normpath(MODULE_FOLDER).split(os.sep)
    if len(parts) > 4 and parts[-2].lower().startswith("slicer-") and parts[-3] == "lib":
        for extension in (BASE_EXTENSION, OLD_BASE_EXTENSION):
            folders.append(os.sep.join(parts[:-4] + [extension] + parts[-3:]))
    clones = os.path.dirname(os.path.dirname(MODULE_FOLDER))
    for repository in BASE_REPOSITORIES:
        folders.append(os.path.join(clones, repository, BASE_MODULE))
    return folders


def requireBaseExtension():
    """Make the PET4D extension (its DICOMGatedDynamicPET module) importable: Slicer may have its module folder on
    sys.path already; otherwise the folder is looked for (baseModuleFolders) and put on sys.path, so this works
    whatever order Slicer reads the modules in. Raises ImportError with a clear message when PET4D is not found."""
    if _importable(BASE_PACKAGE):
        return
    for folder in baseModuleFolders():
        if folder and os.path.isdir(os.path.join(folder, BASE_PACKAGE)) and folder not in sys.path:
            sys.path.insert(0, folder)
            if _importable(BASE_PACKAGE):
                return
    raise ImportError("Mergen (Neuro PET/SPECT Analysis) needs the PET4D extension (SlicerPET4D): "
                      "install PET4D from the Extensions Manager, or add its DICOMGatedDynamicPET folder "
                      "to the additional module paths, and restart Slicer.")
