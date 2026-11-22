"""
The workflow saved with the scene (Save Scene, .mrb) and opened again when that scene is loaded.

    - the step on screen and whether the module was shown are kept in the module's parameter node (saved with the
      scene); the series, CT/MR, template, regions and the review's score map and table are node references there
    - the layouts of the workflow are registered at startup, so a scene saved in one of them opens its views (Slicer
      does not save layout descriptions: an unknown layout shows no views)
    - after a scene is loaded the module is shown again on the saved step, its views built again (the pattern of
      SlicerPETReviewCompare / Epona Compare: never inside a scene notification, from the event loop once the scene
      has been idle for a while)

Helper nodes of the views (3D-SSP models, colored copies, view helpers) are not saved: they are made again.
"""

import logging

import qt
import slicer

STEP_PARAMETER = "DynamicPET.Neuro.Step"            # the step on screen
SHOWN_PARAMETER = "DynamicPET.Neuro.Shown"          # "1" while the module is shown
NEURO_MODULES = ("DynamicNeuroPET", "StaticNeuroPET")

# PET4D's layouts the workflow steps open (Motion Correction, Registration, Modeling): known at startup here too
PET4D_LAYOUTS = (7802, 7803, 7804, 7805)       # their modules' LAYOUT_IDs

SCENE_POLL_MS = 100
SCENE_SETTLE_MS = 300
SCENE_MAX_WAIT_MS = 10 * 60 * 1000


def sceneIsBusy():
    scene = slicer.mrmlScene
    return bool(scene.IsImporting() or scene.IsClosing() or scene.IsBatchProcessing())


def runWhenSceneSettled(callback, settleMs=SCENE_SETTLE_MS):
    """Run callback from the event loop once the scene has been idle for settleMs (dropped when it stays busy for
    SCENE_MAX_WAIT_MS)."""
    state = {"elapsed": 0, "idleSince": None}

    def poll():
        if sceneIsBusy():
            state["idleSince"] = None
            if state["elapsed"] >= SCENE_MAX_WAIT_MS:
                logging.warning("Neuro: the scene stayed busy, the workflow was not restored")
                return
        else:
            if state["idleSince"] is None:
                state["idleSince"] = state["elapsed"]
            if state["elapsed"] - state["idleSince"] >= settleMs:
                try:
                    callback()
                except Exception:
                    logging.exception("Neuro: could not restore the workflow after loading the scene")
                return
        state["elapsed"] += SCENE_POLL_MS
        qt.QTimer.singleShot(SCENE_POLL_MS, poll)

    qt.QTimer.singleShot(0, poll)


def parameterNode(moduleName):
    """The module's parameter node in the scene, None when there is none (not made here)."""
    try:
        return slicer.mrmlScene.GetSingletonNode(moduleName, "vtkMRMLScriptedModuleNode")
    except Exception:
        return None


def currentLayout():
    layoutManager = slicer.app.layoutManager()
    return layoutManager.layout if layoutManager is not None else None


def addLayout(layoutId, description):
    """A layout made known to Slicer (left as it is when it is known already: the module that shows it updates it)."""
    layoutManager = slicer.app.layoutManager()
    if layoutManager is None:
        return
    layoutNode = layoutManager.layoutLogic().GetLayoutNode()
    if not layoutNode.IsLayoutDescription(layoutId):
        layoutNode.AddLayoutDescription(layoutId, description)


def registerPet4dLayouts():
    """The layouts of the PET4D modules the steps show (placeholders until a step opens them with its own views)."""
    from DICOMGatedDynamicPETLib import dynamicviews
    from DICOMGatedDynamicPETLib import registration as reg
    try:
        import DynamicPETRegistration as dpr
        addLayout(dpr.LAYOUT_ID, reg.layoutXml(False))
        addLayout(dpr.CHECKER_LAYOUT_ID, reg.layoutXml(True))
    except Exception:
        logging.debug("Neuro: could not register the registration layouts", exc_info=True)
    for moduleName, kwargs in (("DynamicPETMotionCorrection", dict(mainLabel="Ref")),
                               ("DynamicPETModeling", dict(mainLabel="S"))):
        try:
            module = __import__(moduleName)
            if moduleName == "DynamicPETModeling":
                kwargs["mainShare"] = module.MAIN_SHARE
            addLayout(module.LAYOUT_ID, dynamicviews.layoutXml(
                16, orientation=module.DEFAULT_ORIENTATION, mainView=module.MAIN_VIEW,
                framePrefix=module.FRAME_VIEW_PREFIX, emptyPrefix=module.EMPTY_VIEW_PREFIX,
                viewGroup=module.VIEW_GROUP, **kwargs))
        except Exception:
            logging.debug(f"Neuro: could not register the layout of {moduleName}", exc_info=True)


def install(moduleName, registerLayouts=None, layoutSteps=None):
    """Called by the module (at startup) and its panel (when made): the module's layouts registered and scene loads
    observed (once; a developer reload replaces the observer). layoutSteps {layout ID: step} names the step of a
    scene saved before the step was kept (by the layout it was saved in)."""
    for register in (registerPet4dLayouts, registerLayouts):
        if register is None:
            continue
        try:
            register()
        except Exception:
            logging.exception(f"{moduleName}: could not register the layouts")
    _observeImports(moduleName, layoutSteps or {})


def installAtStartup(moduleName, registerLayouts=None, layoutSteps=None):
    """install() once Slicer has started (the layout manager and the other modules are there then)."""
    slicer.app.connect("startupCompleted()", lambda: install(moduleName, registerLayouts, layoutSteps))


def _observeImports(moduleName, layoutSteps):
    attribute = f"_{moduleName}SceneRestoreObserver"
    old = getattr(slicer.modules, attribute, None)
    if isinstance(old, int):
        slicer.mrmlScene.RemoveObserver(old)

    def onImported(caller, event):
        runWhenSceneSettled(lambda: restoreModule(moduleName, layoutSteps))

    tag = slicer.mrmlScene.AddObserver(slicer.mrmlScene.EndImportEvent, onImported)
    try:
        setattr(slicer.modules, attribute, tag)
    except Exception:
        pass


def savedStep(moduleName, layoutSteps=None):
    """(step, shown) the scene was saved with; for a scene saved before they were kept, the step of the layout it
    was saved in (when only this Neuro module has a series in the scene), shown then."""
    node = parameterNode(moduleName)
    if node is None:
        return None, False
    step = node.GetParameter(STEP_PARAMETER)
    if step:
        return step, node.GetParameter(SHOWN_PARAMETER) == "1"
    step = (layoutSteps or {}).get(currentLayout())
    if not step:
        return None, False
    withSeries = [name for name in NEURO_MODULES if parameterNode(name) is not None
                  and parameterNode(name).GetNodeReference("DynamicPET.Neuro.Sequence") is not None]
    return (step, True) if withSeries == [moduleName] else (None, False)


def restoreModule(moduleName, layoutSteps=None):
    """A scene was loaded: the module on the step it was saved on, shown again when it was shown."""
    step, shown = savedStep(moduleName, layoutSteps)
    if not step:
        return
    logging.info(f"{moduleName}: scene loaded, back on the '{step}' step")
    widget = getattr(slicer.modules, moduleName + "Widget", None)
    if widget is not None:
        widget.restoreFromScene(step)       # shown: the step opened now, else when the module is shown
    if shown:
        try:
            if slicer.util.selectedModule() != moduleName:
                slicer.util.selectModule(moduleName)        # a new panel takes the saved step in setup()
        except Exception:
            logging.exception(f"{moduleName}: could not show the module after loading the scene")


def unsaved(node):
    """A helper node (and its display nodes) left out of the saved scene: it is made again from the saved nodes."""
    if node is None:
        return node
    node.SetSaveWithScene(False)
    for index in range(node.GetNumberOfDisplayNodes() if hasattr(node, "GetNumberOfDisplayNodes") else 0):
        display = node.GetNthDisplayNode(index)
        if display is not None:
            display.SetSaveWithScene(False)
    return node


def dropOrphanDisplayNodes():
    """Before saving: the display nodes of nodes that are not saved are not saved either (else the scene file keeps
    display nodes of nothing, e.g. of PET4D's helper volumes)."""
    for node in slicer.util.getNodesByClass("vtkMRMLDisplayableNode"):
        if not node.GetSaveWithScene():
            unsaved(node)
