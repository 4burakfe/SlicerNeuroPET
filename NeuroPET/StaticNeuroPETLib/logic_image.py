"""
The image of the workflow: its attributes, patient, age and sex, loading from a file, hardening, the
CT/MR registration and the normalization.

Part of the Static Neuro PET/SPECT module (StaticNeuroPET.py): the methods of ImageLogicMixin, mixed into
StaticNeuroPETLogic there.
"""

import json
import logging
import os

import slicer

from DICOMGatedDynamicPETLib import registration as reg, resampling
from NeuroWorkflowLib.base import (ATLAS_FILE_ATTRIBUTE, METHOD_PET, NEURO_PREFIX, NORMALIZED_SUFFIX, REG_PREFIX,
                                   _ijkToRas, _percentiles, _vtkMatrix)
from NeuroPETLib import staticneuro as sn

from StaticNeuroPETLib.constants import AGE_ATTRIBUTE, ID_ATTRIBUTE, NAME_ATTRIBUTE, REGISTERED_SUFFIX, SEX_ATTRIBUTE


class ImageLogicMixin:
    """The logic image methods of StaticNeuroPETLogic."""

    # --- the image -----------------------------------------------------------------------------
    @staticmethod
    def firstImage(volume):
        return volume

    @staticmethod
    def displayImage(volume):
        return volume

    @staticmethod
    def referenceImage(volume):
        return volume

    @staticmethod
    def attributes(node):
        return {name: node.GetAttribute(name) for name in (node.GetAttributeNames() or ())} if node else {}

    @classmethod
    def imageInfo(cls, node):
        """staticneuro.imageInfo() of a volume (its tags, else the DICOM modality)."""
        if node is None:
            return sn.imageInfo({})
        attributes = cls.attributes(node)
        modality = ""
        if not any(k.startswith((sn.STATIC_PREFIX, sn.SPECT_PREFIX)) for k in attributes):
            modality = attributes.get("DICOM.Modality") or cls.dicomValues(node, ("0008,0060",))["0008,0060"]
        return sn.imageInfo(attributes, modality)

    @classmethod
    def ageAndSex(cls, node):
        """(age in years or None, sex) of the patient: the values entered on the Data page, else the loaders' tags,
        else the DICOM tags."""
        if node is None:
            return None, ""
        attributes = cls.attributes(node)
        age = sn.parseAge(attributes.get(AGE_ATTRIBUTE))
        sex = sn.parseSex(attributes.get(SEX_ATTRIBUTE))
        if age is None or not sex:
            taggedAge, taggedSex = sn.taggedAgeSex(attributes)
            if age is None and taggedAge is None or not sex and not taggedSex:
                values = cls.dicomValues(node, ("0010,1010", "0010,0040"))
                taggedAge = taggedAge if taggedAge is not None else sn.parseAge(values["0010,1010"])
                taggedSex = taggedSex or sn.parseSex(values["0010,0040"])
            age = age if age is not None else taggedAge
            sex = sex or taggedSex
        return age, sex

    @staticmethod
    def setAgeAndSex(node, age, sex):
        if node is None:
            return
        node.SetAttribute(AGE_ATTRIBUTE, f"{age:g}" if age is not None else "")
        node.SetAttribute(SEX_ATTRIBUTE, sex or "")

    @staticmethod
    def _subjectHierarchyPatient(node):
        """(name, ID) of the patient item above the volume in the subject hierarchy (a DICOM load); ('', '')."""
        try:
            shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
            item = shNode.GetItemByDataNode(node)
            root = shNode.GetSceneItemID()
            constants = slicer.vtkMRMLSubjectHierarchyConstants
            nameKey = constants.GetDICOMPatientNameAttributeName()
            idKey = constants.GetDICOMPatientIDAttributeName()
            while item and item != root:
                name, patientId = shNode.GetItemAttribute(item, nameKey), shNode.GetItemAttribute(item, idKey)
                if name or patientId:
                    return sn.formatPatientName(name), (patientId or "").strip()
                item = shNode.GetItemParent(item)
        except Exception:
            logging.debug("Static Neuro: no patient in the subject hierarchy", exc_info=True)
        return "", ""

    @classmethod
    def patientNameAndId(cls, node):
        """(name, ID) of the patient: the values entered on the Data page, else the DICOM tags (the file, else the
        subject hierarchy), else the BIDS subject of an imported file (ID)."""
        if node is None:
            return "", ""
        name, patientId = node.GetAttribute(NAME_ATTRIBUTE), node.GetAttribute(ID_ATTRIBUTE)
        if name is None or patientId is None:
            values = cls.dicomValues(node, ("0010,0010", "0010,0020"))
            taggedName, taggedId = sn.formatPatientName(values["0010,0010"]), values["0010,0020"].strip()
            if not taggedName or not taggedId:
                hierarchyName, hierarchyId = cls._subjectHierarchyPatient(node)
                taggedName, taggedId = taggedName or hierarchyName, taggedId or hierarchyId
            taggedId = taggedId or sn.taggedSubjectId(cls.attributes(node))
            name = taggedName if name is None else name
            patientId = taggedId if patientId is None else patientId
        return name, patientId

    @staticmethod
    def setPatientNameAndId(node, name, patientId):
        if node is None:
            return
        node.SetAttribute(NAME_ATTRIBUTE, (name or "").strip())
        node.SetAttribute(ID_ATTRIBUTE, (patientId or "").strip())

    @classmethod
    def isStaticImage(cls, node):
        """A volume the workflow takes as its image: a static PET / SPECT (not an atlas, a template or a helper)."""
        if node is None or node.IsA("vtkMRMLLabelMapVolumeNode") or node.GetHideFromEditors():
            return False
        if node.GetAttribute(ATLAS_FILE_ATTRIBUTE) or node.GetAttribute("DynamicPETRegistration.Helper"):
            return False
        return bool(cls.imageInfo(node)["modality"])

    @staticmethod
    def loadFromFile(path):
        """An image file as a static PET / SPECT volume, with the tags of its BIDS name, .json sidecar and the
        participants.tsv (age, sex)."""
        sidecar = {}
        sidecarFile = sn.sidecarPath(path)
        if sidecarFile:
            try:
                with open(sidecarFile, encoding="utf-8") as f:
                    sidecar = json.load(f)
            except (OSError, ValueError):
                logging.warning(f"Static Neuro: could not read {sidecarFile}", exc_info=True)
        participant = {}
        entities = sn.parseBidsName(path)
        try:
            participant = sn.readParticipant(sn.findParticipants(path), entities.get("sub"))
        except (OSError, ValueError):
            logging.warning("Static Neuro: could not read participants.tsv", exc_info=True)
        node = slicer.util.loadVolume(path, {"name": sn.importName(path), "show": False})
        if node is None:
            raise RuntimeError(f"Could not load {path}")
        attributes = sn.importAttributes(path, sidecar, participant)
        for key, value in attributes.items():
            node.SetAttribute(key, str(value))
        if attributes.get(sn.STATIC_PREFIX + "Frames") or attributes.get(sn.SPECT_PREFIX + "Frames"):
            logging.warning(f"Static Neuro: {os.path.basename(path)} has several frames (dynamic); only the first "
                            "volume is used")
        node.CreateDefaultDisplayNodes()
        display = node.GetDisplayNode()
        colorNode = slicer.mrmlScene.GetFirstNodeByName("PET-DICOM")
        if display is not None and colorNode is not None:
            display.SetAndObserveColorNodeID(colorNode.GetID())
            display.AutoWindowLevelOff()
            low, high = _percentiles(node)
            display.SetWindowLevelMinMax(0.0, high)
        return node

    # --- harden one volume -----------------------------------------------------------------------
    @staticmethod
    def hardenVolume(volume, fixed, linearNode, action, grid, spacing=None, interpolation=reg.DEFAULT_INTERPOLATION,
                     attributes=None, suffix=REGISTERED_SUFFIX, keepFov=False):
        """Resample volume through the registration chain whose moving end is linearNode onto the Harden grid of
        fixed (DynamicPETRegistrationLogic.outputGeometry). reg.OVERWRITE replaces its voxels and geometry,
        reg.SAVE_SEPARATELY makes a new volume '<name><suffix>' next to it. Returns (undo record, volume)."""
        from DynamicPETRegistration import DynamicPETRegistrationLogic
        regLogic = DynamicPETRegistrationLogic
        corners = regLogic.registeredCorners(volume, linearNode) if keepFov else None
        dims, matrix = regLogic.outputGeometry(fixed, volume, grid, spacing, corners)
        worldToMoving, linear = regLogic.worldToPet(linearNode)
        image = resampling.Resampler(_ijkToRas(volume), dims, matrix, worldToMoving, linear,
                                     interpolation).resample(volume.GetImageData())
        attributes = dict(attributes or {})
        attributes.setdefault(REG_PREFIX + "Interpolation", interpolation)
        attributes.setdefault(REG_PREFIX + "VoxelSizeMm", ", ".join(f"{v:.3g}" for v in reg.gridSpacing(matrix)))
        if action == reg.OVERWRITE:
            undo = {"action": action, "node": volume, "image": volume.GetImageData(), "matrix": _ijkToRas(volume),
                    "attributes": {k: volume.GetAttribute(k) for k in attributes}}
            volume.SetAndObserveImageData(image)
            volume.SetIJKToRASMatrix(_vtkMatrix(matrix))
            for key, value in attributes.items():
                volume.SetAttribute(key, value)
            return undo, volume
        scene = slicer.mrmlScene
        node = scene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", scene.GenerateUniqueName(volume.GetName() + suffix))
        node.SetAndObserveImageData(image)
        node.SetIJKToRASMatrix(_vtkMatrix(matrix))
        for key in volume.GetAttributeNames() or ():
            node.SetAttribute(key, volume.GetAttribute(key))
        for key, value in attributes.items():
            node.SetAttribute(key, value)
        try:
            node.SetVoxelValueUnits(volume.GetVoxelValueUnits())
        except Exception:
            pass
        node.CreateDefaultDisplayNodes()
        if volume.GetDisplayNode() is not None:
            node.GetDisplayNode().CopyContent(volume.GetDisplayNode())
        from DynamicPETMotionCorrection import DynamicPETMotionCorrectionLogic
        DynamicPETMotionCorrectionLogic._placeNextTo(volume, [node])
        return {"action": action, "created": [node]}, node

    @staticmethod
    def revertHarden(undo):
        for node in undo.get("extras") or ():          # the CT / MR sequences normalized with the image
            if node is not None and slicer.mrmlScene.IsNodePresent(node):
                slicer.mrmlScene.RemoveNode(node)
        if undo["action"] == reg.OVERWRITE:
            node = undo["node"]
            if not slicer.mrmlScene.IsNodePresent(node):
                return
            node.SetAndObserveImageData(undo["image"])
            node.SetIJKToRASMatrix(_vtkMatrix(undo["matrix"]))
            for key, value in undo["attributes"].items():
                node.SetAttribute(key, value) if value is not None else node.RemoveAttribute(key)
            return
        for node in undo["created"]:
            if slicer.mrmlScene.IsNodePresent(node):
                slicer.mrmlScene.RemoveNode(node)

    # --- CT/MR registration ----------------------------------------------------------------------
    @staticmethod
    def registerToAnatomy(volume, anatomy, settings=None, progress=None):
        """DynamicPETRegistrationLogic.register() of the image to the CT/MR (iterative rigid)."""
        from DynamicPETRegistration import DynamicPETRegistrationLogic
        if volume.GetParentTransformNode() is not None:
            raise ValueError(f"'{volume.GetName()}' is under a transform. Harden or remove it first.")
        logic = DynamicPETRegistrationLogic()
        return logic.register(anatomy, volume, settings or logic.defaultSettings(), progress)

    def hardenRegistration(self, volume, anatomy, linearNode, action, grid, spacing=None,
                           interpolation=reg.DEFAULT_INTERPOLATION, description=""):
        """The image resampled onto the CT/MR field of view (grown to the image's where it is larger), tagged as
        registered to it."""
        attributes = {REG_PREFIX + "Fixed": anatomy.GetName(), REG_PREFIX + "Transform": description}
        return self.hardenVolume(volume, anatomy, linearNode, action, grid, spacing, interpolation, attributes,
                                 REGISTERED_SUFFIX, keepFov=True)

    # --- normalization ---------------------------------------------------------------------------
    def harden(self, volume, template, linearNode, action, grid, spacing=None, interpolation="Linear",
               description="", method=METHOD_PET, anatomy=None, extras=None):
        """The image resampled through the normalization onto the template's field of view. The patient's CT/MR is
        normalized too with the same transform (on the template grid) whatever the method: it lies on the image
        (registered in step 2, or a PET/CT, PET/MR); so are extras, other sequences lying on the CT/MR (undo
        ["extras"]). Returns (undo, volume, None, normalized CT/MR or None)."""
        attributes = {NEURO_PREFIX + "Template": template.GetName(), NEURO_PREFIX + "Method": method,
                      NEURO_PREFIX + "Transform": description}
        undo, node = self.hardenVolume(volume, template, linearNode, action, grid, spacing, interpolation,
                                       attributes, NORMALIZED_SUFFIX)
        normalized = None
        if anatomy is not None:
            try:
                normalized = self.normalizedAnatomy(anatomy, template, linearNode)
            except Exception:
                logging.exception("Static Neuro: could not normalize the CT/MR")
        undo["extras"] = []
        for extra in extras or ():
            if extra is None or extra is volume or extra is anatomy:
                continue
            try:
                undo["extras"].append(self.normalizedAnatomy(extra, template, linearNode))
            except Exception:
                logging.exception(f"Static Neuro: could not normalize '{extra.GetName()}'")
        return undo, node, None, normalized
