"""
Step 4: regions (with the database's region settings) and the regions step views.

Part of the Static Neuro PET/SPECT module (StaticNeuroPET.py): RegionsPageMixin, mixed into StaticNeuroPETWidget
there; the methods are those of NeuroWorkflowLib.database.RegionsDatabaseMixin (shared with the Dynamic Neuro module).
"""

from NeuroWorkflowLib.database import RegionsDatabaseMixin, _windowOf  # noqa: F401


class RegionsPageMixin(RegionsDatabaseMixin):
    """The page regions methods of StaticNeuroPETWidget (NeuroWorkflowLib.database.RegionsDatabaseMixin)."""
