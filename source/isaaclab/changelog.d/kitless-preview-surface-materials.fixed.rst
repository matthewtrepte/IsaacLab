* Fixed kitless USD file spawns, such as visualization markers, dropping their
  :class:`~isaaclab.sim.spawners.materials.PreviewSurfaceCfg` visual material. Preview surfaces are plain ``UsdShade``
  and are now authored and bound without Kit; only MDL materials are still skipped.
