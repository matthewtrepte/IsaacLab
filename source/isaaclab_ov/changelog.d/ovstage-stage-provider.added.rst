* Added :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.stage_provider` and
  :attr:`~isaaclab_ov.stage.OvstageBackendCfg.stage_provider`, which let an application create the OVStage that an
  OVRTX renderer's scene is populated into, for example a viewer that also reads the stage through its own stage
  context. Population starts after any ordinals the provided stage already holds.
