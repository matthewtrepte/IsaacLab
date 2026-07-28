Changed
^^^^^^^

* Changed the default preset for :class:`~isaaclab_tasks.core.reorient.config.shadow_hand.ShadowHandEventCfg`,
  :class:`~isaaclab_tasks.core.reorient.config.shadow_hand.ShadowHandRobotCfg`,
  :class:`~isaaclab_tasks.core.reorient.config.shadow_hand.ObjectCfg`, and
  :class:`~isaaclab_tasks.core.reorient.config.shadow_hand.PhysicsCfg` from ``newton_mjwarp``
  to ``physx``.  The ``physx`` preset was already the stable default used by CI and training
  runs; this aligns the code default with the intended behavior and fixes golden image tests that
  relied on the physx preset when no explicit preset was specified.
