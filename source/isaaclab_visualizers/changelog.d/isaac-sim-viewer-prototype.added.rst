* Added a prototype ``isaac_sim`` visualizer backend (``--viz isaac-sim``,
  :class:`~isaaclab_visualizers.isaac_sim.IsaacSimBackendCfg`) that shows an Isaac Lab simulation in the Isaac Sim 7
  viewer application, with its stage tree, property and content panes and transport controls. The viewer shows Isaac
  Lab's own OVRTX scene: its cameras are scene cameras on a shared OVRTX renderer whose scene stage the application
  host creates, so the host's panes read the stage Isaac Lab renders. The host runs no scheduler; Isaac Lab drives
  every update.
  Clicking a prim selects and outlines it; the transform gizmo moves and rotates the Isaac Lab asset that owns it
  (an articulation or rigid object, per environment), also while paused.
