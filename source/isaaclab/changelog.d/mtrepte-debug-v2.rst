Fixed
^^^^^

* Fixed :class:`~isaaclab.ui.widgets.ManagerLiveVisualizer` raising ``AttributeError`` on
  ``_debug_vis_handle`` when the debug-vis checkbox is toggled before the visualizer has ever
  been enabled.  The handle is now initialized to ``None`` in ``__init__``.
* Fixed :class:`~isaaclab.ui.widgets.LiveLinePlot` ``_rescale_btn_pressed`` raising
  ``ValueError: min() arg is an empty sequence`` when autoscale runs while all visible series
  are still empty (e.g. immediately after a ``clear()`` call).  The method now skips rescaling
  when no non-empty visible series exist, preventing the Re-Scale button and autoscale from
  appearing broken.
