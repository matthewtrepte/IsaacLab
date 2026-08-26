Fixed
^^^^^

* Fixed ``NVST_R_BUSY`` errors during WebRTC livestream (``LIVESTREAM=1`` or ``LIVESTREAM=2``)
  that occurred when the OS or window manager resized the application window after a client
  connected.  ``AppLauncher`` now passes ``--/app/livestream/allowResize=true`` to Kit so
  the application allows the framebuffer to resize.  Without this setting the app resisted
  OS-initiated resizes while NVST was still trying to capture at the original negotiated
  resolution, producing a mismatched resolution and an unrecoverable flood of
  ``NVST_R_BUSY`` errors (NVBug 6281418).

* Added unit tests in ``source/isaaclab/test/app/test_app_launcher_argv.py`` covering
  all required NVST Kit arguments for ``LIVESTREAM=0``, ``LIVESTREAM=1``, and
  ``LIVESTREAM=2``, ensuring the regression cannot be silently reintroduced.
