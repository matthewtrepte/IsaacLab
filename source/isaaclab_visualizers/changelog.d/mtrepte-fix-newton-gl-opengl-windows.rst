Fixed
^^^^^

* Fixed ``newton_gl`` visualizer failing on Linux CI with
  ``Cannot connect to "None"`` when ``DISPLAY`` points to a non-existent X server
  (e.g. Xvfb not running). ``NewtonGLVisualizer._create_viewer()`` now proactively
  sets ``pyglet.options["headless"] = True`` before constructing the viewer when
  ``runtime_headless=True``, and also catches the ``Cannot connect`` error at
  runtime and retries with the EGL off-screen backend regardless of how
  ``DISPLAY`` was set at module-import time.

* Fixed an unhelpful error when ``newton_gl`` fails on Windows because the GPU
  driver does not export OpenGL 2.0 entry points (e.g. ``glCreateShader`` — a
  regression observed on Blackwell GPUs with driver 595.97 under Isaac Sim 6.1).
  ``NewtonGLVisualizer`` now catches this condition and re-raises with a clear
  message directing users to switch to ``--visualizer newton_rtx``, which does not
  require a legacy OpenGL context.
