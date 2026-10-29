# Known Issues

This page documents common pitfalls when using **robosuite** that are not covered elsewhere in the documentation.

## EGL offscreen rendering produces noise after importing trimesh

### Symptoms

When **robosuite** is used with headless EGL rendering (the default on Linux for offscreen cameras), RGB observations or rendered frames may look like random noise. This often appears when the same script also uses [trimesh](https://trimsh.org/).

### Cause

Importing **trimesh** can initialize an OpenGL context (for example via pyglet). If **trimesh** is imported *after* a **robosuite** environment has already been created, that initialization can interfere with the EGL context that MuJoCo and **robosuite** use for offscreen rendering, leaving GL state inconsistent.

This is especially easy to trigger when `import trimesh` (or `import trimesh.transformations`) happens lazily inside a function that runs only after the environment exists, rather than at the top of the module.

### Workaround

Import **trimesh** at module level *before* you import **robosuite** or construct any **robosuite** environment, so any context **trimesh** creates is established first and **robosuite** can set up EGL afterward:

```python
import trimesh
import trimesh.transformations as tra  # if needed

import robosuite
from robosuite import make

env = make("Lift", robots="Panda", has_renderer=False, has_offscreen_renderer=True)
# ...
```

If you use **trimesh** only in some code paths, still place the imports at the top of the file instead of inside functions that may run after the environment is created.

### Related discussion

See [GitHub issue #798](https://github.com/ARISE-Initiative/robosuite/issues/798).
