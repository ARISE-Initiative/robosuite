"""An OpenGL context created via CGL."""

from mujoco.cgl import GLContext

class CGLGLContext(GLContext):
    """An OpenGL context created via CGL."""

    def __init__(self, max_width, max_height, device_id=0):
        super().__init__(max_width, max_height)
