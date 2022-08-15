"""
Pygame renderer class.
"""
import pygame


class PygameRenderer:
    def __init__(self, sim):
        # set up pygame window to be fullscreen by default
        pygame.init()
        info = pygame.display.Info()
        self.width = info.current_w
        self.height = info.current_h
        # self.screen = pygame.display.set_mode((self.onscreen_width, self.onscreen_height), pygame.FULLSCREEN)
        # self.screen = pygame.display.set_mode((self.onscreen_width, self.onscreen_height), pygame.RESIZABLE)
        self.screen = pygame.display.set_mode((self.width, self.height))

        self.sim = sim
        self.camera_name = self.sim.model.camera_id2name(0)

    def set_camera(self, camera_id):
        """
        Set the camera view to the specified camera ID.
        Args:
            camera_id (int): id of the camera to set the current viewer to
        """
        self.camera_name = self.sim.model.camera_id2name(camera_id)

    def render(self):
        # get frame with offscreen renderer (assumes that the renderer already exists)
        im = self.sim.render(camera_name=self.camera_name, height=self.height, width=self.width)[::-1]

        # write frame to window
        im = im.transpose((1, 0, 2))
        pygame.pixelcopy.array_to_surface(self.screen, im)
        pygame.display.update()
        for event in pygame.event.get():
            pass

    def close(self):
        """
        Any cleanup to close renderer.
        """

        # NOTE: assume that @sim will get cleaned up outside the renderer - just delete the reference
        self.sim = None

        # close window
        pygame.display.quit()
