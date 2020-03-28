import robosuite.utils.env_utils as EU


class InteractiveObject(object):
    def __init__(self, sim):
        self.sim = sim


class ButtonObject(InteractiveObject):
    def __init__(self, sim, body_id, off_rgba=(1, 0, 0, 1), on_rgba=None, cooldown_interval=10):
        InteractiveObject.__init__(self, sim)
        self._activated = False
        self._body_id = body_id
        self._geom_ids = EU.bodyid2geomids(self.sim, body_id)

        self._off_rgba = off_rgba
        self._on_rgba = on_rgba
        if self._on_rgba is None:
            self._on_rgba = off_rgba

        self._cooldown_interval = cooldown_interval
        self._last_event_step = 0
        self.reset()

    def reset(self):
        self._last_event_step = 0
        self.deactivate()

    def toggle(self):
        self.activated = not self.activated

    def activate(self):
        self.activated = True

    def deactivate(self):
        self.activated = False

    @property
    def body_id(self):
        return self._body_id

    @property
    def is_fixed(self):
        return True

    @property
    def activated(self):
        return self._activated

    @activated.setter
    def activated(self, val):
        assert(type(val) == bool)
        self._activated = val

        rgba = self._on_rgba if self._activated else self._off_rgba
        for gid in self._geom_ids:
            self.sim.model.geom_rgba[gid] = rgba

    def step(self, sim_step, collided_body_ids=[]):
        print(collided_body_ids)
        # print([self.sim.model.body_id2name(bid) for bid in collided_body_ids])
        if sim_step - self._last_event_step < self._cooldown_interval:
            return

        self._last_event_step = sim_step
        if len(collided_body_ids) > 0:
            self.on_collision(collided_body_ids)

    def on_collision(self, collided_body_ids):
        self.toggle()

