import robosuite.utils.env_utils as EU
from copy import copy, deepcopy


def satisfies(state, cond):
    for k, v in cond.items():
        if state[k] != v:
            return False
    return True


class InteractiveObject(object):
    """Base class for interactive object"""
    def __init__(self, sim, body_id, cooldown_interval=-1):
        self.sim = sim
        self._cooldown_interval = cooldown_interval
        self._last_event_step = 0
        self._body_id = body_id

    @property
    def body_id(self):
        return self._body_id

    @property
    def body_name(self):
        return self.sim.model.body_id2name[self._body_id]

    @property
    def body_geom_ids(self):
        return EU.bodyid2geomids(self.sim, self._body_id)

    def reset(self):
        self._last_event_step = 0

    def _on_contact(self, contacting_body_ids):
        pass

    def _on_customized_condition(self):
        pass

    def step(self, sim_step, contacting_body_ids=None):
        if sim_step - self._last_event_step < self._cooldown_interval:
            return
        self._last_event_step = sim_step

        if contacting_body_ids is None:
            contacting_body_ids = EU.all_contacting_body_ids(self.sim, self.body_id)

        self._on_contact(contacting_body_ids)
        self._on_customized_condition()


class StatefulObject(object):
    """Base class for object that maintains state"""
    def __init__(self, all_state_names, default_state):
        self._default_state = deepcopy(default_state)
        self._all_state_names = copy(all_state_names)
        self._state = self.default_state

        assert(isinstance(default_state, dict))
        assert(sorted(self._default_state.keys()) == sorted(self._all_state_names))

        self._on_enter_state_funcs = []  # called once when switching to the state
        self._on_exit_state_funcs = []  # called once when leaving the state

        self._on_state_funcs = []  # gets called when the object is in the specified state
        self._on_not_state_funcs = []  # gets called when the object is NOT in the specified state

        self._set_state_behaviors()

        self.reset()

    def _set_state_behaviors(self):
        """Fills in state behaviors"""
        raise NotImplementedError

    def add_on_enter_state_funcs(self, cond, func):
        assert(self.is_valid_condition(cond))
        self._on_enter_state_funcs.append((deepcopy(cond), func))

    def add_on_exit_state_funcs(self, cond, func):
        assert(self.is_valid_condition(cond))
        self._on_exit_state_funcs.append((deepcopy(cond), func))

    def add_on_state_funcs(self, cond, func):
        assert(self.is_valid_condition(cond))
        self._on_state_funcs.append((deepcopy(cond), func))

    def add_on_not_enter_state_funcs(self, cond, func):
        assert(self.is_valid_condition(cond))
        self._on_not_state_funcs.append((deepcopy(cond), func))

    def reset(self):
        # reset state without triggering state functions
        self._state = self.default_state

    def is_valid_state(self, state):
        return sorted(list(state.keys())) == sorted(self._all_state_names)

    def is_valid_condition(self, cond):
        for k in cond:
            if k not in self._all_state_names:
                return False
        return True

    @property
    def default_state(self):
        return deepcopy(self._default_state)

    @property
    def state(self):
        return deepcopy(self._state)

    @state.setter
    def state(self, new_state):
        assert(self.is_valid_state(new_state))

        prev_state = self.state
        self._state = deepcopy(new_state)

        for cond, func in self._on_enter_state_funcs:
            if satisfies(self.state, cond) and not satisfies(prev_state, cond):
                func()

        for cond, func in self._on_exit_state_funcs:
            if not satisfies(self.state, cond) and satisfies(prev_state, cond):
                func()


class StatefulInteractiveObject(StatefulObject, InteractiveObject):
    def __init__(self, sim, body_id, all_state_names, default_state, cooldown_interval=-1):
        StatefulObject.__init__(self, all_state_names=all_state_names, default_state=default_state)
        InteractiveObject.__init__(self, sim=sim, body_id=body_id, cooldown_interval=cooldown_interval)

    def step(self, sim_step, contacting_body_ids=None):
        for cond, func in self._on_state_funcs:
            if satisfies(self.state, cond):
                func()
        for cond, func in self._on_not_state_funcs:
            if not satisfies(self.state, cond):
                func()
        InteractiveObject.step(self, sim_step=sim_step, contacting_body_ids=contacting_body_ids)

    def reset(self):
        InteractiveObject.reset(self)
        StatefulObject.reset(self)


class BinaryStateObject(StatefulInteractiveObject):
    def __init__(self, sim, body_id, state_name='ON', cooldown_interval=-1):
        self._state_name = state_name
        default_state = {self.state_name: False}
        super(BinaryStateObject, self).__init__(
            sim=sim,
            body_id=body_id,
            all_state_names=list(default_state.keys()),
            default_state=default_state,
            cooldown_interval=cooldown_interval
        )

    @property
    def state_name(self):
        return self._state_name

    def _set_state_behaviors(self):
        self.add_on_enter_state_funcs(
            cond={self.state_name: True}, func=self._on_activated
        )
        self.add_on_enter_state_funcs(
            cond={self.state_name: False}, func=self._on_deactivated
        )
        self.add_on_state_funcs(
            cond={self.state_name: True}, func=self._on_active
        )
        self.add_on_state_funcs(
            cond={self.state_name: False}, func=self._on_inactive
        )

    def _on_activated(self):
        """Internal function that gets called once when object gets activated """
        pass

    def _on_deactivated(self):
        """Internal function that gets called once when object gets deactivated """
        pass

    def _on_active(self):
        """Internal function that gets called every simulation step when object is active"""
        pass

    def _on_inactive(self):
        """Internal function that gets called every simulation step when object is inactive"""
        pass

    def activate(self):
        self.state = {self.state_name: True}

    def deactivate(self):
        self.state = {self.state_name: False}

    def toggle(self):
        self.state = {self.state_name: not self.state[self.state_name]}


class ButtonObject(BinaryStateObject):
    """Base class for a button object."""
    def __init__(self, sim, body_id, off_rgba=(1, 0, 0, 1), on_rgba=(0, 1, 0, 1), cooldown_interval=-1):
        super(ButtonObject, self).__init__(sim=sim, body_id=body_id, cooldown_interval=cooldown_interval)
        self._off_rgba = off_rgba
        self._on_rgba = on_rgba
        self._prev_contacts = []

    def reset(self):
        super(ButtonObject, self).reset()
        self._prev_contacts = []

    def _on_active(self):
        """Internal function that gets called once when object becomes activated """
        for gid in self.body_geom_ids:
            self.sim.model.geom_rgba[gid] = self._on_rgba

    def _on_inactive(self):
        """Internal function that gets called once when object becomes deactivated """
        for gid in self.body_geom_ids:
            self.sim.model.geom_rgba[gid] = self._off_rgba

    def _on_enter_contact(self):
        pass

    def _on_exit_contact(self):
        pass

    def _on_contact(self, contacting_body_ids):
        table_id = self.sim.model.body_name2id('table')
        contacting_body_ids = [c for c in contacting_body_ids if c != table_id]
        if len(self._prev_contacts) == 0 and len(contacting_body_ids) > 0:
            self._on_enter_contact()
        if len(self._prev_contacts) > 0 and len(contacting_body_ids) == 0:
            self._on_exit_contact()
        self._prev_contacts = contacting_body_ids


class MomentaryButtonObject(ButtonObject):
    """Activated only when in contact"""
    def _on_enter_contact(self):
        self.activate()

    def _on_exit_contact(self):
        self.deactivate()


class MaintainedButtonObject(ButtonObject):
    """Toggles state at the end of a contact"""
    def _on_exit_contact(self):
        self.toggle()
