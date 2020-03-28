import numpy as np
import robosuite.utils.transform_utils as T
from contextlib import contextmanager


def bodyid2geomids(sim, body_id):
    geoms = []
    for i, bid in enumerate(sim.model.geom_bodyid):
        if bid == body_id:
            geoms.append(i)
    return geoms


def set_body_pose(sim, body_name, pos, quat=None):
    sim_state = sim.get_state()
    qpos_addr = sim.model.get_joint_qpos_addr(body_name)
    pos_addr = qpos_addr[0]
    old_pose = np.array(sim_state.qpos[pos_addr: pos_addr + 7])
    sim_state.qpos[pos_addr: pos_addr + 3] = np.array(pos)
    if quat is not None:
        sim_state.qpos[pos_addr + 3: pos_addr + 7] = T.convert_quat(quat, to="wxyz")
    sim.set_state(sim_state)
    return old_pose[:3], old_pose[3:7]


@contextmanager
def world_saved(sim):
    """
    Context scope for saved world state
    """
    world_state = sim.get_state().flatten()
    yield
    sim.set_state_from_flattened(world_state)
    sim.forward()


def all_contacting_geom_ids(sim, geom_id):
    """
    Returns a list of geom ids that are in contact with the target geom id
    """
    contact_gids = []
    for contact in sim.data.contact[:sim.data.ncon]:
        if contact.geom1 == geom_id:
            contact_gids.append(contact.geom2)
        elif contact.geom2 == geom_id:
            contact_gids.append(contact.geom1)
    return contact_gids


def all_contacting_body_ids(sim, body_id):
    """
    Returns a list of body ids that are in contact with the target body id
    """
    contact_gids = []
    for gid in bodyid2geomids(sim, body_id):
        contact_gids += all_contacting_geom_ids(sim, gid)
    contact_bids = list(set([sim.model.geom_bodyid[gid] for gid in contact_gids]))
    return contact_bids