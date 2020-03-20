import numpy as np
import robosuite.utils.transform_utils as T


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

