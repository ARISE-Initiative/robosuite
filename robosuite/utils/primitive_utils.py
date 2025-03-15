def inverse_scale_action(env, action):
    control_dim = env.env.robots[0].controller.control_dim
    action[:control_dim] = env.env.robots[0].controller.inverse_scale_action(action[:control_dim])
    return action

def scale_action(env, action):
    control_dim = env.env.robots[0].controller.control_dim
    action[:control_dim] = env.env.robots[0].controller.scale_action(action[:control_dim])
    return action

