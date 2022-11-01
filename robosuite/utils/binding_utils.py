"""
Useful classes for supporting DeepMind MuJoCo binding.
"""

import gc
import os
from tempfile import TemporaryDirectory

# DIRTY HACK copied from mujoco-py - a global lock on rendering
from threading import Lock

import mujoco
import numpy as np

_MjSim_render_lock = Lock()

import ctypes
import ctypes.util
import os
import platform
import subprocess

import robosuite.macros as macros

_SYSTEM = platform.system()
if _SYSTEM == "Windows":
    ctypes.WinDLL(os.path.join(os.path.dirname(__file__), "mujoco.dll"))

CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if CUDA_VISIBLE_DEVICES != "":
    MUJOCO_EGL_DEVICE_ID = os.environ.get("MUJOCO_EGL_DEVICE_ID", None)
    if MUJOCO_EGL_DEVICE_ID is not None:
        assert MUJOCO_EGL_DEVICE_ID.isdigit() and (
            MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES
        ), "MUJOCO_EGL_DEVICE_ID needs to be set to one of the device id specified in CUDA_VISIBLE_DEVICES"

if macros.MUJOCO_GPU_RENDERING and os.environ.get("MUJOCO_GL", None) not in ["osmesa", "glx"]:
    # If gpu rendering is specified in macros, then we enforce gpu
    # option for rendering
    os.environ["MUJOCO_GL"] = "egl"
_MUJOCO_GL = os.environ.get("MUJOCO_GL", "").lower().strip()
if _MUJOCO_GL not in ("disable", "disabled", "off", "false", "0"):
    _VALID_MUJOCO_GL = ("enable", "enabled", "on", "true", "1", "glfw", "")
    if _SYSTEM == "Linux":
        _VALID_MUJOCO_GL += ("glx", "egl", "osmesa")
    elif _SYSTEM == "Windows":
        _VALID_MUJOCO_GL += ("wgl",)
    elif _SYSTEM == "Darwin":
        _VALID_MUJOCO_GL += ("cgl",)
    if _MUJOCO_GL not in _VALID_MUJOCO_GL:
        raise RuntimeError(f"invalid value for environment variable MUJOCO_GL: {_MUJOCO_GL}")
    if _SYSTEM == "Linux" and _MUJOCO_GL == "osmesa":
        from robosuite.renderers.context.osmesa_context import OSMesaGLContext as GLContext
    elif _SYSTEM == "Linux" and _MUJOCO_GL == "egl":
        from robosuite.renderers.context.egl_context import EGLGLContext as GLContext
    else:
        from robosuite.renderers.context.glfw_context import GLFWGLContext as GLContext


class MjRenderContext:
    """
    Class that encapsulates rendering functionality for a
    MuJoCo simulation.

    See https://github.com/openai/mujoco-py/blob/4830435a169c1f3e3b5f9b58a7c3d9c39bdf4acb/mujoco_py/mjrendercontext.pyx
    """

    def __init__(self, sim, offscreen=True, device_id=-1, max_width=640, max_height=480):
        assert offscreen, "only offscreen supported for now"
        self.sim = sim
        self.offscreen = offscreen
        self.device_id = device_id

        # setup GL context with defaults for now
        self.gl_ctx = GLContext(max_width=max_width, max_height=max_height, device_id=self.device_id)
        self.gl_ctx.make_current()

        # Ensure the model data has been updated so that there
        # is something to render
        sim.forward()
        # make sure sim has this context
        sim.add_render_context(self)

        self.model = sim.model
        self.data = sim.data

        # create default scene
        self.scn = mujoco.MjvScene(sim.model._model, maxgeom=1000)

        # camera
        self.cam = mujoco.MjvCamera()
        self.cam.fixedcamid = 0
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

        # options for visual / collision mesh can be set externally, e.g. vopt.geomgroup[0], vopt.geomgroup[1]
        self.vopt = mujoco.MjvOption()

        self.pert = mujoco.MjvPerturb()
        self.pert.active = 0
        self.pert.select = 0
        self.pert.skinselect = -1

        # self._markers = []
        # self._overlay = {}

        self._set_mujoco_context_and_buffers()

    def _set_mujoco_context_and_buffers(self):
        self.con = mujoco.MjrContext(self.model._model, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)

    def update_offscreen_size(self, width, height):
        if (width != self.con.offWidth) or (height != self.con.offHeight):
            self.model.vis.global_.offwidth = width
            self.model.vis.global_.offheight = height
            self.con.free()
            del self.con
            self._set_mujoco_context_and_buffers()

    def render(self, width, height, camera_id=None, segmentation=False):
        viewport = mujoco.MjrRect(0, 0, width, height)

        # if self.sim.render_callback is not None:
        #     self.sim.render_callback(self.sim, self)

        # update width and height of rendering context if necessary
        if width > self.con.offWidth or height > self.con.offHeight:
            new_width = max(width, self.model.vis.global_.offwidth)
            new_height = max(height, self.model.vis.global_.offheight)
            self.update_offscreen_size(new_width, new_height)

        if camera_id is not None:
            if camera_id == -1:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = camera_id

        mujoco.mjv_updateScene(
            self.model._model, self.data._data, self.vopt, self.pert, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn
        )

        if segmentation:
            self.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
            self.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1

        # for marker_params in self._markers:
        #     self._add_marker_to_scene(marker_params)

        mujoco.mjr_render(viewport=viewport, scn=self.scn, con=self.con)
        # for gridpos, (text1, text2) in self._overlay.items():
        #     mjr_overlay(const.FONTSCALE_150, gridpos, rect, text1.encode(), text2.encode(), &self._con)

        if segmentation:
            self.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 0
            self.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 0

    def read_pixels(self, width, height, depth=False, segmentation=False):
        viewport = mujoco.MjrRect(0, 0, width, height)
        rgb_img = np.empty((height, width, 3), dtype=np.uint8)
        depth_img = np.empty((height, width), dtype=np.float32) if depth else None

        mujoco.mjr_readPixels(rgb=rgb_img, depth=depth_img, viewport=viewport, con=self.con)

        ret_img = rgb_img
        if segmentation:
            seg_img = rgb_img[:, :, 0] + rgb_img[:, :, 1] * (2**8) + rgb_img[:, :, 2] * (2**16)
            seg_img[seg_img >= (self.scn.ngeom + 1)] = 0
            seg_ids = np.full((self.scn.ngeom + 1, 2), fill_value=-1, dtype=np.int32)

            for i in range(self.scn.ngeom):
                geom = self.scn.geoms[i]
                if geom.segid != -1:
                    seg_ids[geom.segid + 1, 0] = geom.objtype
                    seg_ids[geom.segid + 1, 1] = geom.objid
            ret_img = seg_ids[seg_img]

        if depth:
            return (ret_img, depth_img)
        else:
            return ret_img

    # def upload_texture(self, tex_id):
    #     """ Uploads given texture to the GPU. """
    #     self.opengl_context.make_context_current()
    #     mjr_uploadTexture(self._model_ptr, &self._con, tex_id)

    def __del__(self):
        # free mujoco rendering context and GL rendering context
        self.con.free()
        self.gl_ctx.free()
        del self.con
        del self.gl_ctx
        del self.scn
        del self.cam
        del self.vopt
        del self.pert


class MjRenderContextOffscreen(MjRenderContext):
    def __init__(self, sim, device_id, max_width=640, max_height=480):
        super().__init__(sim, offscreen=True, device_id=device_id, max_width=max_width, max_height=max_height)


class MjSimState:
    """
    A mujoco simulation state.
    """

    def __init__(self, time, qpos, qvel):
        self.time = time
        self.qpos = qpos
        self.qvel = qvel

    @classmethod
    def from_flattened(cls, array, sim):
        """
        Takes flat mjstate array and MjSim instance and
        returns MjSimState.
        """
        idx_time = 0
        idx_qpos = idx_time + 1
        idx_qvel = idx_qpos + sim.model.nq

        time = array[idx_time]
        qpos = array[idx_qpos : idx_qpos + sim.model.nq]
        qvel = array[idx_qvel : idx_qvel + sim.model.nv]
        assert sim.model.na == 0

        return cls(time=time, qpos=qpos, qvel=qvel)

    def flatten(self):
        return np.concatenate([[self.time], self.qpos, self.qvel], axis=0)


class _MjModelMeta(type):
    """
    Metaclass which allows MjModel below to delegate to mujoco.MjModel.

    Taken from dm_control: https://github.com/deepmind/dm_control/blob/main/dm_control/mujoco/wrapper/core.py#L244
    """

    def __new__(cls, name, bases, dct):
        for attr in dir(mujoco.MjModel):
            if not attr.startswith("_"):
                if attr not in dct:
                    # pylint: disable=protected-access
                    fget = lambda self, attr=attr: getattr(self._model, attr)
                    fset = lambda self, value, attr=attr: setattr(self._model, attr, value)
                    # pylint: enable=protected-access
                    dct[attr] = property(fget, fset)
        return super().__new__(cls, name, bases, dct)


class MjModel(metaclass=_MjModelMeta):
    """Wrapper class for a MuJoCo 'mjModel' instance.
    MjModel encapsulates features of the model that are expected to remain
    constant. It also contains simulation and visualization options which may be
    changed occasionally, although this is done explicitly by the user.
    """

    _HAS_DYNAMIC_ATTRIBUTES = True

    def __init__(self, model_ptr):
        """Creates a new MjModel instance from a mujoco.MjModel."""
        self._model = model_ptr

        # make useful mappings such as _body_name2id and _body_id2name
        self.make_mappings()

    @classmethod
    def from_xml_path(cls, xml_path):
        """Creates an MjModel instance from a path to a model XML file."""
        model_ptr = _get_model_ptr_from_xml(xml_path=xml_path)
        return cls(model_ptr)

    def __del__(self):
        # free mujoco model
        del self._model

    """
    Some methods supported by sim.model in mujoco-py.
    Copied from https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi#L2611
    """

    def _extract_mj_names(self, name_adr, num_obj, obj_type):
        """
        See https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi#L1127
        """

        ### TODO: fix this to use @name_adr like mujoco-py - more robust than assuming IDs are continuous ###

        # objects don't need to be named in the XML, so name might be None
        id2name = {i: None for i in range(num_obj)}
        name2id = {}
        for i in range(num_obj):
            name = mujoco.mj_id2name(self._model, obj_type, i)
            name2id[name] = i
            id2name[i] = name

        # # objects don't need to be named in the XML, so name might be None
        # id2name = { i: None for i in range(num_obj) }
        # name2id = {}
        # for i in range(num_obj):
        #     name = self.model.names[name_adr[i]]
        #     decoded_name = name.decode()
        #     if decoded_name:
        #         obj_id = mujoco.mj_name2id(self.model, obj_type, name)
        #         assert (0 <= obj_id < num_obj) and (id2name[obj_id] is None)
        #         name2id[decoded_name] = obj_id
        #         id2name[obj_id] = decoded_name

        # sort names by increasing id to keep order deterministic
        return tuple(id2name[nid] for nid in sorted(name2id.values())), name2id, id2name

    def make_mappings(self):
        """
        Make some useful internal mappings that mujoco-py supported.
        """
        p = self
        self.body_names, self._body_name2id, self._body_id2name = self._extract_mj_names(
            p.name_bodyadr, p.nbody, mujoco.mjtObj.mjOBJ_BODY
        )
        self.joint_names, self._joint_name2id, self._joint_id2name = self._extract_mj_names(
            p.name_jntadr, p.njnt, mujoco.mjtObj.mjOBJ_JOINT
        )
        self.geom_names, self._geom_name2id, self._geom_id2name = self._extract_mj_names(
            p.name_geomadr, p.ngeom, mujoco.mjtObj.mjOBJ_GEOM
        )
        self.site_names, self._site_name2id, self._site_id2name = self._extract_mj_names(
            p.name_siteadr, p.nsite, mujoco.mjtObj.mjOBJ_SITE
        )
        self.light_names, self._light_name2id, self._light_id2name = self._extract_mj_names(
            p.name_lightadr, p.nlight, mujoco.mjtObj.mjOBJ_LIGHT
        )
        self.camera_names, self._camera_name2id, self._camera_id2name = self._extract_mj_names(
            p.name_camadr, p.ncam, mujoco.mjtObj.mjOBJ_CAMERA
        )
        self.actuator_names, self._actuator_name2id, self._actuator_id2name = self._extract_mj_names(
            p.name_actuatoradr, p.nu, mujoco.mjtObj.mjOBJ_ACTUATOR
        )
        self.sensor_names, self._sensor_name2id, self._sensor_id2name = self._extract_mj_names(
            p.name_sensoradr, p.nsensor, mujoco.mjtObj.mjOBJ_SENSOR
        )
        self.tendon_names, self._tendon_name2id, self._tendon_id2name = self._extract_mj_names(
            p.name_tendonadr, p.ntendon, mujoco.mjtObj.mjOBJ_TENDON
        )
        self.mesh_names, self._mesh_name2id, self._mesh_id2name = self._extract_mj_names(
            p.name_meshadr, p.nmesh, mujoco.mjtObj.mjOBJ_MESH
        )

    def body_id2name(self, id):
        if id not in self._body_id2name:
            raise ValueError("No body with id %d exists." % id)
        return self._body_id2name[id]

    def body_name2id(self, name):
        if name not in self._body_name2id:
            raise ValueError('No "body" with name %s exists. Available "body" names = %s.' % (name, self.body_names))
        return self._body_name2id[name]

    def joint_id2name(self, id):
        if id not in self._joint_id2name:
            raise ValueError("No joint with id %d exists." % id)
        return self._joint_id2name[id]

    def joint_name2id(self, name):
        if name not in self._joint_name2id:
            raise ValueError('No "joint" with name %s exists. Available "joint" names = %s.' % (name, self.joint_names))
        return self._joint_name2id[name]

    def geom_id2name(self, id):
        if id not in self._geom_id2name:
            raise ValueError("No geom with id %d exists." % id)
        return self._geom_id2name[id]

    def geom_name2id(self, name):
        if name not in self._geom_name2id:
            raise ValueError('No "geom" with name %s exists. Available "geom" names = %s.' % (name, self.geom_names))
        return self._geom_name2id[name]

    def site_id2name(self, id):
        if id not in self._site_id2name:
            raise ValueError("No site with id %d exists." % id)
        return self._site_id2name[id]

    def site_name2id(self, name):
        if name not in self._site_name2id:
            raise ValueError('No "site" with name %s exists. Available "site" names = %s.' % (name, self.site_names))
        return self._site_name2id[name]

    def light_id2name(self, id):
        if id not in self._light_id2name:
            raise ValueError("No light with id %d exists." % id)
        return self._light_id2name[id]

    def light_name2id(self, name):
        if name not in self._light_name2id:
            raise ValueError('No "light" with name %s exists. Available "light" names = %s.' % (name, self.light_names))
        return self._light_name2id[name]

    def camera_id2name(self, id):
        if id not in self._camera_id2name:
            raise ValueError("No camera with id %d exists." % id)
        return self._camera_id2name[id]

    def camera_name2id(self, name):
        if name not in self._camera_name2id:
            raise ValueError(
                'No "camera" with name %s exists. Available "camera" names = %s.' % (name, self.camera_names)
            )
        return self._camera_name2id[name]

    def actuator_id2name(self, id):
        if id not in self._actuator_id2name:
            raise ValueError("No actuator with id %d exists." % id)
        return self._actuator_id2name[id]

    def actuator_name2id(self, name):
        if name not in self._actuator_name2id:
            raise ValueError(
                'No "actuator" with name %s exists. Available "actuator" names = %s.' % (name, self.actuator_names)
            )
        return self._actuator_name2id[name]

    def sensor_id2name(self, id):
        if id not in self._sensor_id2name:
            raise ValueError("No sensor with id %d exists." % id)
        return self._sensor_id2name[id]

    def sensor_name2id(self, name):
        if name not in self._sensor_name2id:
            raise ValueError(
                'No "sensor" with name %s exists. Available "sensor" names = %s.' % (name, self.sensor_names)
            )
        return self._sensor_name2id[name]

    def tendon_id2name(self, id):
        if id not in self._tendon_id2name:
            raise ValueError("No tendon with id %d exists." % id)
        return self._tendon_id2name[id]

    def tendon_name2id(self, name):
        if name not in self._tendon_name2id:
            raise ValueError(
                'No "tendon" with name %s exists. Available "tendon" names = %s.' % (name, self.tendon_names)
            )
        return self._tendon_name2id[name]

    def mesh_id2name(self, id):
        if id not in self._mesh_id2name:
            raise ValueError("No mesh with id %d exists." % id)
        return self._mesh_id2name[id]

    def mesh_name2id(self, name):
        if name not in self._mesh_name2id:
            raise ValueError('No "mesh" with name %s exists. Available "mesh" names = %s.' % (name, self.mesh_names))
        return self._mesh_name2id[name]

    # def userdata_id2name(self, id):
    #     if id not in self._userdata_id2name:
    #         raise ValueError("No userdata with id %d exists." % id)
    #     return self._userdata_id2name[id]

    # def userdata_name2id(self, name):
    #     if name not in self._userdata_name2id:
    #         raise ValueError("No \"userdata\" with name %s exists. Available \"userdata\" names = %s." % (name, self.userdata_names))
    #     return self._userdata_name2id[name]

    def get_xml(self):
        with TemporaryDirectory() as td:
            filename = os.path.join(td, "model.xml")
            ret = mujoco.mj_saveLastXML(filename.encode(), self._model)
            return open(filename).read()

    def get_joint_qpos_addr(self, name):
        """
        See https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi#L1178

        Returns the qpos address for given joint.
        Returns:
        - address (int, tuple): returns int address if 1-dim joint, otherwise
            returns the a (start, end) tuple for pos[start:end] access.
        """
        joint_id = self.joint_name2id(name)
        joint_type = self.jnt_type[joint_id]
        joint_addr = self.jnt_qposadr[joint_id]
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            ndim = 7
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            ndim = 4
        else:
            assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
            ndim = 1

        if ndim == 1:
            return joint_addr
        else:
            return (joint_addr, joint_addr + ndim)

    def get_joint_qvel_addr(self, name):
        """
        See https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi#L1202

        Returns the qvel address for given joint.
        Returns:
        - address (int, tuple): returns int address if 1-dim joint, otherwise
            returns the a (start, end) tuple for vel[start:end] access.
        """
        joint_id = self.joint_name2id(name)
        joint_type = self.jnt_type[joint_id]
        joint_addr = self.jnt_dofadr[joint_id]
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            ndim = 6
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            ndim = 3
        else:
            assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
            ndim = 1

        if ndim == 1:
            return joint_addr
        else:
            return (joint_addr, joint_addr + ndim)


class _MjDataMeta(type):
    """
    Metaclass which allows MjData below to delegate to mujoco.MjData.

    Taken from dm_control.
    """

    def __new__(cls, name, bases, dct):
        for attr in dir(mujoco.MjData):
            if not attr.startswith("_"):
                if attr not in dct:
                    # pylint: disable=protected-access
                    fget = lambda self, attr=attr: getattr(self._data, attr)
                    fset = lambda self, value, attr=attr: setattr(self._data, attr, value)
                    # pylint: enable=protected-access
                    dct[attr] = property(fget, fset)
        return super().__new__(cls, name, bases, dct)


class MjData(metaclass=_MjDataMeta):
    """Wrapper class for a MuJoCo 'mjData' instance.
    MjData contains all of the dynamic variables and intermediate results produced
    by the simulation. These are expected to change on each simulation timestep.
    """

    def __init__(self, model):
        """Construct a new MjData instance.
        Args:
          model: An MjModel instance.
        """
        self._model = model
        self._data = mujoco.MjData(model._model)

    @property
    def model(self):
        """The parent MjModel for this MjData instance."""
        return self._model

    def __del__(self):
        # free mujoco data
        del self._data

    """
    Some methods supported by sim.data in mujoco-py.
    Copied from https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi#L2611
    """

    @property
    def body_xpos(self):
        """
        Note: mujoco-py used to support sim.data.body_xpos but DM mujoco bindings requires sim.data.xpos,
              so we explicitly expose this as a property
        """
        return self._data.xpos

    @property
    def body_xquat(self):
        """
        Note: mujoco-py used to support sim.data.body_xquat but DM mujoco bindings requires sim.data.xquat,
              so we explicitly expose this as a property
        """
        return self._data.xquat

    @property
    def body_xmat(self):
        """
        Note: mujoco-py used to support sim.data.body_xmat but DM mujoco bindings requires sim.data.xmax,
              so we explicitly expose this as a property
        """
        return self._data.xmat

    def get_body_xpos(self, name):
        bid = self.model.body_name2id(name)
        return self.xpos[bid]

    def get_body_xquat(self, name):
        bid = self.model.body_name2id(name)
        return self.xquat[bid]

    def get_body_xmat(self, name):
        bid = self.model.body_name2id(name)
        return self.xmat[bid].reshape((3, 3))

    def get_body_jacp(self, name):
        bid = self.model.body_name2id(name)
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model._model, self._data, jacp, None, bid)
        return jacp

    def get_body_jacr(self, name):
        bid = self.model.body_name2id(name)
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model._model, self._data, None, jacr, bid)
        return jacr

    def get_body_xvelp(self, name):
        jacp = self.get_body_jacp(name)
        xvelp = np.dot(jacp, self.qvel)
        return xvelp

    def get_body_xvelr(self, name):
        jacr = self.get_body_jacr(name)
        xvelr = np.dot(jacr, self.qvel)
        return xvelr

    def get_geom_xpos(self, name):
        gid = self.model.geom_name2id(name)
        return self.geom_xpos[gid]

    def get_geom_xmat(self, name):
        gid = self.model.geom_name2id(name)
        return self.geom_xmat[gid].reshape((3, 3))

    def get_geom_jacp(self, name):
        gid = self.model.geom_name2id(name)
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacGeom(self.model._model, self._data, jacp, None, gid)
        return jacp

    def get_geom_jacr(self, name):
        gid = self.model.geom_name2id(name)
        jacv = np.zeros((3, self.model.nv))
        mujoco.mj_jacGeom(self.model._model, self._data, None, jacv, gid)
        return jacr

    def get_geom_xvelp(self, name):
        jacp = self.get_geom_jacp(name)
        xvelp = np.dot(jacp, self.qvel)
        return xvelp

    def get_geom_xvelr(self, name):
        jacr = self.get_geom_jacr(name)
        xvelr = np.dot(jacr, self.qvel)
        return xvelr

    def get_site_xpos(self, name):
        sid = self.model.site_name2id(name)
        return self.site_xpos[sid]

    def get_site_xmat(self, name):
        sid = self.model.site_name2id(name)
        return self.site_xmat[sid].reshape((3, 3))

    def get_site_jacp(self, name):
        sid = self.model.site_name2id(name)
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model._model, self._data, jacp, None, sid)
        return jacp

    def get_site_jacr(self, name):
        sid = self.model.site_name2id(name)
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model._model, self._data, None, jacr, sid)
        return jacr

    def get_site_xvelp(self, name):
        jacp = self.get_site_jacp(name)
        xvelp = np.dot(jacp, self.qvel)
        return xvelp

    def get_site_xvelr(self, name):
        jacr = self.get_site_jacr(name)
        xvelr = np.dot(jacr, self.qvel)
        return xvelr

    def get_camera_xpos(self, name):
        cid = self.model.camera_name2id(name)
        return self.cam_xpos[cid]

    def get_camera_xmat(self, name):
        cid = self.model.camera_name2id(name)
        return self.cam_xmat[cid].reshape((3, 3))

    def get_light_xpos(self, name):
        lid = self.model.light_name2id(name)
        return self.light_xpos[lid]

    def get_light_xdir(self, name):
        lid = self.model.light_name2id(name)
        return self.light_xdir[lid]

    def get_sensor(self, name):
        sid = self.model.sensor_name2id(name)
        return self.sensordata[sid]

    def get_mocap_pos(self, name):
        body_id = self.model.body_name2id(name)
        mocap_id = self.model.body_mocapid[body_id]
        return self.mocap_pos[mocap_id]

    def set_mocap_pos(self, name, value):
        body_id = self.model.body_name2id(name)
        mocap_id = self.model.body_mocapid[body_id]
        self.mocap_pos[mocap_id] = value

    def get_mocap_quat(self, name):
        body_id = self.model.body_name2id(name)
        mocap_id = self.model.body_mocapid[body_id]
        return self.mocap_quat[mocap_id]

    def set_mocap_quat(self, name, value):
        body_id = self.model.body_name2id(name)
        mocap_id = self.model.body_mocapid[body_id]
        self.mocap_quat[mocap_id] = value

    def get_joint_qpos(self, name):
        addr = self.model.get_joint_qpos_addr(name)
        if isinstance(addr, (int, np.int32, np.int64)):
            return self.qpos[addr]
        else:
            start_i, end_i = addr
            return self.qpos[start_i:end_i]

    def set_joint_qpos(self, name, value):
        """
        See https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi#L2821
        """
        addr = self.model.get_joint_qpos_addr(name)
        if isinstance(addr, (int, np.int32, np.int64)):
            self.qpos[addr] = value
        else:
            start_i, end_i = addr
            value = np.array(value)
            assert value.shape == (end_i - start_i,), "Value has incorrect shape %s: %s" % (name, value)
            self.qpos[start_i:end_i] = value

    def get_joint_qvel(self, name):
        addr = self.model.get_joint_qvel_addr(name)
        if isinstance(addr, (int, np.int32, np.int64)):
            return self.qvel[addr]
        else:
            start_i, end_i = addr
            return self.qvel[start_i:end_i]

    def set_joint_qvel(self, name, value):
        addr = self.model.get_joint_qvel_addr(name)
        if isinstance(addr, (int, np.int32, np.int64)):
            self.qvel[addr] = value
        else:
            start_i, end_i = addr
            value = np.array(value)
            assert value.shape == (end_i - start_i,), "Value has incorrect shape %s: %s" % (name, value)
            self.qvel[start_i:end_i] = value


class MjSim:
    """
    Meant to somewhat replicate functionality in mujoco-py's MjSim object
    (see https://github.com/openai/mujoco-py/blob/master/mujoco_py/mjsim.pyx).
    """

    def __init__(self, model):
        """
        Args:
            model: should be an MjModel instance created via a factory function
                such as mujoco.MjModel.from_xml_string(xml)
        """
        self.model = MjModel(model)
        self.data = MjData(self.model)

        # offscreen render context object
        self._render_context_offscreen = None

    @classmethod
    def from_xml_string(cls, xml):
        model = mujoco.MjModel.from_xml_string(xml)
        return cls(model)

    @classmethod
    def from_xml_file(cls, xml_file):
        f = open(xml_file, "r")
        xml = f.read()
        f.close()
        return cls.from_xml_string(xml)

    def reset(self):
        """Reset simulation."""
        mujoco.mj_resetData(self.model._model, self.data._data)

    def forward(self):
        """Forward call to synchronize derived quantities."""
        mujoco.mj_forward(self.model._model, self.data._data)

    def step(self, with_udd=True):
        """Step simulation."""
        mujoco.mj_step(self.model._model, self.data._data)

    def render(
        self,
        width=None,
        height=None,
        *,
        camera_name=None,
        depth=False,
        mode="offscreen",
        device_id=-1,
        segmentation=False,
    ):
        """
        Renders view from a camera and returns image as an `numpy.ndarray`.
        Args:
        - width (int): desired image width.
        - height (int): desired image height.
        - camera_name (str): name of camera in model. If None, the free
            camera will be used.
        - depth (bool): if True, also return depth buffer
        - device (int): device to use for rendering (only for GPU-backed
            rendering).
        Returns:
        - rgb (uint8 array): image buffer from camera
        - depth (float array): depth buffer from camera (only returned
            if depth=True)
        """
        if camera_name is None:
            camera_id = None
        else:
            camera_id = self.model.camera_name2id(camera_name)

        assert mode == "offscreen", "only offscreen supported for now"
        assert self._render_context_offscreen is not None
        with _MjSim_render_lock:
            self._render_context_offscreen.render(
                width=width, height=height, camera_id=camera_id, segmentation=segmentation
            )
            return self._render_context_offscreen.read_pixels(width, height, depth=depth, segmentation=segmentation)

    def add_render_context(self, render_context):
        assert render_context.offscreen
        if self._render_context_offscreen is not None:
            # free context
            del self._render_context_offscreen
        self._render_context_offscreen = render_context

    def get_state(self):
        """Return MjSimState instance for current state."""
        return MjSimState(
            time=self.data.time,
            qpos=np.copy(self.data.qpos),
            qvel=np.copy(self.data.qvel),
        )

    def set_state(self, value):
        """
        Set internal state from MjSimState instance. Should
        call @forward afterwards to synchronize derived quantities.
        """
        self.data.time = value.time
        self.data.qpos[:] = np.copy(value.qpos)
        self.data.qvel[:] = np.copy(value.qvel)

    def set_state_from_flattened(self, value):
        """
        Set internal mujoco state using flat mjstate array. Should
        call @forward afterwards to synchronize derived quantities.

        See https://github.com/openai/mujoco-py/blob/4830435a169c1f3e3b5f9b58a7c3d9c39bdf4acb/mujoco_py/mjsimstate.pyx#L54
        """
        state = MjSimState.from_flattened(value, self)

        # do this instead of @set_state to avoid extra copy of qpos and qvel
        self.data.time = state.time
        self.data.qpos[:] = state.qpos
        self.data.qvel[:] = state.qvel

    def free(self):
        # clean up here to prevent memory leaks
        del self._render_context_offscreen
        del self.data
        del self.model
        del self
        gc.collect()
