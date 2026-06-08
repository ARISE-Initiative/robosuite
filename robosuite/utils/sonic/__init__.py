"""SONIC whole-body control engine for the robosuite G1 integration.

- controller.G1SonicController : obs build + per-motor PD + effort clip (ported from
  gear_sonic base_sim; the engine the SonicWholeBodyController composite controller wraps)
- sources : command sources (DDSCommandSource live C++/DDS, ReplayCommandSource,
  ReferenceMockSource) + MotorCommand / CommandSource
- debug.DDSDebug : per-timestep recorder for golden-trajectory tests
"""
from .controller import G1SonicController, MotorCommand, CommandSource
from .sources import (DDSCommandSource, ReplayCommandSource, ReferenceMockSource,
                      init_dds_once)
from .debug import DDSDebug
