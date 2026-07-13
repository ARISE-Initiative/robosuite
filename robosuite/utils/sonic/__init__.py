"""SONIC whole-body control engine for the robosuite G1 integration.

- controller.G1SonicController : obs build + per-motor PD + effort clip (ported from
  gear_sonic base_sim; the engine the SonicWholeBodyController composite controller wraps)
- sources.DDSCommandSource : live C++/DDS command source (+ MotorCommand)
"""
from .controller import G1SonicController, MotorCommand
from .sources import DDSCommandSource, init_dds_once
