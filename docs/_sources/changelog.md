# Changelog

## Version 1.5.1

- **Bug Fixes**:  
  - Fixed segmentation demo, part controller demo, demo renderer, joint and actuator issues, equality removal, and orientation mismatch in Fourier hands.  

- **Documentation Updates**:  
  - Updated `basicusage.md`, overview page, demo docs, teleop usage info, devices, and related docs.  
  - Added a CI doc update workflow and `.nojekyll` fix.  
  - Simplified composite controller keywords and updated robots section and task images.  

- **Features/Enhancements**:  
  - Added GymWrapper support for both `gym` and `gymnasium` with `dict_obs`.  
  - Updated DexMimicGen assets and added a default whole-body Mink IK config.  
  - Improved CI to check all tests and print video save paths in demo recordings. 
  - Add GR1 and spot robot to `demo_random_actions.py` script.

- **Miscellaneous**:  
  - Added troubleshooting for SpaceMouse failures and terminated `mjviewer` on resets.  
  - Adjusted OSC position fixes and updated part controller JSONs. 

## Version 1.5.0

The 1.5 release of **Robosuite** introduces significant advancements to extend flexibility and realism in robotic simulations. Key highlights include support for diverse robot embodiments (e.g., humanoids), custom robot compositions, composite controllers (such as whole-body controllers), expanded teleoperation devices, and photorealistic rendering capabilities.

### New Features
- **Diverse Robot Embodiments**: Support for complex robots, including humanoids, allowing exploration of advanced manipulation and mobility tasks. Please see [robosuite_models](https://github.com/ARISE-Initiative/robosuite_models) for extra robosuite-compatible robot models.
- **Custom Robot Composition**: Users can now build custom robots from modular components, offering extensive configuration options.
- **Composite Controllers**: New controller abstraction includes whole-body controllers, and the ability to control robots with composed body parts, arms, and grippers.
- **Additional Teleoperation Devices**: Expanded compatibility with teleoperation tools like drag-and-drop in the MuJoCo viewer and Apple Vision Pro.
- **Photorealistic Rendering**: Integration of NVIDIA Isaac Sim for enhanced, real-time photorealistic visuals, bringing simulations closer to real-world fidelity.

### Improvements
- **Updated Documentation**: New tutorials and expanded documentation on utilizing advanced controllers, teleoperation, and rendering options.
- **Simulation speed improvement**: By default we set the `lite_physics` flag to True to skip redundant calls to [`env.sim.step()`](https://github.com/ARISE-Initiative/robosuite/blob/29e73bd41f9bc43ba88bb7d2573b868398905819/robosuite/environments/base.py#L444)

### Migration

- Composite controller refactoring: please see example of [usage](https://github.com/ARISE-Initiative/robosuite/blob/29e73bd41f9bc43ba88bb7d2573b868398905819/robosuite/examples/third_party_controller/mink_controller.py#L421)
