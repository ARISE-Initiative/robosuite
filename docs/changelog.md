# Changelog


## Version 1.5.1

- **Documentation Updates**:  
  - Updated `basicusage.md`, overview page, demo docs, teleop usage info, devices, and related docs.  
  - Added a CI doc update workflow and `.nojekyll` fix.  
  - Simplified composite controller keywords and updated robots section and task images.  

- **Bug Fixes**:  
  - Fixed segmentation demo, part controller demo, demo renderer, joint and actuator issues, equality removal, and orientation mismatch in Fourier hands.  

- **Features/Enhancements**:  
  - Added GymWrapper support for both `gym` and `gymnasium` with `dict_obs`.  
  - Updated DexMimicGen assets and included a default whole-body Mink IK config.  
  - Improved CI to check all tests and print video save paths in demo recordings.  

- **Miscellaneous**:  
  - Added troubleshooting for SpaceMouse failures and terminated `mjviewer` on resets.  
  - Adjusted OSC position fixes and updated part controller JSONs.  

## Version 1.5.0

<div class="admonition warning">
<p class="admonition-title">Breaking API changes</p>
<div>
    <ul>New controller design.</ul>
</div>
</div>