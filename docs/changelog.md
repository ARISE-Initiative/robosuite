# Changelog



## Version 1.5.0

<div class="admonition warning">
<p class="admonition-title">Breaking API changes</p>
<div>
    <ul>
        <li>
        New controller design: Introduction of composite controllers, which take in a high-level action vector and converts it into commands for each body part controller.
        </li>
    </ul>
</div>
</div>


<div>
    <ul>
        <li>Introduction of custom robot composition: arms, grippers, and bases can be swapped to create new robots configurations</li>
        <li>Integration of more diverse robot embodiments: including humanoids, quadrupeds, and more</li>
        <li>Support for mobile manipulation, whole body IK, and third-party controllers (e.g. Mink)</li>
        <li>Implementation of MuJoCo viewer drag-drop teleoperation interface</li>
        <li>Support for photo-realistic rendering via USD exporting and NVIDIA Isaac Sim rendering</li>
    </ul>
</div>