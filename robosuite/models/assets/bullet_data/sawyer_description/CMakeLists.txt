cmake_minimum_required(VERSION 2.8.3)

project(sawyer_description)

find_package(catkin REQUIRED)

catkin_package()

foreach(dir config meshes params urdf)
   install(DIRECTORY ${dir}/
      DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)
