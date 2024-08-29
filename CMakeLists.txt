cmake_minimum_required(VERSION 3.0.2)
project(multimodal_emotion_detection)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
)

catkin_python_setup()

add_message_files(
  FILES
  EmotionData.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package()

catkin_package(
  CATKIN_DEPENDS rospy message_runtime std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)




catkin_install_python(PROGRAMS
  scripts/main.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)