import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition

def generate_launch_description():
    # Get the path to the config_dope.yaml file
    config_file = os.path.join(get_package_share_directory('dope_ros2'), 'config', 'config_dope.yaml')

    # Declare the log_level argument
    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Logging level'
    )

    rviz_absolute_path = "/home/sfederico/pick_and_place_ws/src/dope_ros2/rviz/rviz.rviz" #os.path.join(get_package_share_directory('dope_ros2'),'rviz','rviz.rviz')
    rviz_ = LaunchConfiguration("rviz_")

    rviz_launch_arg = DeclareLaunchArgument(
        name='rviz_',
        default_value= "true",
        description='Set to true if you want to visualize on rviz, false otherwise.'
    )

    # Launch the dope_node node
    dope_node = Node(
        package='dope_ros2',
        executable='dope_node',
        name='dope_node',
        output='screen',
        emulate_tty=True,
        parameters=[config_file],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')]
    )

    node_rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_absolute_path],
        condition=IfCondition(rviz_)
    )


    return LaunchDescription([
        log_level_arg,
        rviz_launch_arg,
        node_rviz,
        dope_node
    ])
