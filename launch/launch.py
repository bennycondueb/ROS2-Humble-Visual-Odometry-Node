from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():

    visual_odometry_left_camera = DeclareLaunchArgument(
        'camera_source_left',
        default_value='vimba_038',
        description='Left camera for the visual odometry'
    )

    visual_odometry_right_camera = DeclareLaunchArgument(
        'camera_source_right',
        default_value='vimba_039',
        description='Right camera for the visual odometry'
    )

    enable_plot_arg = DeclareLaunchArgument(
        'enable_vo_yaw_plot',
        default_value='False',
        description='Enable optional live Matplotlib plot for Visual Odometry yaw'
    )

    visual_odometry_node = Node(
        package='visual_odometry',
        executable='visual_odometry',
        parameters=[{
            'camera_source_left' : LaunchConfiguration('camera_source_left'),
            'camera_source_right' : LaunchConfiguration('camera_source_right'),
            'enable_vo_yaw_plot': LaunchConfiguration('enable_vo_yaw_plot'),
        }]
    )

    return LaunchDescription([
        # Parameters
        visual_odometry_left_camera, 
        visual_odometry_right_camera, 
        enable_plot_arg,

        # Nodes 
        visual_odometry_node
    ])