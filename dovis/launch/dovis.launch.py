from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='dovis',
            executable='gpt',
            name='GPT',
            output='screen',
        ),
        Node(
            package='dovis',
            executable='object',
            name='Detection',
            output='screen',
        ),
        Node(
            package='dovis',
            executable='control',
            name='Controller',
            output='screen',
        ),
        Node(
            package='dovis',
            executable='person',
            name='Person',
            output='screen',
        )
    ])