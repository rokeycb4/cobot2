from setuptools import find_packages, setup
import glob
import os
package_name = 'dovis'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=[
        'control', 
        'gpt_processing', 
        'object'
    ]),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
	('share/' + package_name + '/resource', glob.glob('resource/*')),
	('share/' + package_name + '/resource', glob.glob('resource/.env')),
    ('share/' + package_name + '/launch', glob.glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='moon',
    maintainer_email='mjw3723@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'control = control.robot_control:main',
            'gpt = gpt_processing.gpt:main',
            'object = object.detection:main',
            'person = object.person:main',
        ],
    },
)
