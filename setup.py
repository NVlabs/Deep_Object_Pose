from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'dope_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*/*')),
        (os.path.join('share', package_name, 'weights'), glob('weights/*.pth')),
    ],
    install_requires=['setuptools', 'inference_script'],
    zip_safe=True,
    maintainer='sfederico',
    maintainer_email='sfederico@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'dope_node = dope_ros2.dope_node:main',
        ],
    },
)
