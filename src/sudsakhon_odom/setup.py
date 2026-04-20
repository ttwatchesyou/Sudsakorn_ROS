from setuptools import find_packages, setup

package_name = 'sudsakhon_odom'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'sudsakhon_odom_node = sudsakhon_odom.sudsakhon_odom_node:main',
            'sudsakhon_odom_tune = sudsakhon_odom.sudsakhon_odom_node_tune:main',
        ],
    },
)
