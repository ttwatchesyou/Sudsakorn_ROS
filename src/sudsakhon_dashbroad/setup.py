from setuptools import find_packages, setup

package_name = 'sudsakhon_dashbroad'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'flask'],
    zip_safe=True,
    maintainer='Sudsakhon User',
    description='ROS2 Dashboard for systemd service monitoring',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dashboard_node = sudsakhon_dashbroad.dashboard_node:main',
            'api_backend = sudsakhon_dashbroad.api_backend:main'
        ],
    },
)