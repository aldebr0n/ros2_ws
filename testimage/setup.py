from setuptools import find_packages, setup

package_name = 'testimage'

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
    maintainer='afr',
    maintainer_email='afr@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    #tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'talker2 = testimage.testimage:main',
        'publish = testimage.imagepub:main',
        'subb= testimage.photusub:main',
        'graph=testimage.graphsub:main',
        'convert=testimage.conversion:main',
        
        
        ],
    },
)
