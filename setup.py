from setuptools import find_packages
from setuptools import setup
REQUIRED_PACKAGES = ['python_speech_features','scipy','numpy','matlab.engine']
setup(
    name='tf',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='CS224S Accent Conversion'
)
