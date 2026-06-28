from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="EQTransformer",
    author="S. Mostafa Mousavi",
    version="0.1.61",
    author_email="smousavi05@gmail.com",
    description="A python package for making and using attentive deep-learning models for earthquake signal detection and phase picking.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/smousavi05/EQTransformer",
    license="MIT",
    packages=find_packages(),
    keywords='Seismology, Earthquakes Detection, P&S Picking, Deep Learning, Attention Mechanism',
    install_requires=[
	'pytest==7.1.2',
	'numpy==1.23.0',     # appox version: numpy 1.19.x but at least 1.19.2
	'keyring==23.7.0', 
	'pkginfo==1.8.3',
	'scipy==1.10.0',
	# 'tensorflow-deps==2.9.0',
	'tensorflow-estimator==2.8.0',
	# 'tensorflow-macos==2.9.2',
	'tensorflow==2.8.2', # tensorflow <2.7.0 needs numpy <1.20.0
	'keras==2.8.0', 
	'matplotlib==3.7.2',
	'pandas==1.4.3',
	'tqdm==4.64.0', 
	'h5py==3.6.0', 
	'obspy==1.3.0',
	'jupyter==1.0.0',
	'httpcore==0.13.5',
	'anyio==3.* ',
	'ipykernel==6.* '
	], 

    python_requires='~=3.10',
)


