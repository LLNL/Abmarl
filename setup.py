import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='admiral',
    version='0.0.1',
    description='Agent Based Simulation and Reinforcement Learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://lc.llnl.gov/gitlab/rusu1/admiral',
    author='Edward Rusu',
    license='BSD 3',
    packages=setuptools.find_packages(),
    entry_points={
        'console_scripts': [
            'admiral=admiral.scripts.scripts:cli'
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD 3 License',
        'Operating System :: OS Independent',
    ],
    python_required='>=3.7',
)