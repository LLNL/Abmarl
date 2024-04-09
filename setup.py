import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='abmarl',
    version='0.2.8',
    description='Agent Based Simulation and MultiAgent Reinforcement Learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/llnl/abmarl',
    author='Ephraim Rusu',
    author_email='rusu1@llnl.gov',
    license='BSD 3',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    project_urls={
        'Featured Usage': 'https://abmarl.readthedocs.io/en/latest/featured_usage.html',
        'Documentation': 'https://abmarl.readthedocs.io/en/latest/index.html',
    },
    packages=setuptools.find_packages(exclude=['tests']),
    install_requires=[
        'matplotlib',
        'gymnasium'
    ],
    extras_require={
        "rllib": [
            'torch',
            'ray[rllib]==2.9.3',
        ],
        "open-spiel": [
            'open-spiel'
        ]
    },
    python_requires='>=3.7, <3.11',
    entry_points={
        'console_scripts': [
            'abmarl=abmarl.scripts.scripts:cli'
        ]
    },
)
