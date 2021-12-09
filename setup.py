import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='abmarl',
    version='0.2.1',
    description='Agent Based Simulation and MultiAgent Reinforcement Learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/llnl/abmarl',
    author='Edward Rusu',
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
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    project_urls={
        'Featured Usage': 'https://abmarl.readthedocs.io/en/latest/featured_usage.html',
        'Documentation': 'https://abmarl.readthedocs.io/en/latest/index.html',
    },
    packages=setuptools.find_packages(),
    install_requires=[
        'tensorflow',
        'ray[rllib]==1.4.0',
        'matplotlib',
        'seaborn',
    ],
    python_requires='>=3.7, <3.9',
    entry_points={
        'console_scripts': [
            'abmarl=abmarl.scripts.scripts:cli'
        ]
    },
)
