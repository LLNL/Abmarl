import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='abmarl',
    version='0.1.2',
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
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    project_urls={
        'Highlights': 'https://abmarl.readthedocs.io/en/latest/highlights.html',
        'Documentation': 'https://abmarl.readthedocs.io/en/latest/index.html',
    },
    packages=setuptools.find_packages(),
    install_requires=[
        'tensorflow==2.4.0',
        'ray[rllib]==1.2.0',
        'matplotlib',
        'seaborn',
    ],
    python_required='>=3.7',
    entry_points={
        'console_scripts': [
            'abmarl=abmarl.scripts.scripts:cli'
        ]
    },
)
