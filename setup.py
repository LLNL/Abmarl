import setuptools
import subprocess


with open('README.md', 'r') as fh:
    long_description = fh.read()

def custom_find_packages():
    """
    Build external integration modules if those dependencies exist.
    """
    packages=[
        'abmarl',
        'abmarl.algs',
        'abmarl.examples',
        'abmarl.examples.sim',
        'abmarl.managers',
        'abmarl.policies',
        'abmarl.scripts',
        'abmarl.sim',
        'abmarl.sim.gridworld',
        'abmarl.sim.wrappers',
        'abmarl.tools',
        'abmarl.trainers',
    ]
    # TODO: This doesn't work because this function runs before dependencies are installed.
    # Need to see if there is way to build dependencies frist and then run this function....
    x = subprocess.Popen("pip3 list", shell=True, stdout=subprocess.PIPE).stdout.read().decode('utf-8')
    if 'ray' in x:
        packages.append('abmarl.external')
        print("\n\n\n\n\nRllib is installed\n\n\n\n\n")
    print("\n\n\n\n\nNot installed\n\n\n\n\n")


    # x = subprocess.Popen("pip3 list", shell=True, stdout=subprocess.PIPE).stdout.read().decode('utf-8')
    # if 'ray' in x:
    #     print("\n\n\nRllib is installed!")
    # if 'open-spiel' in x:
    #     print("\n\n\nOpen Spiel installed!")

    return packages

setuptools.setup(
    name='abmarl',
    version='0.2.7',
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
    packages=custom_find_packages(),
    install_requires=[
        'importlib-metadata<5.0',
        'numpy<1.24',
        'gym',
        'matplotlib',
        'seaborn',
    ],
    extras_require={
        "rllib": [
            'tensorflow',
            'ray[rllib]==2.0.0',
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
