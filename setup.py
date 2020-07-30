from setuptools import setup, find_packages

# read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# read description
with open("README.md", 'r') as fh:
    long_description = fh.read()

setup(name='oias',
      version='0.0.1',
      packages=find_packages(),
      setup_requires=['wheel>=0.34.0'],
      install_requires=requirements,

      long_description=long_description,
      long_description_content_type='text/markdown',

      author='Daniel Danis',
      author_email='daniel.gordon.danis@protonmail.com',
      url='https://github.com/ielis/OffsetImbalanceAwareRandomForest',
      description='Offset and imbalance-aware random forest classifier',
      license='GPLv3',
      keywords='bioinformatics genomics algorithm random forest',

      package_data={
          '': ['test_data/*']},
      data_files=[('', ['requirements.txt'])]
      )
