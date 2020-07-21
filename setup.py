from setuptools import setup, find_packages

with open('requirements.txt') as fp:
    install_reqs = [r.rstrip() for r in fp.readlines()
                    if not r.startswith('#') and not r.startswith('git+')]

with open('skstab/__version__.py') as fh:
    version = fh.readlines()[-1].split()[-1].strip('\'\'')

setup(
    name='skstab',
    version=version,
    description='Clustering stability analysis with a scikit-learn compatible API',
    author='Florent Forest, Alex Mourer',
    author_email='f@florentfo.rest',
    packages=find_packages(),
    install_requires=install_reqs,
    url='https://github.com/FlorentF9/skstab'
)
