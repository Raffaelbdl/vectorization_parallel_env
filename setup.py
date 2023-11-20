from setuptools import setup, find_namespace_packages

with open("requirements.txt", "r") as f:
    requirements = [package.replace("\n", "") for package in f.readlines()]

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name="vec_parallel_env",
    url="https://github.com/Raffaelbdl/vectorization_parallel_env",
    author="Raffael Bolla Di Lorenzo",
    author_email="raffaelbdl@gmail.com",
    # Needed to actually package something
    packages=find_namespace_packages(),
    # Needed for dependencies
    install_requires=requirements[:],
    # dependency_links=requirements[1:],
    # package_data={"configs": ["*.yaml"]},
    # *strongly* suggested for sharing
    version="0.0.1",
    # The license can be anything you like
    license="MIT",
    description="Vectorize PettingZoo ParallelEnv",
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)
