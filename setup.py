import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="seeker",
    version="0.0.1",
    author="Eduardo Cerqueira",
    author_email="eduardomcerqueira@gmail.com",
    description="seeking for code snippet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eduardomcerqueira/seeker",
    project_urls={
        "Bug Tracker": "https://github.com/eduardomcerqueira/seeker/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Linux",
    ],
    packages=setuptools.find_packages(),
    install_requires=['requests>=2.20.0', 'PyGithub>=1.55', 'click>=8.0.0'],
    py_modules=['seeker'],
    entry_points={
        'console_scripts': [
            'seeker = seeker.main:cli',
        ],
    },
)
