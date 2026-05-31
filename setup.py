import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="seeker",
    version="0.0.2",
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
        "Programming Language :: Python :: 3.12",
        "Operating System :: Linux",
    ],
    python_requires=">=3.12,<3.13",
    packages=setuptools.find_packages(include=["seeker", "seeker.*"]),
    install_requires=[
        "requests==2.32.5",
        "PyGithub==2.8.1",
        "click==8.1.8",
    ],
    py_modules=["seeker"],
    entry_points={
        "console_scripts": [
            "seeker = seeker.main:cli",
        ],
    },
)
