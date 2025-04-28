# Welcome to ProgPy
**2024 NASA Software of the Year!**

NASA’s ProgPy is an open-sourced python package supporting research and development of prognostics and health management and predictive maintenance tools. It implements architectures and common functionality of prognostics, supporting researchers and practitioners. The ProgPy package is a combination of the original prog_models and prog_algs packages.

## Installing ProgPy

The latest stable release of ProgPy is hosted on PyPi. For most users, this version will be adequate. To install via the command line, use the following command:

```bash
pip install progpy
```

## Installing ProgPy- Prerelease

Users who would like to contribute to ProgPy or would like to use pre-release features can do so using the ProgPy GitHub repo. This isn’t recommended for most users as this version may be unstable. To do this, use the following commands:

```bash
git clone https://github.com/nasa/progpy
cd progpy
git checkout dev
pip install -e .```

## Citing this repository

Use the following to cite this repository in LaTeX:

```BibTeX
@misc{2023_nasa_progpy,
author = {Christopher Teubert and Katelyn Jarvis Griffith and Matteo Corbetta and Chetan Kulkarni and Portia Banerjee and Jason Watkins and Matthew Daigle},
title = {{ProgPy Python Prognostics Packages}},
month = Oct,
year = 2023,
version = {1.6},
url = {https://nasa.github.io/progpy}
doi = {10.5281/ZENODO.8097013}
}
```
The corresponding reference should look like this:

Teubert, K. Jarvis Griffith, M. Corbetta, C. Kulkarni, P. Banerjee, J. Watkins, M. Daigle, ProgPy Python Prognostics Packages, v1.6, Oct 2023. URL nasa/progpy.


## Contributing and Partnering

ProgPy was developed by researchers of the NASA Prognostics Center of Excellence (PCoE) and Diagnostics & Prognostics Group, with assistance from our partners. We welcome contributions and are actively interested in partnering with other organizations and researchers. If interested in contibuting, please email Chris Teubert at christopher.a.teubert@nasa.gov.

A big thank you to our partners who have contributed to the design, testing, and/or development of ProgPy:

* German Aerospace Center (DLR) Institute of Maintenance, Repair and Overhaul.
* Northrop Grumman Corporation (NGC)
* Research Institutes of Sweden (RISE)
* Vanderbilt University
