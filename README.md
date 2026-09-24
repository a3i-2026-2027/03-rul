---
title: AI in the Industry (Remaining Useful Life Prediction)
author: michele.lombardi2@unibo.it
---

# Remaining Useful Life Prediction

This a lecture block of the "AI in the Industry" course, from [University of Bologna](https://www.unibo.it). Each lecture consists of a tutorial that tackles a simplified industrial problem and tackles it using AI techniques, from Machine Learning to Combinatorial Optimization (and later on their combination).

While the whole course looks like cookbook, the real goal is using examples to teach how industrial problem can be methodically approaches, analyzed, and tackles using a combination of techniques.

This tutorial in particular tackles the problem of designing maintenance policies based on the Remaining Useful Life of an industrial components. Surrogate-based Bayesian Optimization is introduced as a way to optimize parameters when evaluating their impact involves a computational expensive operation.

# Accessing the Lecture #

## Local Execution (Preferred) ##

Students are strongly encouraged to _run all lectures locally_. Doing this will require to:

* Download or clone this repository, e.g. via the command:
```sh
git clone https://github.com/a3i-2026-2027/03-rul.git
```
* Open a terminal in the main directory of the downloaded/cloned repository
* Install the required dependencies, _by using a virtual environment_. Each lectures already contains configuration files to manage dependencies via [poetry](https://python-poetry.org). If you are using poetry, you can install dependencies via:
```sh
poetry install
```
* Start the notebook server in a virtual environment
```sh
poetry run jupyter notebook
```

If you do not wish to use poetry, a `requirements.txt` file with frozen dependency versions is also provided. However, **you are strongly advised to still use virtual environments**. The course will eventually make use of multiple libraries and package versions between lectures will sometimes be inconsistent.

## Read-only Access and PDF Notes ##

You can inspect the individual notebooks in by just clicking on any `*.ipynb` file in the `notebooks` directory: github provides a notebook viewer that mostly works, though this access method may occasionally have issue when displaying plots.

The repository contains PDF notes for all the notebooks. They can be used for read-only access (with more consistent results compared to the github notebook viewer), but more importantly they can be useful to add annotations. Just keep in mind that in case of updates, cloning the repository will replace the PDF files.
