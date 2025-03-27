.. cdm documentation master file, created by
   sphinx-quickstart on Fri Apr 16 14:18:24 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root ``toctree`` directive.

Installation
============

The **cdm_reader_mapper**  toolbox is a pure Python package, but it has a few dependencies that rely in a specific python and module version.

Stable release
~~~~~~~~~~~~~~

To install the **cdm_reader_mapper** toolbox in your current environment, run this command in your terminal:

.. code-block:: console

  pip install cdm_reader_mapper

This is the preferred method to install the **cdm_reader_mapper** toolbox, as it will always install the most recent stable release.

Alternatively, it can be installed using the `uv`_ package manager:

.. code-block:: console

    uv add cdm_reader_mapper

.. include:: hyperlinks.rst

From source
~~~~~~~~~~~

.. warning:: It is not guaranteed that the version on source will run stably. Therefore, we highly recommend to use the ``Stable release`` installation.

The source for the **cdm_reader_mapper** can be downloaded from the `GitHub repository`_ via git_.

You can either clone the public repository:

.. code-block:: console

    git clone https://github.com/glamod/cdm_reader_mapper

or download th tarball_:

.. code-block:: console

   curl -OJL https://github.com/glamod/cdm_reader_mapper/tarball/master

Once you have a copy of the source, you can install it with pip_:

.. code-block:: console

   pip install -e .

Or using the `uv`_ package manager to install cdm_reader_mapper:

.. code-block:: console

    uv add .

Development mode
~~~~~~~~~~~~~~~~

If you're interested in participating in the development of the **cdm_reader_mapper** toolbox, you can install the package in development mode after cloning the repository from source:

.. code-block:: console

    pip install -e .[dev]      # Install optional development dependencies in addition
    pip install -e .[docs]     # Install optional dependencies for the documentation in addition
    pip install -e .[all]      # Install all the above for complete dependency version

Alternatively, you can use the uv package manager:

.. code-block:: console

    uv sync       # Install in development mode and create a virtual environment

You can specify optional dependency groups with the `--extra` option.

Creating a Conda Environment
----------------------------

To create a conda environment including **cdm_reader_mapper**'s dependencies and and development dependencies, run the following command from within your cloned repo:

.. code-block:: console

    $ conda env create -n my_cdm_env python=3.12 --file=environment.yml
    $ conda activate my_cdm_env
    (my_cdm_env) $ python -m pip install -e --no-deps .
