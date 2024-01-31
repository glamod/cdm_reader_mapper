.. cdm documentation master file, created by
   sphinx-quickstart on Fri Apr 16 14:18:24 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root ``toctree`` directive.

Tool set up
===========

The **cdm**  toolbox is a pure Python package, but it has a few dependencies that rely in a specific python and module version.

From source
~~~~~~~~~~~

The source for the **mdf_reader** can be downloaded from the `GitHub repository`_ via git_.

You can either clone the public repository:

.. code-block:: console

    git clone https://github.com/glamod/cdm_reader_mapper

or download th tarball_:

.. code-block:: console

   curl -OJL https://github.com/glamod/cdm_reader_mapper/tarball/master

Once you have a copy of the source, you caninstall it with pip_:

.. code-block:: console

   pip install -e .

Stable release (not possible yet)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install the **cdm** toolbox in your current conda_ environment, run this command in your terminal:

.. code-block:: console

  pip install cdm

In the future, this will be the preferred method to install the **cdm** toolbox, as it will always install the moste recent stable release.

.. include:: hyperlinks.rst
