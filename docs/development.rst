.. _dev_docs:

Developer documentation
=======================
We'd love your help, either as ideas, documentation, or code. If you have a new
module or want to add or fix existing code, please do. *skultrafast* tries to
follow Python's PEP-8 closely. New code should have numpy-doc styled docstrings
and unit tests. The aim is to increase the quality of the code base over time.

The easiest way to contribute is file bugs, questions and ideas as an issue on
_github. If the issue involves code, please provide small, complete, working
example that illustrates the problem.

Contributing code
-----------------
Contributing code is done via pull-requests on
`github <https://github.com/tillsten/skultrafast>`_. A detailed description of
the workflow can be found in the `Matplotlib documentation
<https://matplotlib.org/devel/gitwash/development_workflow.html#development-workflow>`_.


Documentation
-------------
The documentation is in `docs` directory and uses the Sphinx documentation
generator. Sphinx uses reStructuredText (`.rst`) as its makeup language. Simple
changes to the documentation can be done by using the github web interface
directly. Browse to the file, click it and use the `Edit this file` button. Use
the "Create a new branch for this commit and start a pull request." option for
submitting the change.

The code itself uses the numpy-doc style doc-strings for public functions and
classes. These doc-strings are in the source files itself. This part of can be
found in the :ref:`api_docs` section.

Running tests
-------------
Run ``pytest`` in the source directory.
