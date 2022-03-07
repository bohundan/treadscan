.. treadscan documentation master file, created by
   sphinx-quickstart on Fri Feb  4 14:01:31 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Treadscan documentation
=======================

.. image:: _static/treadscan.jpg

Required Python version: ``python3.8`` or higher.

Required packages: ``numpy``, ``opencv-python``, ``opencv-contrib-python``, ``improutils``.

To build this documentation, you need ``Sphinx``, ``numpydoc`` and ``sphinx-rtd-theme``.

**Treadscan modules:**

.. autosummary::
   treadscan.detector
   treadscan.segmentor
   treadscan.extractor
   treadscan.utilities

**Treadscan classes:**

.. autosummary::
   treadscan.detector.Detector
   treadscan.detector.InputType
   treadscan.detector.FrameExtractor
   treadscan.segmentor.Segmentor
   treadscan.segmentor.RCNNSegmentor
   treadscan.extractor.Extractor
   treadscan.extractor.CameraPosition
   treadscan.utilities.Ellipse

.. toctree::
   :maxdepth: 2
   :caption: Table of Contents
   :name: mastertoc
   :includehidden:

   detector
   segmentor
   extractor
   utilities

Indices and tables
==================

* :ref:`genindex`
