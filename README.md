Mapping cell locations via multi-layer regionalization constraints
=========================================================================

Introduction
------------
Resolving spatial cell arrangement is crucial for understanding physiological and pathological processes. While scRNA-seq captures gene expression at single-cell resolution, it loses spatial context, and current spatial transcriptomics methods often compromise on throughput or resolution. Existing integration methods face challenges with accuracy and scalability due to noise from molecular diffusion, cell segmentation errors, and disproportionate cell-type representation. We present Polyomino, an algorithm framework employing multi-layered regional constraints to accurately assign cell locations, enhancing spatial accuracy and resilience to noise. Comparative analysis on benchmark datasets demonstrates Polyomino’s superior accuracy and scalability over existing methods. Applied to liver cancer tissue, Polyomino revealed spatial heterogeneity of cDC cells, a detail missed by deconvolution-based techniques, and achieved cell-cell interaction resolution beyond traditional mapping approaches. Additionally, Polyomino outperforms current techniques in computational efficiency and resource usage, particularly with large-scale stereo-seq data, underscoring its potential for broad application.

![](https://s2.loli.net/2024/08/28/mN2Bsg1W6aVpljh.png)

Installation
------------
Polyomino can be installed either through GitHub or PyPI.

To install from GitHub:

    git clone https://github.com/caiquanyou/Polyomino
    cd Polyomino
    python setup.py install # or pip install .

Alternatively, install via PyPI using:

    pip install Polyomino

Usage
-----
After installation, Polyomino can be used in Python as follows:

    import Polyomino as plmo

Contributing
------------
Contributions to Polyomino are welcome. Please refer to the project's issues and pull requests for areas where you can help.

License
-------
**Free Software**

Support and Contact
-------------------
For support or to contact the developers, please use the project's GitHub Issues page.