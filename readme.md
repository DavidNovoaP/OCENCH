# OCENCH

OCENCH: A One-Class Classification method based on Expanded Non-Convex Hulls

## Pip Installation
Pypi url: https://pypi.org/project/ocench/. To install, run the command

    pip install ocench

## Cite

If you plan to use this code, please cite the following paper:

    @article{NOVOAPARADELA20231,
        title = {A One-Class Classification method based on Expanded Non-Convex Hulls},
        journal = {Information Fusion},
        volume = {89},
        pages = {1-15},
        year = {2023},
        issn = {1566-2535},
        doi = {https://doi.org/10.1016/j.inffus.2022.07.023},
        url = {https://www.sciencedirect.com/science/article/pii/S1566253522000896},
        author = {David Novoa-Paradela and Oscar Fontenla-Romero and Bertha Guijarro-Berdiñas},
        keywords = {Machine learning, One-Class Classification, Convex Hull, Delaunay triangulation, Random projections, Ensemble learning},
        abstract = {This paper presents an intuitive, robust and efficient One-Class Classification algorithm. The method developed is called OCENCH (One-class Classification via Expanded Non-Convex Hulls) and bases its operation on the construction of subdivisible and expandable non-convex hulls to represent the target class. The method begins by reducing the dimensionality of the data to two-dimensional spaces using random projections. After that, an iterative process based on Delaunay triangulations is applied to these spaces to obtain simple polygons that characterizes the non-convex shape of the normal class data. In addition, the method subdivides the non-convex hulls to represent separate regions in space if necessary. The method has been evaluated and compared to several main algorithms of the field using real data sets. In contrast to other methods, OCENCH can deal with non-convex and disjointed shapes. Finally, its execution can be carried out in a parallel way, which is interesting to reduce the execution time.}
    }