import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.spatial import Delaunay, ConvexHull # Triangulizacion de Delaunay y calculo del Convex Hull
from bentley_ottmann.planar import segments_intersect # Implementacion del algoritmo Bentley Ottmann

from calcular_NCH import *
from aux_functions import *
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import random

def OCENCH_train (X, n_projections, l, extend):
    contraer_SCH = False
    plot = False
    subdividir = True
    # Generamos las proyecciones bidimensionales
    random.seed(10)
    projections = generate_Projections(n_projections, X.shape[1])
    # Proyectamos los datos en estos espacios 2D
    dataset_projected = project_Dataset(X, projections)
    # Calculamos el NCH y SNCH en cada proyección
    l_vertices = []
    l_aristas = []
    l_vertices_expandidos = []
    l_orden_vertices = []
    l_factor_expansion = []
    l_normalizadores = []
    l_cierres_separados = []
    l_asociacion_vertices_e_indices = []
    for i in range (0, n_projections):
        model_normalizer = NormalizeData_Train(dataset_projected[i])
        dataset_projected[i] = NormalizeData(dataset_projected[i], model_normalizer)
        l_normalizadores.append(model_normalizer)
    #print("Projection: ", 0) 
    tic1 = time.perf_counter()
    vertices_aux, aristas_aux, vertices_expandidos, orden_vertices_aux, factor_expansion_aux, cierres_separados, asociacion_vertices_e_indices =  calcular_NCH((dataset_projected[0], l, extend, contraer_SCH, subdividir, 0, plot))
    
    l_vertices.append(vertices_aux)
    l_aristas.append(aristas_aux)
    l_vertices_expandidos.append(vertices_expandidos)
    l_orden_vertices.append(orden_vertices_aux)
    l_factor_expansion.append(factor_expansion_aux)
    l_cierres_separados.append(cierres_separados)
    l_asociacion_vertices_e_indices.append(asociacion_vertices_e_indices)
    for i in range (1, n_projections):
        #print("Projection: ", i)
        vertices_aux, aristas_aux, vertices_expandidos, orden_vertices_aux, factor_expansion_aux, cierres_separados, asociacion_vertices_e_indices =  calcular_NCH((dataset_projected[i], l, extend, contraer_SCH, subdividir, i, plot)) # añadir el argumento i si se trata de pintar graficas
        l_vertices.append(vertices_aux)
        l_aristas.append(aristas_aux)
        l_vertices_expandidos.append(vertices_expandidos)
        l_orden_vertices.append(orden_vertices_aux)
        l_factor_expansion.append(factor_expansion_aux)
        l_cierres_separados.append(cierres_separados)
        l_asociacion_vertices_e_indices.append(asociacion_vertices_e_indices)
    toc1 = time.perf_counter()
    # Chekear si todos los factores de expansion son el mismo para emplear el NCH o el SNCH
    if (all(v == 0 for v in l_factor_expansion)):
        l_vertices_expandidos = [False] * len(l_factor_expansion)

    return l_cierres_separados, projections, l_vertices, l_aristas, l_vertices_expandidos, l_orden_vertices, l_normalizadores, l_asociacion_vertices_e_indices

def OCENCH_classify (X, model):        
    plot = False
    n_projections = len(model[0])
    l_cierres_separados, projections, l_vertices, l_aristas, l_vertices_expandidos, l_orden_vertices, l_normalizadores, l_asociacion_vertices_e_indices = model
    # Proyectamos los datos a clasificar
    tic = time.perf_counter() 
    # Proyectamos los datos a clasificar
    dataset_projected = project_Dataset(X, projections)
    toc = time.perf_counter() 
    for i in range (0, len(l_normalizadores)):
        dataset_projected[i] = NormalizeData(dataset_projected[i], l_normalizadores[i])
    mitad = int(dataset_projected[0].shape[0]/2)
    if plot == True: 
        plt.plot(dataset_projected[i][0:mitad,0], dataset_projected[i][0:mitad,1], 'yo', markersize=1)
        plt.plot(dataset_projected[i][mitad:,0], dataset_projected[i][mitad:,1], 'mo', markersize=1)
    result = check_if_points_are_inside_polygons_matplotlib_sin_paralelizar(dataset_projected, model, n_projections)
    result = combinar_clasificaciones(result) 
    return result
