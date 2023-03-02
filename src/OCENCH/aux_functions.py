
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import math
import multiprocessing as mp
import matplotlib.path as mpltPath
import copy
import time

# ##################################################################
# Entrenamiento estandarizador
def NormalizeData_Train(dataframe_proccessed):
    # Entrenamos un normalizador de media cero y desviacion tipica 1 con el dataframe de entrada
    scaler = preprocessing.StandardScaler().fit(dataframe_proccessed) 
    return scaler

# ##################################################################
# Aplicación de un estandarizador
def NormalizeDataframe(dataframe_proccessed, model):
    columns = dataframe_proccessed.columns
    # Normalizamos el dataframe de entrada mediante el normalizador recibido como argumento
    data = model.transform(dataframe_proccessed.astype(float))
    data = pd.DataFrame(data, columns=columns)
    return data

def NormalizeData(data, model):
    # Normalizamos el dataframe de entrada mediante el normalizador recibido como argumento
    data = model.transform(data)
    return data

# ##################################################################
# Inversión de la normalización
def inverse_transform(dataframe_proccessed, model):
    columns = dataframe_proccessed.columns
    data = model.inverse_transform(dataframe_proccessed.astype(float))
    data = pd.DataFrame(data, columns=columns)
    return data.to_numpy()

def change_target_value_GH(df):
    if df == 'g':
        return 0
    elif df == 'h':
        return 1
    
def change_target_value_01(df):
    if df == 1:
        return 0
    elif df == -1:
        return 1
 
def change_target_value_MNIST(df):
    if df == 1 or df == 7:
        return 1
    else:
        return 0
    
def array_to_sequence_of_vertices (data):
    from ground.base import get_context
    context = get_context()    
    Point = context.point_cls
    Contour = context.contour_cls
    aux_list = []
    for i in range (0, data.shape[0]):
        aux_list.append(Point(data[i, 0], data[i, 1]))
    aux_list = Contour(aux_list)
    return aux_list

def array_to_sequence_of_vertices2 (data):
    # Función auxiliar para transformar una matriz de numpy de vértices en una lista con el formato [(X1,Y1), (X2,Y2), ... , (Xn,Yn)]
    aux_list = []
    for i in range (0, data.shape[0]):
        aux_list.append((data[i, 0],data[i, 1]))
    return aux_list

def generate_Projections (n_projections, n_dim):
    # Función que genera n matrices de proyección bidimensionales
    import numpy as np
    np.random.seed(1)
    projections = np.random.randn(n_dim, 2, n_projections)
    return projections    

def project_Dataset (dataset, projections):
    # Función que proyecta un conjunto de datos a partir de matrices de proyección
    import numpy as np
    n_projections = projections.shape[2]
    dataset_projected = []
    for j in range(0, n_projections):
        one_projection = np.matmul(dataset, projections[:, :, j])
        dataset_projected.append(one_projection)
    return dataset_projected

def check_if_points_are_inside_polygons (dataset, model):
    # Función que determina si uno o varios datos pasados como matriz de numpy se encuentran dentro de un modelo NCH entrenado
    projections, l_vertices, l_aristas, l_vertices_expandidos, l_orden_vertices = model
    l_results = []
    if (dataset[0].ndim == 1):
        num_datos = 1
    else:
        num_datos = dataset[0].shape[0]
    for i in range (0, len(l_vertices)):
        aux = []
        if (l_vertices_expandidos != False): # Si los cierres SI se expandieron durante el entrenamiento, utilizamos el SNCH para clasificar
            # Construimos el polígono a partir de los vértices del SNCH
            polygon = Polygon(array_to_sequence_of_vertices2(l_vertices_expandidos[i]))
        elif (l_vertices_expandidos == False): # Si los cierres NO se expandieron durante el entrenamiento, utilizamos el NCH para clasificar
            # Construimos el polígono a partir de los vértices del NCH
            polygon = Polygon(array_to_sequence_of_vertices2(l_vertices[i][l_orden_vertices[i]]))
        for j in range (0, num_datos): # Clasificamos cada uno de los puntos
            if (num_datos == 1):
                point = Point(dataset[i])
            else:
                point = Point(dataset[i][j])
            aux.append(polygon.contains(point)) # Lista de comprobaciones para una proyección
        l_results.append(aux) # Lista de listas de proyecciones
    return l_results

def check_if_points_are_inside_polygons_p (dataset, model, process_pool):
    # Función que determina si uno o varios datos pasados como matriz de numpy se encuentran dentro de un modelo NCH entrenado
    projections, l_vertices, l_aristas, l_vertices_expandidos, l_orden_vertices = model
    if (dataset[0].ndim == 1):
        num_datos = 1
    else:
        num_datos = dataset[0].shape[0]
    arguments_iterable = []
    for i in range (0, projections[0].shape[1]):
        if (l_vertices_expandidos != False):
            parameter = l_vertices_expandidos[i]
        else:
            parameter = l_vertices[i][l_orden_vertices[i]]
        arguments_iterable.append((l_vertices_expandidos, l_vertices[i], parameter, num_datos, dataset[i]))
    result = list(process_pool.imap(check_one_projection, arguments_iterable))
    return result

def check_one_projection(args):   
    l_vertices_ex, vertices, l_vertices_x, n_datos, dataset = args
    aux = []
    # Construimos el polígono a partir de los vértices del NCH
    polygon = Polygon(array_to_sequence_of_vertices2(l_vertices_x))
    for j in range (0, n_datos): # Clasificamos cada uno de los puntos
        if (n_datos == 1):
            point = Point(dataset)
        else:
            point = Point(dataset[j])
        aux.append(polygon.contains(point)) # Lista de comprobaciones para una proyección
    return aux # Lista de listas de proyecciones

# ################################################################################
    
def check_if_points_are_inside_polygons_matplotlib_sin_paralelizar (dataset, model, num_p):
    # Función que determina si uno o varios datos pasados como matriz de numpy se encuentran dentro de un modelo NCH entrenado
    aux = []
    l_cierres_separados, projections, l_vertices, l_aristas, l_vertices_expandidos, l_orden_vertices, _, l_asociacion_vertices_e_indices = model
    expandido = False
    expandido_local = []
    if num_p == 1:
        if isinstance(l_vertices_expandidos[0], np.ndarray) == False:
            if l_vertices_expandidos != [False]:
                expandido = True
    elif num_p > 1:
        for i in range (0, num_p):
            if isinstance(l_vertices_expandidos[i], np.ndarray):
                expandido_local.append(True)
            else:
                expandido_local.append(False)
    
    if sum(expandido_local) == num_p:
        expandido = True
            
    for i in range (0, num_p): # para cada proyeccion
        aux_p = []    
        numero_vertices = len([item for sublist in l_cierres_separados[0] for item in sublist])
        cierre_count = 0
        clasificacion_una_proyeccion = []
        for cierre in l_cierres_separados[i]: # para cada cierre de una proyeccion
            if expandido: # Si los cierres SI se expandieron durante el entrenamiento, utilizamos el SNCH para clasificar
                # Construimos el polígono a partir de los vértices del SNCH
                polygon = mpltPath.Path(vertices = l_vertices_expandidos[i][l_asociacion_vertices_e_indices[i][cierre_count]]) 
                
            else: # Si los cierres NO se expandieron durante el entrenamiento, utilizamos el NCH para clasificar
                # Construimos el polígono a partir de los vértices del NCH
                polygon = mpltPath.Path(vertices = l_vertices[i][cierre, :])
            clasificacion = list(polygon.contains_points(dataset[i]))
            #print("cierre ", cierre_count)
            aux_result = [elem for elem in clasificacion]
            aux_p.append(aux_result)
            cierre_count = cierre_count + 1   
        suma_de_clasificaciones = np.array(aux_p).sum(axis = 0)
        for k in range (0, len(suma_de_clasificaciones)):
            if suma_de_clasificaciones[k] == 0:
                suma_de_clasificaciones[k] = 1
            else:
                suma_de_clasificaciones[k] = 0
        aux.append(suma_de_clasificaciones)
    return aux

def combinar_clasificaciones(result):
    # Funcion que recibe una lista de listas y combinar los resultados -> si un dato es clasificado en alguna proyección
    # como anómalo, el resultado será anómalo
    result_sum = np.array(result).sum(axis = 0)
    result_sum[result_sum > 0] = 1
    return result_sum


def calcular_metricas (Y_test, result, titulo):
    cm = confusion_matrix(Y_test, result).ravel()
    if cm.shape[0] > 1:
        TN, FP, FN, TP = confusion_matrix(Y_test, result).ravel()
    else:
        if Y_test.iloc[0] == 0:
            TN, FP, FN, TP = Y_test.shape[0], 0, 0, 0
        elif Y_test.iloc[0] == 1:
            TN, FP, FN, TP = 0, 0, 0, Y_test.shape[0]
    print("")
    print(titulo)
    print("-TN, FP, FN, TP: ", TN, FP, FN, TP)
    #print("-Sensibilidad TP/(TP+FN): ", TP/(TP+FN))
    #print("-Especificidad TN/(TN+FP): ", TN/(TN+FP))
    #print("-Precisión (TP+TN)/(TP+TN+FP+FN): ", (TP+TN)/(TP+TN+FP+FN))
    #print("-Similitud: ", 1-(math.sqrt((1-(TP+TN)/(TP+TN+FP+FN))**2+(1-TP/(TP+FN))**2)/math.sqrt(2)))
    print("")
    return [TN, FP, FN, TP]
    
def cargar_resultados_txt (path):
    import ast
    lines = []
    with open(path, "r") as reader:
        lines.append(reader.readline())
    return ast.literal_eval(lines[0])

def weird_division(n, d):
    return n / d if d else 0

def parsear_y_calcular_metricas (list_results):
    desired_output = []
    for result in list_results:
        TN, FP, FN, TP, info = result 
        sensibilidad = TP/(TP+FN)
        especificidad = TN/(TN+FP)
        accuracy = (TP+TN)/(TP+TN+FP+FN)
        similitud = 1-(math.sqrt((1-(TP+TN)/(TP+TN+FP+FN))**2+(1-TP/(TP+FN))**2)/math.sqrt(2))
        precision = weird_division(TP, TP+FP)
        F1 = weird_division((2*precision*sensibilidad), (precision+sensibilidad))
        
        desired_output.append([sensibilidad, especificidad, accuracy, similitud, F1, info])
    return desired_output

def parsear_y_calcular_metricas2 (list_results):
    desired_output = []
    for k in list_results:
        for result in k:
            TN, FP, FN, TP, info = result 
            sensibilidad = TP/(TP+FN)
            especificidad = TN/(TN+FP)
            accuracy = (TP+TN)/(TP+TN+FP+FN)
            similitud = 1-(math.sqrt((1-(TP+TN)/(TP+TN+FP+FN))**2+(1-TP/(TP+FN))**2)/math.sqrt(2))
            precision = weird_division(TP, TP+FP)
            F1 = weird_division((2*precision*sensibilidad), (precision+sensibilidad))
            
            desired_output.append([sensibilidad, especificidad, accuracy, similitud, F1, info])
    return desired_output

def obtener_mejor_metodo (list_results, index_metric):
    
    if index_metric != -1:
        list_results_target_metric = []
        for result in list_results:
            list_results_target_metric.append(result[index_metric])
        
    else:
        list_results_target_metric = []
        for result in list_results:
            list_results_target_metric.append((result[0] + result[1])/2)

    max_value = max(list_results_target_metric)
    max_index = list_results_target_metric.index(max_value)
    return list_results[max_index], max_value

def calcula_cuantos_cierres_hay (lista_aristas):
    count = 0
    lista_cierres = [] 
    principio_cierre = lista_aristas[0, 0]
    valor_actual = lista_aristas[0, 1]
    lista_aux = [valor_actual]
    lista_aristas = np.delete(lista_aristas, 0, 0)
    while len(lista_aristas)>0:
        count = count + 1
        target = np.where(lista_aristas==valor_actual) ### index
        arista = lista_aristas[target[0], :][0] ### value
        
        tic2 = time.perf_counter()
        if (pd.Series(valor_actual).isin(arista).any() == True):
            if valor_actual == arista[0]:
                valor_actual = arista[1]
            else:
                valor_actual = arista[0]
            lista_aux.append(valor_actual)
            index = np.where(np.all(arista==lista_aristas,axis=1))[0][0]
            lista_aristas = np.delete(lista_aristas, index, 0)
            if (valor_actual ==  principio_cierre):
                lista_cierres.append(lista_aux)
                if len(lista_aristas) > 0:
                    principio_cierre = lista_aristas[0, 0]
                    valor_actual = lista_aristas[0, 1]
                    lista_aux = [valor_actual]
                    lista_aristas = np.delete(lista_aristas, 0, 0)
    return lista_cierres

def calcular_indices_tras_separar_boundaries (lista_original, lista_separados):
    lista_final = [ [] for _ in range(len(lista_separados)) ]
    for i in range (0, len(lista_separados)):
        for j in range (0, len(lista_separados[i])):
            lista_final[i].append(lista_original.index(lista_separados[i][j]))
    return lista_final
    
    



