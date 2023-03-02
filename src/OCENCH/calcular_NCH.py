import math
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, ConvexHull
from IPython import get_ipython
from bentley_ottmann.planar import contour_self_intersects # Implementacion del algoritmo Bentley Ottmann
from calcular_NCH import *
from aux_functions import *

# #############################################################################

def etapa_podado (boundary_e, boundary_v, boundary_final, dist_sorted, triangles, X, l):
    lista_aristas_sospechosas_division = []
    lista_triangulos_sospechosas_division = []
    while len(boundary_e)>0:
            edge = boundary_e[-1,:] # Se obtiene la arista de borde de mayor longitud
            dist_e = dist_sorted[-1] # Se obtiene la distancia de la arista de borde seleccionada
            boundary_e = np.delete(boundary_e, -1, 0) # Se elimina la última arista (la mayor por orden de longitud)
            dist_sorted = np.delete(dist_sorted,-1,0) # Se elimina la distancia corresponcdiente a la arista eliminada
            find_e = np.isin(triangles, edge) # Buscar los triángulos que contengan la arista 
            index = np.where(find_e) # Proporciona los índices de los triángulos que contienen algún vértice de la arista
            # Busca el índice duplicado que indicará el triángulo que contiene ambos vértices de la arista
            u, c = np.unique(index[0], return_counts=True)
            triangle_ix = u[c > 1]
            # Se obtiene el triángulo buscado (el que contiene la arista borde seleccionada)
            triangle_e = triangles[triangle_ix,:]
            # Se obtiene el tercer vértice del triángulo (es el que no está en la arista seleccionada)
            vertex = np.setdiff1d(triangle_e,edge)
            # Si la distancia de la arista es mayor que el umbral establecido y el tercer vértice no es de borde (para mantener regularidad)
            supera_l = dist_e > l
            pertenece_a_cierre_externo = vertex in boundary_v
            if ((supera_l == True) and (pertenece_a_cierre_externo == True)):
                # Marcar la arista como susceptible de ser dividida
                lista_aristas_sospechosas_division.append(edge)
                lista_triangulos_sospechosas_division.append(triangle_e)
            if (supera_l and not(pertenece_a_cierre_externo) and max(c)>1):
                triangles = np.delete(triangles, triangle_ix, axis=0)  # Elimina el triángulo del polígono
                new_b_edge1 = [vertex[0], edge[0]] # Se obtiene la arista 1 del triángulo
                new_b_edge2 = [vertex[0], edge[1]] # Se obtiene la arista 2 del triángulo
                dist_edge1 = np.linalg.norm(X[new_b_edge1[0]]-X[new_b_edge1[1]]) # Calcula la longitud de arista 1
                dist_edge2 = np.linalg.norm(X[new_b_edge2[0]]-X[new_b_edge2[1]]) # Calcula la longitud de arista 2
                idx1 = np.searchsorted(dist_sorted,dist_edge1) # Busca la posición de la arista en la lista ordenada de longitudes
                boundary_e = np.insert(boundary_e,idx1,new_b_edge1,axis=0) # Inserta la arista 1 ordenada en el lista de aristas de borde
                dist_sorted = np.insert(dist_sorted,idx1,dist_edge1) # Inserta la longitud de arista 1 en la lista de longitudes
                idx2 = np.searchsorted(dist_sorted,dist_edge2) # Buscar la posición de la arista en la lista ordenada de longitudes
                boundary_e = np.insert(boundary_e,idx2,new_b_edge2,axis=0) # Inserta la arista 2 ordenada en el lista de aristas de borde
                dist_sorted = np.insert(dist_sorted,idx2,dist_edge2) # Inserta la longitud de arista 1 en la lista de longitudes
                v1 = np.where(boundary_v == edge[0])[0][0]
                v2 = np.where(boundary_v == edge[1])[0][0]
                if (np.abs(v2-v1) > 1 and v1 != len(boundary_v)-1 and v2 != len(boundary_v)-1):
                    print('Error!. Vértices no contiguos: ', v2, v1, edge, boundary_v)  
                if ((max(v2,v1) == len(boundary_v)-1) and ((min(v2,v1) == 0))):
                    boundary_v = np.insert(boundary_v,0,vertex)
                else:
                    boundary_v = np.insert(boundary_v,max(v2,v1),vertex)
            else:
                boundary_final = np.append(boundary_final,np.reshape(edge, (-1, 2)),axis=0)
    if lista_aristas_sospechosas_division != []:
        lista_aristas_sospechosas_division = np.vstack(lista_aristas_sospechosas_division)
        lista_triangulos_sospechosas_division = np.vstack(lista_triangulos_sospechosas_division)
    return boundary_final, boundary_v, triangles, lista_aristas_sospechosas_division, lista_triangulos_sospechosas_division      


def etapa_subdivision (boundary_final, l, triangles, X, lista_aristas_sospechosas_division, lista_triangulos_sospechosas_division):
    salir1 = False
    distancias_aristas = np.linalg.norm(X[boundary_final[:, 0]] - X[boundary_final[:, 1]], axis = 1)
    count = 0
    if lista_aristas_sospechosas_division == [] or lista_triangulos_sospechosas_division == []:
        print("No suspicious edges to be droped.")
    else:
        for suspiciousEdge in lista_aristas_sospechosas_division:
            if (distancias_aristas[count] > l): 
                pdsuspiciousEdge = pd.Series(suspiciousEdge)
    
                tri = lista_triangulos_sospechosas_division[count]
                if pdsuspiciousEdge.isin(tri).all() == True:
                    third_vertex = np.setdiff1d(tri, np.array(suspiciousEdge))[0]
                    arista_interna11 = np.array([suspiciousEdge[0], third_vertex])
                    arista_interna12 = np.array([third_vertex, suspiciousEdge[0]])
                    arista_interna21 = np.array([suspiciousEdge[1], third_vertex])
                    arista_interna22 = np.array([third_vertex, suspiciousEdge[1]])
                    comprobacion_arista1 = sum(np.all(boundary_final==arista_interna11,axis=1)) + sum(np.all(boundary_final==arista_interna12,axis=1))
                    #SI las dos otras aristas que componen el triangulo no pertenecen al boundary  
                    if (comprobacion_arista1 == 0):
                        comprobacion_arista2 = sum(np.all(boundary_final==arista_interna21,axis=1)) + sum(np.all(boundary_final==arista_interna22,axis=1))
                        if (comprobacion_arista2 == 0):
                            for suspiciousEdge2 in boundary_final:
                                pdsuspiciousEdge2 = pd.Series(suspiciousEdge2)
                                if (len(np.setdiff1d(suspiciousEdge, suspiciousEdge2)) == 2) and (pd.Series(third_vertex).isin(suspiciousEdge2).all() == True):
                                    arraysuspiciousEdge2 = np.array(suspiciousEdge2)
                                    for tri2 in lista_triangulos_sospechosas_division:
                                        if (tri != tri2).any():
                                            if (pdsuspiciousEdge2.isin(tri2).all() == True):
                                                third_vertex2 = np.setdiff1d(tri2, arraysuspiciousEdge2)[0]
                                                if pd.Series(third_vertex2).isin(suspiciousEdge).all() == True:
                                                    arista_interna211 = np.array([suspiciousEdge2[0], third_vertex2])
                                                    arista_interna212 = np.array([third_vertex2, suspiciousEdge2[0]])
                                                    arista_interna221 = np.array([suspiciousEdge2[1], third_vertex2])
                                                    arista_interna222 = np.array([third_vertex2, suspiciousEdge2[1]])
                                                    comprobacion_arista21 = sum(np.all(boundary_final==arista_interna211,axis=1)) + sum(np.all(boundary_final==arista_interna212,axis=1))
                                                    #SI las dos otras aristas que componen el triangulo no pertenecen al boundary  
                                                    if comprobacion_arista21  == 0:
                                                        comprobacion_arista22 = sum(np.all(boundary_final==arista_interna221,axis=1)) + sum(np.all(boundary_final==arista_interna222,axis=1))
                                                        if comprobacion_arista22  == 0:
                                                            edge1_index =  np.where(np.all(boundary_final==suspiciousEdge,axis=1))
                                                            boundary_final = np.delete(boundary_final, edge1_index, axis=0)
                                                            edge2_index = np.where(np.all(boundary_final==suspiciousEdge2,axis=1))
                                                            boundary_final = np.delete(boundary_final, edge2_index, axis=0)
                                                            # eliminar triangulos
                                                            tri1_index =  np.where(np.all(tri==triangles,axis=1))
                                                            triangles = np.delete(triangles, tri1_index, axis=0)
                                                            tri2_index =  np.where(np.all(tri2==triangles,axis=1))
                                                            triangles = np.delete(triangles, tri2_index, axis=0)
                                                            # unir nuevas aristas
                                                            v1 = np.setdiff1d(tri, tri2)
                                                            v2 = np.setdiff1d(tri2, tri)
                                                            new_edge = np.array([[v1[0], third_vertex]])
                                                            new_edge2 = np.array([[v2[0], third_vertex2]])
                                                            boundary_final = np.append(boundary_final, new_edge,axis=0)
                                                            boundary_final = np.append(boundary_final, new_edge2,axis=0)
                                                            salir1 = True
                                                            break
            count = count + 1                             
    return boundary_final, triangles                            

def calcular_NCH (args):
    X, l, extend, contraer_SCH, subdividir, proyeccion, plot = args
    # Delaunay tesselation of X
    tri = Delaunay(X)
    triangles = tri.simplices.copy()
    # Calcula el cierre convexo: aristas de borde del polígono
    CH = ConvexHull(X)
    triangles = tri.simplices.copy()
    CH_e = CH.simplices  # Aristas de borde del polígono
    # Cálculo de las longitudes de las aristas de borde
    dist = np.zeros(len(CH_e))
    j = 0
    for i in range(len(CH_e)):
        dist[j] = np.linalg.norm(X[CH_e[i,0]]-X[CH_e[i,1]])
        j = j + 1
    # Se ordenan de menor a mayor las aristas del cierre convexo en función de su longitud
    index_sorted = np.argsort(dist)#[::-1]
    dist_sorted = np.sort(dist)#[::-1]
    triangles = tri.simplices.copy()
    # Se crea una lista con las aristas de borde ordenadas por distancia (de menor a mayor)
    boundary_e = CH_e[index_sorted,:]
    # Se crea una lista con los vértices de borde
    boundary_v = CH.vertices
    if plot == True:
        marker_size = 1
        plt.figure()
        plt.plot(X[:,0], X[:,1], 'go', markersize=marker_size)
        plt.axis('equal')
        plt.title('Data')
        # Muestra la triangulazión inicial
        plt.figure()
        plt.triplot(X[:,0], X[:,1], tri.simplices.copy())
        plt.plot(X[:,0], X[:,1], 'go', markersize=marker_size)
        plt.plot(X[boundary_v,0], X[boundary_v,1], 'ro', markersize=marker_size)
        plt.axis('equal')
        plt.title('Delaunay triangulation')
    # Se crea un array vacío para contener las aristas del cierre no convexo final
    boundary_final = np.empty(shape=[0, 2],dtype=np.int32)
    boundary_final, boundary_v, triangles, lista_aristas_sospechosas_division, lista_triangulos_sospechosas_division = etapa_podado(boundary_e, boundary_v, boundary_final, dist_sorted, triangles, X, l)
    if ((subdividir == True) and (lista_aristas_sospechosas_division != []) and (lista_triangulos_sospechosas_division != [])):
        boundary_final, triangles = etapa_subdivision(boundary_final, l, triangles, X, lista_aristas_sospechosas_division, lista_triangulos_sospechosas_division)
    cierres_separados = calcula_cuantos_cierres_hay(boundary_final)
    asociacion_vertices_e_indices = calcular_indices_tras_separar_boundaries (boundary_v.tolist(), cierres_separados)
    if plot == True:
        # Muestra la triangulazión final
        plt.figure(str(proyeccion))
        plt.triplot(X[:,0], X[:,1], triangles)
        plt.plot(X[:,0], X[:,1], 'go', markersize=marker_size)
        plt.plot(X[boundary_v,0], X[boundary_v,1], 'ro', markersize=marker_size)
        plt.axis('equal')
        plt.title('Non-convex clousure with l=%s' %l)
        # Muestra el borde del cierre no convexo final
        plt.plot([X[boundary_final[:,0],0],X[boundary_final[:,1],0]],[X[boundary_final[:,0],1],X[boundary_final[:,1],1]],'r-')
    if (contraer_SCH == True):
        extend_list = np.arange(extend, 0, -extend/5)
        extend_list = np.append(extend_list, 0)
    else:
        extend_list = np.array([extend])
        extend_list = np.append(extend_list, 0)
    e_valido = False
   
    for e in extend_list: 
        # Recorre los vértices externos del polígono final
        if (e != 0) and (e_valido == False):
            sign_ang = []
            incenter_l = np.empty(shape=[0, 2])
            extVertex_l = np.empty(shape=[0, 2])
            z = 0
            count = 0
            Xordenado = np.zeros(X.shape)
            for i in boundary_v:
                find_v = np.isin(triangles, i)  # Busca los triángulos que contengan ese vértice
                index_t = np.where(find_v)      # Localiza las posiciones de los triángulos
                index_t = index_t[0]            # Se queda con el primer índice ya que indica el número de triángulo
                sum_angle = 0
                # Recorre los triángulos seleccioandos para calcular el ángulo interior del vértice externos
                for j in index_t:              
                    vertices = np.setdiff1d(triangles[j,:],i)   # Obtiene los otros vértices del triángulo que no son el seleccionado
                    a = np.linalg.norm(X[vertices[0]]-X[i])     # Calcula la longitud del primer lado del triángulo
                    b = np.linalg.norm(X[vertices[1]]-X[i])     # Calcula la longitud del segundo lado del triángulo
                    c = np.linalg.norm(X[vertices[0]]-X[vertices[1]]) # Calcula la longitud del tercer lado del triángulo
                    aux_ang = ( a**2 + b**2 - c**2 ) / (2*a*b)
                    if aux_ang > 1:
                        aux_ang = 1
                    elif aux_ang < -1:
                        aux_ang = -1
                    angle = np.degrees ( math.acos( aux_ang ) ) # Cácula el ángulo para el vértice dado
                    sum_angle = sum_angle + angle
                # Cálculos previos para determinar el vértice extendido a partir del vértice externo
                find_e = np.isin(boundary_final, i)
                index_e = np.where(find_e)
                index_e = index_e[0]
                edges = boundary_final[index_e]
                points = np.setdiff1d(edges,i)
                edges = np.append(edges,[points],axis=0)
                lenEdges = np.linalg.norm(X[edges[:,0]]-X[edges[:,1]],axis=1)        
                incenter = (X[i,:]*lenEdges[2]+X[np.setdiff1d(edges[0],i),:][0]*lenEdges[1]+X[np.setdiff1d(edges[1],i),:][0]*lenEdges[0])/ sum(lenEdges) # Basado en: https://es.wikipedia.org/wiki/Incentro
                lenAB = np.linalg.norm(X[i]-incenter) 
                # Indica si el vértice es cóncavo a convexo (función de la suma de todos los ángulos de los triángulos)
                if sum_angle>180:
                    if plot == True:
                        plt.text(X[i,0],X[i,1],'Concave',fontsize=8,fontweight='bold')  
                    sign_ang = np.append(sign_ang,-1) # Si el ángulo es cóncava se restará sobre el vértice externo
                else:
                    if plot == True:
                        plt.text(X[i,0],X[i,1],'Convex',fontsize=8,fontweight='bold')   
                    sign_ang = np.append(sign_ang,1) # Si el ángulo es convexo se sumará sobre el vértice externo
                # Calcula el vértice extendido en función de si es cóncavo o convexo (valor de sign_ang)
                if lenAB == 0:
                    lenAB = 0.0001
                extVertex = X[i] + sign_ang[z] * (X[i] - incenter) / lenAB * e # Basado en: https://stackoverflow.com/questions/7740507/extend-a-line-segment-a-specific-distance
                z = z + 1
                incenter_l = np.append(incenter_l,np.reshape(incenter, (-1, 2)),axis=0)
                extVertex_l = np.append(extVertex_l, np.reshape(extVertex, (-1, 2)),axis=0) 
                # Almacenamos el vértice extendido
                if plot == True:
                    # Dibuja el vértice extendido y el incentro usado para calcularlo
                    plt.plot(extVertex[0],extVertex[1],'bo')
            
            # PARA CADA CIERRE DE ESTA PROYECCIÓN COMPROBAR SI ES SIMPLE O COMPLEJO
            algun_complejo = False
            for indices in asociacion_vertices_e_indices:
                if contour_self_intersects(array_to_sequence_of_vertices(extVertex_l[indices])) == True:
                    algun_complejo = True
                    break
            if algun_complejo == False:
                e_valido = True # Si todos los SNCHs son simples podemos dejar de evaluar diferentes valores de expansion
                extend = e
        elif (e == 0) and (e_valido == False):
            extend = 0
            extVertex_l = False
    
    return X, boundary_final, extVertex_l , boundary_v, extend, cierres_separados, asociacion_vertices_e_indices


