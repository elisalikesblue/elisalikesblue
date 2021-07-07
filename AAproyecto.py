# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 09:15:05 2021

@author: brand
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from random import randrange
import numpy as np

import math

def menu():
  print("1. Busqueda Lineal")
  print("2. Busqueda Binaria")
  print("3. Factorial iterativo")
  print("4. Factorial recursivo")
  print("5. Fibonacci recursivo")
  print("6. Fibonacci iterativo")
  print("7. Merge Sort")
  print("8. Heap Sort")
  print("9. Quick Sort")
  print("10. Randomized Quicksort")
  print("11. Counting sort")
  print("12. Radix sort")
  print("13. Reinas")
  print("14. Activity Selector")
  print("15. Multiplicación de matrices")
  print("16. Matrix chain order")
  print("17. Cut Rod")
  print("18. Huffman")
  print("19. LCS")
  print("0. Salir")



def pedirArreglo():
  A=[] 
  A = [int(item) for item in input("Ingrese los elementos del arreglo(enteros con espacio y enter para terminar sin espacio al final) : ").split()] 
  return A



#busqueda
def busquedaLineal():
  arr= pedirArreglo()
  x = float(input("Qué elemanto deseas encontrar?"))

  for i in range(0,len(arr)): 
        if arr[i] == x: 
          print ("Tu elemento se encuentra en la posición ", i + 1) 
          return
  
  print("Ese elemento no está")
  



#busqueda binaria
def busqueda_binaria_r(A, x, i, j):
    if i > j:
        return -1
    m = (i + j) // 2
    am = A[m]
    if am == x:
        return m
    if x < am:
        return busqueda_binaria_r(A, x, i, m - 1)
    else:
        return busqueda_binaria_r(A, x, m + 1, j)

    



#factorial
def factorialI(n):
    x = 1
    for i in range(1,n+1):
        x = i*x
    return x


def factorialR(n):
    if n == 0:
        return 1
    else:
        return n*factorialR(n-1)







#fibonacci


def fibonacciR(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        x = fibonacciR(n-1)
        y = fibonacciR(n-2)
        return x + y
        

def fibonacciI(n):
    if n == 0:
        return 0
    else:
        x = 0
        y = 1
        for i in range(1,n):
            z= x + y
            x= y
            y= z
        return y




 


#merge sort   

def merge(A, p, q, r) :
	temp = [0] * (r - p + 1)

	i, j, k = p, q+1, 0

	while(i <= q and j <= r) :
		if(A[i] <= A[j]) :
			temp[k] = A[i]
			k += 1; i += 1
		else :
			temp[k] = A[j]
			k += 1; j += 1

	while(i <= q) :
		temp[k] = A[i]
		k += 1; i += 1

	while(j <= r) :
		temp[k] = A[j]
		k += 1; j += 1

	for i in range (p, r+1) :
		A[i] = temp[i - p]
    
    

def mergeSort(A, p, r):
    if p < r:
        q = math.floor((p+r)/2)
        mergeSort(A, p, q)
        mergeSort(A, q+1, r)
        merge(A, p, q, r)





#heap sort

def parent(i):
    return math.ceiling(i/2) - 1  
def left(i):
    return (2*i) + 1
def right(i):
    return (2*i) + 2
    


def exchange(a, i, j):
    aux = a[i]
    a[i] = a[j]
    a[j] = aux

def max_heapify(A, i, size):
    l = left(i)
    r = right(i)
    if l <= size-1 and A[l]>A[i]:
        largest = l
    else:
        largest = i
    if r <= size-1 and A[r] > A[largest]:
        largest = r
    if largest != i:
        exchange(A, i, largest)
        max_heapify(A, largest, size)

def build_max_heap(A):
    size = len(A)
    for i in range(math.floor(size/2), -1, -1):
        max_heapify(A, i, size)


def heap_sort(A):
    build_max_heap(A)
    size = len(A)
    for i in range(len(A)-1, -1, -1):
        exchange(A, 0, i)
        size -= 1
        max_heapify(A, 0, size)
        print(A)






#quick sort

def partition(A, p, r):
    x = A[r]
    i = p-1
    for j in range(p, r):
        if A[j] <= x:
            i = i+1
            exchange(A, i, j)
    exchange(A, i+1, r)
    return i + 1

def quicksort(A, p, r):
    if p < r:
        q = partition(A, p, r)
        quicksort(A, p, q-1)
        quicksort(A, q+1, r)




def randomized_partition(A, p, r):
    i = randrange(p, r, 1)
    exchange(A, r, i)
    return partition(A, p, r)

def randomized_quicksort(A, p, r):
    if p < r :
        q = randomized_partition(A, p, r)
        randomized_quicksort(A, p, q-1)
        randomized_quicksort(A, q+1, r)








#counting sort

def counting_sort(A, B, k):
    C = [0]*(k+1)
    size = len(A)
    for j in range(0, size):
        C[A[j]]= C[A[j]] + 1
    for i in range(1, k+1):
        C[i] = C[i] + C[i-1]
    for j in range(size-1, -1, -1):
        B[C[A[j]]-1] = A[j]
        C[A[j]] = C[A[j]]-1










#radix sort

def counting_sort_ce(A, d):
    size = len(A)
    B = [0] * size
    C = [0] * 10

    for i in range(0, size):
        index = A[i] // d
        C[index % 10] += 1

    for i in range(1, 10):
        C[i] += C[i - 1]

    i = size - 1
    while i >= 0:
        index = A[i] // d
        B[C[index % 10] - 1] = A[i]
        C[index % 10] -= 1
        i -= 1

    for i in range(0, size):
        A[i] = B[i]


def radixSort(A):
    maximo = max(A)

    d = 1
    while maximo // d > 0:
        counting_sort_ce(A, d)
        d *= 10










#queens

def queens(size):
    Q = [-1]*size
    return recursiveNqueens(Q, 0, size)
    

def recursiveNqueens(Q, r, size):
    if r == size:
        print(Q)
        
    else:
        for i in range(size):
            Q[r] = i
            if isLegal(Q, r):
                done = recursiveNqueens(Q, r+1, size)
                if(done):
                    return True
        return False
    
def isLegal(Q, r):
        for i in range(r):
            if Q[i] == Q[r]:
                return False
            if (r-i) == abs(Q[r]-Q[i]):
                return False
        return True












#activity selector

def greedy_activity_selector(s, f): 
    n = len(s) 
    print ("Se excogen las siguientes actividades ")
  
    k = 0
    print(k+1)
  
    for m in range(n): 
  
        if s[m] >= f[k]: 
            print (m+1) 
            k = m









#matrix multiply

def pedir_matriz():
    c = int(input("Columnas "))
    r = int(input("Renglones "))
    C = np.zeros((r, c), float)
    for i in range(r):
        for j in range(c):
            print("Elemento ", i, j)
            C[i,j] = int(input())
    return C 


def matrix_multiply(A, B):
    ar = len(A[0])
    bc = len(B)
    if ar != bc:
        print("Matrices incompatibles")
        return
    else:
        C = np.zeros((ar, bc), float)
    for i in range(ar):
        for j in range(bc):
            C[i,j] = 0
            for k in range(ar):
                C[i,j] = C[i,j] + (A[i,k]*B[k,j])
                
    return C

def imprime_matriz(A):
    for i in range(len(A)):
        print(A[i])













#Matrix chain order


inf = float('inf')


def print_optimal_parens(s, i, j):
    if i == j:
        print ("A", i, end =" ")
    else: 
        print("(", end = "")
        print_optimal_parens(s, i, s[i,j])
        print_optimal_parens(s, s[i,j]+1, j)
        print(")", end = "")
        
        
def matrix_chain_order(p):
    n = len(p)-1
    m = np.zeros((n,n), float)
    s = np.zeros((n, n), int)

    for l in range(1,n):
        for i in range(n-l):
            j = i + l
            m[i, j] = inf
            for k in range(i, j):
                q = m[i,k] + m[k+1,j] + (p[i]*p[k+1]*p[j+1])
                if q < m[i,j]:
                    m[i,j] = q
                    s[i,j] = k
    return(m,s)















#Cut Rod
minus_inf = float('-inf')

def cutRod(p, n):
    if n <= 0:
        return 0
    q = minus_inf
    for i in range(n):
        q = max(q, p[i] + cutRod(p, n-i-1))
    return q









#Huffman
class Nodo:
    def __init__ (self, frec, letra):
        self.letra=letra
        self.frec=frec
    def imprimir(self):
        print("["+self.letra+","+str(self.frec)+"]", end = "")



class NodoA:
    def __init__ (self, frec, izq, der ):
        self.frec = frec
        self.izq=izq
        self.der=der
    def imprimir(self):
        print("(", self.frec, end = "")
        self.izq.imprimir()
        self.der.imprimir()
        print(")", end = "")






    

def insertar(c, nodito):
    i = 0
    if len(c) == 0:
        c.append(nodito)
        return 
    while i<len(c):
        
        if c[i].frec > nodito.frec:
            c.insert(i, nodito)
            break
        i += 1
        if i == len(c):
            c.insert(i, nodito)
            break
   
        
def huffman(c):
    n = len(c)
    q = c
    for i in range(n-1):
        x = q[0]
        q.pop(0)
        y = q[0]
        q.pop(0)
        z = NodoA(x.frec + y.frec, x, y)
        insertar(q, z)
        for x in q:
            x.imprimir()
        print("")
           
    return q[0]















#Lcs


def printLCS(b, X, i, j):
    if i == 0  or j == 0 :
        return 
    if b[i, j] == 'i':
        printLCS(b, X, i-1, j-1)
        print(X[i], end = "")
        
    elif b[i, j] == 'a':
        printLCS(b, X, i-1, j)
        
    else:
        printLCS(b, X, i, j-1)
    
            
    
def LCS(X, Y):
    m = len(X)
    n = len(Y)
    b = np.array([['']*n]*m)
    c = np.zeros((m, n))
    
    for i in range(1, m):
        for j in range(1, n):
            if X[i] == Y[j]:
                c[i, j] = c[i-1, j-1] +1
                b[i, j] = 'i'
            elif c[i-1, j] >= c[i, j-1]:
                c[i, j] = c[i-1, j]
                b[i, j] = 'a'
            else:
                c[i, j] = c[i, j-1]
                b[i, j] = 'h'
    
    return (c, b)


def pedirCadena():
    i = 0
    x=[]
    x.append('xi')
    n = int(input("Cuántos elementos tiene la cadena? "))
    print("Presiona enter despues de agregar un elemento")
    for i in range(1, n+1):
        x.append(input())
    return x



















menu()
option= int(input("Elegir Algoritmo "))

while option != 0:
    
  if option == 1:
    busquedaLineal()
    
    
    
  elif option == 2:
    A = [int(item) for item in input("Ingresa una lista ordenada(enteros con espacio y enter para terminar sin espacio al final) : ").split()] 
    x=int(input("Qué elemento deseas encontrar? "))
    i = busqueda_binaria_r(A, x, 0, len(A)-1)
    if i == -1:
        print ("No está ese elemento")
    else:
        print ("Está en la posición ", i + 1)
      
    
        
  elif option == 3:
      n = int(input("Numero entero "))
      print("El factorial de ", n,  " es ", factorialI(n))
    
      
    
  elif option == 4:
      n = int(input("Numero entero "))
      print("El factorial de ", n,  " es ", factorialR(n))



  elif option == 5:
      n = int(input("Numero entero "))
      print("El termino", n, " de fibonacci es ", fibonacciR(n))



  elif option == 6:
      n = int(input("Numero entero "))
      print("El termino", n, " de fibonacci es ", fibonacciI(n))


  elif option == 7:
      A = [int(item) for item in input("Ingresa una lista de numeros (enteros con espacio y enter para terminar sin espacio al final) : ").split()] 
      mergeSort(A, 0, len(A)-1)
      print (A)
      
      
  elif option == 8:
      A = pedirArreglo()
      heap_sort(A)



  elif option == 9:
      A = pedirArreglo()
      quicksort(A, 0, len(A)-1)
      print(A)
      
      
  elif option == 10:
      A = pedirArreglo()
      randomized_quicksort(A, 0, len(A)-1)
      print(A)
      
     
  elif option == 11:
      A = pedirArreglo()
      B = [0]*(len(A))
      k = int(input("Número máximo "))
      counting_sort(A, B, k)
      print(B)
 
        
  elif option == 12:
      A= pedirArreglo()
      radixSort(A)
      print(A)
      
      
  elif option == 13:
      n = int(input("Tamaño del tablero "))
      queens(n)
      
      
  elif option == 14:
     s = [int(item) for item in input("Ingrese los inciios en el orden correspondiente (enteros con espacio y enter para terminar sin espacio al final) : ").split()] 
     f = [int(item) for item in input("Ingrese los finales en el orden correspondiente (enteros con espacio y enter para terminar sin espacio al final) : ").split()] 
     greedy_activity_selector(s, f)
     
     
  elif option == 15:
      print("Primera matriz")
      A = pedir_matriz()
      print("Segunda matriz")
      B = pedir_matriz()
      C = matrix_multiply(A, B)
      print("AxB\n")
      imprime_matriz(C)


  elif option == 16:
      p = [int(item) for item in input("Ingrese en vector p válido (enteros con espacio y enter para terminar sin espacio al final) : ").split()] 
      m,s = matrix_chain_order(p)
      print("Los paréntesis deben ser así: ")
      print_optimal_parens(s, 0, len(p)-2)
      
      
  elif option == 17:
      p = [int(item) for item in input("Ingresa los precios (enteros con espacio y enter para terminar sin espacio al final) : ").split()] 
      q=cutRod(p, len(p))
      print("Lo más que se puede obtener es ", q)
   
  elif option == 18:
      n=int (input("Ingrese la cantidad de nodos: "))
      i=0
      c=[]
      while i<n:
          letra=input("Ingrese la letra: ")
          numero=int(input("Ingrese la frecuencia: "))
          nodo=Nodo(numero, letra)
          c.append(nodo)
          i+=1
          
      print("Se imprimiran los pasos")
      print("El arbol se lee: raiz, izquierdo, derecho. Un parentesis indica un nodo de arbol")
      q = huffman(c)
      
      
  elif option == 19:
      x = pedirCadena()
      y = pedirCadena()
      c, b = LCS(x, y)
      print("La subsecuencia mas larga es ")
      printLCS(b, x, len(x)-1, len(y)-1)
      
   
  else:
    print("Opción invalida")

  print()
  menu()
  option=int(input("Elegir algoritmo "))

print("Adiós")