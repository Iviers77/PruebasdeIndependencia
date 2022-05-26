# -*- coding: utf-8 -*-
"""
Created on Mon May 23 14:06:19 2022

@author: Julián Morales Hernández

Prueba de independencia ji-cuadrdada
"""
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import fisher_exact
from scipy.stats import chi2_contingency, chi2
from tabulate import tabulate
from statsmodels.graphics.mosaicplot import mosaic
from scipy.stats import hypergeom
import pylab as mp
from scipy.stats.contingency import association


class ji_cuadrada_test():
    def __init__(self,set_datos,alpha,plot):
        self.set_datos=set_datos
        self.plot=plot
        self.alpha=alpha  
        
    def frecuencias_esperadas(self):
        '''
        Returns frecuencias_esperadas
        -------
        frecuencias_esperadas : Data Frame
            Contiene las frecuenciás esperadas matematicamente frecuencia esperada = Total[i] * frecuencia [i],[j] / total [j].

        '''
        analisis = sm.stats.Table(self)
        frecuencias_esperadas=analisis.fittedvalues
        return frecuencias_esperadas
        
    def chi_square_statistic(self):
        '''
        Returns
        valor de la estadística ji-cuadrada,p-value de la prueba, los grados de libertad y las frecuencias esperadas
        -------
        None.

        '''
        ji, p_value, grados_libertad , frecuencias_esperadas = chi2_contingency(observed = self.set_datos, correction=False)
        return ji, p_value, grados_libertad , frecuencias_esperadas
    
    def scatter_plot(self):
        '''
        

        Objeto que grafica la ditribución de probabilidad, la cual contiene un booleano
        para saber si es o no necesario graficar la grafica, imprimiendo el alfa usado,
        los grados de libertad y la ji critical
        -------
        Graphics.

        '''
        
        ji, p_value_ji, grados_libertad_ji , frecuencias_esperadas_1 =self.chi_square_statistic()
        x = np.arange(0, 4*grados_libertad_ji, 0.001) #lista de valores del 0 a n veces los grados de libertad de 0.001 a 0.001 unidades
        y = chi2.pdf(x, df=grados_libertad_ji) # #pdf(x, grados_libertad_ji , loc=0, escala=1) Función de densidad de probabilidad.
        z = chi2.ppf(1-self.alpha, df=grados_libertad_ji) #ppf(q=1-prob, grados_libertad_ji , loc=0, escala=1)
        if self.plot ==True:
            mp.scatter(x, y, color='red', alpha=0.5)
            xarea = np.arange(z, 4*grados_libertad_ji, 0.001) #lista apartir del punto z a  n veces los grados de libertad de 0.001 a 0.001 unidades
            yarea = chi2.pdf(xarea, df=grados_libertad_ji)  #Función de densidad de probabilidad. apartir de el  xarea
            xarea = np.append(xarea, xarea[-1])
            yarea = np.append(0, yarea)
            xarea1 = np.arange(0,z, 0.001) #lista apartir del punto 0 al z
            yarea1 = chi2.pdf(xarea1, df=grados_libertad_ji) #Función de densidad de probabilidad. apartir de el  xarea1
            xarea1 = np.append(xarea1, xarea1[-1])
            yarea1 = np.append(0, yarea1)
            plt.plot(x, y, label = "Función de densidad de probabilidad") #Función de densidad de probabilidad
            plt.title("Función de densidad ji-cuadrada con " + str(grados_libertad_ji) +" grados de libertad ")
            #plt.fill_between(xarea, yarea, color='y',alpha=0.3 )
            plt.fill_between(xarea, yarea, color='y', alpha=0.3, label = "Región de rechazo con alfa = " +str(self.alpha) )
            plt.fill_between(xarea1, yarea1, color='C0', alpha=0.2, label = "Región de aceptación")
            plt.legend(loc='upper right')
            plt.xlabel("X")
            plt.text(z, 0, round(z, 3), {'color': 'b'})
        
        else:
            print("Para un alfa de " + str(self.alpha) + " y con " + str(grados_libertad_ji) + " grados de libertad se tiene un valor de ji_critical = " + str(z))
            
    def ji_square_and_ji_critical(self):
        '''
        ji_square , ji_square_critical
        -------
        doblue
            Calculo del valor de ji_square respecto a los datos.
        doblue
            Calculo del valor de ji_square_critical.

        '''
        
        self.ji_square, self.p_value_ji, self.grados_libertad_ji , self.frecuencias_esperadas_1 =self.chi_square_statistic()
        self.ji_critical = chi2.ppf(1-self.alpha, df=self.grados_libertad_ji) #ppf(q=1-prob, grados_libertad_ji , loc=0, escala=1)   
        return self.ji_square,self.ji_critical
            
    def test_hipotesis(self):
        '''
        Returns Validación de las hipotesis
        -------
        None.
        '''
        print("Para un alfa de " + str(self.alpha) + " con " + str(self.grados_libertad_ji) + " grados de libertad se tiene un valor de ji_critical = " + str(self.ji_critical))
        if self.ji_square > self.ji_critical:
            print("Aceptación de H1 = Es decir la variable X es DEPENDIENTE de la varible Y. valor de xi_critical = " +str(self.ji_critical) + " valor ji obtenida = "+ str(self.ji_square))
            
            if self.set_datos.size==4:
            
                phi=self.Coeficiente_phi()
                print("Con un grado de  asociación del "+ str(phi))
                return phi
            elif self.set_datos.size>4:
                cramer=self.Coeficiente_cramer()
                if cramer<=0.2 :    #x < 5 and  x < 10
                    print("Con un grado de  asociación debil con  valor de "+ str(cramer))
                elif cramer>0.2 and cramer<=0.6:
                    print("Con un grado de  asociación moderado con valor de "+ str(cramer))
                elif cramer>0.6 and cramer<=1:
                    print("Con un grado de  asociación fuerte con valor de "+ str(cramer))
            
            return cramer
                    
                
        else:
            print("Aceptación de HO HIPOTESIS NULA = Es decir la variable X es independiente de la varible Y. valor de xi_critical = ", str(self.ji_critical) + " valor ji obtenida = "+ str(self.ji_square) )
               
        
    def Coeficiente_phi(self):
        '''
        phi
        -------
        phi : doblue [-1,1] tabla 2x2
            Mide la fuerza de asociación entre dos variables.
            Coeficiente phi
            -1 indica una relación perfectamente negativa entre las dos variables.
            0 indica que no hay asociación entre las dos variables.
            1 indica una relación perfectamente positiva entre las dos variables.
        '''
        phi = association(observed = self.set_datos, method='cramer', correction=False, lambda_=None)
        
        return phi
    
    def Coeficiente_cramer(self):
        '''
        cramer
        -------
        cramer :doblue [0,1]
            Si cramer ≤0.2 la fuerza de asociación entre las dos variables es débil
            Si cramer 0.2<𝑉≤0.6 la fuerza de asociación entre las dos variables es moderada.
            Si cramer 0.6<𝑉≤1 entonces la asociación es fuerte.
        '''
        cramer = association(observed = self.set_datos, method='cramer', correction=False, lambda_=None)
        return cramer
        
            

        

 
