import tensorflow as tf
from keras.models import Sequential 
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from random import randint

def redNeuronal(hospital):
    #HOSPITAL 1: público, clínica universitaria: 18 médicos
    #HOSPITAL 2: público, 8 médicos, 3 de los cuales son especialistas
    
    numMedicos = 0
    
    if hospital == 1:
        numMedicos = 18
    elif hospital == 2:
        numMedicos = 8

    #Definir modelo y capas
    modelo = Sequential()

    #7 entradas: doctores, pacientes por hora, y categoría de triage más frecuente
    modelo.add(Dense(12, input_dim=3, activation='relu'))
    modelo.add(Dense(6, activation='relu'))
    modelo.add(Dense(6, activation='relu'))
    #4 salidas, probabilidad de colapso en cada etapa
    modelo.add(Dense(4, activation='sigmoid'))

    modelo.summary()

    modelo.compile(loss = 'binary_crossentropy', optimizer= 'adam', metrics= ['accuracy'])

    #Definir datos de entrenamiento
    
    x = np.array([[20, 6, 4], 
                  [7, 32, 5],
                  [7, 27, 1],
                  [8, 18, 2],
                  [18, 55, 3],
                  [14, 68, 4]
                  ])
    y = np.array([[0, 0, 0, 0], 
                [1, 1, 1, 0], 
                [0, 0, 1, 1], 
                [1, 0, 1, 1], 
                [1, 1, 1, 0],
                [1, 1, 1, 1]
                ])

        
    #Entrenar el modelo
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss',verbose = 1, patience= 1)
    historial = modelo.fit(x, y, epochs= 1000, batch_size = 10, callbacks = [callback])


    #Evaluar
    medicion = modelo.evaluate(x, y)
    print(f'medicion: {medicion[1]*100}%')


    #Hacer una predicción
    
    if hospital == 1:
        pacientesPorHora = randint(20, 60)
    
    elif hospital == 2:
        pacientesPorHora = randint(20, 40)
    
    categoriaTriage= randint(1,5)
    
    print(f'Para un hospital con capacidad de {numMedicos} médicos, {pacientesPorHora} pacientes por hora y categoría de triage {categoriaTriage}:')
    item= [numMedicos, pacientesPorHora, categoriaTriage]
    resultado = modelo.predict([item]) 
    np.set_printoptions(formatter={'all':lambda x: str(float("{:.1f}".format(x))*100)+ '%'})
    print(resultado)
    
    return historial


def graficarPerdida(historial):
    #Graficar pérdida respecto a número de épocas
    font1 = {'color':'darkblue','size':17}
    font2 = {'color':'mediumvioletred','size':14}

    plt.title("Pérdida respecto a épocas", fontdict = font1)
    plt.xlabel("Época" , fontdict = font2)
    plt.ylabel("Magnitud de pérdida" , fontdict = font2)
    plt.plot(historial.history["loss"], color = "darkorchid", linewidth = '4.5' )
    plt.grid(axis = 'y')
    plt.show()
    
if __name__ == '__main__':
    
    hospital = int(input("¿Qué hospital desea evaluar? "))
    
    
    info = redNeuronal(hospital)
    
    graficarPerdida(info)
    




