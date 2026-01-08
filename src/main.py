# 1.0 Generic Modules
from math import trunc
# 1.1 My Modules
from etl import load
from tree import DecisionTree, DecisionTreePruning
from inference import Engine


# 2.0 Constants
SIZE= 100000

# 3.0 Main Function
def main():
    # 3.1 Initial Prints
    # Only this part will be in Spanish -><-
    print("Hola muy Buenas, Soy Carlos y este es el árbol de decision ID3 que he hecho.")
    print("Siempre trabajamos con la misma base de datos pero cambia: ")
    print(" - Con que porcentaje entrenamos y con cual validamos")
    print(" - Si queremos usar un ID3 puro o un ID3 con prepoda.")
    print("\nPor favor, escriba un número decimal entre el 0 y el 1 que represente el porcentaje\n de la base de datos que vamos a usar para entrenar el árbol:")
    
    # 3.2 Inputs
    while True:
        try:
            percent= float(input())
            if percent not in [0.0, 1.0]:
                print("VALOR VÁLIDO: PORCENTAJE")
                break

        except ValueError:
            print("ERROR: Parece que has metido el número mal,\n Por favor escribe el número con este formato: 0.384\nGracias.")

    print("\n Por favor, ahora escriba una sola letra mayúscula A si prefiere el árbol puro o una \nletra mayúscula B si prefiere la versión con prepoda:")
    while True:
        try:
            kind = str(input())
            if kind not in ["A", "B"]:
                print("VALOR VÁLIDO: TIPO")
                break
        
        except Exception:
            print("ERROR: Parece que lo escrito no corresponde a ninguna de las opciones.\nPor favor, escriba una sola letra mayúscula A o B sin espacios.\nGracias.")

    # 3.3 Post Selection Prints
    print("TODO LISTO: Ya podemos empezar con el proceso de entrenamiento. Los pasos que se van a seguir son: ")
    print("\n\t- Se leera el archivo diabetes.csv en un DataFrame de la librería Pandas")
    print("\n\t- Se seleccionará el porcentaje de información que usted ha especificado.")
    print("\n\t- Se instansciará la clase del árbol de decisión que haya decidido.")
    print("\n\t- Se entrenará el modelo y se guardará en un json.")
    print("\n\t- Después se cargará el modelo y se hará inferencia para las filas que se\n\tguardaron para validar el modelo.")
    print("\n\t- Por último se imprimiran las métricas MSE (Mean Squared Error) y \n\tRMSE (Root MSE) demostrando como de bien se entrenó el modelo")

    # 4.0 Main Function
    # Preprocessing
    n_rows= trunc(SIZE* percent)
    # Read it and Load it (and Categorize it)
    training, validation= load(n_rows)
    # We instance the Decision Tree
    model= DecisionTree(training) if kind=='A' else DecisionTreePruning(training)
    # We train it and store the model
    model.run()
    # We load the model and instance the Inference Engine
    engine= Engine(validation)
    # We Inference the Results and get the precision of the model
    report, blunders, desperates= engine.run()
    # Ta Chaaan! Here you have it ; )
    print(report)
    print(f"Blunders: {blunders}")
    print(f"Resorts: {desperates}")

if __name__=='__main__':
    main()