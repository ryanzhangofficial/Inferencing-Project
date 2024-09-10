from algorithm import algorithm
from utilities.models import createDefaultModelList

def main():
    T = int(input("Input total amount of requests: "))  
    c = float(input("Input hyperparameter c: "))  
    m_list = createDefaultModelList()  

    algorithm(T, c, m_list)  

if __name__ == "__main__":
    main()
