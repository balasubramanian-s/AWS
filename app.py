import numpy as np
import pickle
import streamlit as st

    #loading the saved model
loaded_model = pickle.load(open('C:/Users/sabarishmanogaran/OneDrive - revature.com/Desktop/ML/finalized_model.sav', 'rb'))
#model1 = pickle.load(open('C:/Users/sabarishmanogaran/OneDrive - revature.com/Desktop/DS/Projectthree/model1.pkl','rb'))

#creating a function for predection
def procurement_preduiction(input_data):
    
    input_data_as_numpy_as_array = np.asarray(input_data)

    #resahpe the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_as_array.reshape(1,-1)

    Prediction = loaded_model.predict(input_data_reshaped)
    print(Prediction)

    if (Prediction[0] == 0):    
       return 'Procurement Fraud does not happen'
    else: 
        return 'Procurement Fraud happens'
        
        
def main():
    
    #giving a title
    st.title('Procuremnt Fruad Web App')
    
    #getting the input data from the user
    
    UnitPrice = st.text_input('Number of Unit Price')
    InflatedInvoice = st.text_input('InflatedInvoice')
    Employeescolludingwithsupplierswithhighercost = st.text_input('Employees colluding with suppliers with higher cost')
    
    print(type(UnitPrice))
    print(type(InflatedInvoice))
    print(type(Employeescolludingwithsupplierswithhighercost))
    
    #code for Prediction
    Fraudness = ''
    
    #creating a button for Prediction
    if st.button('Predict'):
        Fraudness = procurement_preduiction([int(UnitPrice), int(InflatedInvoice), int(Employeescolludingwithsupplierswithhighercost)])
        print('Format: ', Fraudness)
        print('Type: ', type(Fraudness))
        st.success(Fraudness)
        
       
        
     
if __name__ == '__main__':
    main()          
        
            




    
    
    
    


    
       
    

    
    



    

     
    

    

     



    
    
    
    
    
