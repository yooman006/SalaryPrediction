import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
import numpy as np
import streamlit as st

class MyPredictor:

    def __init__(self,file_path):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.X = self.data[['Years of Experience']].values
        self.y = self.data['Salary'].values
        self.scalar = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train_imputed = None
        self.ridge_model = None
        self.scale_data()
        self.build_model()
    
    def scale_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y, test_size=0.2, random_state=42)
        imputer = SimpleImputer(strategy='mean')
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)


        self.y_train_imputed = imputer.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_imputed = imputer.transform(y_test.reshape(-1, 1)).ravel()


        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        self.X_test_scaled = self.scaler.transform(X_test_imputed)

    def build_model(self):
        self.ridge_model = Ridge(alpha=1.0)
        self.ridge_model.fit(self.X_train_scaled, self.y_train_imputed)

    def make_prediction(self,value):
        user_experience = float(value)
        user_input_scaled = self.scaler.transform(np.array([[user_experience]]))
        predicted_salary = self.ridge_model.predict(user_input_scaled)
        result = f'Predicted Salary for {user_experience} years of experience: {predicted_salary[0]:.2f}'
        return result


def main():
    st.title('Salary Prediction App')
    m = MyPredictor("Salary.csv")
    st.sidebar.header('Input Parameters')
    experience = st.sidebar.number_input('Years of Experience', min_value=0, step=1)

    # Make prediction
    if st.sidebar.button('Predict'):
        prediction = m.make_prediction(experience)
        st.success(prediction)

if __name__ == '__main__':
    main()
