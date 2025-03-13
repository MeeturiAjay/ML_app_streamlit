import streamlit as st
import pickle as pkl

st.header("🌿 Machine Learning Web Application - Iris Species Prediction")

st.sidebar.title("🌸 Choose Feature Values")
sepal_length = st.sidebar.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, step=0.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)", min_value=2.0, max_value=4.5, step=0.1)
petal_length = st.sidebar.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, step=0.1)
petal_width = st.sidebar.slider("Petal Width (cm)", min_value=0.1, max_value=2.5, step=0.1)


with open("iris_model.pkl", "rb") as file:
    model = pkl.load(file)

features = {
    0 : 'setosa',
    1 : 'versicolor', 
    2 : 'virginica', 
}

if st.sidebar.button("🔍 Predict Species"):
    # Perform prediction
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    predicted_species = features[int(prediction[0])]  

    # Display result
    st.success(f"🌸 **Predicted Species:** {predicted_species}")

