#___IMPORTING DEPENDENCIES___#
import pickle
import numpy as np 
import streamlit as st

#___LOADING THE TRAINED MODEL___#
load_model = pickle.load(open("train_model.sav", 'rb'))
#___LABEL ENCODING___#
encoder ={'Gender': {'Male':1, 
              'Female':0},
'Sleep Duration': {"'Less than 5 hours'": 0, "'5-6 hours'": 1, 
    "'7-8 hours'": 2, "'More than 8 hours'": 3, 
    'Others': 4
    },
'Dietary Habits': {'Unhealthy': 0, 'Moderate': 1, 'Healthy': 2, 'Others': 3},
'Sucidal Thought':{'Yes':1, 'No':0},
'Family History':{'Yes':1, 'No':0}}

#___FORM TITLE___#
st.title('🎓STUDENTS DEPRESSION DETECTOR')
st.markdown("**Provide the detail's below to predict the student Depression Risk**")
st.write('---')
#___FORM LAYOUT___#
col1, col2 = st.columns(2)
with col1:
  Gender = st.selectbox("**Gender**", ['Select Gender','Male', 'Female'], index=0)
  Age = st.number_input('**Age Of Person**', min_value=10, max_value=50, value=None)
  Academic_Pressure = st.slider('**Academic Pressure (0-5)**',0,5, value=None)
  CGPA = st.slider('**CGPA**',0.0,10.0, value=0.0, step=0.01)
  Study_Statisfaction = st.slider('**Study Satisfaction (0-5)**',0,5, value=None)
  Sleep_Duration = st.selectbox("**Sleep Duration**", ['Select Sleep Duration',"'Less than 5 hours'", "'5-6 hours'", 
    "'7-8 hours'", "'More than 8 hours'", 
    'Others'],index=0)  
with col2:
  Dietary_Habits = st.selectbox("**Dietary Habits**", ['Select Dietary Habit','Unhealthy', 'Moderate', 'Healthy', 'Others'], index=0)
  suicidal_thoughts  = st.radio('**Sucidal Thought?**', ['Yes', 'No'], index=None)
  StudyHours = st.slider('**Study Hours/Day**',0,24, value=None)
  Financial_Stress = st.slider('**Financial Stress (0-5)**',0,5, value=0)
  Mental_Illness = st.radio('**Family History Of Mental Illness?**',['Yes', 'No'], index=None)

def encode_inputs():
     return[
        encoder['Gender'][Gender],
        float(Age),
        float(Academic_Pressure), 
        float(CGPA),
        float(Study_Statisfaction), 
        encoder['Sleep Duration'][Sleep_Duration],
        encoder['Dietary Habits'][Dietary_Habits],
        encoder['Sucidal Thought'][suicidal_thoughts],
        float(StudyHours),
        float(Financial_Stress),
        encoder['Family History'][Mental_Illness]]
st.markdown('---')
#____CREATING THE BUTTON___#
predict = st.button("🔍PREDICT DEPRESSION RISK", type='secondary', use_container_width=True, key='predict_btn')
if predict:
   try: 
      #___PREDICTIVE SYSTEM___#
      input_encoded = encode_inputs()
      input_data_as_numpy_array = np.asarray(input_encoded)
      input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
      prediction = load_model.predict(input_data_reshaped)
      st.markdown('---')
      
       #___PREDICTION OUTPUT AND SOLUTIONS___#
      if prediction[0]== 0:
        st.success("✅The Student is NOT Depressed.")
        st.info("Maintain healthy habits and balance.")
      else:
         st.error('⚠️The Student is at Risk of Depression ')
         st.warning('Consider seeking support from campus counseling services.')
      #___FILLING ALL THE FIELDS REQUIREMENT___#
   except (TypeError, ValueError, KeyError, TabError, IndexError):
          st.error('❌Please fill all the fields before predicting!')


