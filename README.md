# 🎓 Student Depression Prediction Web App
This is a Machine Learning Web Application that predicts a student's risk of depression based on academic, lifestyle, and psychological input data.
It uses a trained Logistic Regression model and a Streamlit web interface for user-friendly interaction.

---

# 🚀 Project Overview
• **Model Type**: Supervised Machine Learning (Classification)

• **Web Framework**: Streamlit

• **Core Libraries**: Scikit-learn, Pandas, NumPy, Pickle

• **Language**: Python

_The user provides details such as academic pressure, sleep duration, dietary habits, and other factors through a web form. 
The app processes the inputs and predicts whether the student is at high or low risk of depression, offering relevant guidance._

---

# 🧠 How It Works
1. A pre-trained Logistic Regression model (trained on a dataset of over 27,000 student records) is loaded using `pickle`.
2. The user inputs their information via the interactive Streamlit form.
3. The app encodes the categorical data (like Gender, Sleep Duration) into the numeric format the model expects.
4. The model makes a prediction, and the result is displayed instantly with a risk assessment and actionable recommendations.

---

# 💻 How to Run the App Locally

## 1️⃣ Clone the Repository
### Open your terminal (Command Prompt, PowerShell, or Terminal) and run the following commands:
``git clone https://github.com/Olamilekan-23-ML/student-depression-prediction.git``

``cd Student-s-Depression-Risk-Prediction ``
## 2️⃣ Install Dependencies
### Ensure you have Python installed, then run:
``pip install -r requirements.txt`` 
## 3️⃣ Run the Streamlit App
### Start the application with the following command:
``streamlit run Depressed.py`` 

_Then open the URL shown in your terminal (usually http://localhost:8501) in your web browser._

---

# 📂 Project Structure

| File | Description |
| :--- | :--- |
| `Depressed.py` | The main Streamlit app script that runs the web interface. |
| `train_model.sav` | The serialized, trained machine learning model. |
| `Student_Depression.py` | The Python file containing the complete data analysis and model training code. |
| `student_depression_dataset.csv` | The dataset used for training the model. |
| `requirements.txt` | List of Python dependencies required to run the app. |
| `README.md` | This file. |

---

# 🧰 Technologies Used
• Python

• Streamlit

• Scikit-learn

• NumPy & Pandas

• Pickle

---

# ⚠️ Important Disclaimer
_This tool is for educational and screening purposes only. It is not a substitute for professional medical advice, 
diagnosis, or treatment. Always seek the advice of qualified mental health providers with any questions you may have._

---

# 👤 Author
*_OLAMILEKAN_*

*_GitHub:@Olamilekan-23-ML_*
