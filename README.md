
# 💬 AI Echo – Sentiment Analysis Web App

AI Echo is an **NLP-powered Sentiment Analysis System** that classifies user reviews or text inputs into **Positive 😊, Negative 😠, or Neutral 😐** sentiments.
The project uses **Machine Learning and Natural Language Processing (NLP)** to analyze textual data, extract features, and visualize insights interactively through **Streamlit**.

---

## 🔧 Tech Stack

![Python](https://img.shields.io/badge/Python-3.8%2B-gray?logo=python&logoColor=white&labelColor=3776AB)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-gray?logo=pandas&logoColor=white&labelColor=150458)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-gray?logo=numpy&logoColor=white&labelColor=013243)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-ML%20Models-gray?logo=scikit-learn&logoColor=white&labelColor=f89939)
![NLTK](https://img.shields.io/badge/NLTK-Text%20Processing-gray?logo=nltk&logoColor=white&labelColor=154D2E)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-gray?logo=plotly&logoColor=white&labelColor=11557c)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-gray?logo=plotly&logoColor=white&labelColor=3F4F75)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-gray?logo=streamlit&logoColor=white&labelColor=FF4B4B)
![Joblib](https://img.shields.io/badge/Joblib-Model%20Serialization-gray?logo=python&logoColor=white&labelColor=3776AB)
![Google%20Colab](https://img.shields.io/badge/Google%20Colab-Notebook-gray?logo=google-colab&logoColor=white&labelColor=f9ab00)


---

## 📁 Project Structure

```
📂 Sentiment-Analysis-AI-Echo
│
├── 📁 app/                         # Streamlit app files
│   └── streamlit_app.py
│
├── 📁 dataset/                     # Data used for EDA and modeling
│   ├── cleaned_senti_mapped_data.csv       # Cleaned + sentiment mapped data (TextBlob)
│   ├── filtered_data_for_model.csv         # Review & sentiment data for model building
│   └── raw_data.csv                        # Raw review data
│
├── 📁 model/                       # Trained ML models
│   └── sentiment_model.pkl
│
├── 📁 notebook/                    # Jupyter/Colab notebooks for EDA and model building
│   └── Sentiment_Analysis.ipynb
│
├── requirements.txt                # Python dependencies
├── .gitignore                      # Ignored files for Git
├── LICENSE                         # Open-source license
└── README.md                       # Project documentation
```

---

## 📷 App Screenshots

### 📄 Overview
![Overview](https://github.com/user-attachments/assets/62d287af-c1d6-4505-b1dc-59c9e094ddb1)

---

### 📊 EDA Charts
![EDA](https://github.com/user-attachments/assets/1ab8699b-e7ec-4185-b5e8-9fc1568b7483)

##
![EDA](https://github.com/user-attachments/assets/16b209a6-0235-4715-8ac0-1291b757c069)

---

### ⚖️ Sentiment Prediction
![Prediction](https://github.com/user-attachments/assets/af405f62-5d66-4b04-9fc4-ca40a402b883)

---

## 🚀 How to Run

1️⃣ Clone the repository  
```
git clone https://github.com/Infant-Joshva/Project_5-AI-Echo.git
cd Project_5-AI-Echo
```

2️⃣ Install dependencies  
```
pip install -r requirements.txt
```

3️⃣ Run the Streamlit app  
```
streamlit run app/main.py
```

---

## 📊 Features

- 🧹 **Text Preprocessing** (Tokenization, stopword removal, lemmatization using NLTK)
- 🔠 **Feature Extraction** (TF-IDF Vectorization)
- 🧮 **Model Training** (Naïve Bayes, Logistic Regression, Random Forest)
- 💬 **Live Prediction Interface** (User inputs text → Sentiment with emoji & probability)
- 📈 **EDA Visuals** (Word clouds, sentiment distribution charts)
- ⚡ **Streamlit Dashboard** for interactive analysis

---

## 👤 Author

**Infant Joshva**  
📧 infantjoshva2024@gmail.com  
🐙 [GitHub](https://github.com/Infant-Joshva)  
🔗 [LinkedIn](https://www.linkedin.com/in/infant-joshva)

---

## ⭐ Give a Star!

If you like this project, please ⭐ the repo — it motivates me to build more! 🚀

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.
