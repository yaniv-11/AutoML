# Customizable Machine Learning Playground

An interactive **AutoML Playground** built with Streamlit and scikit-learn—designed for easy experimentation with **classification**, **regression**, and **ensemble modeling**.

---

##  Live Demo  
Explore the app here: (https://customizable-machine-learning-playground.streamlit.app/)

---

##  Features  
- **Upload any CSV dataset** (drag & drop) and view a live preview  
- **Target selection**: auto-detects classification or regression tasks  
- **Per-column control**: choose how to impute missing values, encode categorical features, and scale numeric features  
- **Model selection**: support for Logistic Regression (L1/L2/ElasticNet), Decision Trees, Random Forest, Ridge, Lasso, ElasticNet  
- **Ensemble options**: Voting and Bagging with easy switching  
- **Instant evaluation metrics**:
  - **Classification**: Accuracy, Precision, Recall, F1-score, ROC-AUC  
  - **Regression**: MAE, RMSE, R²  
- **Downloadable results** for easy sharing or reporting  

---

##  Tech Stack  
- **Frontend**: Streamlit for UI and rapid deployment  
- **Backend**: scikit-learn for modeling and preprocessing  
- **Data handling**: pandas + NumPy  
- **Deployment**: Streamlit Community Cloud (aka Streamlit Cloud) — fully functional app published and accessible publicly  

---

##  How to Run Locally  
```bash
git clone <your-repo-url>
cd <project-folder>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
