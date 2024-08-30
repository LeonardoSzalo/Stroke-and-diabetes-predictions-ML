import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import base64
from io import BytesIO

def run():
    st.title("Machine Learning Models")

    def preprocess_data(df, alvo):
        categorical_vars_dict = {
            'DIABETES': [
                'ALCOHOL_FREQUENCY', 'VIGOROUS_ACTIVITY', 'MODERATE_ACTIVITY', 'SMOKE', 
                'KIDNEY_DISEASE', 'DIALYSES', 'GENDER', 'RACE', 'EDUCATION', 
                'MARITAL_STATUS', 'HIGH_PRESSURE', 'HIGH_CHOLESTEROL', 
                'ASTHMA', 'BLOOD_TRANSFUSION', 'ARTHRITIS', 'CONGESTIVE_HEART_FAILURE', 
                'CORONARY_DISEASE', 'HEART_ATTACK', 'STROKE', 'THYROID_PROBLEMA', 
                'COPD', 'LIVER_DISEASE', 'ABDOMINAL_PAIN', 'GALLSTONES', 'CANCER', 
                'HEALTH_INSURANCE', 'SNORE', 'SNORT', 'TROUBLE_SLEEPING', 'OVERLY_SLEEPY'
            ],
            'HEART_ATTACK': [
                'ALCOHOL_FREQUENCY', 'VIGOROUS_ACTIVITY', 'MODERATE_ACTIVITY', 'SMOKE', 
                'KIDNEY_DISEASE', 'DIALYSES', 'GENDER', 'RACE', 'EDUCATION', 
                'MARITAL_STATUS', 'HIGH_PRESSURE', 'HIGH_CHOLESTEROL', 
                'ASTHMA', 'BLOOD_TRANSFUSION', 'ARTHRITIS', 'CONGESTIVE_HEART_FAILURE', 
                'CORONARY_DISEASE', 'DIABETES', 'STROKE', 'THYROID_PROBLEMA', 
                'COPD', 'LIVER_DISEASE', 'ABDOMINAL_PAIN', 'GALLSTONES', 'CANCER', 
                'HEALTH_INSURANCE', 'SNORE', 'SNORT', 'TROUBLE_SLEEPING', 'OVERLY_SLEEPY'
            ],
            'STROKE': [
                'ALCOHOL_FREQUENCY', 'VIGOROUS_ACTIVITY', 'MODERATE_ACTIVITY', 'SMOKE', 
                'KIDNEY_DISEASE', 'DIALYSES', 'GENDER', 'RACE', 'EDUCATION', 
                'MARITAL_STATUS', 'HIGH_PRESSURE', 'HIGH_CHOLESTEROL', 
                'ASTHMA', 'BLOOD_TRANSFUSION', 'ARTHRITIS', 'CONGESTIVE_HEART_FAILURE', 
                'CORONARY_DISEASE', 'DIABETES', 'HEART_ATTACK', 'THYROID_PROBLEMA', 
                'COPD', 'LIVER_DISEASE', 'ABDOMINAL_PAIN', 'GALLSTONES', 'CANCER', 
                'HEALTH_INSURANCE', 'SNORE', 'SNORT', 'TROUBLE_SLEEPING', 'OVERLY_SLEEPY'
            ]
        }

        categorical_vars = categorical_vars_dict.get(alvo, [])

        y = df[alvo]
        X = df.drop(columns=[alvo])
        
        X_encoded = pd.get_dummies(X, columns=categorical_vars, drop_first=True)
        
        numerical_vars = X_encoded.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()
        X_encoded[numerical_vars] = scaler.fit_transform(X_encoded[numerical_vars])
        
        return X_encoded, y

    def plot_roc_curve(model, X_test, y_test, model_name, num_classes=None):
        fig, ax = plt.subplots(figsize=(8, 6))
        if num_classes and num_classes > 2:
            from sklearn.preprocessing import label_binarize
            
            y_test_bin = label_binarize(y_test, classes=model.classes_)
            y_prob = model.predict_proba(X_test)
            
            fpr = {}
            tpr = {}
            roc_auc = {}
            for i in range(y_test_bin.shape[1]):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            for i in range(y_test_bin.shape[1]):
                ax.plot(fpr[i], tpr[i], label=f'Class {model.classes_[i]} (AUC = {roc_auc[i]:.4f})')
            ax.plot([0, 1], [0, 1], color='#e85d04', linestyle='--')
            ax.fill_between(fpr[i], tpr[i], alpha=0.2, color='#023047')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
            ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
            ax.set_title(f'ROC AUC - {model_name} (Multiclass)', fontsize=14)
            ax.legend(loc='lower right')
            ax.grid(False)
        else:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_prob = model.decision_function(X_test)
                y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
            else:
                raise ValueError(f"O modelo {model_name} não suporta previsão de probabilidade.")
            
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')
            ax.plot([0, 1], [0, 1], color='#e85d04', linestyle='--')
            ax.fill_between(fpr, tpr, alpha=0.2, color='#023047')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
            ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
            ax.set_title(f'ROC AUC - {model_name}', fontsize=14)
            ax.legend(loc='lower right')
            ax.grid(False)

        # Salvar imagem em um buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        st.markdown(f'<div style="text-align: center;"><img src="data:image/png;base64,{img_base64}" style="max-width: 100%; height: auto;" /></div>', unsafe_allow_html=True)

        # Texto específico para cada combinação de modelo e doença
        texts = {
            ('DIABETES', 'Logistic Regression'): "This  was the most accurate model. To check the model tunning, F1 score and Recall, go to 'more information' and check my GitHub repository. Logistic Regression is a statistical model used for binary classification tasks. It predicts the probability that an instance belongs to a particular class (e.g., disease or no disease) by applying a logistic function to a linear combination of input features. It's simple, interpretable, and often used as a baseline model in classification problems. The most influencial variables were: age, waist circunference, high blood pressure, being asian and heart attack.",
            ('DIABETES', 'Random Forest'): "Random Forest is an ensemble learning method that builds multiple decision trees and merges them to get a more accurate and stable prediction. Each tree is trained on a random subset of data and features. The final prediction is made by averaging the predictions of all the trees (for regression) or by majority voting (for classification). Random Forest is known for its robustness and ability to handle large datasets with high dimensionality.",
            ('DIABETES', 'Decision Tree'): "A Decision Tree is a flowchart-like structure where internal nodes represent decisions based on feature values, branches represent outcomes of these decisions, and leaf nodes represent final predictions or classes. It's intuitive and easy to interpret but can be prone to overfitting, especially with complex datasets.",
            ('DIABETES', 'SVM'): "SVM is a supervised learning model used for classification and regression tasks. It works by finding the hyperplane that best separates the data into different classes. SVM is effective in high-dimensional spaces and is versatile, as it can be used with different kernel functions to handle non-linear relationships.",
            ('DIABETES', 'XGBoost'): "XGBoost (Extreme Gradient Boosting) is an advanced implementation of gradient boosting, which is an ensemble learning technique. It builds models sequentially, with each new model trying to correct the errors made by the previous ones. XGBoost is highly efficient and often performs exceptionally well in machine learning competitions due to its ability to handle large datasets, reduce overfitting, and improve predictive accuracy.",



            ('HEART_ATTACK', 'Logistic Regression'): "This  was the most accurate model. To check the model tunning, F1 score and Recall, go to 'more information' and check my GitHub repository. Logistic Regression is a statistical model used for binary classification tasks. It predicts the probability that an instance belongs to a particular class (e.g., disease or no disease) by applying a logistic function to a linear combination of input features. It's simple, interpretable, and often used as a baseline model in classification problems. The most influencial variables were: age, presence of diabetes, high blood pressure, waist circunference and height.",
            ('HEART_ATTACK', 'Random Forest'): "Random Forest is an ensemble learning method that builds multiple decision trees and merges them to get a more accurate and stable prediction. Each tree is trained on a random subset of data and features. The final prediction is made by averaging the predictions of all the trees (for regression) or by majority voting (for classification). Random Forest is known for its robustness and ability to handle large datasets with high dimensionality.",
            ('HEART_ATTACK', 'Decision Tree'): "A Decision Tree is a flowchart-like structure where internal nodes represent decisions based on feature values, branches represent outcomes of these decisions, and leaf nodes represent final predictions or classes. It's intuitive and easy to interpret but can be prone to overfitting, especially with complex datasets.",
            ('HEART_ATTACK', 'SVM'): "SVM is a supervised learning model used for classification and regression tasks. It works by finding the hyperplane that best separates the data into different classes. SVM is effective in high-dimensional spaces and is versatile, as it can be used with different kernel functions to handle non-linear relationships.",
            ('HEART_ATTACK', 'XGBoost'): "XGBoost (Extreme Gradient Boosting) is an advanced implementation of gradient boosting, which is an ensemble learning technique. It builds models sequentially, with each new model trying to correct the errors made by the previous ones. XGBoost is highly efficient and often performs exceptionally well in machine learning competitions due to its ability to handle large datasets, reduce overfitting, and improve predictive accuracy.",


            ('STROKE', 'Logistic Regression'): "  Logistic Regression is a statistical model used for binary classification tasks. It predicts the probability that an instance belongs to a particular class (e.g., disease or no disease) by applying a logistic function to a linear combination of input features. It's simple, interpretable, and often used as a baseline model in classification problems. The most influencial variables were: age, high blood pressure, being black, almost always being overly sleepy during day and high systolic pressure.",
            ('STROKE', 'Random Forest'): "Random Forest is an ensemble learning method that builds multiple decision trees and merges them to get a more accurate and stable prediction. Each tree is trained on a random subset of data and features. The final prediction is made by averaging the predictions of all the trees (for regression) or by majority voting (for classification). Random Forest is known for its robustness and ability to handle large datasets with high dimensionality.",
            ('STROKE', 'Decision Tree'): "A Decision Tree is a flowchart-like structure where internal nodes represent decisions based on feature values, branches represent outcomes of these decisions, and leaf nodes represent final predictions or classes. It's intuitive and easy to interpret but can be prone to overfitting, especially with complex datasets.",
            ('STROKE', 'SVM'): "SVM is a supervised learning model used for classification and regression tasks. It works by finding the hyperplane that best separates the data into different classes. SVM is effective in high-dimensional spaces and is versatile, as it can be used with different kernel functions to handle non-linear relationships.",
            ('STROKE', 'XGBoost'): "This  was the most accurate model. To check the model tunning, F1 score and Recall, go to 'more information' and check my GitHub repository. XGBoost (Extreme Gradient Boosting) is an advanced implementation of gradient boosting, which is an ensemble learning technique. It builds models sequentially, with each new model trying to correct the errors made by the previous ones. XGBoost is highly efficient and often performs exceptionally well in machine learning competitions due to its ability to handle large datasets, reduce overfitting, and improve predictive accuracy.",


        }
        
        key = (alvo, model_name)
        st.write(texts.get(key, "Texto padrão para essa combinação não disponível."))

        plt.close(fig)

    def plot_top_features(coef_df, top_n=5):
        color_palette = ['#023047', '#e85d04', '#0077b6', '#0096c7', '#ff9c33']
        top_features = coef_df.sort_values(by='Coefficient', ascending=False).head(top_n)
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x='Coefficient', y='Feature', data=top_features, palette=color_palette, ax=ax)
        ax.set_title('Top 5 most influential variables', fontsize=14)
        ax.set_xlabel('Coefficient', fontsize=12)
        ax.grid(False)
        plt.yticks(rotation = 45)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        
        # Salvar imagem em um buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        st.markdown(f'<div style="text-align: center;"><img src="data:image/png;base64,{img_base64}" style="max-width: 100%; height: auto;" /></div>', unsafe_allow_html=True)

        plt.close(fig)

    # Carregar dados e realizar o pré-processamento
    df = pd.read_excel(r"Streamlit_Nhanes/dados_demograficos_streamlit1.xlsx")
    
    # Selecione a doença e o modelo a partir das seleções do usuário
    model_name = st.selectbox("Choose the model", ["Logistic Regression", "Random Forest", "Decision Tree", "SVM", "XGBoost"])
    alvo = st.selectbox("Choose the disease", ["DIABETES", "HEART_ATTACK", "STROKE"])

    # Pré-processar dados para a doença selecionada
    X, y = preprocess_data(df, alvo)

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Treinar o modelo selecionado
    if model_name == "Logistic Regression":
        model = LogisticRegression(C=0.1, penalty='l1', solver='liblinear')
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == "SVM":
        model = SVC(probability=True, kernel='linear', random_state=42)
    elif model_name == "XGBoost":
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    else:
        st.error("Modelo não suportado")
        return

    model.fit(X_train, y_train)
    
    # Plotar a curva ROC
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy {model_name}: {accuracy:.2f}")
    plot_roc_curve(model, X_test, y_test, model_name, num_classes=len(set(y)))
    
    
    # Plotar gráfico das principais características se for Regressão Logística
    if model_name == "Logistic Regression":
        coef_df = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': model.coef_[0]
        })
        plot_top_features(coef_df)

if __name__ == "__main__":
    run()

