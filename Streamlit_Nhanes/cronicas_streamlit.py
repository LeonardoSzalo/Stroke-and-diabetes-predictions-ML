import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
import io
import base64



def run():


    st.title("Chronic Disease Analysis")

    # Introdução
    st.write("""
    
    Chronic diseases are long-lasting conditions that typically progress slowly over time. They include a wide range of health issues such as diabetes, heart disease, stroke, and hypertension, among others. Unlike acute conditions, chronic diseases require ongoing management and can lead to severe complications if not properly controlled.

    The impact of chronic diseases on public health is profound, accounting for the majority of healthcare costs and contributing to significant morbidity and mortality worldwide. These diseases often share common risk factors, including unhealthy diet, physical inactivity, tobacco use, and excessive alcohol consumption. Addressing chronic diseases requires a comprehensive approach that includes prevention, early detection, and effective management to reduce the burden on individuals and healthcare systems.

    

    """)

    # Carregar os dados
    df = pd.read_excel(r"Streamlit_Nhanes/dados_demograficos_streamlit1.xlsx")
    color_palette = ['#023047', '#e85d04', '#0077b6', '#0096c7', '#ff9c33']
    sns.set_palette(sns.color_palette(color_palette))

    def categorize_bmi(bmi):
        if bmi < 18.5:
            return 'Underweight'
        elif 18.5 <= bmi < 24.9:
            return 'Normal weight'
        elif 25 <= bmi < 29.9:
            return 'Overweight'
        else:
            return 'Obesity'


    # Adicionar a coluna de categorias
    df['BMI_Category'] = df['BMI'].apply(categorize_bmi)


    # Função para plotar gráficos e retornar a imagem como bytes
    def plot_countplot(df, column, title, xticks_labels, hue='GENDER'):
        plt.figure(figsize=(6, 6), dpi=100)  # Ajuste o tamanho e a resolução da imagem

        # Criar o gráfico de contagem
        ax = sns.countplot(data=df, x=column, hue=hue, palette=color_palette[:2], dodge=True)

        # Ajustar o título e os ticks
        ax.set_title(title, fontweight='bold')

        unique_values = df[column].unique()
        tick_positions = np.arange(len(unique_values))
        ax.set_xticks(tick_positions)

        if len(xticks_labels) == len(unique_values):
            ax.set_xticklabels(xticks_labels, ha='center', rotation=45, fontweight='bold')
        else:
            st.warning(f"Number of xticks_labels ({len(xticks_labels)}) does not match number of unique values ({len(unique_values)}) in the column '{column}'")

        ax.set_xlabel('')
        ax.grid(False)
        ax.yaxis.set_visible(False)

        bar_width = 0.35
        for patch in ax.patches:
            patch.set_width(bar_width)

        total = len(df)
        counts = [patch.get_height() for patch in ax.patches]
        percentages = [(count / total) * 100 for count in counts]

        for patch, percentage in zip(ax.patches, percentages):
            x = patch.get_x() + patch.get_width() / 2
            y = patch.get_height()
            if percentage > 0:
                ax.annotate(f"{percentage:.1f}%", (x, y + 0.5), ha='center', va='bottom', xytext=(0, 3), textcoords='offset points')

        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, ['Male', 'Female'], title=hue, loc='upper right')

        plt.tight_layout(pad=1.0)

        # Salvar a figura em um buffer de bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)

        return img

    # Checkboxes para seleção dos gráficos
    selected_graphs = []
    if st.checkbox('Diabetes'):
        selected_graphs.append('DIABETES')
    if st.checkbox('Kidney Disease'):
        selected_graphs.append('KIDNEY_DISEASE')
    if st.checkbox('Stroke'):
        selected_graphs.append('STROKE')
    if st.checkbox('High Cholesterol'):
        selected_graphs.append('HIGH_CHOLESTEROL')
    if st.checkbox('Liver Condition'):
        selected_graphs.append('LIVER_DISEASE')
    if st.checkbox('Heart Attack'):
        selected_graphs.append('HEART_ATTACK')
    if st.checkbox('BMI'):
        selected_graphs.append('BMI_Category')
    if st.checkbox('Blood Pressure'):
        selected_graphs.append('HIGH_PRESSURE')

    # Plotar os gráficos selecionados e exibir descrições
    for graph in selected_graphs:
        img = None
        description = ""
        if graph == 'DIABETES':
            img = plot_countplot(df, column='DIABETES', title='Diabetes Distribution', xticks_labels=['Diabetic', 'Not Diabetic', 'Borderline', "Don't Know"])
            description = "Diabetes is a chronic condition characterized by high levels of glucose in the blood due to either insufficient insulin production (Type 1) or insulin resistance (Type 2). It can lead to severe complications such as heart disease, nerve damage, and kidney failure. The cost to the public health system is substantial, with the U.S. spending approximately $327 billion on diabetes-related medical expenses and lost productivity in the last year."
        elif graph == 'KIDNEY_DISEASE':
            img = plot_countplot(df, column='KIDNEY_DISEASE', title='Kidney Disease Distribution', xticks_labels=['Failing Kidneys', 'Healthy Kidneys', "Don't Know"])
            description = "Kidney disease refers to a range of conditions that impair the kidney’s ability to filter waste from the blood. Chronic kidney disease (CKD) can progress to end-stage renal disease (ESRD), requiring dialysis or a kidney transplant. The annual cost to the public health system for kidney disease, including dialysis and transplant procedures, is estimated at over $50 billion in the U.S."
        elif graph == 'STROKE':
            img = plot_countplot(df, column='STROKE', title='Stroke Distribution', xticks_labels=['Stroke', 'Never had a stroke', "Don't Know"])
            description = "A stroke occurs when blood flow to a part of the brain is interrupted, leading to brain cell damage. It can result in long-term disability, including paralysis and cognitive impairments. The annual cost of stroke care, including hospitalizations, rehabilitation, and long-term care, amounts to about $45 billion in the U.S."
        elif graph == 'HIGH_CHOLESTEROL':
            img = plot_countplot(df, column='HIGH_CHOLESTEROL', title='High Cholesterol Distribution', xticks_labels=['High Cholesterol', 'Normal Cholesterol', 'Refused', "Don't Know"])
            description = "High cholesterol is a condition where there are elevated levels of cholesterol in the blood, increasing the risk of heart disease and stroke. Managing high cholesterol typically involves lifestyle changes and medication. The U.S. spends roughly $30 billion annually on treating high cholesterol and its related complications."
        elif graph == 'LIVER_DISEASE':
            img = plot_countplot(df, column='LIVER_DISEASE', title='Liver Conditions Distribution', xticks_labels=['Have any liver condition', 'No liver condition', "Don't Know"])
            description = "Liver disease encompasses various conditions affecting the liver, including hepatitis, cirrhosis, and fatty liver disease. Chronic liver disease can lead to liver failure and requires treatments such as medications or liver transplants. The cost of liver disease to the public health system is estimated at over $20 billion per year in the U.S."
        elif graph == 'HEART_ATTACK':
            img = plot_countplot(df, column='HEART_ATTACK', title='Heart attack Distribution', xticks_labels=['Had a heart Attack', 'Never had a heart attack', "Don't Know"])
            description = "A heart attack, or myocardial infarction, occurs when blood flow to the heart is blocked, causing damage to the heart muscle. It is often caused by coronary artery disease. The cost of heart attacks, including emergency care, hospitalization, and long-term treatment, is about $220 billion annually in the U.S."
        elif graph == 'BMI_Category':
            img = plot_countplot(df, column='BMI_Category', title='BMI Distribution', xticks_labels=['Obesity', 'Overweight', 'Normal Weight', 'Underweight'])
            description = "Obesity is characterized by excessive body fat accumulation and is associated with various health problems such as diabetes, heart disease, and hypertension. The annual economic burden of obesity in the U.S., including direct medical costs and lost productivity, is estimated at over $170 billion."
        elif graph == 'HIGH_PRESSURE':
            img = plot_countplot(df, column='HIGH_PRESSURE', title='Hypertension Distribution', xticks_labels=['High Blood Pressure', 'Normal Blood Pressure', "Don't Know"])
            description = "High blood pressure, or hypertension, is a condition where the force of the blood against the artery walls is consistently too high. It can lead to heart disease, stroke, and kidney damage. The cost of managing high blood pressure and its complications in the U.S. is estimated to be over $50 billion per year."

        if img:
            # Converter a imagem para bytes
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            # Mostrar a imagem centralizada
            st.markdown(
                f'<div style="text-align: center;"><img src="data:image/png;base64,{base64.b64encode(img_buffer.getvalue()).decode()}" style="width: 80%; max-width: 800px;" /></div>',
                unsafe_allow_html=True
            )
            # Mostrar a descrição centralizada
            st.markdown(
                f'<div style="text-align: center;">{description}</div>',
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    run()
