import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from io import BytesIO
import base64

def run():
    # Configurações iniciais
    color_palette = ['#023047', '#e85d04', '#0077b6', '#0096c7', '#ff9c33']
    sns.set_palette(sns.color_palette(color_palette))
    pd.set_option('display.max_columns', None)

    # Carregamento dos dados
    df_concatenado = pd.read_excel(r"Streamlit_Nhanes/dados_demograficos_streamlit1.xlsx")

    # Função para criar os gráficos
    def create_plot(data, x_col, hue_col=None, plot_type='count', title='', xlabel='', ylabel='', tick_labels=None, rotation=0, kde=False, bins=30):
        fig, ax = plt.subplots(figsize=(10, 8))  # Ajuste o tamanho conforme necessário

        if plot_type == 'count':
            num_bars = data[x_col].nunique()
            dodge = False if num_bars == 2 else True
            
            sns.countplot(data=data, x=x_col, hue=hue_col, palette=color_palette[:2], dodge=dodge, ax=ax)
            ax.grid(False)
            ax.yaxis.set_visible(False)

            total = len(data)
            for p in ax.patches:
                percentage = 100 * p.get_height() / total
                if percentage > 0:
                    ax.annotate(f'{percentage:.1f}%', 
                                (p.get_x() + p.get_width() / 2, p.get_height() + 0.5),
                                ha='center', va='bottom', xytext=(0, 3), textcoords='offset points')

            if hue_col:
                handles, _ = ax.get_legend_handles_labels()
                ax.legend(handles, ['Male', 'Female'], title=hue_col.capitalize(), loc='upper right')
                
            # Ajustar os rótulos dos ticks para gráficos com duas barras
            if hue_col is None and num_bars == 2:
                ax.set_xticks([0, 1])
                ax.set_xticklabels(['Male', 'Female'], ha='center')

        elif plot_type == 'hist':
            sns.histplot(data=data, x=x_col, hue=hue_col, multiple='stack', kde=kde, bins=bins, palette=color_palette[:2], ax=ax)
            ax.legend(title=hue_col.capitalize(), labels=['Female', 'Male'])

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if tick_labels is not None:
            ax.set_xticks(np.arange(len(tick_labels)))
            ax.set_xticklabels(tick_labels, ha='center', rotation=rotation, fontweight='bold')

        # Ajustar o layout para evitar cortes
        plt.tight_layout()

        # Salvar a figura em um buffer para exibir no Streamlit
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')  # Ajuste o bbox_inches para evitar cortes
        buf.seek(0)
        plt.close(fig)
        
        return buf

    # Definir os gráficos disponíveis e seus textos descritivos
    graphs = {
        'Gender distribution': {
            'plot_func': lambda: create_plot(df_concatenado, 'GENDER', hue_col=None, plot_type='count', title='Gender distribution'),
            'description': 'The participants were evenly divided between genders, ensuring balanced representation in the study.'
        },
        'Age distribution': {
            'plot_func': lambda: create_plot(df_concatenado, 'AGE', hue_col='GENDER', plot_type='hist', title='Age distribution', xlabel='AGE', kde=True),
            'description': 'The participants were evenly divided between age, ensuring balanced representation in the study. Participants aged 80 years or older were categorized as being 80 years old. '
        },
        'Race distribution': {
            'plot_func': lambda: create_plot(df_concatenado, 'RACE', hue_col='GENDER', plot_type='count', title='Race distribution', tick_labels=['Mexican American', 'Other Hispanic', 'Non-Hispanic White', 'Non-Hispanic Black', 'Non-Hispanic Asian', 'Other Race - Including Multi-Racial'], rotation=45),
            'description': 'The race distribution in the dataset mirrors the demographic composition of the U.S. population.'
        },
        'Education distribution': {
            'plot_func': lambda: create_plot(df_concatenado, 'EDUCATION', hue_col='GENDER', plot_type='count', title='Education distribution', tick_labels=['Less than 9th grade', '9-11th grade', 'High school graduate/GED or equivalent', 'Some college or AA degree', 'College graduate or above'], rotation=45),
            'description': 'The education distribution in the dataset mirrors the educational composition of the U.S. population.'
        },
        'Poverty Ratio': {
            'plot_func': lambda: create_plot(df_concatenado, 'POVERT_RATIO', hue_col='GENDER', plot_type='hist', title='Poverty Ratio', xlabel='Poverty Ratio', kde=True),
            'description': 'The poverty ratio in the dataset reflects the socioeconomic status of participants by indicating their income relative to the federal poverty level. Values in the dataset are categorized to represent different levels of poverty. To simplify the analysis, any poverty ratio greater than 5 was standardized to a value of 5. This approach ensures consistency in data representation and facilitates clearer interpretation of socioeconomic status across the dataset.'
        }
    }

    # Interface do Streamlit
    st.title('Interactive Demographic Analysis')
    st.write('Select the charts you want to view:')

    # Caixa de seleção múltipla para escolher os gráficos
    selected_graphs = st.multiselect('Choose the charts:', options=list(graphs.keys()))

    # Exibir gráficos e textos associados
    for graph in selected_graphs:
        st.write(f"### {graph}")
        img_buf = graphs[graph]['plot_func']()
        
        # Exibir imagem centralizada
        st.markdown(
            f"<div style='display: flex; flex-direction: column; align-items: center;'><img src='data:image/png;base64,{base64.b64encode(img_buf.getvalue()).decode()}' width='800'/></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='text-align: center; margin-top: 10px;'>{graphs[graph]['description']}</div>",
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    run()

