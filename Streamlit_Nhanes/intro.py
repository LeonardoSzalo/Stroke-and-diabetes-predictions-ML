import streamlit as st

def run():

    # Function to create a centered header in Streamlit using Markdown
    def centered_header(title):
        st.markdown(f"<h2 style='text-align: center; font-family: Arial, sans-serif;'>{title}</h2>", unsafe_allow_html=True)
    
    # Function to create styled text using Markdown
    def styled_text(text):
        return f"<p style='font-family: Arial, sans-serif;'>{text}</p>"

    # Page title
    centered_header("About NHANES")
    
    # Show sections progressively
    show_section = st.selectbox("Select Section", ["Introduction", "Importance", "Project Overview", "More Information"])
    
    if show_section == "Introduction":
        st.markdown(styled_text(
            """
            What is NHANES?
            
            The National Health and Nutrition Examination Survey (NHANES) is a research program conducted by the
            National Center for Health Statistics (NCHS) in the United States. It is designed to assess the health and nutrition
            of the American population through a combination of physical examinations, health and dietary questionnaires, and
            laboratory tests.

            NHANES collects a wide range of data about the health of participants, including:
            - Demographic data
            - Dietary habits
            - Health conditions
            - Anthropometric measurements (such as height and weight)
            - Laboratory tests
            - Clinical examinations

            The data are collected through personal interviews and physical exams conducted at a mobile examination center.
            """
        ), unsafe_allow_html=True)
    
    elif show_section == "Importance":
        st.markdown(styled_text(
            """
            Importance of Monitoring Assessed Conditions
            
            Monitoring the conditions assessed by NHANES is crucial for several reasons:

            - Public Health Monitoring: It allows for ongoing monitoring of health conditions and nutritional patterns
              in the population, helping to identify trends and changes over time.

            - Identification of Health Issues: It helps identify and understand the prevalence of chronic diseases and
              health conditions such as diabetes, hypertension, and heart disease.

            - Health Policy Formulation: NHANES data provides valuable information for the formulation of public health
              policies and strategies, supporting the development of prevention and treatment programs.

            - Research and Development: The data are used in research to better understand the relationships between diet,
              health conditions, and other factors, aiding in the development of new treatments and interventions.
            """
        ), unsafe_allow_html=True)
    
    elif show_section == "Project Overview":
        st.markdown(styled_text(
            """
            About This Project
            
            This project utilized the NHANES dataset from 2017 to 2020, which provides representative data of the U.S. population.

            This is a learning-driven project aimed at gaining deeper insights into data collection concepts and real-world problems. 
            The goal was to apply exploratory data analysis and data preprocessing techniques, identify important parameters for machine 
            learning models, and predict conditions such as diabetes, stroke, and heart attack.

            The project involved:
            - Exploring the NHANES dataset to understand its structure and content.
            - Preprocessing and cleaning the data to ensure quality and consistency.
            - Identifying key features that could be used in predictive models.
            - Applying machine learning techniques to develop and evaluate models for predicting the mentioned conditions.
            """
        ), unsafe_allow_html=True)
    
    elif show_section == "More Information":
        st.markdown(styled_text(
            """
            More Information
            
            The steps undertaken in this project were:

            1. Data Collection: Gathering relevant data for predicting the mentioned diseases, based on existing literature.
            2. Data Cleaning and Preprocessing: Preparing the data for analysis by handling missing values, outliers, and inconsistencies.
            3. Data Aggregation: Combining the data into a single file for ease of analysis.
            4. Model Training: Training various machine learning models to identify the most effective one.
            5. Hyperparameter Tuning: Optimizing the chosen models to improve recall and F1-score (details available in the GitHub repository).
            
            """
        ), unsafe_allow_html=True)
        st.markdown(
        """
        To view the complete work and analyses step by step, access my GitHub repository: 
        [GitHub Repository](https://github.com/LeonardoSzalo)


         For updates and more information, connect with me on LinkedIn: 
        [LinkedIn Profile](https://www.linkedin.com/in/leonardo-szalo-11a3aa156)
        

        """,
        unsafe_allow_html=True  # Permite a inclusão de HTML e Markdown
    )
