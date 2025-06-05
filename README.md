# Student Dropout Prediction System

A machine learning-powered web application built with Streamlit to predict student dropout risks and academic success in higher education institutions. This system helps educational administrators and counselors identify at-risk students early and implement timely interventions to improve retention rates.

## Overview

Student dropout is a critical issue affecting educational institutions worldwide, impacting not only individual students but also families, institutions, and society as a whole. This project leverages machine learning classification algorithms to analyze student data and predict the likelihood of dropout, enabling proactive support and intervention strategies.

## Features

- **Interactive Web Interface**: User-friendly Streamlit dashboard for easy data input and visualization
- **Real-time Predictions**: Instant dropout risk assessment based on student characteristics
- **Multiple ML Models**: Implementation of various classification algorithms for optimal accuracy
- **Data Visualization**: Comprehensive charts and graphs to understand prediction factors
- **Batch Processing**: Ability to analyze multiple student records simultaneously
- **Risk Assessment**: Detailed probability scores and risk categorization
- **Export Functionality**: Download prediction results and reports

## Technology Stack

- **Frontend**: Streamlit 2.3+
- **Backend**: Python 3.7+
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Data Processing**: Pandas, NumPy

## Dataset Features

The system analyzes various student attributes including:

### Demographic Information
- Age at enrollment
- Gender
- Marital status
- Nationality

### Academic Background
- Previous qualification grades
- Application mode
- Course selection
- Academic performance (1st and 2nd semester)

### Socio-Economic Factors
- Parents' education level
- Parents' occupation
- Tuition fees status
- Scholarship status

### Macro-Economic Indicators
- Unemployment rate
- Inflation rate
- GDP growth

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/AShirsat96/Student_Dropout_Prediction_System.git
   cd Student_Dropout_Prediction_System
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run Streamlit_app_V2.3.py
   ```

5. **Access the application**
   Open your browser and navigate to `http://localhost:8501`

## Usage

### Single Student Prediction
1. Navigate to the "Single Prediction" tab
2. Fill in the student information form
3. Click "Predict" to get the dropout risk assessment
4. View the probability scores and risk category

### Batch Processing
1. Go to the "Batch Prediction" tab
2. Upload a CSV file with student data
3. Download the results with predictions for all students

### Data Analysis
1. Explore the "Data Analysis" section
2. View statistical insights and feature importance
3. Understand key factors influencing dropout risk

## Machine Learning Models

The system implements and compares multiple classification algorithms:

- **Random Forest Classifier**
- **Support Vector Machine (SVM)**
- **Logistic Regression**
- **Gradient Boosting**
- **Neural Networks**

Model performance is evaluated using:
- Accuracy
- Precision and Recall
- F1-Score
- ROC-AUC Score

## Model Performance

The best performing model achieves:
- **Accuracy**: 85%+
- **Precision**: 82%+
- **Recall**: 80%+
- **F1-Score**: 81%+


## Key Insights

Based on the analysis, the most significant factors influencing dropout risk include:

1. **Academic Performance**: 1st semester grades are highly predictive
2. **Financial Status**: Tuition payment delays increase dropout risk
3. **Previous Education**: Prior qualification grades matter significantly
4. **Economic Factors**: Unemployment rates in the region affect retention

## Business Impact

This system enables educational institutions to:

- **Reduce Dropout Rates**: Early identification allows for timely interventions
- **Optimize Resources**: Focus support efforts on high-risk students
- **Improve Student Success**: Implement targeted academic and financial support
- **Data-Driven Decisions**: Make informed policy and program decisions

## Recommended Actions

For institutions using this system:

1. **Academic Support**: Provide tutoring for students with low 1st semester performance
2. **Financial Aid**: Offer flexible payment plans and emergency financial assistance
3. **Mentorship Programs**: Assign advisors to high-risk students
4. **Early Warning System**: Implement regular monitoring and check-ins


**Note**: This system is designed to assist educational institutions in identifying at-risk students. It should be used as a supplementary tool alongside human judgment and institutional knowledge for making decisions about student support and interventions.
