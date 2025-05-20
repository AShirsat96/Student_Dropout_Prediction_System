import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

@st.cache_data
def load_data(data_path):
    return pd.read_csv(data_path, encoding='utf-8')

class StudentDropoutPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.features = None
        self.best_features = None
    
    def engineer_features(self, data):
        """Engineer features consistently for both training and prediction"""
        df = data.copy()
        
        # Calculate success ratios
        df['first_sem_success_ratio'] = (
            df['Curricular units 1st sem (approved)'] / 
            df['Curricular units 1st sem (enrolled)'].replace(0, 1)
        )
        
        df['second_sem_success_ratio'] = (
            df['Curricular units 2nd sem (approved)'] / 
            df['Curricular units 2nd sem (enrolled)'].replace(0, 1)
        )
        
        # Calculate grades and changes
        df['average_grade'] = df['Curricular units 1st sem (grade)'].fillna(0) + df['Curricular units 2nd sem (grade)'].fillna(0)
        df['performance_change'] = df['Curricular units 2nd sem (grade)'].fillna(0) - df['Curricular units 1st sem (grade)'].fillna(0)
        
        # Calculate economic factor
        df['economic_factor'] = df['Unemployment rate'] * (1 - df['Scholarship holder']) * (1 - df['Tuition fees up to date'])
        
        return df
    
    def prepare_features(self, df):
        """Prepare feature set"""
        self.features = [
            'Age at enrollment',
            'Previous qualification (grade)',
            'Admission grade',
            'first_sem_success_ratio',
            'second_sem_success_ratio',
            'average_grade',
            'performance_change',
            'economic_factor',
            'Scholarship holder',
            'Tuition fees up to date'
        ]
        return df[self.features]
    
    def fit(self, X, y):
        """Fit the model with scaling"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.best_features = self.features
        return self
    
    def predict(self, X):
        """Make predictions with scaling"""
        X_scaled = self.scaler.transform(X[self.best_features])
        return self.model.predict(X_scaled), self.model.predict_proba(X_scaled)

class DataPreprocessor:
    def __init__(self):
        # Application mode mapping
        self.application_mode_mapping = {
            1: '1st phase - general',
            2: 'Ordinance 612/93',
            5: '1st phase - Azores',
            7: 'Other higher courses',
            10: 'Ordinance 854-B/99',
            15: 'Intl. student (bachelor)',
            16: '1st phase - Madeira',
            17: '2nd phase - general',
            18: '3rd phase - general',
            26: 'Ordinance 533-A/99 (Plan)',
            27: 'Ordinance 533-A/99 (Institution)',
            39: 'Over 23 years old',
            42: 'Transfer',
            43: 'Change of course',
            44: 'Technological diploma',
            51: 'Change institution/course',
            53: 'Short cycle diploma',
            57: 'Change institution (Intl.)'
        }
        
        # Marital status mapping
        self.marital_status_mapping = {
            1: 'Single',
            2: 'Married',
            3: 'Widower',
            4: 'Divorced',
            5: 'Facto union',
            6: 'Legally separated'
        }
        
        # Course mappings
        self.course_mapping = {
            33: 'Biofuel Production Tech',
            171: 'Animation & Multimedia',
            8014: 'Social Service (evening)',
            9003: 'Agronomy',
            9070: 'Comm. Design',
            9085: 'Veterinary Nursing',
            9119: 'Informatics Eng.',
            9130: 'Equinculture',
            9147: 'Mgmt',
            9238: 'Social Service',
            9254: 'Tourism',
            9500: 'Nursing',
            9556: 'Oral Hygiene',
            9670: 'Advertising & Marketing',
            9773: 'Journalism & Comm.',
            9853: 'Basic Education',
            9991: 'Mgmt (evening)'
        }
        
        self.course_numeric_assignment = {
            33: 1, 171: 2, 8014: 3, 9003: 1, 9070: 2, 9085: 3,
            9119: 2, 9130: 1, 9147: 4, 9238: 3, 9254: 4, 9500: 3,
            9556: 3, 9670: 4, 9773: 5, 9853: 6, 9991: 4
        }
        
        self.course_condensed_mapping = {
            1: 'Agriculture',
            2: 'Tech',
            3: 'Health',
            4: 'Business',
            5: 'Journalism',
            6: 'Education'
        }
    
    def apply_mappings(self, df):
        """Apply all mappings to the dataframe"""
        df = df.copy()
        
        # Apply basic mappings
        df['Application mode'] = df['Application mode'].map(self.application_mode_mapping)
        df['Marital status'] = df['Marital status'].map(self.marital_status_mapping)
        df['Course_Name'] = df['Course'].map(self.course_mapping)
        
        # Apply course categorization
        df['Course_Category'] = df['Course'].map(self.course_numeric_assignment)
        df['Course_Category'] = df['Course_Category'].map(self.course_condensed_mapping)
        
        # Fill missing values
        categorical_columns = [
            'Application mode', 'Marital status', 'Course_Name', 
            'Course_Category'
        ]
        
        for col in categorical_columns:
            df[col] = df[col].fillna('Unknown')
        
        return df
    
    def get_feature_groups(self):
        """Return groups of features for analysis"""
        return {
            'Demographic Features': [
                'Marital status',
                'Age at enrollment',
                'International',
                'Displaced'
            ],
            'Application Features': [
                'Application mode',
                'Course_Category',
                'Course_Name',
                'Previous qualification (grade)',
                'Admission grade'
            ],
            'Academic Performance': [
                'Curricular units 1st sem (enrolled)',
                'Curricular units 1st sem (evaluations)',
                'Curricular units 1st sem (approved)',
                'Curricular units 1st sem (grade)',
                'Curricular units 2nd sem (enrolled)',
                'Curricular units 2nd sem (evaluations)',
                'Curricular units 2nd sem (approved)',
                'Curricular units 2nd sem (grade)',
                'first_sem_success_ratio',
                'second_sem_success_ratio',
                'average_grade',
                'performance_change'
            ],
            'Economic Factors': [
                'Scholarship holder',
                'Tuition fees up to date',
                'Debtor',
                'Unemployment rate',
                'Inflation rate',
                'GDP',
                'economic_factor'
            ]
        }

def plot_category_proportion(df, category_col, target_col):
    """Create a proportion plot for categorical variables"""
    counts = pd.crosstab(df[category_col], df[target_col], normalize='index') * 100
    counts = counts.reset_index()
    
    fig = go.Figure()
    
    for target in df[target_col].unique():
        fig.add_trace(go.Bar(
            name=target,
            x=counts[target],
            y=counts[category_col],
            orientation='h'
        ))
    
    fig.update_layout(
        title=f'{category_col} Distribution by {target_col}',
        barmode='stack',
        height=max(400, len(counts) * 30),
        margin=dict(l=200, r=50),
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def create_categorical_analysis_plots(df, preprocessor):
    """Create categorical analysis plots"""
    st.header("Categorical Analysis")
    
    feature_groups = preprocessor.get_feature_groups()
    
    for group_name, features in feature_groups.items():
        st.subheader(f"{group_name}")
        
        categorical_features = [f for f in features 
                              if f in df.columns and 
                              (df[f].dtype == 'object' or df[f].nunique() < 10)]
        
        if categorical_features:
            selected_feature = st.selectbox(
                f"Select {group_name} Feature",
                categorical_features,
                key=f"select_{group_name}"
            )
            
            fig = plot_category_proportion(df, selected_feature, 'Target')
            st.plotly_chart(fig, use_container_width=True)

def train_model(df):
    """Train the model with interface"""
    st.header("Model Training")
    
    # Create predictor instance
    predictor = StudentDropoutPredictor()
    
    # Engineer features
    df_processed = predictor.engineer_features(df)
    
    # Prepare features and target
    X = predictor.prepare_features(df_processed)
    y = df['Target']
    
    # Model parameters selection
    st.subheader("Model Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        n_estimators = st.slider("Number of trees", 100, 500, 200, 50)
        max_depth = st.slider("Maximum depth", 5, 30, 10, 1)
        
    with col2:
        min_samples_split = st.slider("Minimum samples to split", 2, 20, 5, 1)
        test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
    
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            # Update model parameters
            predictor.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            )
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Fit model
            predictor.fit(X_train, y_train)
            
            # Save model
            joblib.dump(predictor, 'trained_model.joblib')
            
            # Make predictions
            y_pred_train, _ = predictor.predict(X_train)
            y_pred_test, _ = predictor.predict(X_test)
            
            # Show results
            st.success("Model trained successfully!")
            
            # Model performance
            st.subheader("Model Performance")
            
            # Training metrics
            st.write("Training Set Performance:")
            st.code(classification_report(y_train, y_pred_train))
            
            # Test metrics
            st.write("Test Set Performance:")
            st.code(classification_report(y_test, y_pred_test))
            
            # Feature importance
            importance_df = pd.DataFrame({
                'Feature': predictor.best_features,
                'Importance': predictor.model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Feature Importance')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred_test)
            fig = px.imshow(cm,
                           labels=dict(x="Predicted", y="Actual"),
                           x=['Not Dropout', 'Dropout'],
                           y=['Not Dropout', 'Dropout'],
                           title="Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)

def train_logistic_model(df):
    """Train logistic regression model with interface"""
    st.header("Logistic Regression Model Training")
    
    # Create predictor instance
    predictor = StudentDropoutPredictor()
    
    # Engineer features
    df_processed = predictor.engineer_features(df)
    
    # Prepare features and target
    X = predictor.prepare_features(df_processed)
    y = (df['Target'] == 'Dropout').astype(int)  # Convert to binary
    
    # Model parameters selection
    st.subheader("Model Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        C = st.select_slider(
            "Inverse of regularization strength (C)",
            options=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            value=1.0,
            help="Smaller values specify stronger regularization"
        )
        solver = st.selectbox(
            "Solver",
            options=['lbfgs', 'liblinear', 'newton-cg', 'sag'],
            help="Algorithm to use in optimization"
        )
        
    with col2:
        max_iter = st.slider(
            "Maximum iterations",
            min_value=100,
            max_value=1000,
            value=200,
            step=100,
            help="Maximum number of iterations for solver"
        )
        test_size = st.slider(
            "Test set size",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05
        )
    
    if st.button("Train Logistic Regression Model"):
        with st.spinner("Training model..."):
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Train model
            model = LogisticRegression(
                C=C,
                solver=solver,
                max_iter=max_iter,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Save model info
            model_info = {
                'model': model,
                'scaler': scaler,
                'features': predictor.features,
                'model_type': 'logistic'
            }
            joblib.dump(model_info, 'trained_logistic_model.joblib')
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            y_pred_proba_test = model.predict_proba(X_test)
            
            # Show results
            st.success("Model trained successfully!")
            
            # Model performance
            st.subheader("Model Performance")
            
            # Training metrics
            st.write("Training Set Performance:")
            st.code(classification_report(y_train, y_pred_train))
            
            # Test metrics
            st.write("Test Set Performance:")
            st.code(classification_report(y_test, y_pred_test))
            
            # Feature importance (coefficients)
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': predictor.features,
                'Coefficient': model.coef_[0]
            }).sort_values('Coefficient', ascending=True)
            
            fig = px.bar(importance_df,
                        x='Coefficient',
                        y='Feature',
                        orientation='h',
                        title='Feature Coefficients')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # ROC Curve
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test[:, 1])
            auc_score = auc(fpr, tpr)
            
            fig = px.line(
                x=fpr, y=tpr,
                labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
                title=f'ROC Curve (AUC = {auc_score:.3f})'
            )
            fig.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred_test)
            fig = px.imshow(cm,
                           labels=dict(x="Predicted", y="Actual"),
                           x=['Not Dropout', 'Dropout'],
                           y=['Not Dropout', 'Dropout'],
                           text=cm,
                           aspect="auto",
                           title="Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)


def make_prediction(df, model_type='rf'):
    """Make predictions with interface for both Random Forest and Logistic Regression"""
    st.subheader("Student Information")
    
    try:
        # Load appropriate model
        if model_type == 'rf':
            model_info = joblib.load('trained_model.joblib')
            model_name = "Random Forest"
        else:
            model_info = joblib.load('trained_logistic_model.joblib')
            model_name = "Logistic Regression"
        
        # Get model components
        model = model_info['model']
        scaler = model_info['scaler']
        features = model_info['features']
        
        # Create input form
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age at enrollment", 15, 70, 20)
            prev_grade = st.number_input("Previous qualification grade", 0.0, 200.0, 120.0)
            admission_grade = st.number_input("Admission grade", 0.0, 200.0, 120.0)
            units_1st = st.number_input("Units enrolled (1st sem)", 0, 20, 6)
            units_1st_approved = st.number_input("Units approved (1st sem)", 0, 20, 5)
        
        with col2:
            grade_1st = st.number_input("Average grade (1st sem)", 0.0, 20.0, 12.0)
            units_2nd = st.number_input("Units enrolled (2nd sem)", 0, 20, 6)
            units_2nd_approved = st.number_input("Units approved (2nd sem)", 0, 20, 5)
            grade_2nd = st.number_input("Average grade (2nd sem)", 0.0, 20.0, 12.0)
        
        with col3:
            scholarship = st.selectbox("Scholarship holder", ['No', 'Yes'])
            tuition = st.selectbox("Tuition fees up to date", ['No', 'Yes'])
            unemployment = st.number_input("Unemployment rate", 0.0, 30.0, 10.0)
            international = st.selectbox("International student", ['No', 'Yes'])
        
        if st.button(f"Predict using {model_name}"):
            # Prepare input data
            input_data = pd.DataFrame({
                'Age at enrollment': [age],
                'Previous qualification (grade)': [prev_grade],
                'Admission grade': [admission_grade],
                'Curricular units 1st sem (enrolled)': [units_1st],
                'Curricular units 1st sem (approved)': [units_1st_approved],
                'Curricular units 1st sem (grade)': [grade_1st],
                'Curricular units 2nd sem (enrolled)': [units_2nd],
                'Curricular units 2nd sem (approved)': [units_2nd_approved],
                'Curricular units 2nd sem (grade)': [grade_2nd],
                'Scholarship holder': [1 if scholarship == 'Yes' else 0],
                'Tuition fees up to date': [1 if tuition == 'Yes' else 0],
                'International': [1 if international == 'Yes' else 0],
                'Unemployment rate': [unemployment]
            })
            
            # Engineer features using the same process as training
            predictor = StudentDropoutPredictor()
            input_processed = predictor.engineer_features(input_data)
            
            # Extract required features in the correct order
            X = input_processed[features]
            
            # Scale the features
            X_scaled = scaler.transform(X)
            
            # Make prediction
            probability = model.predict_proba(X_scaled)[0]
            prediction = model.predict(X_scaled)[0]
            
            # Show results
            st.subheader("Prediction Results")
            
            # Create columns for results
            col1, col2 = st.columns(2)
            
            with col1:
                if model_type == 'rf':
                    result = "Dropout Risk" if prediction == 'Dropout' else "Likely to Graduate"
                else:
                    result = "Dropout Risk" if prediction == 1 else "Likely to Graduate"
                    
                color = "red" if "Dropout" in result else "green"
                st.markdown(f"**Prediction:** <span style='color:{color}'>{result}</span>", 
                          unsafe_allow_html=True)
            
            with col2:
                if model_type == 'rf':
                    dropout_prob = probability[1] if "Dropout" in result else probability[0]
                else:
                    dropout_prob = probability[1]
                st.markdown(f"**Dropout Probability:** {dropout_prob:.1%}")
            
            # Visualization of probability
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=['Graduate', 'Dropout'],
                y=[probability[0], probability[1]],
                marker_color=['blue', 'red']
            ))
            
            fig.update_layout(
                title=f'Prediction Probabilities ({model_name})',
                yaxis_title='Probability',
                yaxis_range=[0, 1]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance analysis
            if "Dropout" in result:
                st.subheader("Key Risk Factors")
                
                if model_type == 'rf':
                    # For Random Forest, use feature importances
                    importance = pd.DataFrame({
                        'Feature': features,
                        'Value': input_processed[features].iloc[0],
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                else:
                    # For Logistic Regression, use coefficients
                    importance = pd.DataFrame({
                        'Feature': features,
                        'Value': input_processed[features].iloc[0],
                        'Coefficient': model.coef_[0]
                    }).sort_values('Coefficient', ascending=False)
                    
                    # Add impact column (coefficient * value)
                    importance['Impact'] = importance['Coefficient'] * importance['Value']
                    importance = importance.sort_values('Impact', ascending=False)
                
                # Display feature importance table
                if model_type == 'rf':
                    st.dataframe(importance[['Feature', 'Value', 'Importance']])
                else:
                    st.dataframe(importance[['Feature', 'Value', 'Coefficient', 'Impact']])
                
                # Feature impact visualization
                if model_type == 'logistic':
                    st.subheader("Feature Impact Analysis")
                    fig = px.bar(
                        importance,
                        x='Feature',
                        y='Impact',
                        title='Feature Impact on Dropout Prediction',
                        color='Impact',
                        color_continuous_scale=['blue', 'red']
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.info("Please ensure all inputs are valid and try again.")
def format_sample_data(df):
    """Format sample data by converting numerical categories to labels"""
    df_display = df.copy()
    
    # Binary columns
    binary_columns = [
        'Displaced', 'Educational special needs',
        'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder',
        'International'
    ]
    
    binary_map = {
        0: 'No',
        1: 'Yes'
    }
    
    gender_map = {
        0: 'Female',
        1: 'Male'
    }
    
    # Course mapping
    course_map = {
        33: 'Biofuel Production Tech',
        171: 'Animation & Multimedia',
        8014: 'Social Service (evening)',
        9003: 'Agronomy',
        9070: 'Comm. Design',
        9085: 'Veterinary Nursing',
        9119: 'Informatics Eng.',
        9130: 'Equinculture',
        9147: 'Mgmt',
        9238: 'Social Service',
        9254: 'Tourism',
        9500: 'Nursing',
        9556: 'Oral Hygiene',
        9670: 'Advertising & Marketing',
        9773: 'Journalism & Comm.',
        9853: 'Basic Education',
        9991: 'Mgmt (evening)'
    }
    
    # Previous qualification mapping
    prev_qual_map = {
        1: 'Secondary Education',
        2: "Bachelor's Degree",
        3: 'Degree',
        4: "Master's",
        5: 'Doctorate',
        6: 'Higher Education Frequency',
        9: '12th Year (Not Completed)',
        10: '11th Year (Not Completed)',
        12: 'Other - 11th Year',
        14: '10th Year',
        15: '10th Year (Not Completed)',
        19: 'Basic Education 3rd Cycle',
        38: 'Basic Education 2nd Cycle',
        39: 'Tech Specialization',
        40: 'Degree (1st Cycle)',
        42: 'Professional Tech',
        43: 'Master (2nd Cycle)'
    }
    
    try:
        # Convert numeric columns and apply mappings
        df_display['Course'] = pd.to_numeric(df_display['Course'], errors='coerce')
        df_display['Previous qualification'] = pd.to_numeric(df_display['Previous qualification'], errors='coerce')
        
        # Apply mappings
        df_display['Course'] = df_display['Course'].map(course_map)
        df_display['Previous qualification'] = df_display['Previous qualification'].map(prev_qual_map)
        
        # Apply binary mappings
        for col in binary_columns:
            if col in df_display.columns:
                df_display[col] = pd.to_numeric(df_display[col], errors='coerce')
                if col == 'Gender':
                    df_display[col] = df_display[col].map(gender_map)
                else:
                    df_display[col] = df_display[col].map(binary_map)
    
    except Exception as e:
        print(f"Error in mapping: {e}")
    
    # Select and reorder columns for display
    display_columns = [
        'Course', 
        'Previous qualification',
        'Gender', 
        'Age at enrollment', 
        'International',
        'Scholarship holder', 
        'Tuition fees up to date', 
        'Displaced',
        'Educational special needs', 
        'Debtor', 
        'Target'
    ]
    
    # Only include columns that exist in the dataframe
    display_columns = [col for col in display_columns if col in df_display.columns]
    df_display = df_display[display_columns]
    
    return df_display

def main():
    st.set_page_config(page_title="Student Dropout Prediction", layout="wide")
    
    st.title("🎓 Student Dropout Prediction System")
    st.markdown("""
    This application analyzes and predicts student dropout patterns using various academic and demographic factors.
    """)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", 
                           ["Data Overview",
                            "Categorical Analysis",
                            "Random Forest Model",
                            "Logistic Regression Model",
                            "Make Prediction"])
    
    try:
        # Load and preprocess data
        df = load_data("C:\Fall_Semester_2024\Intro_DataScience\Academic_Success_Data.csv")
        df = df[df['Target'].isin(['Dropout', 'Graduate'])]
        df_processed = preprocessor.apply_mappings(df)
        
        if page == "Data Overview":
            st.header("Dataset Overview")
            
            # Basic statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Students", len(df))
            with col2:
                st.metric("Dropout Rate", f"{(df['Target'] == 'Dropout').mean():.1%}")
            with col3:
                st.metric("Graduate Rate", f"{(df['Target'] == 'Graduate').mean():.1%}")
            
            # Create two columns for the pie chart and sample data
            col_left, col_right = st.columns([1, 2])
            
            with col_left:
                # Data Distribution Pie Chart
                st.subheader("Data Distribution")
                target_counts = df['Target'].value_counts()
                fig = px.pie(
                    values=target_counts.values,
                    names=target_counts.index,
                    title="Distribution of Dropout vs Graduate",
                    color_discrete_map={'Dropout': '#FF6B6B', 'Graduate': '#4CAF50'}
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(
                    showlegend=False,
                    margin=dict(t=40, b=0, l=0, r=0)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_right:
                # Show sample data with categorical labels
                st.subheader("Sample Data")
                df_display = format_sample_data(df_processed)
                st.dataframe(df_display.head())
        
        elif page == "Categorical Analysis":
            create_categorical_analysis_plots(df_processed, preprocessor)
        
        elif page == "Random Forest Model":
            st.header("Random Forest Model")
            train_model(df_processed)
        
        elif page == "Logistic Regression Model":
            train_logistic_model(df_processed)
        
        elif page == "Make Prediction":
            st.header("Make Predictions")
            
            # Model selection
            model_type = st.radio(
                "Select Model for Prediction",
                ["Random Forest", "Logistic Regression"],
                horizontal=True
            )
            
            # Check for model files
            rf_model_exists = os.path.exists('trained_model.joblib')
            log_model_exists = os.path.exists('trained_logistic_model.joblib')
            
            if model_type == "Random Forest" and not rf_model_exists:
                st.warning("No trained Random Forest model found. Please train the model first.")
                if st.button("Go to Random Forest Training"):
                    st.session_state.page = "Random Forest Model"
                return
                
            if model_type == "Logistic Regression" and not log_model_exists:
                st.warning("No trained Logistic Regression model found. Please train the model first.")
                if st.button("Go to Logistic Regression Training"):
                    st.session_state.page = "Logistic Regression Model"
                return
            
            # Load appropriate model and make prediction
            if model_type == "Random Forest":
                make_prediction(df_processed, model_type='rf')
            else:
                make_prediction(df_processed, model_type='logistic')
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please check the data file path and format.")

if __name__ == "__main__":
    main()