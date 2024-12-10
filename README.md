# Student Dropout Prediction System

## Overview
A machine learning-based web application that predicts and analyzes student dropout patterns in higher education using XGBoost. The system identifies students at risk of dropping out, enabling educational institutions to implement timely interventions.

Below is the link to our streamlit application:
https://student-graduation-prediction.streamlit.app/
 

### Key Stakeholder Benefits
- **Academic Administrators**: Early identification of at-risk students for targeted support
- **Student Support Services**: Data-driven insights for resource allocation
- **Faculty Members**: Understanding key factors influencing student retention
- **Financial Planning**: Better prediction of enrollment patterns and resource needs

## Data Description
The system analyzes student data including:
- Demographic information (age, gender, international status)
- Academic performance (semester grades, units enrolled/approved)
- Economic indicators (scholarship status, tuition payment history)
- Institutional factors (course type, admission details)

Key features engineered from raw data:
- Semester success ratios
- Performance change metrics
- Economic factor index combining unemployment rate and financial indicators

## Algorithm Description
The system implements an XGBoost classifier optimized for dropout prediction:

### XGBoost Model Features
- Gradient boosting for complex pattern recognition
- Tree-based feature importance analysis
- Hyperparameter optimization for:
  - Number of estimators (100-500)
  - Learning rate
  - Maximum depth (5-30)
  - Minimum child weight
- Cross-validation for model validation
- Feature importance ranking for interpretability

## Tools Used
- **Streamlit**: Web application framework and user interface
- **Pandas/NumPy**: Data manipulation and numerical operations
- **XGBoost**: Machine learning model implementation
- **Plotly**: Interactive data visualization
- **Joblib**: Model persistence and loading
- **SciPy**: Statistical analysis and testing

## Ethical Concerns and Mitigation

### Data Privacy and Security
- **Risk**: Student personal and academic information exposure
- **Mitigation**: 
  - Implement data anonymization
  - Restrict access to authorized personnel
  - Secure data storage and transmission

### Bias and Fairness
- **Risk**: Model bias against certain demographic groups
- **Mitigation**:
  - Regular model auditing for demographic parity
  - Balanced training data representation
  - Cross-validation to ensure model stability

### Intervention Impact
- **Risk**: Self-fulfilling prophecies from predictions
- **Mitigation**:
  - Focus on support rather than labels
  - Regular model retraining with new data
  - Clear communication about prediction limitations

### Economic Impact
- **Risk**: Resource allocation bias based on predictions
- **Mitigation**:
  - Transparent decision-making processes
  - Regular review of support program effectiveness
  - Equal access to academic support resources
