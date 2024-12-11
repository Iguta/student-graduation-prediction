# Student Dropout Prediction System

## Overview
A machine learning-based web application that predicts and analyzes student dropout patterns in higher education using Logistic Regression. The system identifies students at risk of dropping out, providing educational institutions with interpretable results for implementing timely interventions.

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
The system implements a Logistic Regression model optimized for dropout prediction:

Model Details:

- Binary classification with probability outputs
- Feature standardization using StandardScaler
- Model parameters:
  - Regularization strength (C parameter)
  - Solver selection (lbfgs, liblinear, newton-cg)
  - Maximum iterations


- Model evaluation metrics:
  - ROC curve and AUC score
  - Classification report (precision, recall, F1-score)
  - Confusion matrix


- Interpretable coefficients showing feature importance
- Cross-validation for model validation
  
## Tools Used
- **Streamlit**: Web application framework and user interface
- **Pandas/NumPy**: Data manipulation and numerical operations
- **Scikit-learn**: Logistic Regression implementation and preprocessing
- **Plotly**: Interactive data visualization
- **Pickle**: Model serialization and persistence
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

