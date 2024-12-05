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