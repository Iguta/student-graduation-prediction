from xgboost import XGBClassifier

def train_xgboost_model(df):
    """Train XGBoost model with interface"""
    st.header("XGBoost Model Training")
    
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
        n_estimators = st.slider("Number of trees", 100, 500, 200, 50)
        max_depth = st.slider("Maximum depth", 3, 15, 6, 1)
        
    with col2:
        learning_rate = st.slider("Learning rate", 0.01, 0.3, 0.1, 0.01)
        test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
    
    if st.button("Train XGBoost Model"):
        with st.spinner("Training model..."):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Train model
            model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            
            model.fit(X_train, y_train)
            
            # Save model
            joblib.dump(model, 'trained_xgboost_model.joblib')
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Show results
            st.success("Model trained successfully!")
            
            st.subheader("Model Performance")
            st.write("Training Set Performance:")
            st.code(classification_report(y_train, y_pred_train))
            
            st.write("Test Set Performance:")
            st.code(classification_report(y_test, y_pred_test))
            
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': predictor.features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(importance_df,
                         x='Importance',
                         y='Feature',
                         orientation='h',
                         title='Feature Importance')
            st.plotly_chart(fig, use_container_width=True)