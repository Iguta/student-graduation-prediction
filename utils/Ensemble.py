from sklearn.ensemble import VotingClassifier

def train_ensemble_model(df):
    """Train ensemble model with interface"""
    st.header("Ensemble Model Training (Voting Classifier)")
    
    # Create predictor instance
    predictor = StudentDropoutPredictor()
    
    # Engineer features
    df_processed = predictor.engineer_features(df)
    
    # Prepare features and target
    X = predictor.prepare_features(df_processed)
    y = (df['Target'] == 'Dropout').astype(int)  # Convert to binary
    
    # Model parameters
    test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
    
    if st.button("Train Ensemble Model"):
        with st.spinner("Training model..."):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Define base models
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
            lr = LogisticRegression(max_iter=200, random_state=42)
            
            # Ensemble model
            ensemble = VotingClassifier(estimators=[
                ('rf', rf), ('xgb', xgb), ('lr', lr)
            ], voting='soft')
            
            ensemble.fit(X_train, y_train)
            
            # Save model
            joblib.dump(ensemble, 'trained_ensemble_model.joblib')
            
            # Make predictions
            y_pred_train = ensemble.predict(X_train)
            y_pred_test = ensemble.predict(X_test)
            
            st.success("Model trained successfully!")
            
            st.subheader("Model Performance")
            st.write("Training Set Performance:")
            st.code(classification_report(y_train, y_pred_train))
            
            st.write("Test Set Performance:")
            st.code(classification_report(y_test, y_pred_test))