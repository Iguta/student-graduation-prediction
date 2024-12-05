
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
        df = data.copy()
        =
        df['first_sem_success_ratio'] = (
            df['Curricular units 1st sem (approved)'] / 
            df['Curricular units 1st sem (enrolled)'].replace(0, 1)
        )

        ## creating 'second_sem_success_ratio' variable based on Curricular units approved for 2nd Sem from Curricular units that the student enrolled in.
        df['second_sem_success_ratio'] = (
            df['Curricular units 2nd sem (approved)'] / 
            df['Curricular units 2nd sem (enrolled)'].replace(0, 1)
        )
        
        df['average_grade'] = df['Curricular units 1st sem (grade)'].fillna(0) + df['Curricular units 2nd sem (grade)'].fillna(0)
        df['performance_change'] = df['Curricular units 2nd sem (grade)'].fillna(0) - df['Curricular units 1st sem (grade)'].fillna(0)

        ## creating 'economic_factor' variable based on Unemployment rate, Scholarship holder status and Tuition fees up to date
        df['economic_factor'] = df['Unemployment rate'] * (1 - df['Scholarship holder']) * (1 - df['Tuition fees up to date'])
        
        return df
    
    def prepare_features(self, df):
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
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.best_features = self.features
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X[self.best_features])
        return self.model.predict(X_scaled), self.model.predict_proba(X_scaled)
