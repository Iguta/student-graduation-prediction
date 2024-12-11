from robot.api.deco import keyword, library
from utils.models.logistic_regression import LogisticRegressionModel

model = LogisticRegressionModel()
    
@keyword('Create Model Instance')
def create_model():
    return model
    
@keyword('Prepare Data')
def prepare_data():
    return model.prepare_data()
    
@keyword('Split And Scale Data')
def split_and_scale_data(self, X, y):
    return model.split_and_scale_data(X, y)
    
@keyword('Fit Model')
def fit_model(self, X, y):
    return model.fit(X, y)
    
@keyword('Predict')
def predict(X):
    return model.predict(X)
    
@keyword('Evaluate Model')
def evaluate_model(X, y):
    return model.evaluate(X, y)
    
@keyword('Save Model')
def save_model(self, model_path="model.pkl", scaler_path="scaler.pkl"):
    self.model.save(model_path, scaler_path)
    
@keyword('Load Model')
def load_model(self, model_path="model.pkl", scaler_path="scaler.pkl"):
    return LogisticRegressionModel.load(model_path, scaler_path)