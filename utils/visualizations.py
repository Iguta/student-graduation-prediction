import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

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

def plot_feature_importance(feature_names, importance_scores, title='Feature Importance'):
    """Plot feature importance"""
    fig = px.bar(
        x=importance_scores,
        y=feature_names,
        orientation='h',
        title=title
    )
    return fig

def plot_confusion_matrix(cm, labels=['Not Dropout', 'Dropout']):
    """Plot confusion matrix"""
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual"),
        x=labels,
        y=labels,
        title="Confusion Matrix",
        text=cm
    )
    return fig

def plot_roc_curve(fpr, tpr, auc_score):
    """Plot ROC curve"""
    fig = px.line(
        x=fpr, 
        y=tpr,
        labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
        title=f'ROC Curve (AUC = {auc_score:.3f})'
    )
    fig.add_shape(
        type='line', 
        line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    return fig

def plot_prediction_probabilities(probabilities, model_name):
    """Plot prediction probabilities"""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Graduate', 'Dropout'],
        y=[probabilities[0], probabilities[1]],
        marker_color=['blue', 'red']
    ))
    
    fig.update_layout(
        title=f'Prediction Probabilities ({model_name})',
        yaxis_title='Probability',
        yaxis_range=[0, 1]
    )
    return fig

def plot_feature_impact(impact_df):
    """Plot feature impact analysis"""
    fig = px.bar(
        impact_df,
        x='Feature',
        y='Impact',
        title='Feature Impact on Dropout Prediction',
        color='Impact',
        color_continuous_scale=['blue', 'red']
    )
    return fig