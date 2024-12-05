import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_category_proportion(df, category_col, target_col):
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