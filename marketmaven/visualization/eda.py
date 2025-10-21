import plotly.express as px
import pandas as pd 
import numpy as np

def correlation_matrix(data: pd.DataFrame, title: str = "Correlation Heatmap"):
    
    # Select only continuous variables
    continuous_data = data.select_dtypes(include=[np.number])
    
    
    corr_matrix = continuous_data.corr()
    # Create a heatmap using Plotly
    fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale="RdYlBu",
        title=title,
        zmin=-1,
        zmax=1,
    )

    fig.update_layout(width=600, height=600)
    return fig