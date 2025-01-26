"""Utility functions for data visualization."""
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

def create_survival_distribution(df: pd.DataFrame, feature: str, title: str) -> go.Figure:
    """Create a bar plot showing survival distribution for a feature."""
    survival_rates = df.groupby(feature)['Survived'].agg(['mean', 'count']).reset_index()
    
    fig = go.Figure([
        go.Bar(
            x=survival_rates[feature],
            y=survival_rates['mean'],
            text=survival_rates['count'],
            textposition='auto',
            name='Survival Rate'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title=feature,
        yaxis_title='Survival Rate',
        showlegend=True
    )
    
    return fig

def create_age_distribution(df: pd.DataFrame) -> go.Figure:
    """Create an age distribution plot by survival status."""
    fig = go.Figure()
    
    for survived in [0, 1]:
        fig.add_trace(go.Histogram(
            x=df[df['Survived'] == survived]['Age'],
            name=f"{'Survived' if survived else 'Did not survive'}",
            opacity=0.7,
            nbinsx=30
        ))
    
    fig.update_layout(
        title='Age Distribution by Survival Status',
        xaxis_title='Age',
        yaxis_title='Count',
        barmode='overlay'
    )
    
    return fig

def create_correlation_heatmap(df: pd.DataFrame, features: List[str]) -> go.Figure:
    """Create a correlation heatmap for specified features."""
    corr_matrix = df[features].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=features,
        y=features,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Feature Correlation Heatmap',
        xaxis_title='Features',
        yaxis_title='Features'
    )
    
    return fig

def create_fare_vs_age_scatter(df: pd.DataFrame) -> go.Figure:
    """Create a scatter plot of Fare vs Age colored by survival status."""
    fig = px.scatter(
        df,
        x='Age',
        y='Fare',
        color='Survived',
        size='FamilySize',
        hover_data=['Pclass', 'Sex', 'FamilySize'],
        title='Fare vs Age by Survival Status'
    )
    
    return fig

def create_survival_probability_map(df: pd.DataFrame) -> go.Figure:
    """Create a survival probability heatmap based on Age and Fare."""
    age_bins = np.linspace(df['Age'].min(), df['Age'].max(), 11)
    fare_bins = np.linspace(df['Fare'].min(), df['Fare'].max(), 11)
    
    df['Age_bin'] = pd.cut(df['Age'], bins=age_bins, labels=age_bins[:-1])
    df['Fare_bin'] = pd.cut(df['Fare'], bins=fare_bins, labels=fare_bins[:-1])
    
    heatmap_data = df.pivot_table(
        values='Survived',
        index='Age_bin',
        columns='Fare_bin',
        aggfunc='mean'
    ).round(2)
    
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Fare ($)", y="Age", color="Survival Probability"),
        aspect="auto",
        title="Survival Probability Map"
    )
    
    fig.update_layout(
        xaxis_title="Fare ($)",
        yaxis_title="Age",
        coloraxis_colorbar_title="Survival<br>Probability"
    )
    
    return fig

def create_3d_scatter(df: pd.DataFrame) -> go.Figure:
    """Create 3D scatter plot of Age, Fare, and Pclass."""
    fig = go.Figure(data=[go.Scatter3d(
        x=df['Age'],
        y=df['Fare'],
        z=df['Pclass'],
        mode='markers',
        marker=dict(
            size=5,
            color=df['Survived'],
            colorscale='Viridis',
            opacity=0.8
        ),
        text=[f"Survived: {s}<br>Age: {a}<br>Fare: ${f:.2f}<br>Class: {p}" 
              for s, a, f, p in zip(df['Survived'], df['Age'], df['Fare'], df['Pclass'])]
    )])
    
    fig.update_layout(
        title='3D Visualization of Survival Factors',
        scene=dict(
            xaxis_title='Age',
            yaxis_title='Fare',
            zaxis_title='Class'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    return fig

def create_feature_importance_plot(feature_importance: Dict[str, float]) -> go.Figure:
    """Create a horizontal bar plot of feature importance."""
    features = list(feature_importance.keys())
    importance = list(feature_importance.values())
    
    fig = go.Figure(go.Bar(
        x=importance,
        y=features,
        orientation='h'
    ))
    
    fig.update_layout(
        title='Feature Importance Analysis',
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        height=400
    )
    
    return fig

def create_confusion_matrix_plot(confusion_matrix: List[List[int]]) -> go.Figure:
    """Create a confusion matrix visualization."""
    z = confusion_matrix
    x = ['Predicted Not Survived', 'Predicted Survived']
    y = ['Actually Not Survived', 'Actually Survived']
    
    fig = ff.create_annotated_heatmap(
        z,
        x=x,
        y=y,
        colorscale='Blues',
        showscale=True
    )
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual'
    )
    
    return fig
