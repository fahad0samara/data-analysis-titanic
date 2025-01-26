import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Set wide page layout and custom theme
st.set_page_config(
    page_title="Titanic Survival Analysis",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .st-emotion-cache-16idsys p {
        font-size: 14px;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('data/raw/titanic.csv')

def create_3d_scatter(df):
    """Create 3D scatter plot"""
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

def create_feature_importance_plot(df):
    """Create feature importance plot using Random Forest"""
    # Prepare data
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    X = df[features].copy()
    y = df['Survived']
    
    # Handle missing values
    X['Age'].fillna(X['Age'].median(), inplace=True)
    X['Fare'].fillna(X['Fare'].median(), inplace=True)
    
    # Encode categorical variables
    le = LabelEncoder()
    X['Sex'] = le.fit_transform(X['Sex'])
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Create importance plot
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=importance_df['Importance'],
        y=importance_df['Feature'],
        orientation='h'
    ))
    
    fig.update_layout(
        title='Feature Importance Analysis',
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        height=400
    )
    return fig

def create_survival_probability_map(df):
    """Create a survival probability heatmap based on Age and Fare"""
    # Create age and fare bins
    age_bins = np.linspace(df['Age'].min(), df['Age'].max(), 11)
    fare_bins = np.linspace(df['Fare'].min(), df['Fare'].max(), 11)
    
    # Cut the data into bins
    df['Age_bin'] = pd.cut(df['Age'], bins=age_bins, labels=age_bins[:-1])
    df['Fare_bin'] = pd.cut(df['Fare'], bins=fare_bins, labels=fare_bins[:-1])
    
    # Create pivot table
    heatmap_data = df.pivot_table(
        values='Survived',
        index='Age_bin',
        columns='Fare_bin',
        aggfunc='mean'
    ).round(2)
    
    # Convert index and columns to float for better visualization
    heatmap_data.index = heatmap_data.index.astype(float)
    heatmap_data.columns = heatmap_data.columns.astype(float)
    
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Fare ($)", y="Age", color="Survival Probability"),
        aspect="auto",
        title="Survival Probability Map"
    )
    
    # Update layout for better readability
    fig.update_layout(
        xaxis_title="Fare ($)",
        yaxis_title="Age",
        coloraxis_colorbar_title="Survival<br>Probability"
    )
    
    return fig

def main():
    st.title("🚢 Advanced Titanic Survival Analysis Dashboard")
    
    try:
        # Load data
        df = load_data()
        
        # Sidebar
        st.sidebar.header("📊 Advanced Filters")
        
        # Advanced filtering options
        filter_container = st.sidebar.container()
        with filter_container:
            # Passenger Class filter with counts
            class_counts = df['Pclass'].value_counts()
            pclass_options = [f"Class {c} ({class_counts[c]} passengers)" for c in sorted(df['Pclass'].unique())]
            selected_pclass = st.multiselect(
                "Passenger Class",
                options=pclass_options,
                default=pclass_options
            )
            selected_pclass = [int(p.split()[1]) for p in selected_pclass]
            
            # Gender filter with counts
            gender_counts = df['Sex'].value_counts()
            gender_options = [f"{g} ({gender_counts[g]} passengers)" for g in sorted(df['Sex'].unique())]
            selected_gender = st.multiselect(
                "Gender",
                options=gender_options,
                default=gender_options
            )
            selected_gender = [g.split()[0] for g in selected_gender]
            
            # Age range filter
            age_range = st.slider(
                "Age Range",
                float(df['Age'].min()),
                float(df['Age'].max()),
                (float(df['Age'].min()), float(df['Age'].max()))
            )
            
            # Fare range filter
            fare_range = st.slider(
                "Fare Range ($)",
                float(df['Fare'].min()),
                float(df['Fare'].max()),
                (float(df['Fare'].min()), float(df['Fare'].max()))
            )
            
            # Family size filter
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            family_size = st.slider(
                "Family Size",
                int(df['FamilySize'].min()),
                int(df['FamilySize'].max()),
                (int(df['FamilySize'].min()), int(df['FamilySize'].max()))
            )
        
        # Apply filters
        mask = (
            (df['Pclass'].isin(selected_pclass)) & 
            (df['Sex'].isin(selected_gender)) &
            (df['Age'].between(age_range[0], age_range[1])) &
            (df['Fare'].between(fare_range[0], fare_range[1])) &
            (df['FamilySize'].between(family_size[0], family_size[1]))
        )
        filtered_df = df[mask].copy()  # Create a copy to avoid SettingWithCopyWarning
        
        # Main content
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Overview",
            "🔍 Advanced Analysis",
            "🎯 ML Insights",
            "📊 Statistics",
            "🌍 3D Visualization"
        ])
        
        # Tab 1: Overview (Enhanced)
        with tab1:
            # Key metrics with context
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Passengers",
                    len(filtered_df),
                    delta=f"{len(filtered_df) - len(df)}" if len(filtered_df) != len(df) else None,
                    help="Number of passengers matching the selected filters"
                )
            
            with col2:
                overall_survival = filtered_df['Survived'].mean()
                st.metric(
                    "Survival Rate",
                    f"{overall_survival:.1%}",
                    delta=f"{(overall_survival - df['Survived'].mean())*100:.1f}%",
                    help="Percentage of passengers who survived"
                )
            
            with col3:
                avg_age = filtered_df['Age'].mean()
                st.metric(
                    "Average Age",
                    f"{avg_age:.1f}",
                    delta=f"{avg_age - df['Age'].mean():.1f}",
                    help="Average age of passengers"
                )
            
            with col4:
                avg_fare = filtered_df['Fare'].mean()
                st.metric(
                    "Average Fare",
                    f"${avg_fare:.2f}",
                    delta=f"${avg_fare - df['Fare'].mean():.2f}",
                    help="Average fare paid by passengers"
                )
            
            # Enhanced visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                survival_by_class = filtered_df.groupby('Pclass')['Survived'].agg(['mean', 'count']).reset_index()
                fig = px.bar(
                    survival_by_class,
                    x='Pclass',
                    y='mean',
                    text='count',
                    title='Survival Rate by Passenger Class',
                    labels={'mean': 'Survival Rate', 'Pclass': 'Passenger Class'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                survival_by_gender = filtered_df.groupby('Sex')['Survived'].agg(['mean', 'count']).reset_index()
                fig = px.bar(
                    survival_by_gender,
                    x='Sex',
                    y='mean',
                    text='count',
                    title='Survival Rate by Gender',
                    labels={'mean': 'Survival Rate', 'Sex': 'Gender'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Tab 2: Advanced Analysis
        with tab2:
            st.header("Advanced Passenger Analysis")
            
            # Survival Probability Map
            st.plotly_chart(create_survival_probability_map(filtered_df), use_container_width=True)
            
            # Enhanced correlation analysis
            st.subheader("Feature Correlations")
            numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']
            corr = filtered_df[numeric_cols].corr()
            
            fig = px.imshow(
                corr,
                labels=dict(color="Correlation"),
                x=numeric_cols,
                y=numeric_cols,
                aspect="auto",
                title="Correlation Matrix"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Tab 3: ML Insights
        with tab3:
            st.header("Machine Learning Insights")
            
            # Feature importance plot
            st.plotly_chart(create_feature_importance_plot(filtered_df), use_container_width=True)
            
            # Cross-validation results
            st.subheader("Model Performance Analysis")
            features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
            X = filtered_df[features].copy()
            y = filtered_df['Survived']
            
            # Handle missing values and encoding
            X['Age'].fillna(X['Age'].median(), inplace=True)
            X['Fare'].fillna(X['Fare'].median(), inplace=True)
            X['Sex'] = LabelEncoder().fit_transform(X['Sex'])
            
            # Perform cross-validation
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            scores = cross_val_score(model, X, y, cv=5)
            
            # Display cross-validation results
            cv_col1, cv_col2 = st.columns(2)
            with cv_col1:
                st.metric("Average CV Score", f"{scores.mean():.2f}")
            with cv_col2:
                st.metric("CV Score Std", f"±{scores.std():.2f}")
        
        # Tab 4: Statistics
        with tab4:
            st.header("Statistical Analysis")
            
            # Summary statistics with explanation
            st.subheader("Summary Statistics")
            summary_stats = filtered_df.describe()
            st.dataframe(summary_stats)
            
            # Advanced survival analysis
            st.subheader("Advanced Survival Analysis")
            
            # Create survival rate pivot table
            pivot = pd.pivot_table(
                filtered_df,
                values='Survived',
                index=['Sex', 'Pclass'],
                aggfunc=['mean', 'count']
            ).round(3)
            
            st.write("Survival Rates by Gender and Class")
            st.dataframe(pivot)
            
            # Download options
            st.subheader("Export Analysis")
            
            # Create Excel file with multiple sheets
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                filtered_df.to_excel(writer, sheet_name='Passenger Data', index=False)
                summary_stats.to_excel(writer, sheet_name='Summary Statistics')
                pivot.to_excel(writer, sheet_name='Survival Analysis')
            
            st.download_button(
                label="📥 Download Complete Analysis",
                data=output.getvalue(),
                file_name="titanic_advanced_analysis.xlsx",
                mime="application/vnd.ms-excel",
                help="Download all analysis results as Excel file"
            )
        
        # Tab 5: 3D Visualization
        with tab5:
            st.header("3D Data Visualization")
            
            # 3D scatter plot
            st.plotly_chart(create_3d_scatter(filtered_df), use_container_width=True)
            
            # Additional controls for 3D visualization
            st.info("💡 Tip: You can rotate, zoom, and pan the 3D plot using your mouse!")
        
        # Footer with dataset information
        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; color: #666;'>
                <p>Dataset: RMS Titanic Passenger Data | Total Records: {} | Filtered Records: {}</p>
            </div>
        """.format(len(df), len(filtered_df)), unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.write("Please make sure the data file exists at data/raw/titanic.csv")

if __name__ == "__main__":
    main()