import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Mall Customer Analytics Dashboard",
    page_icon="🏬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Indian mall theme
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B35;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #FF6B35, #F7931E);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6B35;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the mall customers dataset"""
    df = pd.read_csv('Mall_Customers.csv')
    # Convert income to INR (assuming 1 USD = 83 INR for Indian context)
    df['Annual_Income_INR'] = df['Annual Income (k$)'] * 83000
    df['Annual_Income_Lakhs'] = df['Annual Income (k$)'] * 8.3
    return df

@st.cache_data
def perform_clustering(df, n_clusters=5):
    """Perform K-means clustering on customer data"""
    # Prepare features for clustering
    features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].copy()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(features_scaled)
    
    # Name clusters based on characteristics
    cluster_names = {}
    for i in range(n_clusters):
        cluster_data = df[df['Cluster'] == i]
        avg_income = cluster_data['Annual Income (k$)'].mean()
        avg_spending = cluster_data['Spending Score (1-100)'].mean()
        avg_age = cluster_data['Age'].mean()
        
        if avg_income > 60 and avg_spending > 60:
            cluster_names[i] = "Premium Customers"
        elif avg_income < 40 and avg_spending > 60:
            cluster_names[i] = "Budget-Conscious Shoppers"
        elif avg_income > 60 and avg_spending < 40:
            cluster_names[i] = "High Income, Low Spenders"
        elif avg_age > 50:
            cluster_names[i] = "Mature Customers"
        else:
            cluster_names[i] = "Regular Customers"
    
    df['Segment'] = df['Cluster'].map(cluster_names)
    return df, kmeans, scaler

def main():
    # Header
    st.markdown('<h1 class="main-header">🏬 Mall Customer Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem;">Comprehensive Customer Insights for Indian Malls</p>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    # Sidebar
    st.sidebar.header("📊 Dashboard Controls")
    st.sidebar.markdown("---")
    
    # Clustering options
    n_clusters = st.sidebar.slider("Number of Customer Segments", 3, 8, 5)
    
    # Filter options
    st.sidebar.subheader("🔍 Filters")
    gender_filter = st.sidebar.multiselect("Gender", options=df['Gender'].unique(), default=df['Gender'].unique())
    age_range = st.sidebar.slider("Age Range", int(df['Age'].min()), int(df['Age'].max()), (int(df['Age'].min()), int(df['Age'].max())))
    income_range = st.sidebar.slider("Annual Income (k$)", int(df['Annual Income (k$)'].min()), int(df['Annual Income (k$)'].max()), (int(df['Annual Income (k$)'].min()), int(df['Annual Income (k$)'].max())))
    
    # Apply filters
    filtered_df = df[
        (df['Gender'].isin(gender_filter)) &
        (df['Age'] >= age_range[0]) &
        (df['Age'] <= age_range[1]) &
        (df['Annual Income (k$)'] >= income_range[0]) &
        (df['Annual Income (k$)'] <= income_range[1])
    ].copy()
    
    # Perform clustering
    df_clustered, kmeans, scaler = perform_clustering(filtered_df, n_clusters)
    
    # Key Metrics
    st.markdown("---")
    st.subheader("📈 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", len(filtered_df))
        st.metric("Average Age", f"{filtered_df['Age'].mean():.1f} years")
    
    with col2:
        avg_income = filtered_df['Annual Income (k$)'].mean()
        st.metric("Average Annual Income", f"₹{avg_income * 8.3:.1f} Lakhs", 
                  f"{avg_income:.1f}k USD")
        st.metric("Average Spending Score", f"{filtered_df['Spending Score (1-100)'].mean():.1f}/100")
    
    with col3:
        gender_dist = filtered_df['Gender'].value_counts()
        female_pct = (gender_dist.get('Female', 0) / len(filtered_df)) * 100
        st.metric("Female Customers", f"{female_pct:.1f}%")
        st.metric("Male Customers", f"{100-female_pct:.1f}%")
    
    with col4:
        high_spenders = len(filtered_df[filtered_df['Spending Score (1-100)'] >= 70])
        st.metric("High Spenders (70+)", f"{high_spenders}", 
                  f"{(high_spenders/len(filtered_df)*100):.1f}%")
        potential_revenue = filtered_df['Annual Income (k$)'].sum() * 8.3
        st.metric("Total Potential Revenue", f"₹{potential_revenue:.0f} Lakhs")
    
    st.markdown("---")
    
    # Customer Segmentation Analysis
    st.subheader("👥 Customer Segmentation Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Segment distribution
        segment_counts = df_clustered['Segment'].value_counts()
        fig_pie = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title="Customer Segment Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Segment characteristics
        segment_stats = df_clustered.groupby('Segment').agg({
            'Age': 'mean',
            'Annual Income (k$)': 'mean',
            'Spending Score (1-100)': 'mean'
        }).round(2)
        segment_stats.columns = ['Avg Age', 'Avg Income (k$)', 'Avg Spending Score']
        st.dataframe(segment_stats, use_container_width=True)
    
    # 3D Clustering Visualization
    st.subheader("🎯 Customer Clusters (3D Visualization)")
    fig_3d = px.scatter_3d(
        df_clustered,
        x='Age',
        y='Annual Income (k$)',
        z='Spending Score (1-100)',
        color='Segment',
        size='Spending Score (1-100)',
        hover_data=['Gender', 'CustomerID'],
        title="Customer Segmentation in 3D Space",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_3d.update_layout(scene=dict(
        xaxis_title="Age",
        yaxis_title="Annual Income (k$)",
        zaxis_title="Spending Score"
    ))
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Detailed Analysis Tabs
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Demographics", "💰 Income Analysis", "🛒 Spending Patterns", "🎯 Business Insights"])
    
    with tab1:
        st.subheader("Demographic Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution
            fig_age = px.histogram(
                filtered_df,
                x='Age',
                nbins=20,
                title="Age Distribution",
                color='Gender',
                barmode='overlay',
                color_discrete_map={'Male': '#3498db', 'Female': '#e74c3c'}
            )
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            # Gender distribution
            gender_counts = filtered_df['Gender'].value_counts()
            fig_gender = px.bar(
                x=gender_counts.index,
                y=gender_counts.values,
                title="Gender Distribution",
                labels={'x': 'Gender', 'y': 'Count'},
                color=gender_counts.index,
                color_discrete_map={'Male': '#3498db', 'Female': '#e74c3c'}
            )
            st.plotly_chart(fig_gender, use_container_width=True)
        
        # Age vs Spending by Gender
        fig_age_spend = px.scatter(
            filtered_df,
            x='Age',
            y='Spending Score (1-100)',
            color='Gender',
            size='Annual Income (k$)',
            hover_data=['CustomerID'],
            title="Age vs Spending Score by Gender",
            color_discrete_map={'Male': '#3498db', 'Female': '#e74c3c'}
        )
        st.plotly_chart(fig_age_spend, use_container_width=True)
    
    with tab2:
        st.subheader("Income Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Income distribution
            fig_income = px.histogram(
                filtered_df,
                x='Annual Income (k$)',
                nbins=20,
                title="Annual Income Distribution",
                color='Gender',
                barmode='overlay',
                color_discrete_map={'Male': '#3498db', 'Female': '#e74c3c'}
            )
            st.plotly_chart(fig_income, use_container_width=True)
        
        with col2:
            # Income vs Spending
            fig_income_spend = px.scatter(
                filtered_df,
                x='Annual Income (k$)',
                y='Spending Score (1-100)',
                color='Gender',
                size='Age',
                hover_data=['CustomerID'],
                title="Income vs Spending Score",
                color_discrete_map={'Male': '#3498db', 'Female': '#e74c3c'}
            )
            st.plotly_chart(fig_income_spend, use_container_width=True)
        
        # Income statistics by gender
        income_stats = filtered_df.groupby('Gender')['Annual Income (k$)'].describe()
        st.subheader("Income Statistics by Gender")
        st.dataframe(income_stats, use_container_width=True)
    
    with tab3:
        st.subheader("Spending Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Spending score distribution
            fig_spending = px.histogram(
                filtered_df,
                x='Spending Score (1-100)',
                nbins=20,
                title="Spending Score Distribution",
                color='Gender',
                barmode='overlay',
                color_discrete_map={'Male': '#3498db', 'Female': '#e74c3c'}
            )
            st.plotly_chart(fig_spending, use_container_width=True)
        
        with col2:
            # Spending categories
            def categorize_spending(score):
                if score >= 80:
                    return "Very High"
                elif score >= 60:
                    return "High"
                elif score >= 40:
                    return "Medium"
                else:
                    return "Low"
            
            filtered_df['Spending_Category'] = filtered_df['Spending Score (1-100)'].apply(categorize_spending)
            spending_cat = filtered_df['Spending_Category'].value_counts()
            fig_cat = px.bar(
                x=spending_cat.index,
                y=spending_cat.values,
                title="Spending Categories",
                labels={'x': 'Category', 'y': 'Count'},
                color=spending_cat.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_cat, use_container_width=True)
        
        # Heatmap: Age vs Income vs Spending
        fig_heatmap = px.density_heatmap(
            filtered_df,
            x='Age',
            y='Annual Income (k$)',
            z='Spending Score (1-100)',
            histfunc='avg',
            title="Heatmap: Age, Income, and Spending Score",
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab4:
        st.subheader("Business Insights & Recommendations")
        
        # Key insights
        insights = []
        
        # Segment insights
        for segment in df_clustered['Segment'].unique():
            seg_data = df_clustered[df_clustered['Segment'] == segment]
            insights.append(f"**{segment}**: {len(seg_data)} customers | "
                          f"Avg Income: ₹{seg_data['Annual Income (k$)'].mean() * 8.3:.1f}L | "
                          f"Avg Spending: {seg_data['Spending Score (1-100)'].mean():.1f}")
        
        st.markdown("### 📋 Segment Summary")
        for insight in insights:
            st.markdown(f"- {insight}")
        
        st.markdown("---")
        
        # Recommendations
        st.markdown("### 💡 Strategic Recommendations")
        
        premium_customers = df_clustered[df_clustered['Segment'] == 'Premium Customers']
        if len(premium_customers) > 0:
            st.success(f"🎯 **Premium Customers ({len(premium_customers)})**: "
                      f"Focus on exclusive products, VIP services, and loyalty programs. "
                      f"These customers have high income and high spending potential.")
        
        budget_customers = df_clustered[df_clustered['Segment'] == 'Budget-Conscious Shoppers']
        if len(budget_customers) > 0:
            st.info(f"💰 **Budget-Conscious Shoppers ({len(budget_customers)})**: "
                   f"Offer discounts, value deals, and bundle packages. "
                   f"These customers spend well despite lower income.")
        
        high_income_low_spend = df_clustered[df_clustered['Segment'] == 'High Income, Low Spenders']
        if len(high_income_low_spend) > 0:
            st.warning(f"📈 **High Income, Low Spenders ({len(high_income_low_spend)})**: "
                      f"Engage with personalized marketing, premium brand showcases, "
                      f"and targeted promotions to increase spending.")
        
        # Top customers
        st.markdown("---")
        st.markdown("### 🏆 Top 10 Customers by Spending Score")
        top_customers = filtered_df.nlargest(10, 'Spending Score (1-100)')[
            ['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        ]
        top_customers['Annual Income (₹ Lakhs)'] = top_customers['Annual Income (k$)'] * 8.3
        st.dataframe(top_customers[['CustomerID', 'Gender', 'Age', 'Annual Income (₹ Lakhs)', 'Spending Score (1-100)']], 
                    use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🏬 Mall Customer Analytics Dashboard | Built for Indian Mall Operations</p>
        <p>Data-driven insights to optimize customer experience and business performance</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

