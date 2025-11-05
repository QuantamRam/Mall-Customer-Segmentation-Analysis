"""
Utility functions for Mall Customer Analytics
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def calculate_rfm_scores(df):
    """
    Calculate RFM (Recency, Frequency, Monetary) scores
    Note: This is a placeholder since we don't have transaction data
    For now, we'll use available features to estimate RFM
    """
    # For this dataset, we'll use:
    # Recency: Inverse of age (younger = more recent)
    # Frequency: Spending score (higher = more frequent)
    # Monetary: Annual income
    
    df['R_Score'] = pd.qcut(df['Age'].rank(method='first'), q=5, labels=[5,4,3,2,1])
    df['F_Score'] = pd.qcut(df['Spending Score (1-100)'].rank(method='first'), q=5, labels=[1,2,3,4,5])
    df['M_Score'] = pd.qcut(df['Annual Income (k$)'].rank(method='first'), q=5, labels=[1,2,3,4,5])
    
    df['RFM_Score'] = df['R_Score'].astype(int) + df['F_Score'].astype(int) + df['M_Score'].astype(int)
    
    return df

def get_optimal_clusters(df, max_k=10):
    """
    Find optimal number of clusters using elbow method and silhouette score
    """
    features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].copy()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(features_scaled, labels))
    
    # Find optimal k (highest silhouette score)
    optimal_k = k_range[np.argmax(silhouette_scores)]
    
    return optimal_k, inertias, silhouette_scores, list(k_range)

def categorize_customers(df):
    """
    Categorize customers into meaningful groups
    """
    def get_category(row):
        age = row['Age']
        income = row['Annual Income (k$)']
        spending = row['Spending Score (1-100)']
        
        if age < 25:
            age_group = "Young"
        elif age < 40:
            age_group = "Adult"
        elif age < 60:
            age_group = "Middle-aged"
        else:
            age_group = "Senior"
        
        if income < 40:
            income_group = "Low"
        elif income < 70:
            income_group = "Medium"
        else:
            income_group = "High"
        
        if spending < 40:
            spending_group = "Low"
        elif spending < 70:
            spending_group = "Medium"
        else:
            spending_group = "High"
        
        return f"{age_group}-{income_group} Income-{spending_group} Spender"
    
    df['Category'] = df.apply(get_category, axis=1)
    return df

def calculate_customer_lifetime_value(df):
    """
    Estimate Customer Lifetime Value (CLV)
    Simple calculation: (Annual Income * Spending Score / 100) * Average Mall Visit Frequency Estimate
    """
    # Assuming average visit frequency based on spending score
    # Higher spending score = more frequent visits
    visit_frequency = df['Spending Score (1-100)'] / 20  # Scale to reasonable visit frequency
    annual_spend = df['Annual Income (k$)'] * (df['Spending Score (1-100)'] / 100) * 0.1
    clv = annual_spend * visit_frequency
    
    df['Estimated_CLV'] = clv
    return df

def get_segment_insights(df):
    """
    Generate detailed insights for each customer segment
    """
    insights = {}
    
    for segment in df['Segment'].unique():
        seg_data = df[df['Segment'] == segment]
        
        insights[segment] = {
            'count': len(seg_data),
            'avg_age': seg_data['Age'].mean(),
            'avg_income': seg_data['Annual Income (k$)'].mean(),
            'avg_spending': seg_data['Spending Score (1-100)'].mean(),
            'gender_distribution': seg_data['Gender'].value_counts().to_dict(),
            'recommendations': generate_recommendations(segment, seg_data)
        }
    
    return insights

def generate_recommendations(segment_name, segment_data):
    """
    Generate business recommendations based on segment characteristics
    """
    recommendations = []
    
    avg_income = segment_data['Annual Income (k$)'].mean()
    avg_spending = segment_data['Spending Score (1-100)'].mean()
    avg_age = segment_data['Age'].mean()
    
    if 'Premium' in segment_name:
        recommendations.extend([
            "Focus on exclusive products and premium brands",
            "Offer VIP membership and loyalty programs",
            "Personalized shopping assistance",
            "Early access to new collections"
        ])
    
    if 'Budget-Conscious' in segment_name:
        recommendations.extend([
            "Offer value deals and bundle packages",
            "Discount campaigns and seasonal sales",
            "Affordable product ranges",
            "Reward programs for frequent purchases"
        ])
    
    if avg_age > 50:
        recommendations.extend([
            "Comfortable seating areas",
            "Senior citizen discounts",
            "Accessibility features",
            "Health and wellness products"
        ])
    
    if avg_spending < 40:
        recommendations.extend([
            "Engage with personalized marketing campaigns",
            "Showcase products that match their income level",
            "Offer financing options",
            "Create urgency with limited-time offers"
        ])
    
    return recommendations

