# 🏬 Mall Customer Analytics Dashboard

A comprehensive data analytics and visualization platform designed for Indian mall operations. This project provides deep insights into customer behavior, spending patterns, and segmentation to help mall management make data-driven decisions.

## 📋 Features

### 1. **Customer Segmentation**
- Automatic customer clustering using K-means algorithm
- Identifies different customer segments (Premium, Budget-Conscious, High Income Low Spenders, etc.)
- Dynamic segment analysis with customizable cluster numbers

### 2. **Interactive Visualizations**
- **3D Scatter Plot**: Visualize customer clusters in age, income, and spending dimensions
- **Demographic Analysis**: Age and gender distribution charts
- **Income Analysis**: Income patterns and correlations with spending
- **Spending Patterns**: Comprehensive spending score analysis
- **Heatmaps**: Density analysis across multiple dimensions

### 3. **Key Performance Indicators**
- Total customer count
- Average age and income metrics
- Gender distribution
- High spender identification
- Total potential revenue calculations

### 4. **Business Intelligence**
- Strategic recommendations for each customer segment
- Top customers identification
- Segment-specific marketing insights
- Revenue optimization suggestions

### 5. **Advanced Filtering**
- Filter by gender, age range, and income levels
- Real-time data updates based on filters
- Dynamic clustering based on filtered data

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone or download this repository**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure your data file is in the root directory:**
   - File name: `Mall_Customers.csv`
   - Should contain columns: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser:**
   - The app will automatically open at `http://localhost:8501`
   - If it doesn't, navigate to the URL shown in the terminal

## 📊 Dataset Format

The application expects a CSV file with the following columns:
- `CustomerID`: Unique identifier for each customer
- `Gender`: Male or Female
- `Age`: Customer age in years
- `Annual Income (k$)`: Annual income in thousands of US dollars
- `Spending Score (1-100)`: Spending behavior score (1-100)

## 🎯 Use Cases

### For Mall Management:
1. **Customer Segmentation**: Identify different customer groups for targeted marketing
2. **Product Placement**: Optimize store locations based on customer demographics
3. **Marketing Campaigns**: Design campaigns for specific customer segments
4. **Revenue Optimization**: Focus on high-value customers and convert low spenders

### For Store Owners:
1. **Target Audience**: Understand which customer segments visit which areas
2. **Pricing Strategy**: Adjust pricing based on customer income levels
3. **Inventory Management**: Stock products aligned with customer preferences
4. **Promotional Activities**: Plan discounts and offers for specific segments

### For Marketing Teams:
1. **Customer Profiling**: Create detailed customer personas
2. **Campaign Effectiveness**: Measure impact on different segments
3. **Retention Strategies**: Develop loyalty programs for high-value customers
4. **Acquisition Plans**: Identify potential customer segments to target

## 📈 Key Insights Provided

1. **Premium Customers**: High income, high spending - focus on exclusive products
2. **Budget-Conscious Shoppers**: Lower income but good spending - offer value deals
3. **High Income, Low Spenders**: Potential for conversion with targeted marketing
4. **Mature Customers**: Age-specific preferences and needs
5. **Regular Customers**: Standard segment requiring balanced approach

## 🔧 Customization

### Adjusting Clusters
- Use the sidebar slider to change the number of customer segments (3-8)
- Clusters automatically adjust based on filtered data

### Income Conversion
- The app converts USD to INR (₹) for Indian context
- Conversion rate: 1 USD = 83 INR (can be adjusted in code)

### Adding New Features
- The modular code structure allows easy addition of new visualizations
- Extend the clustering logic to include more features
- Add export functionality for reports

## 📱 Dashboard Sections

1. **KPI Overview**: Quick metrics at the top
2. **Customer Segmentation**: Cluster distribution and characteristics
3. **3D Visualization**: Interactive cluster exploration
4. **Demographics Tab**: Age and gender analysis
5. **Income Analysis Tab**: Income patterns and correlations
6. **Spending Patterns Tab**: Spending behavior analysis
7. **Business Insights Tab**: Strategic recommendations

## 🛠️ Technical Stack

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning (K-means clustering)

## 📝 Notes

- The application uses caching for better performance
- All visualizations are interactive and responsive
- Data is processed in real-time based on filters
- Clustering is performed on standardized features

## 🤝 Contributing

Feel free to extend this project with:
- Additional ML models (e.g., RFM analysis)
- Predictive analytics
- Customer lifetime value calculations
- Integration with real-time data sources
- Export functionality for reports

## 📄 License

This project is open source and available for use in mall operations and analytics.

## 🎉 Future Enhancements

- [ ] Customer lifetime value prediction
- [ ] RFM (Recency, Frequency, Monetary) analysis
- [ ] Real-time data integration
- [ ] Export reports to PDF/Excel
- [ ] Email notifications for insights
- [ ] Mobile-responsive design
- [ ] Multi-mall comparison features
- [ ] Seasonal trend analysis

---

**Built with ❤️ for Indian Mall Operations**

For questions or support, please refer to the code documentation or contact your development team.

