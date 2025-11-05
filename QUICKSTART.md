# 🚀 Quick Start Guide

## Installation (Windows)

### Step 1: Install Python
If you don't have Python installed:
1. Download Python 3.8+ from [python.org](https://www.python.org/downloads/)
2. During installation, check "Add Python to PATH"
3. Verify installation: Open Command Prompt and type `python --version`

### Step 2: Install Dependencies
Open Command Prompt or PowerShell in the project folder and run:
```bash
pip install -r requirements.txt
```

### Step 3: Run the Application

**Option A: Using the batch file (Easiest)**
- Double-click `run.bat`
- The dashboard will open automatically in your browser

**Option B: Using Command Line**
```bash
streamlit run app.py
```

### Step 4: Access the Dashboard
- The app will automatically open at `http://localhost:8501`
- If it doesn't, copy the URL from the terminal and paste it in your browser

## First Time Setup

1. **Verify Data File**: Make sure `Mall_Customers.csv` is in the same folder as `app.py`

2. **Check Data Format**: The CSV should have these columns:
   - CustomerID
   - Gender
   - Age
   - Annual Income (k$)
   - Spending Score (1-100)

3. **Run the App**: Use `run.bat` or `streamlit run app.py`

## Using the Dashboard

### Sidebar Controls
- **Number of Segments**: Adjust customer clusters (3-8)
- **Filters**: 
  - Select gender(s)
  - Adjust age range
  - Adjust income range

### Main Dashboard Sections

1. **Key Metrics**: Overview of customer statistics
2. **Customer Segmentation**: Pie chart and segment characteristics
3. **3D Visualization**: Interactive cluster view
4. **Tabs**:
   - **Demographics**: Age and gender analysis
   - **Income Analysis**: Income patterns
   - **Spending Patterns**: Spending behavior
   - **Business Insights**: Recommendations and top customers

### Tips for Best Experience

1. **Start with Defaults**: View all data first, then apply filters
2. **Experiment with Clusters**: Try different numbers of segments (5-6 is usually optimal)
3. **Use Filters**: Narrow down to specific customer groups
4. **Explore 3D View**: Rotate and zoom to see customer clusters
5. **Check Insights Tab**: Get actionable business recommendations

## Troubleshooting

### Error: "Module not found"
```bash
pip install -r requirements.txt
```

### Error: "File not found"
- Make sure `Mall_Customers.csv` is in the project folder
- Check the file name (case-sensitive)

### Dashboard won't open
- Check if port 8501 is already in use
- Try closing other Streamlit apps
- Restart the application

### Slow performance
- Reduce the number of clusters
- Apply filters to reduce data size
- Close other applications

## Need Help?

1. Check the main README.md for detailed documentation
2. Review the code comments in app.py
3. Ensure all dependencies are installed correctly

## Next Steps

After exploring the dashboard:
1. Export insights for your team
2. Use segment recommendations for marketing
3. Adjust filters for specific analysis needs
4. Share findings with mall management

Enjoy analyzing your mall customer data! 🏬📊

