"""
Shared utilities for data analysis tests.
Contains common test data creation and validation functions.
"""

import os
import tempfile
from pathlib import Path


def create_sales_data(data_dir):
    """Create comprehensive sales data for analysis."""
    data_path = Path(data_dir)
    
    # Create a detailed CSV file
    csv_file = data_path / "sales_data.csv"
    csv_content = """product,sales,region,month,category,price
Laptop,1200,North,January,Electronics,800
Phone,800,South,January,Electronics,600
Tablet,600,East,January,Electronics,400
Monitor,300,West,January,Electronics,200
Laptop,1400,North,February,Electronics,800
Phone,900,South,February,Electronics,600
Tablet,550,East,February,Electronics,400
Monitor,350,West,February,Electronics,200
Keyboard,150,North,January,Accessories,50
Mouse,100,South,January,Accessories,30
Headphones,250,East,January,Accessories,80
Cable,75,West,January,Accessories,25
Keyboard,180,North,February,Accessories,50
Mouse,120,South,February,Accessories,30
Headphones,280,East,February,Accessories,80
Cable,90,West,February,Accessories,25"""
    csv_file.write_text(csv_content)
    
    # Create documentation
    readme_file = data_path / "README.md"
    readme_content = """# Sales Data Analysis Dataset

This dataset contains monthly sales information for Q1 analysis with the following columns:

## Data Schema
- **product**: Product name (Laptop, Phone, Tablet, Monitor, Keyboard, Mouse, Headphones, Cable)
- **sales**: Sales amount in USD
- **region**: Sales region (North, South, East, West)
- **month**: Month of sale (January, February)
- **category**: Product category (Electronics, Accessories)
- **price**: Unit price in USD

## Analysis Goals
Use this data to analyze:
1. Sales performance by product and region
2. Month-over-month growth trends
3. Category performance comparison
4. Regional market analysis

## Data Quality Notes
- All sales figures are in USD
- Data covers January-February period
- No missing values in the dataset
"""
    readme_file.write_text(readme_content)
    
    # Create additional context file
    context_file = data_path / "business_context.txt"
    context_content = """Business Context for Sales Analysis:

Company: TechCorp Electronics
Period: Q1 2024 (Jan-Feb)
Market: North American regions

Key Business Questions:
1. Which products are driving revenue growth?
2. Are there regional preferences for certain products?
3. How is the accessories category performing vs electronics?
4. What's the month-over-month growth rate?

Success Metrics:
- Total revenue growth > 5%
- Electronics category should represent 80%+ of revenue
- All regions should show positive growth
"""
    context_file.write_text(context_content)
    
    print(f"✅ Created comprehensive sales data in: {data_dir}")
    return data_dir


def test_data_exploration(agent, data_dir):
    """Test the agent's data exploration capabilities."""
    print("\n🔍 Testing data exploration...")
    
    exploration_task = f"""
    Explore the data available in {data_dir}. Follow the exploration phase of your workflow:
    
    1. List all files in the directory
    2. Read the documentation to understand the data structure
    3. Load and examine the sales data
    4. Provide a summary of what data is available
    
    Use the final_answer() function to return your exploration findings.
    """
    
    try:
        print("🤖 Agent exploring data...")
        result = agent.run(exploration_task)
        
        # Check if exploration was comprehensive
        result_str = str(result)
        exploration_indicators = [
            "sales_data.csv",
            "README.md", 
            "business_context.txt",
            "product", "sales", "region"
        ]
        
        found_indicators = sum(1 for indicator in exploration_indicators if indicator in result_str)
        
        if found_indicators >= 4:
            print("✅ Data exploration completed successfully")
            print(f"📋 Exploration result: {result}")
            return True
        else:
            print("❌ Data exploration incomplete - missing key elements")
            return False
            
    except Exception as e:
        print(f"❌ Data exploration failed: {e}")
        return False


def test_sales_analysis(agent, data_dir):
    """Test comprehensive sales analysis."""
    print("\n📊 Testing sales analysis...")
    
    analysis_task = f"""
    Perform a comprehensive analysis of the sales data in {data_dir}. Follow your standard workflow:
    
    1. Explore: You've already seen the data structure
    2. Plan: Create a plan to analyze the sales performance
    3. Execute: Perform analysis to find:
       - Total sales by product
       - Best performing region
       - Month-over-month growth by category
       - Top 3 products by revenue
    4. Conclude: Provide actionable business insights
    
    Use the final_answer() function to return your complete analysis.
    """
    
    try:
        print("🤖 Agent performing sales analysis...")
        result = agent.run(analysis_task)
        
        # Check if analysis contains expected elements
        result_str = str(result)
        analysis_indicators = [
            "total", "growth", "region", "product", 
            "revenue", "performance", "analysis"
        ]
        
        found_indicators = sum(1 for indicator in analysis_indicators if indicator.lower() in result_str.lower())
        
        if found_indicators >= 4:
            print("✅ Sales analysis completed successfully")
            print(f"📋 Analysis result: {result}")
            return True
        else:
            print("❌ Sales analysis incomplete - missing key metrics")
            return False
            
    except Exception as e:
        print(f"❌ Sales analysis failed: {e}")
        return False


def test_business_insights(agent, data_dir):
    """Test the agent's ability to generate business insights."""
    print("\n💡 Testing business insights generation...")
    
    insights_task = f"""
    Based on the sales data in {data_dir}, generate strategic business insights:
    
    1. Read the business context file to understand company goals
    2. Analyze the data against the success metrics mentioned
    3. Identify trends and patterns
    4. Provide 3 specific recommendations for business improvement
    
    Use the final_answer() function to return your business recommendations.
    """
    
    try:
        print("🤖 Agent generating business insights...")
        result = agent.run(insights_task)
        
        # Check if insights are meaningful
        result_str = str(result)
        insight_indicators = [
            "recommend", "improve", "strategy", "trend", 
            "growth", "opportunity", "insight"
        ]
        
        found_indicators = sum(1 for indicator in insight_indicators if indicator.lower() in result_str.lower())
        
        if found_indicators >= 3:
            print("✅ Business insights generated successfully")
            print(f"📋 Insights: {result}")
            return True
        else:
            print("❌ Business insights incomplete - lacking strategic depth")
            return False
            
    except Exception as e:
        print(f"❌ Business insights generation failed: {e}")
        return False


def run_analysis_tests(agent, data_dir):
    """Run all analysis tests and return results."""
    exploration_passed = test_data_exploration(agent, data_dir)
    analysis_passed = test_sales_analysis(agent, data_dir)
    insights_passed = test_business_insights(agent, data_dir)
    
    return exploration_passed, analysis_passed, insights_passed


def print_test_summary(exploration_passed, analysis_passed, insights_passed, agent_type):
    """Print test summary results."""
    print("\n" + "=" * 65)
    print(f"📊 {agent_type.upper()} DATA ANALYSIS TEST SUMMARY")
    print(f"🔍 Data exploration: {'✅ PASSED' if exploration_passed else '❌ FAILED'}")
    print(f"📈 Sales analysis: {'✅ PASSED' if analysis_passed else '❌ FAILED'}")
    print(f"💡 Business insights: {'✅ PASSED' if insights_passed else '❌ FAILED'}")
    
    all_passed = exploration_passed and analysis_passed and insights_passed
    if all_passed:
        print(f"\n🎉 All {agent_type} data analysis tests passed! Your agent can perform comprehensive analysis.")
    else:
        print(f"\n⚠️  Some {agent_type} analysis tests failed. Check the output above for details.")
    
    return all_passed


def setup_test_environment():
    """Set up common test environment and configuration."""
    from utils.execution import get_env
    
    # Get LLM configuration
    config = get_env()
    api_key = config["API_KEY"]
    model_id = f"{config['LLM_GATEWAY']}/{config['MODEL']}"
    api_base = config["BASE_URL"]
    
    print(f"🔧 Using model: {model_id}")
    if api_base:
        print(f"🔧 Using API base: {api_base}")
    
    # Show tracing endpoint
    tracing_endpoint = os.getenv("OTLP_ENDPOINT", "http://0.0.0.0:6006/v1/traces")
    phoenix_ui = tracing_endpoint.replace("/v1/traces", "").replace("0.0.0.0", "localhost")
    print(f"📊 Phoenix UI available at: {phoenix_ui}")
    print(f"🔍 Traces endpoint: {tracing_endpoint}")
    
    return config, api_key, model_id, api_base
