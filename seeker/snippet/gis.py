#date: 2025-07-15T17:00:09Z
#url: https://api.github.com/gists/113256ac7babe74950d11870bf5d9dbe
#owner: https://api.github.com/users/RajChowdhury240

#!/usr/bin/env python3
"""
IAM Roles Data Analyzer & Visualizer
Analyzes the CSV output from AWS IAM Role Scanner and creates beautiful visualizations
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime
import os
import sys
from pathlib import Path

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class IAMDataAnalyzer:
    def __init__(self, csv_file: str = "iam_roles_report.csv"):
        """Initialize the analyzer with CSV file"""
        self.csv_file = csv_file
        self.df = None
        self.output_dir = "iam_analysis_charts"
        
        # Create output directory
        Path(self.output_dir).mkdir(exist_ok=True)
        
        # Load and process data
        self.load_data()
        
    def load_data(self):
        """Load and preprocess the CSV data"""
        try:
            if not os.path.exists(self.csv_file):
                print(f"❌ CSV file '{self.csv_file}' not found!")
                print("Please run the IAM scanner first to generate the CSV file.")
                sys.exit(1)
            
            self.df = pd.read_csv(self.csv_file)
            
            # Check if dataframe is empty
            if self.df.empty:
                print("✅ CSV file is empty - all roles have standard 4-hour session duration!")
                return
            
            # Add duration categories for better analysis
            self.df['Duration Category'] = self.df['Maximum Session Duration (seconds)'].apply(
                self._categorize_duration
            )
            
            # Add risk level based on duration
            self.df['Risk Level'] = self.df['Maximum Session Duration (seconds)'].apply(
                self._assess_risk_level
            )
            
            print(f"✅ Loaded {len(self.df)} records from {self.csv_file}")
            print(f"📊 Found {self.df['Account Name'].nunique()} unique accounts")
            print(f"🔍 Found {self.df['Role Name'].nunique()} unique roles")
            
        except Exception as e:
            print(f"❌ Error loading CSV file: {e}")
            sys.exit(1)
    
    def _categorize_duration(self, seconds: int) -> str:
        """Categorize duration into readable categories"""
        if seconds == 3600:
            return "1 Hour"
        elif seconds == 7200:
            return "2 Hours"
        elif seconds == 10800:
            return "3 Hours"
        elif seconds > 14400:
            return f"{seconds//3600} Hours"
        else:
            return f"{seconds//60} Minutes"
    
    def _assess_risk_level(self, seconds: int) -> str:
        """Assess risk level based on session duration"""
        if seconds <= 3600:
            return "High Risk"
        elif seconds <= 7200:
            return "Medium Risk"
        elif seconds < 14400:
            return "Low Risk"
        else:
            return "Extended Duration"
    
    def generate_summary_stats(self):
        """Generate summary statistics"""
        if self.df.empty:
            return
            
        print("\n" + "="*60)
        print("📈 SUMMARY STATISTICS")
        print("="*60)
        
        # Basic stats
        print(f"Total Accounts with Non-4-Hour Roles: {self.df['Account Name'].nunique()}")
        print(f"Total Non-4-Hour Roles: {len(self.df)}")
        print(f"Average Roles per Account: {len(self.df) / self.df['Account Name'].nunique():.2f}")
        
        # Duration distribution
        print(f"\nDuration Distribution:")
        duration_counts = self.df['Duration Category'].value_counts()
        for duration, count in duration_counts.items():
            print(f"  {duration}: {count} roles ({count/len(self.df)*100:.1f}%)")
        
        # Risk level distribution
        print(f"\nRisk Level Distribution:")
        risk_counts = self.df['Risk Level'].value_counts()
        for risk, count in risk_counts.items():
            print(f"  {risk}: {count} roles ({count/len(self.df)*100:.1f}%)")
        
        # Top accounts with most non-4-hour roles
        print(f"\nTop 10 Accounts with Most Non-4-Hour Roles:")
        top_accounts = self.df.groupby('Account Name').size().sort_values(ascending=False).head(10)
        for account, count in top_accounts.items():
            print(f"  {account}: {count} roles")
    
    def create_pivot_tables(self):
        """Create and save pivot tables"""
        if self.df.empty:
            return
            
        print("\n📊 Creating Pivot Tables...")
        
        # Pivot Table 1: Account vs Duration Categories
        pivot1 = pd.pivot_table(
            self.df,
            values='Role Name',
            index='Account Name',
            columns='Duration Category',
            aggfunc='count',
            fill_value=0
        )
        pivot1.to_csv(f"{self.output_dir}/pivot_account_vs_duration.csv")
        print(f"✅ Saved: {self.output_dir}/pivot_account_vs_duration.csv")
        
        # Pivot Table 2: Account vs Risk Level
        pivot2 = pd.pivot_table(
            self.df,
            values='Role Name',
            index='Account Name',
            columns='Risk Level',
            aggfunc='count',
            fill_value=0
        )
        pivot2.to_csv(f"{self.output_dir}/pivot_account_vs_risk.csv")
        print(f"✅ Saved: {self.output_dir}/pivot_account_vs_risk.csv")
        
        # Summary pivot table
        summary_pivot = self.df.groupby('Account Name').agg({
            'Role Name': 'count',
            'Maximum Session Duration (seconds)': ['min', 'max', 'mean'],
            'Risk Level': lambda x: x.value_counts().index[0]  # Most common risk level
        }).round(2)
        summary_pivot.columns = ['Total_Roles', 'Min_Duration', 'Max_Duration', 'Avg_Duration', 'Primary_Risk_Level']
        summary_pivot.to_csv(f"{self.output_dir}/summary_by_account.csv")
        print(f"✅ Saved: {self.output_dir}/summary_by_account.csv")
        
        return pivot1, pivot2, summary_pivot
    
    def create_matplotlib_charts(self):
        """Create matplotlib/seaborn charts"""
        if self.df.empty:
            return
            
        print("\n🎨 Creating Matplotlib/Seaborn Charts...")
        
        # Set up the plotting style
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        
        # 1. Roles per Account (Top 15)
        plt.figure(figsize=(14, 8))
        account_counts = self.df['Account Name'].value_counts().head(15)
        
        ax = sns.barplot(x=account_counts.values, y=account_counts.index, palette='viridis')
        plt.title('Top 15 Accounts by Number of Non-4-Hour Roles', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Roles', fontsize=12)
        plt.ylabel('Account Name', fontsize=12)
        
        # Add value labels on bars
        for i, v in enumerate(account_counts.values):
            ax.text(v + 0.1, i, str(v), va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/roles_per_account.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Duration Categories Distribution
        plt.figure(figsize=(12, 8))
        duration_counts = self.df['Duration Category'].value_counts()
        
        # Create pie chart
        plt.subplot(2, 2, 1)
        colors = sns.color_palette('Set3', len(duration_counts))
        plt.pie(duration_counts.values, labels=duration_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        plt.title('Duration Categories Distribution', fontweight='bold')
        
        # Create bar chart
        plt.subplot(2, 2, 2)
        sns.barplot(x=duration_counts.index, y=duration_counts.values, palette='Set2')
        plt.title('Duration Categories Count', fontweight='bold')
        plt.xticks(rotation=45)
        
        # Risk Level Distribution
        plt.subplot(2, 2, 3)
        risk_counts = self.df['Risk Level'].value_counts()
        colors = ['#ff4444', '#ffaa44', '#44ff44', '#4444ff']
        sns.barplot(x=risk_counts.index, y=risk_counts.values, palette=colors[:len(risk_counts)])
        plt.title('Risk Level Distribution', fontweight='bold')
        plt.xticks(rotation=45)
        
        # Session Duration Histogram
        plt.subplot(2, 2, 4)
        plt.hist(self.df['Maximum Session Duration (seconds)'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Session Duration Distribution', fontweight='bold')
        plt.xlabel('Duration (seconds)')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/duration_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Heatmap of Account vs Duration Category
        if self.df['Account Name'].nunique() > 1:
            plt.figure(figsize=(14, 10))
            pivot_data = pd.pivot_table(
                self.df, 
                values='Role Name', 
                index='Account Name', 
                columns='Duration Category', 
                aggfunc='count', 
                fill_value=0
            )
            
            sns.heatmap(pivot_data, annot=True, cmap='YlOrRd', fmt='d', 
                       cbar_kws={'label': 'Number of Roles'})
            plt.title('Heatmap: Accounts vs Duration Categories', fontsize=16, fontweight='bold')
            plt.xlabel('Duration Category', fontsize=12)
            plt.ylabel('Account Name', fontsize=12)
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/heatmap_account_duration.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Matplotlib charts saved in {self.output_dir}/")
    
    def create_plotly_charts(self):
        """Create interactive Plotly charts"""
        if self.df.empty:
            return
            
        print("\n🚀 Creating Interactive Plotly Charts...")
        
        # 1. Interactive Bar Chart - Roles per Account
        account_counts = self.df['Account Name'].value_counts()
        
        fig1 = px.bar(
            x=account_counts.values[:20], 
            y=account_counts.index[:20],
            orientation='h',
            title='Top 20 Accounts by Number of Non-4-Hour Roles',
            labels={'x': 'Number of Roles', 'y': 'Account Name'},
            color=account_counts.values[:20],
            color_continuous_scale='viridis'
        )
        fig1.update_layout(height=600, showlegend=False)
        fig1.write_html(f"{self.output_dir}/interactive_roles_per_account.html")
        
        # 2. Interactive Sunburst Chart
        fig2 = px.sunburst(
            self.df,
            path=['Risk Level', 'Duration Category', 'Account Name'],
            title='Hierarchical View: Risk Level → Duration → Account',
            color='Risk Level',
            color_discrete_map={
                'High Risk': '#ff4444',
                'Medium Risk': '#ffaa44', 
                'Low Risk': '#44ff44',
                'Extended Duration': '#4444ff'
            }
        )
        fig2.update_layout(height=600)
        fig2.write_html(f"{self.output_dir}/interactive_sunburst.html")
        
        # 3. Interactive Treemap
        account_summary = self.df.groupby(['Account Name', 'Risk Level']).size().reset_index(name='count')
        
        fig3 = px.treemap(
            account_summary,
            path=['Account Name', 'Risk Level'],
            values='count',
            title='Treemap: Account Distribution by Risk Level',
            color='Risk Level',
            color_discrete_map={
                'High Risk': '#ff4444',
                'Medium Risk': '#ffaa44',
                'Low Risk': '#44ff44',
                'Extended Duration': '#4444ff'
            }
        )
        fig3.update_layout(height=600)
        fig3.write_html(f"{self.output_dir}/interactive_treemap.html")
        
        # 4. Interactive Scatter Plot
        account_stats = self.df.groupby('Account Name').agg({
            'Role Name': 'count',
            'Maximum Session Duration (seconds)': 'mean',
            'Risk Level': lambda x: (x == 'High Risk').sum()
        }).reset_index()
        account_stats.columns = ['Account_Name', 'Total_Roles', 'Avg_Duration', 'High_Risk_Count']
        
        fig4 = px.scatter(
            account_stats,
            x='Total_Roles',
            y='Avg_Duration',
            size='High_Risk_Count',
            hover_name='Account_Name',
            title='Account Analysis: Total Roles vs Average Duration',
            labels={
                'Total_Roles': 'Total Number of Roles',
                'Avg_Duration': 'Average Session Duration (seconds)',
                'High_Risk_Count': 'High Risk Roles Count'
            },
            color='High_Risk_Count',
            color_continuous_scale='Reds'
        )
        fig4.update_layout(height=600)
        fig4.write_html(f"{self.output_dir}/interactive_scatter.html")
        
        # 5. Interactive Dashboard
        fig5 = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Duration Distribution', 'Risk Level Distribution', 
                          'Top 10 Accounts', 'Session Duration Histogram'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "histogram"}]]
        )
        
        # Duration pie chart
        duration_counts = self.df['Duration Category'].value_counts()
        fig5.add_trace(
            go.Pie(labels=duration_counts.index, values=duration_counts.values, name="Duration"),
            row=1, col=1
        )
        
        # Risk level bar chart
        risk_counts = self.df['Risk Level'].value_counts()
        fig5.add_trace(
            go.Bar(x=risk_counts.index, y=risk_counts.values, name="Risk Level"),
            row=1, col=2
        )
        
        # Top 10 accounts
        top_accounts = self.df['Account Name'].value_counts().head(10)
        fig5.add_trace(
            go.Bar(x=top_accounts.index, y=top_accounts.values, name="Top Accounts"),
            row=2, col=1
        )
        
        # Duration histogram
        fig5.add_trace(
            go.Histogram(x=self.df['Maximum Session Duration (seconds)'], name="Duration Hist"),
            row=2, col=2
        )
        
        fig5.update_layout(
            height=800,
            title_text="IAM Roles Analysis Dashboard",
            showlegend=False
        )
        
        # Update x-axis labels for better readability
        fig5.update_xaxes(tickangle=45, row=2, col=1)
        
        fig5.write_html(f"{self.output_dir}/interactive_dashboard.html")
        
        print(f"✅ Interactive Plotly charts saved in {self.output_dir}/")
    
    def create_detailed_report(self):
        """Create a detailed HTML report combining all analysis"""
        if self.df.empty:
            return
            
        print("\n📋 Creating Detailed HTML Report...")
        
        # Generate summary statistics
        total_accounts = self.df['Account Name'].nunique()
        total_roles = len(self.df)
        avg_roles_per_account = total_roles / total_accounts
        
        duration_stats = self.df['Duration Category'].value_counts()
        risk_stats = self.df['Risk Level'].value_counts()
        
        # Top 10 accounts
        top_accounts = self.df['Account Name'].value_counts().head(10)
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>IAM Roles Analysis Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 15px;
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                    overflow: hidden;
                }}
                .header {{
                    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                    color: white;
                    padding: 30px;
                    text-align: center;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 2.5em;
                    font-weight: 300;
                }}
                .summary {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    padding: 30px;
                    background: #f8f9fa;
                }}
                .summary-card {{
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                .summary-number {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #495057;
                }}
                .summary-label {{
                    color: #6c757d;
                    margin-top: 5px;
                }}
                .section {{
                    padding: 30px;
                    border-bottom: 1px solid #e9ecef;
                }}
                .section h2 {{
                    color: #495057;
                    border-bottom: 2px solid #007bff;
                    padding-bottom: 10px;
                }}
                .table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                .table th, .table td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #dee2e6;
                }}
                .table th {{
                    background-color: #495057;
                    color: white;
                }}
                .charts-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .chart-link {{
                    display: block;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    text-decoration: none;
                    border-radius: 10px;
                    text-align: center;
                    transition: transform 0.3s ease;
                }}
                .chart-link:hover {{
                    transform: translateY(-2px);
                }}
                .footer {{
                    text-align: center;
                    padding: 20px;
                    background: #f8f9fa;
                    color: #6c757d;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 IAM Roles Analysis Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="summary">
                    <div class="summary-card">
                        <div class="summary-number">{total_accounts}</div>
                        <div class="summary-label">Accounts Analyzed</div>
                    </div>
                    <div class="summary-card">
                        <div class="summary-number">{total_roles}</div>
                        <div class="summary-label">Non-4-Hour Roles</div>
                    </div>
                    <div class="summary-card">
                        <div class="summary-number">{avg_roles_per_account:.1f}</div>
                        <div class="summary-label">Avg Roles/Account</div>
                    </div>
                    <div class="summary-card">
                        <div class="summary-number">{risk_stats.get('High Risk', 0)}</div>
                        <div class="summary-label">High Risk Roles</div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🎯 Key Findings</h2>
                    <div class="charts-grid">
                        <div>
                            <h3>Duration Distribution</h3>
                            <ul>
        """
        
        for duration, count in duration_stats.items():
            html_content += f"<li>{duration}: {count} roles ({count/total_roles*100:.1f}%)</li>"
        
        html_content += """
                            </ul>
                        </div>
                        <div>
                            <h3>Risk Level Distribution</h3>
                            <ul>
        """
        
        for risk, count in risk_stats.items():
            html_content += f"<li>{risk}: {count} roles ({count/total_roles*100:.1f}%)</li>"
        
        html_content += f"""
                            </ul>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🏆 Top 10 Accounts with Most Non-4-Hour Roles</h2>
                    <table class="table">
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Account Name</th>
                                <th>Role Count</th>
                                <th>Percentage</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        
        for i, (account, count) in enumerate(top_accounts.items(), 1):
            percentage = (count / total_roles) * 100
            html_content += f"""
                            <tr>
                                <td>{i}</td>
                                <td>{account}</td>
                                <td>{count}</td>
                                <td>{percentage:.1f}%</td>
                            </tr>
            """
        
        html_content += """
                        </tbody>
                    </table>
                </div>
                
                <div class="section">
                    <h2>📊 Interactive Charts</h2>
                    <div class="charts-grid">
                        <a href="interactive_roles_per_account.html" class="chart-link">
                            <h3>📈 Roles per Account</h3>
                            <p>Interactive bar chart showing role distribution</p>
                        </a>
                        <a href="interactive_sunburst.html" class="chart-link">
                            <h3>🌅 Sunburst Chart</h3>
                            <p>Hierarchical view of risk levels and durations</p>
                        </a>
                        <a href="interactive_treemap.html" class="chart-link">
                            <h3>🗺️ Treemap</h3>
                            <p>Account distribution by risk level</p>
                        </a>
                        <a href="interactive_scatter.html" class="chart-link">
                            <h3>🔍 Scatter Plot</h3>
                            <p>Roles vs Duration analysis</p>
                        </a>
                        <a href="interactive_dashboard.html" class="chart-link">
                            <h3>📊 Dashboard</h3>
                            <p>Combined analysis dashboard</p>
                        </a>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📁 Generated Files</h2>
                    <ul>
                        <li><strong>CSV Files:</strong> pivot_account_vs_duration.csv, pivot_account_vs_risk.csv, summary_by_account.csv</li>
                        <li><strong>Static Charts:</strong> roles_per_account.png, duration_analysis.png, heatmap_account_duration.png</li>
                        <li><strong>Interactive Charts:</strong> All HTML files for interactive visualization</li>
                    </ul>
                </div>
                
                <div class="footer">
                    <p>📊 IAM Roles Analysis Report | Generated by AWS Security Analytics Tool</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(f"{self.output_dir}/analysis_report.html", 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Detailed report saved: {self.output_dir}/analysis_report.html")
    
    def run_full_analysis(self):
        """Run the complete analysis workflow"""
        if self.df.empty:
            print("✅ No data to analyze - all roles have standard 4-hour session duration!")
            return
        
        print("\n🚀 Starting Full IAM Roles Analysis...")
        print("="*60)
        
        # Generate summary statistics
        self.generate_summary_stats()
        
        # Create pivot tables
        self.create_pivot_tables()
        
        # Create matplotlib charts
        self.create_matplotlib_charts()
        
        # Create plotly charts
        self.create_plotly_charts()
        
        # Create detailed report
        self.create_detailed_report()
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE!")
        print("="*60)
        print(f"📁 All files saved in: {self.output_dir}/")
        print(f"🌐 Open '{self.output_dir}/analysis_report.html' to view the complete report")
        print(f"📊 Interactive charts available in the same directory")

def main():
    """Main function"""
    print("🔬 IAM Roles Data Analyzer & Visualizer")
    print("="*50)
    
    # Check if CSV file exists
    csv_file = "iam_roles_report.csv"
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    try:
        # Initialize analyzer
        analyzer = IAMDataAnalyzer(csv_file)
        
        # Run full analysis
        analyzer.run_full_analysis()
        
    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()