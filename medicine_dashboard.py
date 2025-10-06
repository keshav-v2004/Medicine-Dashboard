
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# PART 1: DATA PREPARATION AND NORMALIZATION
# =====================================================================

class MedicineDataPrep:
    """Handles all data loading, cleaning, and normalization"""
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df_main = None
        self.df_uses = None
        self.df_side_effects = None
        self.df_ingredients = None
        
    def load_and_clean(self):
        """Load CSV and perform initial cleaning"""
        print("Loading data...")
        
        # Load CSV
        self.df_main = pd.read_csv(self.csv_path)
        
        # Strip whitespace from all string columns
        str_cols = self.df_main.select_dtypes(include=['object']).columns
        for col in str_cols:
            self.df_main[col] = self.df_main[col].str.strip()
        
        # Ensure review columns are numeric
        review_cols = ['Excellent Review %', 'Average Review %', 'Poor Review %']
        for col in review_cols:
            # Remove % sign if present
            if self.df_main[col].dtype == 'object':
                self.df_main[col] = self.df_main[col].str.replace('%', '').astype(float)
        
        # Add review sum validation
        self.df_main['ReviewSum'] = (
            self.df_main['Excellent Review %'] + 
            self.df_main['Average Review %'] + 
            self.df_main['Poor Review %']
        )
        self.df_main['IsValid'] = np.where(
            np.abs(self.df_main['ReviewSum'] - 100) < 0.1, 
            'Valid', 
            'Invalid'
        )
        
        # Report data quality
        invalid_count = (self.df_main['IsValid'] == 'Invalid').sum()
        print(f"Data loaded: {len(self.df_main)} medicines")
        print(f"Invalid review sums: {invalid_count}")
        
        return self
    
    def normalize_uses(self):
        """Create normalized uses table"""
        print("Normalizing uses...")
        
        uses_list = []
        for _, row in self.df_main.iterrows():
            medicine = row['Medicine Name']
            uses = str(row['Uses']).split(',')
            
            for use in uses:
                use_clean = use.strip().title()
                if use_clean and use_clean.lower() != 'nan':
                    uses_list.append({
                        'Medicine Name': medicine,
                        'Indication': use_clean
                    })
        
        self.df_uses = pd.DataFrame(uses_list)
        print(f"Created {len(self.df_uses)} medicine-indication pairs")
        
        return self
    
    def normalize_side_effects(self):
        """Create normalized side effects table"""
        print("Normalizing side effects...")
        
        effects_list = []
        for _, row in self.df_main.iterrows():
            medicine = row['Medicine Name']
            effects = str(row['Side_effects']).split(',')
            
            for effect in effects:
                effect_clean = effect.strip().title()
                if effect_clean and effect_clean.lower() != 'nan':
                    effects_list.append({
                        'Medicine Name': medicine,
                        'Side Effect': effect_clean
                    })
        
        self.df_side_effects = pd.DataFrame(effects_list)
        print(f"Created {len(self.df_side_effects)} medicine-side effect pairs")
        
        return self
    
    def normalize_ingredients(self):
        """Parse and normalize ingredients"""
        print("Normalizing ingredients...")
        
        ingredients_list = []
        for _, row in self.df_main.iterrows():
            medicine = row['Medicine Name']
            composition = str(row['Composition'])
            
            # Split by + or comma
            parts = composition.replace('+', ',').split(',')
            
            for part in parts:
                part = part.strip()
                if not part or part.lower() == 'nan':
                    continue
                
                # Try to extract ingredient name and strength
                tokens = part.split()
                if len(tokens) > 1:
                    # Last token might be strength (contains numbers)
                    if any(char.isdigit() for char in tokens[-1]):
                        ingredient = ' '.join(tokens[:-1]).title()
                        strength = tokens[-1]
                    else:
                        ingredient = part.title()
                        strength = None
                else:
                    ingredient = part.title()
                    strength = None
                
                ingredients_list.append({
                    'Medicine Name': medicine,
                    'Ingredient Name': ingredient,
                    'Strength': strength
                })
        
        self.df_ingredients = pd.DataFrame(ingredients_list)
        print(f"Created {len(self.df_ingredients)} medicine-ingredient pairs")
        
        return self
    
    def compute_derived_metrics(self):
        """Add computed columns to main dataframe"""
        print("Computing derived metrics...")
        
        # Count ingredients per medicine
        ingredient_counts = self.df_ingredients.groupby('Medicine Name').size()
        self.df_main['Ingredient Count'] = self.df_main['Medicine Name'].map(ingredient_counts).fillna(0)
        
        # Classify as single or combination
        self.df_main['Medicine Type'] = np.where(
            self.df_main['Ingredient Count'] == 1,
            'Single Ingredient',
            np.where(self.df_main['Ingredient Count'] > 1, 'Combination', 'Unknown')
        )
        
        # Calculate review entropy (Shannon entropy)
        def calc_entropy(row):
            exc = row['Excellent Review %'] / 100
            avg = row['Average Review %'] / 100
            poor = row['Poor Review %'] / 100
            
            values = [exc, avg, poor]
            entropy = 0
            for v in values:
                if v > 0:
                    entropy -= v * np.log(v)
            return entropy
        
        self.df_main['Review Entropy'] = self.df_main.apply(calc_entropy, axis=1)
        
        # Calculate polarization (standard deviation)
        def calc_polarization(row):
            values = [
                row['Excellent Review %'],
                row['Average Review %'],
                row['Poor Review %']
            ]
            return np.std(values)
        
        self.df_main['Review Polarization'] = self.df_main.apply(calc_polarization, axis=1)
        
        print("Derived metrics computed")
        
        return self
    
    def get_all_tables(self):
        """Return all normalized tables"""
        return {
            'main': self.df_main,
            'uses': self.df_uses,
            'side_effects': self.df_side_effects,
            'ingredients': self.df_ingredients
        }

# =====================================================================
# PART 2: ANALYTICS AND KPI CALCULATIONS
# =====================================================================

class MedicineAnalytics:
    """Calculate all KPIs and metrics"""
    
    def __init__(self, tables):
        self.main = tables['main']
        self.uses = tables['uses']
        self.side_effects = tables['side_effects']
        self.ingredients = tables['ingredients']
        self.kpis = {}
    
    def calculate_kpis(self):
        """Calculate all key performance indicators"""
        
        # Basic counts
        self.kpis['total_medicines'] = len(self.main)
        self.kpis['unique_manufacturers'] = self.main['Manufacturer'].nunique()
        self.kpis['unique_ingredients'] = self.ingredients['Ingredient Name'].nunique()
        self.kpis['unique_indications'] = self.uses['Indication'].nunique()
        self.kpis['unique_side_effects'] = self.side_effects['Side Effect'].nunique()
        
        # Average reviews
        self.kpis['avg_excellent'] = self.main['Excellent Review %'].mean()
        self.kpis['avg_average'] = self.main['Average Review %'].mean()
        self.kpis['avg_poor'] = self.main['Poor Review %'].mean()
        
        # Median reviews
        self.kpis['median_excellent'] = self.main['Excellent Review %'].median()
        self.kpis['median_average'] = self.main['Average Review %'].median()
        self.kpis['median_poor'] = self.main['Poor Review %'].median()
        
        # Data quality
        self.kpis['invalid_count'] = (self.main['IsValid'] == 'Invalid').sum()
        self.kpis['data_quality_pct'] = (1 - self.kpis['invalid_count'] / self.kpis['total_medicines']) * 100
        
        # Polarization
        self.kpis['avg_entropy'] = self.main['Review Entropy'].mean()
        self.kpis['avg_polarization'] = self.main['Review Polarization'].mean()
        self.kpis['highly_polarized_pct'] = (self.main['Review Polarization'] > 30).sum() / len(self.main) * 100
        
        return self.kpis
    
    def get_manufacturer_performance(self):
        """Aggregate metrics by manufacturer"""
        manufacturer_stats = self.main.groupby('Manufacturer').agg({
            'Medicine Name': 'count',
            'Excellent Review %': 'mean',
            'Average Review %': 'mean',
            'Poor Review %': 'mean',
            'Review Polarization': 'mean'
        }).round(1)
        
        manufacturer_stats.columns = [
            'Medicine Count', 'Avg Excellent %', 'Avg Average %', 
            'Avg Poor %', 'Avg Polarization'
        ]
        
        return manufacturer_stats.sort_values('Avg Excellent %', ascending=False)
    
    def get_ingredient_performance(self):
        """Analyze performance by ingredient"""
        # Merge ingredients with main table
        ing_merged = self.ingredients.merge(
            self.main[['Medicine Name', 'Excellent Review %', 'Poor Review %']], 
            on='Medicine Name'
        )
        
        ingredient_stats = ing_merged.groupby('Ingredient Name').agg({
            'Medicine Name': 'count',
            'Excellent Review %': 'mean',
            'Poor Review %': 'mean'
        }).round(1)
        
        ingredient_stats.columns = ['Medicine Count', 'Avg Excellent %', 'Avg Poor %']
        
        return ingredient_stats.sort_values('Avg Excellent %', ascending=False)
    
    def get_indication_analysis(self):
        """Analyze most common indications and their performance"""
        use_merged = self.uses.merge(
            self.main[['Medicine Name', 'Excellent Review %', 'Poor Review %']], 
            on='Medicine Name'
        )
        
        indication_stats = use_merged.groupby('Indication').agg({
            'Medicine Name': 'count',
            'Excellent Review %': 'mean',
            'Poor Review %': 'mean'
        }).round(1)
        
        indication_stats.columns = ['Frequency', 'Avg Excellent %', 'Avg Poor %']
        
        return indication_stats.sort_values('Frequency', ascending=False)
    
    def get_side_effect_analysis(self):
        """Analyze side effects"""
        effect_merged = self.side_effects.merge(
            self.main[['Medicine Name', 'Poor Review %']], 
            on='Medicine Name'
        )
        
        effect_stats = effect_merged.groupby('Side Effect').agg({
            'Medicine Name': 'count',
            'Poor Review %': ['mean', 'median']
        }).round(1)
        
        effect_stats.columns = ['Frequency', 'Avg Poor %', 'Median Poor %']
        effect_stats['Prevalence %'] = (effect_stats['Frequency'] / len(self.main) * 100).round(1)
        
        return effect_stats.sort_values('Frequency', ascending=False)
    
    def get_single_vs_combo(self):
        """Compare single vs combination medicines"""
        comparison = self.main.groupby('Medicine Type').agg({
            'Medicine Name': 'count',
            'Excellent Review %': ['mean', 'median'],
            'Poor Review %': ['mean', 'median'],
            'Review Polarization': 'mean'
        }).round(1)
        
        return comparison

# =====================================================================
# PART 3: VISUALIZATION FUNCTIONS
# =====================================================================

def create_kpi_cards(kpis):
    """Create KPI card visualizations"""
    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=(
            f"Total Medicines<br><b>{kpis['total_medicines']}</b>",
            f"Manufacturers<br><b>{kpis['unique_manufacturers']}</b>",
            f"Ingredients<br><b>{kpis['unique_ingredients']}</b>",
            f"Avg Excellent %<br><b>{kpis['avg_excellent']:.1f}%</b>"
        ),
        specs=[[{"type": "indicator"}, {"type": "indicator"}, 
                {"type": "indicator"}, {"type": "indicator"}]]
    )
    
    fig.update_layout(height=200, showlegend=False)
    return fig

def create_manufacturer_stacked_bar(df_manufacturer, top_n=15):
    """Stacked bar chart of manufacturer performance - top N only"""
    
    # Limit to top N by Excellent %
    df_top = df_manufacturer.nlargest(top_n, 'Avg Excellent %')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Excellent',
        y=df_top.index,
        x=df_top['Avg Excellent %'],
        orientation='h',
        marker_color='#34A853'
    ))
    
    fig.add_trace(go.Bar(
        name='Average',
        y=df_top.index,
        x=df_top['Avg Average %'],
        orientation='h',
        marker_color='#FBBC04'
    ))
    
    fig.add_trace(go.Bar(
        name='Poor',
        y=df_top.index,
        x=df_top['Avg Poor %'],
        orientation='h',
        marker_color='#EA4335'
    ))
    
    fig.update_layout(
        barmode='stack',
        title=f'Review Distribution by Top {top_n} Manufacturers',
        xaxis_title='Percentage',
        yaxis_title='Manufacturer',
        height=450,  # Fixed, manageable height
        showlegend=True,
        yaxis={
            'automargin': True,
            'autorange': 'reversed'  # Best performers on top
        }
    )
    
    return fig

def create_top_medicines_chart(df_main, metric='Excellent Review %', top_n=15):
    """Bar chart of top medicines"""
    df_sorted = df_main.nlargest(top_n, metric)
    
    color = '#34A853' if 'Excellent' in metric else '#EA4335'
    
    fig = go.Figure(go.Bar(
        y=df_sorted['Medicine Name'],
        x=df_sorted[metric],
        orientation='h',
        marker_color=color,
        text=df_sorted[metric].round(1),
        textposition='auto'
    ))
    
    fig.update_layout(
        title=f'Top {top_n} Medicines by {metric}',
        xaxis_title=metric,
        yaxis_title='Medicine',
        height=max(400, top_n * 30),
        yaxis={'autorange': 'reversed'}
    )
    
    return fig

def create_ingredient_leaderboard(df_ingredient_stats, top_n=20):
    """Bar chart of ingredient performance"""
    df_top = df_ingredient_stats.nlargest(top_n, 'Avg Excellent %')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_top.index,
        x=df_top['Avg Excellent %'],
        orientation='h',
        marker_color='#34A853',
        name='Avg Excellent %',
        text=df_top['Avg Excellent %'].round(1),
        textposition='auto'
    ))
    
    # Add medicine count as annotation
    fig.add_trace(go.Scatter(
        y=df_top.index,
        x=[105] * len(df_top),  # Position to the right
        mode='text',
        text=df_top['Medicine Count'].astype(str) + ' meds',
        textposition='middle right',
        showlegend=False
    ))
    
    fig.update_layout(
        title=f'Top {top_n} Ingredients by Performance',
        xaxis_title='Average Excellent %',
        yaxis_title='Ingredient',
        height=max(400, top_n * 25),
        yaxis={'autorange': 'reversed'}
    )
    
    return fig

def create_single_vs_combo_chart(df_comparison):
    """Comparison chart for single vs combination medicines"""
    fig = go.Figure()
    
    categories = df_comparison.index.tolist()
    
    fig.add_trace(go.Bar(
        name='Median Excellent %',
        x=categories,
        y=df_comparison[('Excellent Review %', 'median')],
        marker_color='#34A853'
    ))
    
    fig.add_trace(go.Bar(
        name='Median Average %',
        x=categories,
        y=df_comparison[('Excellent Review %', 'median')],
        marker_color='#FBBC04'
    ))
    
    fig.add_trace(go.Bar(
        name='Median Poor %',
        x=categories,
        y=df_comparison[('Poor Review %', 'median')],
        marker_color='#EA4335'
    ))
    
    fig.update_layout(
        title='Single vs Combination Medicine Reviews',
        xaxis_title='Medicine Type',
        yaxis_title='Percentage',
        barmode='group',
        height=400
    )
    
    return fig

def create_indication_chart(df_indication, top_n=15):
    """Combo chart for indication frequency and performance"""
    df_top = df_indication.nlargest(top_n, 'Frequency')
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            x=df_top.index,
            y=df_top['Frequency'],
            name='Frequency',
            marker_color='#4285F4'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_top.index,
            y=df_top['Avg Excellent %'],
            name='Avg Excellent %',
            marker_color='#34A853',
            mode='lines+markers'
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title=f'Top {top_n} Indications: Frequency & Performance',
        xaxis_title='Indication',
        height=500,
        xaxis_tickangle=-45
    )
    
    fig.update_yaxes(title_text="Frequency", secondary_y=False)
    fig.update_yaxes(title_text="Avg Excellent %", secondary_y=True)
    
    return fig

def create_side_effect_scatter(df_side_effect):
    """Scatter plot of side effect prevalence vs poor reviews"""
    fig = px.scatter(
        df_side_effect.reset_index(),
        x='Prevalence %',
        y='Median Poor %',
        size='Frequency',
        hover_name='Side Effect',
        labels={
            'Prevalence %': 'Prevalence (% of medicines)',
            'Median Poor %': 'Median Poor Review %',
            'Frequency': 'Number of Medicines'
        },
        title='Side Effect Risk Analysis',
        color='Median Poor %',
        color_continuous_scale='Reds'
    )
    
    # Add quadrant lines (average values)
    avg_prevalence = df_side_effect['Prevalence %'].mean()
    avg_poor = df_side_effect['Median Poor %'].mean()
    
    fig.add_hline(y=avg_poor, line_dash="dash", line_color="gray", 
                  annotation_text="Avg Poor %")
    fig.add_vline(x=avg_prevalence, line_dash="dash", line_color="gray",
                  annotation_text="Avg Prevalence")
    
    fig.update_layout(height=600)
    
    return fig

def create_polarization_chart(df_main):
    """Distribution of polarization scores"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df_main['Review Polarization'],
        nbinsx=30,
        marker_color='#4285F4',
        name='Distribution'
    ))
    
    fig.add_vline(
        x=df_main['Review Polarization'].mean(),
        line_dash="dash",
        line_color="red",
        annotation_text="Mean"
    )
    
    fig.update_layout(
        title='Distribution of Review Polarization',
        xaxis_title='Polarization Score (Std Dev)',
        yaxis_title='Count of Medicines',
        height=400
    )
    
    return fig

# ===========================
# Additional Visualizations
# ===========================

def create_medicine_type_pie(df_main):
    """Pie chart of Single vs Combination medicines"""
    type_counts = df_main['Medicine Type'].value_counts()
    fig = px.pie(
        names=type_counts.index,
        values=type_counts.values,
        title="Medicine Type Distribution",
        hole=0.4  # makes it a donut chart
    )
    fig.update_traces(textinfo='percent+label', pull=[0.05, 0.05])
    return fig

def create_review_distribution_pie(df_main):
    """Donut chart of overall review distribution"""
    review_totals = [
        df_main['Excellent Review %'].sum(),
        df_main['Average Review %'].sum(),
        df_main['Poor Review %'].sum()
    ]
    labels = ['Excellent', 'Average', 'Poor']
    fig = go.Figure(go.Pie(
        labels=labels,
        values=review_totals,
        hole=0.4,
        textinfo='percent+label',
        marker_colors=['#34A853', '#FBBC04', '#EA4335']
    ))
    fig.update_layout(title="Overall Review Distribution")
    return fig

def create_top_manufacturers_chart(df_main, top_n=10):
    """Bar chart of top manufacturers by medicine count"""
    top_manuf = df_main['Manufacturer'].value_counts().nlargest(top_n)
    fig = go.Figure(go.Bar(
        x=top_manuf.index,
        y=top_manuf.values,
        marker_color='#4285F4',
        text=top_manuf.values,
        textposition='auto'
    ))
    fig.update_layout(title=f"Top {top_n} Manufacturers by Medicine Count",
                      xaxis_title='Manufacturer', yaxis_title='Number of Medicines')
    return fig

def create_top_ingredients_chart(df_ingredients, top_n=15):
    """Bar chart of top ingredients by medicine usage"""
    top_ingredients = df_ingredients['Ingredient Name'].value_counts().nlargest(top_n)
    fig = go.Figure(go.Bar(
        x=top_ingredients.values,
        y=top_ingredients.index,
        orientation='h',
        marker_color='#FBBC04',
        text=top_ingredients.values,
        textposition='auto'
    ))
    fig.update_layout(title=f"Top {top_n} Ingredients by Medicine Count",
                      xaxis_title='Number of Medicines', yaxis_title='Ingredient',
                      yaxis={'autorange':'reversed'})
    return fig

def create_side_effect_pie(df_side_effect, top_n=10):
    """Pie chart for most common side effects"""
    top_effects = df_side_effect.nlargest(top_n, 'Frequency')
    fig = px.pie(top_effects, names=top_effects.index, values='Frequency',
                 title=f"Top {top_n} Side Effects", hole=0.3)
    fig.update_traces(textinfo='percent+label')
    return fig

def create_indication_bar_chart(df_indication, top_n=15):
    """Bar chart of top indications"""
    df_top = df_indication.nlargest(top_n, 'Frequency')
    fig = go.Figure(go.Bar(
        x=df_top['Frequency'],
        y=df_top.index,
        orientation='h',
        marker_color='#34A853',
        text=df_top['Frequency'],
        textposition='auto'
    ))
    fig.update_layout(title=f"Top {top_n} Indications by Frequency",
                      xaxis_title='Frequency', yaxis_title='Indication',
                      yaxis={'autorange':'reversed'})
    return fig

def create_ingredient_treemap(df_ingredients):
    """Treemap showing ingredient composition across medicines"""
    ingredient_counts = df_ingredients['Ingredient Name'].value_counts().reset_index()
    ingredient_counts.columns = ['Ingredient', 'Medicine Count']
    
    fig = px.treemap(
        ingredient_counts,
        path=['Ingredient'],
        values='Medicine Count',
        title='Ingredient Composition Treemap',
        color='Medicine Count',
        color_continuous_scale='Blues'
    )
    
    fig.update_traces(textinfo='label+value')
    return fig

def create_polarization_boxplot(df_main):
    """Box plot for review polarization scores"""
    fig = px.box(
        df_main,
        y='Review Polarization',
        points='all',
        color_discrete_sequence=['#EA4335'],
        title='Review Polarization Box Plot'
    )
    fig.update_layout(yaxis_title='Polarization Score (Std Dev)')
    return fig

def create_medicine_type_donut(df_main):
    """Donut chart showing proportion of Single vs Combination medicines"""
    type_counts = df_main['Medicine Type'].value_counts().reset_index()
    type_counts.columns = ['Medicine Type', 'Count']

    fig = px.pie(
        type_counts,
        names='Medicine Type',
        values='Count',
        hole=0.4,
        title='Medicine Type Distribution (Donut Chart)',
        color='Medicine Type',
        color_discrete_map={'Single Ingredient':'#34A853','Combination':'#4285F4','Unknown':'#FBBC04'}
    )
    fig.update_traces(textinfo='percent+label')
    return fig

def create_review_distribution_donut(df_main):
    """Donut chart showing overall review percentages across all medicines"""
    review_sums = {
        'Excellent': df_main['Excellent Review %'].mean(),
        'Average': df_main['Average Review %'].mean(),
        'Poor': df_main['Poor Review %'].mean()
    }
    review_df = pd.DataFrame(list(review_sums.items()), columns=['Review','Percentage'])

    fig = px.pie(
        review_df,
        names='Review',
        values='Percentage',
        hole=0.4,
        title='Overall Review Distribution (Donut Chart)',
        color='Review',
        color_discrete_map={'Excellent':'#34A853','Average':'#FBBC04','Poor':'#EA4335'}
    )
    fig.update_traces(textinfo='percent+label')
    return fig

def create_top_indications_bar(df_indication, top_n=15):
    """Bar chart of top indications colored by average Excellent %"""
    df_top = df_indication.nlargest(top_n,'Frequency')
    colors = px.colors.sequential.Blues_r
    fig = px.bar(
        df_top,
        x='Frequency',
        y=df_top.index,
        orientation='h',
        text='Avg Excellent %',
        color='Avg Excellent %',
        color_continuous_scale=colors,
        title=f'Top {top_n} Indications Colored by Avg Excellent %'
    )
    fig.update_layout(yaxis={'autorange':'reversed'})
    return fig

def create_side_effect_bar(df_side_effect, top_n=15):
    """Bar chart of top side effects colored by Median Poor %"""
    df_top = df_side_effect.nlargest(top_n,'Frequency')
    colors = px.colors.sequential.Reds
    fig = px.bar(
        df_top,
        x='Frequency',
        y=df_top.index,
        orientation='h',
        text='Median Poor %',
        color='Median Poor %',
        color_continuous_scale=colors,
        title=f'Top {top_n} Side Effects Colored by Median Poor %'
    )
    fig.update_layout(yaxis={'autorange':'reversed'})
    return fig




# =====================================================================
# PART 4: DASH APPLICATION
# =====================================================================

def create_dashboard(tables, analytics):
    """Enhanced Dash dashboard with a sidebar theme and interactive charts"""

    # Use the FLATLY theme for a clean base, plus Font Awesome icons
    FA = "https://use.fontawesome.com/releases/v5.15.4/css/all.css"
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY, FA])
    
    # --- Data (same as your original code) ---
    df_main = tables['main']
    df_uses = tables['uses']
    df_side_effects = tables['side_effects']
    df_ingredients = tables['ingredients']
    
    kpis = analytics.calculate_kpis()
    df_manufacturer = analytics.get_manufacturer_performance()
    df_comparison = analytics.get_single_vs_combo()
 
    # ==============================================================================
    # 1. LAYOUT DEFINITION
    # ==============================================================================

    # --- SIDEBAR ---
    sidebar = html.Div(
        [
            html.H2("💊 MedReview", className="text-primary text-center"),
            html.P("Dashboard & Analytics", className="text-center text-muted"),
            html.Hr(),
            
            # Filters are now in the sidebar
            html.H4("Filters", className="mt-4"),
            dbc.Label("Manufacturer:"),
            dcc.Dropdown(
                id='manufacturer-filter',
                options=[{'label': m, 'value': m} for m in sorted(df_main['Manufacturer'].unique())],
                value=[], multi=True, placeholder="Select Manufacturer(s)..."
            ),
            dbc.Label("Medicine Type:", className="mt-3"),
            dcc.Dropdown(
                id='type-filter',
                options=[{'label': t, 'value': t} for t in df_main['Medicine Type'].unique()],
                value=[], multi=True, placeholder="Select Type(s)..."
            ),
            dbc.Label("Min Excellent %:", className="mt-3"),
            dcc.Slider(
                id='excellent-slider', min=0, max=100, step=5, value=0,
                marks={i: str(i) for i in range(0, 101, 20)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Hr(),
            dbc.Label("Search Medicine Name:", className="mt-3"),
            dbc.Input(id='medicine-search', placeholder='e.g., Aspirin...', debounce=True),
            
            dbc.Label("Search Ingredient:", className="mt-3"),
            dbc.Input(id='ingredient-search', placeholder='e.g., Paracetamol...', debounce=True),
            
            dbc.Label("Search Uses/Indication:", className="mt-3"),
            dbc.Input(id='indication-search', placeholder='e.g., headache...', debounce=True),
        ],
        className="sidebar",
    )

    # --- MAIN CONTENT ---
    content = html.Div(
        [
            # --- KPI Cards using the new theme ---
            dbc.Row([
                dbc.Col(html.Div([
                    html.I(className="fas fa-pills card-icon"),
                    html.Div([
                        html.P("Total Medicines", className="card-title"),
                        html.H3(kpis['total_medicines'], className="card-value"),
                    ])
                ], className="kpi-card"), lg=3, md=6),
                dbc.Col(html.Div([
                    html.I(className="fas fa-industry card-icon"),
                    html.Div([
                        html.P("Unique Manufacturers", className="card-title"),
                        html.H3(kpis['unique_manufacturers'], className="card-value"),
                    ])
                ], className="kpi-card"), lg=3, md=6),
                dbc.Col(html.Div([
                    html.I(className="fas fa-flask card-icon"),
                    html.Div([
                        html.P("Unique Ingredients", className="card-title"),
                        html.H3(kpis['unique_ingredients'], className="card-value"),
                    ])
                ], className="kpi-card"), lg=3, md=6),
                dbc.Col(html.Div([
                    html.I(className="fas fa-check-circle card-icon"),
                    html.Div([
                        html.P("Avg. Excellent %", className="card-title"),
                        html.H3(f"{kpis['avg_excellent']:.1f}%", className="card-value"),
                    ])
                ], className="kpi-card"), lg=3, md=6),
            ]),
            
            # --- Charts ---
            # All chart cards now use the new `chart-card` class from style.css
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardHeader("Distribution by Medicine Type"), dbc.CardBody(dcc.Loading(dcc.Graph(id='medicine-type-donut')))], className="chart-card"), lg=6, className="mb-4"),
                dbc.Col(dbc.Card([dbc.CardHeader("Overall Review Distribution"), dbc.CardBody(dcc.Loading(dcc.Graph(id='review-distribution-donut')))], className="chart-card"), lg=6, className="mb-4"),
                dbc.Col(dbc.Card([dbc.CardHeader("Top 10 Manufacturers by Medicine Count"), dbc.CardBody(dcc.Loading(dcc.Graph(id='top-manufacturers')))], className="chart-card"), lg=6, className="mb-4"),
                dbc.Col(dbc.Card([dbc.CardHeader("Top 15 Most Common Ingredients"), dbc.CardBody(dcc.Loading(dcc.Graph(id='top-ingredients')))], className="chart-card"), lg=6, className="mb-4"),
                dbc.Col(dbc.Card([dbc.CardHeader("Top 15 Common Uses/Indications"), dbc.CardBody(dcc.Loading(dcc.Graph(id='top-indications')))], className="chart-card"), lg=6, className="mb-4"),
                dbc.Col(dbc.Card([dbc.CardHeader("Top 15 Reported Side Effects"), dbc.CardBody(dcc.Loading(dcc.Graph(id='side-effects-bar')))], className="chart-card"), lg=6, className="mb-4"),
                dbc.Col(dbc.Card([dbc.CardHeader("Review Distribution by Manufacturer"), dbc.CardBody(dcc.Loading(dcc.Graph(id='manufacturer-chart')))], className="chart-card"), lg=6, className="mb-4"),
                dbc.Col(dbc.Card([dbc.CardHeader("Single vs. Combination Medicine Analysis"), dbc.CardBody(dcc.Loading(dcc.Graph(id='single-combo-chart')))], className="chart-card"), lg=6, className="mb-4"),
                dbc.Col(dbc.Card([dbc.CardHeader("Review Polarization Distribution"), dbc.CardBody(dcc.Loading(dcc.Graph(id='polarization-chart')))], className="chart-card"), lg=6, className="mb-4"),
                dbc.Col(dbc.Card([dbc.CardHeader("Polarization by Medicine Type (Box Plot)"), dbc.CardBody(dcc.Loading(dcc.Graph(id='polarization-boxplot')))], className="chart-card"), lg=6, className="mb-4"),
                dbc.Col(dbc.Card([dbc.CardHeader("Ingredient Composition Treemap"), dbc.CardBody(dcc.Loading(dcc.Graph(id='ingredient-treemap')))], className="chart-card"), width=12, className="mb-4"),
            ]),
            
            # --- Data Table ---
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Detailed Medicine Data"),
                    dbc.CardBody(dcc.Loading(dash_table.DataTable(
                        id='medicine-table',
                        columns=[{'name': c, 'id': c} for c in df_main.columns],
                        page_size=15, sort_action='native',
                        style_cell={'textAlign': 'left', 'padding': '10px'},
                        style_header={'fontWeight': 'bold'},
                        style_table={'overflowX': 'auto'}
                    )))
                ], className="chart-card"), width=12, className="mb-5")
            ])
        ],
        className="content",
    )

    # --- Final Layout ---
    app.layout = html.Div([sidebar, content])

    # ==============================================================================
    # 2. CALLBACKS
    # ==============================================================================
    
    # Note: Your callback logic is excellent and complex. I've adapted it to the
    # new filter values (e.g., using empty lists `[]` for multi-select defaults).
    # I have removed the 'filter-results-count' Output as it was part of the old layout.
    @app.callback(
        [Output('medicine-type-donut','figure'),
         Output('review-distribution-donut','figure'),
         Output('top-manufacturers','figure'),
         Output('top-ingredients','figure'),
         Output('top-indications','figure'),
         Output('side-effects-bar','figure'),
         Output('manufacturer-chart','figure'),
         Output('single-combo-chart','figure'),
         Output('polarization-chart','figure'),
         Output('polarization-boxplot','figure'),
         Output('ingredient-treemap','figure'),
         Output('medicine-table','data')],
        
        [Input('manufacturer-filter','value'),
         Input('type-filter','value'),
         Input('excellent-slider','value'),
         Input('medicine-search','value'),
         Input('ingredient-search','value'),
         Input('indication-search','value')]
    )
    def update_charts(manufacturer_filter, type_filter, excellent_min, 
                      medicine_search, ingredient_search, indication_search):
        
        filtered_df = df_main.copy()
        
        # --- Apply Filters ---
        if manufacturer_filter:
            filtered_df = filtered_df[filtered_df['Manufacturer'].isin(manufacturer_filter)]
        
        if type_filter:
            filtered_df = filtered_df[filtered_df['Medicine Type'].isin(type_filter)]
            
        if excellent_min > 0:
            filtered_df = filtered_df[filtered_df['Excellent Review %'] >= excellent_min]
        
        # --- Apply Search ---
        if medicine_search:
            filtered_df = filtered_df[filtered_df['Medicine Name'].str.contains(medicine_search, case=False, na=False)]
        
        if ingredient_search:
            matching_meds = df_ingredients[df_ingredients['Ingredient Name'].str.contains(ingredient_search, case=False, na=False)]['Medicine Name'].unique()
            filtered_df = filtered_df[filtered_df['Medicine Name'].isin(matching_meds)]
            
        if indication_search:
            matching_meds = df_uses[df_uses['Indication'].str.contains(indication_search, case=False, na=False)]['Medicine Name'].unique()
            filtered_df = filtered_df[filtered_df['Medicine Name'].isin(matching_meds)]
        
        # --- Handle Empty Results ---
        if filtered_df.empty:
            empty_fig = go.Figure().add_annotation(text="No results found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return [empty_fig] * 11 + [[]] # 11 empty figures, 1 empty list for the table data

        # --- Update Related Tables ---
        filtered_med_names = filtered_df['Medicine Name'].unique()
        filtered_ingredients = df_ingredients[df_ingredients['Medicine Name'].isin(filtered_med_names)]
        filtered_uses = df_uses[df_uses['Medicine Name'].isin(filtered_med_names)]
        filtered_side_effects = df_side_effects[df_side_effects['Medicine Name'].isin(filtered_med_names)]

        # --- Recalculate Analytics on Filtered Data ---
        analytics_filtered = analytics.__class__({ # Re-instantiate analytics with filtered data
            'main': filtered_df, 'uses': filtered_uses,
            'side_effects': filtered_side_effects, 'ingredients': filtered_ingredients
        })
        df_indication_filtered = analytics_filtered.get_indication_analysis()
        df_side_effect_filtered = analytics_filtered.get_side_effect_analysis()
        df_manufacturer_filtered = analytics_filtered.get_manufacturer_performance()
        df_comparison_filtered = analytics_filtered.get_single_vs_combo()
        

        return [
            create_medicine_type_donut(filtered_df),
            create_review_distribution_donut(filtered_df),
            create_top_manufacturers_chart(filtered_df, 10),
            create_top_ingredients_chart(filtered_ingredients, 15),
            create_top_indications_bar(df_indication_filtered, 15),
            create_side_effect_bar(df_side_effect_filtered, 15),
            create_manufacturer_stacked_bar(df_manufacturer_filtered),
            create_single_vs_combo_chart(df_comparison_filtered),
            create_polarization_chart(filtered_df),
            create_polarization_boxplot(filtered_df),
            create_ingredient_treemap(filtered_ingredients),
            filtered_df.to_dict('records')
        ]
        
    return app


# =====================================================================
# PART 5: MAIN EXECUTION
# =====================================================================

def main():
    """Main execution function"""
    
    # Step 1: Load and prepare data
    print("=" * 60)
    print("MEDICINE DASHBOARD - DATA PREPARATION")
    print("=" * 60)
    
    prep = MedicineDataPrep('Medicine_Details.csv')
    prep.load_and_clean()
    prep.normalize_uses()
    prep.normalize_side_effects()
    prep.normalize_ingredients()
    prep.compute_derived_metrics()
    
    tables = prep.get_all_tables()
    
    # Step 2: Calculate analytics
    print("\n" + "=" * 60)
    print("CALCULATING ANALYTICS")
    print("=" * 60)
    
    analytics = MedicineAnalytics(tables)
    kpis = analytics.calculate_kpis()
    
    # Print summary
    print("\nKEY METRICS:")
    print(f"Total Medicines: {kpis['total_medicines']}")
    print(f"Unique Manufacturers: {kpis['unique_manufacturers']}")
    print(f"Unique Ingredients: {kpis['unique_ingredients']}")
    print(f"Average Excellent %: {kpis['avg_excellent']:.1f}%")
    print(f"Data Quality: {kpis['data_quality_pct']:.1f}%")
    print(f"Highly Polarized Medicines: {kpis['highly_polarized_pct']:.1f}%")
    
    # Step 3: Create and run dashboard
    print("\n" + "=" * 60)
    print("LAUNCHING DASHBOARD")
    print("=" * 60)
    print("Dashboard will open at: http://127.0.0.1:8050/")
    print("Press Ctrl+C to stop the server")
    
    app = create_dashboard(tables, analytics)
    # app.run(port = 8050)
    return app

if __name__ == '__main__':
    app = main()
    app.run(debug=False, host="0.0.0.0", port=8050)
else:
    # For deployment - Gunicorn will use this
    app = main()
    server = app.server




# =====================================================================
# PART 6: EXPORT FUNCTIONS (OPTIONAL)
# =====================================================================

def export_to_excel(tables, analytics, filename='medicine_analysis.xlsx'):
    """Export all analysis to Excel with multiple sheets"""
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Main data
        tables['main'].to_excel(writer, sheet_name='Medicines', index=False)
        
        # Normalized tables
        tables['uses'].to_excel(writer, sheet_name='Uses', index=False)
        tables['side_effects'].to_excel(writer, sheet_name='Side Effects', index=False)
        tables['ingredients'].to_excel(writer, sheet_name='Ingredients', index=False)
        
        # Analytics
        analytics.get_manufacturer_performance().to_excel(writer, sheet_name='Manufacturer Analysis')
        analytics.get_ingredient_performance().to_excel(writer, sheet_name='Ingredient Analysis')
        analytics.get_indication_analysis().to_excel(writer, sheet_name='Indication Analysis')
        analytics.get_side_effect_analysis().to_excel(writer, sheet_name='Side Effect Analysis')
        analytics.get_single_vs_combo().to_excel(writer, sheet_name='Single vs Combo')
        
        # KPIs
        kpis_df = pd.DataFrame.from_dict(analytics.calculate_kpis(), orient='index', columns=['Value'])
        kpis_df.to_excel(writer, sheet_name='KPIs')
    
    print(f"Analysis exported to {filename}")

def export_charts_to_html(tables, analytics, filename='medicine_dashboard.html'):
    """Export all charts to a standalone HTML file"""
    from plotly.offline import plot
    
    df_main = tables['main']
    df_manufacturer = analytics.get_manufacturer_performance()
    df_ingredient_stats = analytics.get_ingredient_performance()
    df_indication = analytics.get_indication_analysis()
    df_side_effect = analytics.get_side_effect_analysis()
    
    # Create all charts
    charts = [
        create_manufacturer_stacked_bar(df_manufacturer),
        create_top_medicines_chart(df_main, 'Excellent Review %', 15),
        create_top_medicines_chart(df_main, 'Poor Review %', 15),
        create_ingredient_leaderboard(df_ingredient_stats, 20),
        create_indication_chart(df_indication, 15),
        create_side_effect_scatter(df_side_effect),
        create_polarization_chart(df_main)
    ]
    
    # Combine into HTML
    html_content = '<html><head><title>Medicine Dashboard</title></head><body>'
    html_content += '<h1 style="text-align:center">Medicine Review Dashboard</h1>'
    
    for chart in charts:
        html_content += plot(chart, output_type='div', include_plotlyjs='cdn')
    
    html_content += '</body></html>'
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Charts exported to {filename}")