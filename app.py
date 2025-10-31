# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from talent_identifier import TalentIdentifier
from supabase import create_client
import time
import functools
warnings.filterwarnings("ignore")
from streamlit.runtime.caching import cache_data

# REPLACE THE CACHING DECORATOR with this corrected version:

def cache_model_training(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Get data from session state for cache key
        bowling_df = st.session_state.bowling_df
        batting_df = st.session_state.batting_df
        
        # Create cache key based on data characteristics
        if bowling_df is not None and batting_df is not None:
            bowling_hash = hash(str(bowling_df.shape) + str(bowling_df.columns.tolist()))
            batting_hash = hash(str(batting_df.shape) + str(batting_df.columns.tolist()))
            cache_key = f"{bowling_hash}_{batting_hash}"
        else:
            cache_key = "no_data"
        
        # Check if we have cached models for this data
        if hasattr(self, '_model_cache') and self._model_cache.get('key') == cache_key:
            st.session_state.bowling_ti = self._model_cache['bowling_ti']
            st.session_state.batting_ti = self._model_cache['batting_ti']
            st.session_state.models_trained = True
            st.info("✅ Using cached models (data unchanged)")
            return True
        
        # If no cache hit, run the actual training
        result = func(self, *args, **kwargs)
        
        # Cache the results if training was successful
        if result and st.session_state.models_trained:
            self._model_cache = {
                'key': cache_key,
                'bowling_ti': st.session_state.bowling_ti,
                'batting_ti': st.session_state.batting_ti
            }
            st.info("✅ Models trained and cached for future use")
        
        return result
    return wrapper
# Configure page
st.set_page_config(
    page_title="CrickVision League Talent",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 3rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #343a40;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 600;
        border-left: 5px solid #667eea;
        padding-left: 1rem;
    }
    .player-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    .player-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stSelectbox, .stSlider {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class CricketScoutApp:
    def __init__(self):
        self._model_cache = {}  
        
        self.supabase_url = "https://dxudswxuovyeglvzczpy.supabase.co"
        self.supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImR4dWRzd3h1b3Z5ZWdsdnpjenB5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTc2NjgzOTIsImV4cCI6MjA3MzI0NDM5Mn0.puHDwRUAr0yfmBclAjMSGbcxUI6TV8eBUPXxL-e0voY"
        self.supabase = None
        
        # Initialize TalentIdentifier instances for league data access
        self.temp_ti = TalentIdentifier()
        
        # Initialize session state
        self._init_session_state()
        
    def _init_session_state(self):
        """Initialize session state variables"""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'models_trained' not in st.session_state:
            st.session_state.models_trained = False
        if 'run_analysis' not in st.session_state:
            st.session_state.run_analysis = False
        if 'bowling_df' not in st.session_state:
            st.session_state.bowling_df = None
        if 'batting_df' not in st.session_state:
            st.session_state.batting_df = None
        if 'bowling_ti' not in st.session_state:
            st.session_state.bowling_ti = None
        if 'batting_ti' not in st.session_state:
            st.session_state.batting_ti = None
        
    def init_supabase(self):
        """Initialize Supabase connection"""
        try:
            self.supabase = create_client(self.supabase_url, self.supabase_key)
            return True
        except Exception as e:
            st.error(f"Failed to connect to Supabase: {e}")
            return False
            
    def load_data(self, force_reload=False):
        """Load data from Supabase with optimized parallel loading and caching"""
        # Check if data is already cached
        if not force_reload and 'data_cache_timestamp' in st.session_state:
            cache_age = time.time() - st.session_state.data_cache_timestamp
            if cache_age < 3600:  # Cache for 1 hour
                st.info(f"✅ Using cached data (loaded {int(cache_age/60)} minutes ago)")
                return True
        
        if not self.init_supabase():
            return False
        
        import concurrent.futures
        import threading
        
        # ULTRA FAST: Bigger batches + parallel chunks
        BATCH_SIZE = 50000  # Massive batches - way bigger!
        MAX_PARALLEL_CHUNKS = 10  # Load 10 chunks at once per table
        
        def get_table_count(table_name):
            """Get total row count to calculate chunks"""
            try:
                resp = self.supabase.table(table_name).select("*", count="exact").limit(1).execute()
                return resp.count if hasattr(resp, 'count') else 100000
            except:
                return 100000  # Default estimate
        
        def load_chunk(table_name, start, end):
            """Load a single chunk of data"""
            try:
                resp = self.supabase.table(table_name).select("*").range(start, end).execute()
                return resp.data if resp.data else []
            except Exception as e:
                st.warning(f"Chunk {start}-{end} failed: {e}")
                return []
        
        def load_table_data_ultra_fast(table_name, progress_callback=None):
            """Load entire table with MAXIMUM parallelization"""
            # Get total count
            total_count = get_table_count(table_name)
            
            # Create chunk ranges
            chunks = []
            for start in range(0, total_count, BATCH_SIZE):
                end = min(start + BATCH_SIZE - 1, total_count - 1)
                chunks.append((start, end))
            
            all_data = []
            
            # Load chunks in parallel batches
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_CHUNKS) as chunk_executor:
                futures = [chunk_executor.submit(load_chunk, table_name, start, end) 
                          for start, end in chunks]
                
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    chunk_data = future.result()
                    all_data.extend(chunk_data)
                    # Note: Can't update Streamlit UI from threads - progress happens after
            
            return pd.DataFrame(all_data), table_name, len(all_data)
        
        # Progress tracking setup
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Load both tables in ULTRA PARALLEL MODE
        status_text.text("⚡ Loading bowling and batting data with maximum parallelization...")
        progress_bar.progress(0.1)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks (no callbacks - threads can't update UI)
            bowling_future = executor.submit(load_table_data_ultra_fast, "bowlers_stats")
            batting_future = executor.submit(load_table_data_ultra_fast, "batting_stats")
            
            # Wait for both to complete
            progress_bar.progress(0.5)
            status_text.text("⚡ Loading... Almost there...")
            
            bowling_df, _, bowling_count = bowling_future.result()
            progress_bar.progress(0.75)
            
            batting_df, _, batting_count = batting_future.result()
            progress_bar.progress(0.95)
        
        # Final progress update
        progress_bar.progress(1.0)
        status_text.text(f"✅ Loaded {len(bowling_df):,} bowling + {len(batting_df):,} batting records")
        
        # Store in session state
        st.session_state.bowling_df = bowling_df
        st.session_state.batting_df = batting_df
        
        # Debug info in sidebar
        if len(bowling_df) > 0 or len(batting_df) > 0:
            with st.sidebar:
                st.markdown("### 📊 Data Summary")
                st.metric("Bowling Records", f"{len(bowling_df):,}")
                st.metric("Batting Records", f"{len(batting_df):,}")
                
                if len(bowling_df) > 0 and 'player_name' in bowling_df.columns:
                    st.metric("Unique Bowlers", bowling_df['player_name'].nunique())
                if len(batting_df) > 0 and 'player_name' in batting_df.columns:
                    st.metric("Unique Batsmen", batting_df['player_name'].nunique())
                    
                # Show available columns for debugging
                if len(bowling_df) > 0:
                    with st.expander("🔍 Bowling Columns"):
                        st.write(list(bowling_df.columns))
                if len(batting_df) > 0:
                    with st.expander("🔍 Batting Columns"):
                        st.write(list(batting_df.columns))
        
        # Update session state
        st.session_state.data_loaded = True
        st.session_state.models_trained = False
        st.session_state.run_analysis = False
        st.session_state.data_cache_timestamp = time.time()  # Cache timestamp
        
        return True

    @cache_model_training 
    def train_models(self):
        """Train the machine learning models"""
        # Get data from session state
        bowling_df = st.session_state.bowling_df
        batting_df = st.session_state.batting_df
        
        # Check if data is actually loaded
        if bowling_df is None or batting_df is None or len(bowling_df) == 0 or len(batting_df) == 0:
            st.error("Data not properly loaded. Please load data first!")
            st.session_state.data_loaded = False
            return False
            
        with st.spinner("Training AI models... This may take a few minutes."):
            # Bowling model
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Training bowling performance model...")
            try:
                bowling_ti = TalentIdentifier()
                bowling_ti.fit(
                    bowling_df,
                    player_type='bowling',
                    horizon=5,
                    form_window=3,
                    verbose=False
                )
                progress_bar.progress(0.5)
                # Store in session state
                st.session_state.bowling_ti = bowling_ti
            except Exception as e:
                st.error(f"Error training bowling model: {e}")
                return False
            
            status_text.text("Training batting performance model...")
            try:
                batting_ti = TalentIdentifier()
                batting_ti.fit(
                    batting_df,
                    player_type='batting',
                    horizon=5,
                    form_window=3,
                    verbose=False
                )
                progress_bar.progress(1.0)
                # Store in session state
                st.session_state.batting_ti = batting_ti
            except Exception as e:
                st.error(f"Error training batting model: {e}")
                return False
            
            status_text.text("✅ Models trained successfully!")
            
        # Update session state
        st.session_state.models_trained = True
        return True
    
    def get_available_venues(self, league):
        """Get available venues for a specific league"""
        bowling_df = st.session_state.bowling_df
        batting_df = st.session_state.batting_df
        
        if bowling_df is None or not st.session_state.data_loaded:
            return ["All Venues"]
            
        # Get teams for the selected league
        league_teams = self.temp_ti.leagues.get(league, [])
        
        # Get venues from both datasets - handle missing venue column
        bowling_venues = []
        batting_venues = []
        
        if len(bowling_df) > 0:
            # Check if venue column exists
            if 'venue' in bowling_df.columns:
                bowling_venues = bowling_df[
                    bowling_df['Team'].isin(league_teams)
                ]['venue'].dropna().unique()
            else:
                # Try alternative column names
                venue_columns = ['venue', 'Venue', 'ground', 'stadium', 'location']
                for col in venue_columns:
                    if col in bowling_df.columns:
                        bowling_venues = bowling_df[
                            bowling_df['Team'].isin(league_teams)
                        ][col].dropna().unique()
                        break
        
        if len(batting_df) > 0:
            # Check if venue column exists
            if 'venue' in batting_df.columns:
                batting_venues = batting_df[
                    batting_df['Team'].isin(league_teams)
                ]['venue'].dropna().unique()
            else:
                # Try alternative column names
                venue_columns = ['venue', 'Venue', 'ground', 'stadium', 'location']
                for col in venue_columns:
                    if col in batting_df.columns:
                        batting_venues = batting_df[
                            batting_df['Team'].isin(league_teams)
                        ][col].dropna().unique()
                        break
        
        all_venues = set(bowling_venues) | set(batting_venues)
        venues_list = ["All Venues"] + sorted([str(v) for v in all_venues if v and str(v) != 'nan' and str(v) != ''])
        
        # If no venues found, return only "All Venues"
        if len(venues_list) == 1:
            return ["All Venues"]
        
        return venues_list
    
    def run_analysis(self, league, format_type, venue, top_n):
        """Run the talent scouting analysis"""
        bowling_df = st.session_state.bowling_df
        batting_df = st.session_state.batting_df
        bowling_ti = st.session_state.bowling_ti
        batting_ti = st.session_state.batting_ti
        
        if not st.session_state.models_trained or bowling_ti is None or batting_ti is None:
            st.error("Models not trained. Please train models first!")
            return None, None
            
        try:
            # Get top players
            with st.spinner(f"🔍 Scouting top {top_n} talents in {league}..."):
                # Bowling analysis
                try:
                    top_bowlers = bowling_ti.league_scouting(
                        df=bowling_df,
                        league=league,
                        format_type=format_type,
                        venue=venue if venue != "All Venues" else None,
                        top_n=top_n,
                        verbose=False
                    )
                except Exception as e:
                    st.error(f"❌ Bowling analysis failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    return None, None
                
                # Batting analysis
                try:
                    top_batsmen = batting_ti.league_scouting(
                        df=batting_df,
                        league=league,
                        format_type=format_type,
                        venue=venue if venue != "All Venues" else None,
                        top_n=top_n,
                        verbose=False
                    )
                except Exception as e:
                    st.error(f"❌ Batting analysis failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    return None, None
                
            return top_bowlers, top_batsmen
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return None, None
    
    def display_player_card(self, player, player_type, rank):
        """Display a player card with metrics"""
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            
            with col1:
                st.markdown(f"**#{rank} {player['player_name']}**")
                st.markdown(f"*{player['Team']}*")
                
            with col2:
                if player_type == 'bowling':
                    st.metric("Predicted Wickets", f"{player['predicted_score']:.2f}")
                else:
                    st.metric("Predicted Runs", f"{player['predicted_score']:.2f}")
                    
            with col3:
                st.metric("Performance Score", f"{player['predicted_score']:.2f}")
                
            with col4:
                if st.button("📊 Analyze", key=f"{player_type}_{player['player_name']}_{rank}"):
                    self.show_player_analysis(player['player_name'], player_type)
            
            st.markdown("---")
    
    def show_player_analysis(self, player_name, player_type):
        """Show detailed player analysis with FULL-WIDTH modern UI"""
        bowling_df = st.session_state.bowling_df
        batting_df = st.session_state.batting_df
        bowling_ti = st.session_state.bowling_ti
        batting_ti = st.session_state.batting_ti
        
        # Use expander for full-width modal-like experience
        with st.expander(f"🎯 **DETAILED ANALYSIS: {player_name.upper()}**", expanded=True):
            st.markdown("---")
            
            if player_type == 'bowling':
                df = bowling_df
                ti = bowling_ti
            else:
                df = batting_df
                ti = batting_ti
                
            try:
                # Get recent matches
                recent_matches = ti.get_player_recent_matches(df, player_name, 10)
                
                if len(recent_matches) > 0:
                    # Top metrics row - 4 columns
                    st.markdown("### 📊 Key Performance Indicators")
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    
                    if player_type == 'bowling':
                        avg_wickets = recent_matches['Wickets Taken'].mean() if 'Wickets Taken' in recent_matches.columns else 0
                        avg_economy = recent_matches['Economy Rate'].mean() if 'Economy Rate' in recent_matches.columns else 0
                        avg_dot_pct = recent_matches['Dot Ball %'].mean() if 'Dot Ball %' in recent_matches.columns else 0
                        
                        with metric_col1:
                            st.metric("🎯 Avg Wickets", f"{avg_wickets:.2f}")
                        with metric_col2:
                            st.metric("📉 Avg Economy", f"{avg_economy:.2f}")
                        with metric_col3:
                            st.metric("⏱️ Dot Ball %", f"{avg_dot_pct:.1f}%")
                        with metric_col4:
                            if avg_wickets > 2.5:
                                st.metric("⭐ Rating", "Excellent", delta="Top Tier")
                            elif avg_wickets > 1.5:
                                st.metric("⭐ Rating", "Good", delta="Solid")
                            else:
                                st.metric("⭐ Rating", "Developing", delta="Growth")
                    else:  # batting
                        avg_runs = recent_matches['Total Runs'].mean() if 'Total Runs' in recent_matches.columns else 0
                        avg_sr = recent_matches['Strike Rate'].mean() if 'Strike Rate' in recent_matches.columns else 0
                        avg_boundary_pct = recent_matches['Boundary %'].mean() if 'Boundary %' in recent_matches.columns else 0
                        
                        with metric_col1:
                            st.metric("🏏 Avg Runs", f"{avg_runs:.2f}")
                        with metric_col2:
                            st.metric("⚡ Strike Rate", f"{avg_sr:.2f}")
                        with metric_col3:
                            st.metric("🎯 Boundary %", f"{avg_boundary_pct:.1f}%")
                        with metric_col4:
                            if avg_runs > 40:
                                st.metric("⭐ Rating", "Excellent", delta="Top Tier")
                            elif avg_runs > 25:
                                st.metric("⭐ Rating", "Good", delta="Solid")
                            else:
                                st.metric("⭐ Rating", "Developing", delta="Growth")
                    
                    st.markdown("---")
                    
                    # Recent matches table - FULL WIDTH
                    st.markdown("### 📈 Recent Match Performance")
                    
                    if player_type == 'bowling':
                        display_cols = ['match_id', 'venue', 'Wickets Taken', 'Economy Rate', 'Dot Ball %']
                    else:
                        display_cols = ['match_id', 'venue', 'Total Runs', 'Strike Rate', 'Boundary %']
                    
                    # Only show columns that exist
                    available_cols = [col for col in display_cols if col in recent_matches.columns]
                    
                    if available_cols:
                        st.dataframe(
                            recent_matches[available_cols].head(10),
                            use_container_width=True,
                            height=350
                        )
                    
                    st.markdown("---")
                    
                    # Show trends chart in full width
                    st.markdown("### 📊 Performance Trends")
                    try:
                        fig = ti.plot_player_trends(df, player_name, player_type)
                        if fig:
                            # Increase figure size for better visibility
                            fig.set_size_inches(16, 10)
                            st.pyplot(fig)
                        else:
                            st.info("📊 Trend chart not available for this player")
                    except Exception as chart_error:
                        st.info(f"📊 Trend visualization not available: {chart_error}")
                        
                    # Additional insights section - FULL WIDTH
                    st.markdown("### 💡 AI-Powered Insights")
                    
                    insight_col1, insight_col2, insight_col3 = st.columns(3)
                    
                    if player_type == 'bowling':
                        with insight_col1:
                            if avg_wickets > 2:
                                st.success("🎯 **High Impact**\n\nConsistent wicket-taker who can change games")
                            else:
                                st.info("📈 **Growing**\n\nShows potential for improvement")
                        with insight_col2:
                            if avg_economy < 7:
                                st.success("💰 **Economical**\n\nGreat at building pressure and controlling runs")
                            else:
                                st.warning("⚠️ **Watch Economy**\n\nNeeds better control")
                        with insight_col3:
                            if avg_dot_pct > 50:
                                st.success("⏱️ **Pressure Builder**\n\nExcellent at creating dot-ball pressure")
                            else:
                                st.info("🎯 **Wicket Hunter**\n\nFocuses on taking wickets")
                    else:
                        with insight_col1:
                            if avg_runs > 35:
                                st.success("🏆 **Match Winner**\n\nConsistent high scorer who builds innings")
                            else:
                                st.info("📈 **Role Player**\n\nContributes to team success")
                        with insight_col2:
                            if avg_sr > 140:
                                st.success("⚡ **Power Hitter**\n\nAggressive striker who accelerates innings")
                            else:
                                st.info("🎯 **Accumulator**\n\nBuilds innings steadily")
                        with insight_col3:
                            if avg_boundary_pct > 20:
                                st.success("🎯 **Boundary Hunter**\n\nFinds boundaries regularly")
                            else:
                                st.info("🏃 **Run Rotator**\n\nExcels at finding gaps")
                            
                else:
                    st.warning("📭 No recent match data available for detailed analysis.")
                    
            except Exception as e:
                st.error(f"❌ Could not generate detailed analysis: {str(e)}")
                st.info("💡 This might be due to missing data columns or insufficient match history")
    
    def run(self):
        """Main application runner"""
        # Header
        st.markdown('<h1 class="main-header">🏏 Cricket Talent Scout Pro</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">AI-Powered Talent Identification & Performance Prediction</p>', unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.markdown("## ⚙️ Configuration")
            
            # Data loading section
            st.markdown("### 📊 Data Management")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("🔄 Load Data", use_container_width=True):
                    with st.spinner("Loading data..."):
                        if self.load_data():
                            st.success("Data loaded successfully!")
                        else:
                            st.error("Failed to load data!")
            with col2:
                if st.button("🔃", help="Force reload data", use_container_width=True):
                    with st.spinner("Reloading..."):
                        if self.load_data(force_reload=True):
                            st.success("Reloaded!")
                        else:
                            st.error("Failed!")
            
            if st.session_state.data_loaded:
                st.markdown("### 🎯 Scouting Filters")
                
                # League selection
                leagues = list(self.temp_ti.leagues.keys())
                selected_league = st.selectbox(
                    "Select League",
                    leagues,
                    help="Choose the T20 league for talent scouting"
                )
                
                # Format selection
                formats = ["All", "T20"]
                selected_format = st.selectbox(
                    "Select Format",
                    formats,
                    help="Filter by match format"
                )
                
                # Venue selection - with error handling
                try:
                    available_venues = self.get_available_venues(selected_league)
                    selected_venue = st.selectbox(
                        "Select Venue",
                        available_venues,
                        help="Filter by specific venue"
                    )
                except Exception as e:
                    st.error(f"Error loading venues: {e}")
                    available_venues = ["All Venues"]
                    selected_venue = "All Venues"
                
                # Number of players
                top_n = st.slider(
                    "Number of Top Players",
                    min_value=5,
                    max_value=20,
                    value=10,
                    help="Number of top players to display in each category"
                )
                
                # Model training
                st.markdown("### 🤖 AI Models")
                if not st.session_state.models_trained:
                    if st.button("Train AI Models", use_container_width=True):
                        with st.spinner("Training models..."):
                            if self.train_models():
                                st.success("Models trained successfully!")
                            else:
                                st.error("Failed to train models. Check your data.")
                
                # Analysis button
                if st.session_state.models_trained:
                    if st.button("🚀 Run Talent Analysis", use_container_width=True):
                        st.session_state.run_analysis = True
                else:
                    st.info("Please train AI models first to run analysis")
            
            # Data overview
            if st.session_state.data_loaded:
                st.markdown("### 📈 Data Overview")
                bowling_df = st.session_state.bowling_df
                batting_df = st.session_state.batting_df
                
                if bowling_df is not None and len(bowling_df) > 0:
                    if 'player_name' in bowling_df.columns:
                        st.metric("Total Bowlers", len(bowling_df['player_name'].unique()))
                    else:
                        st.metric("Bowling Records", len(bowling_df))
                        st.caption(f"Columns: {', '.join(bowling_df.columns[:5].tolist())}...")
                        
                if batting_df is not None and len(batting_df) > 0:
                    if 'player_name' in batting_df.columns:
                        st.metric("Total Batsmen", len(batting_df['player_name'].unique()))
                    else:
                        st.metric("Batting Records", len(batting_df))
                        st.caption(f"Columns: {', '.join(batting_df.columns[:5].tolist())}...")
        
        # Main content area
        if st.session_state.data_loaded and st.session_state.models_trained:
            if st.session_state.run_analysis:
                # Run analysis
                top_bowlers, top_batsmen = self.run_analysis(
                    selected_league, selected_format, selected_venue, top_n
                )
                
                if top_bowlers is not None and top_batsmen is not None:
                    # Display results in tabs
                    tab1, tab2 = st.tabs(["🎯 Top Bowlers", "🏏 Top Batsmen"])
                    
                    with tab1:
                        st.markdown(f'<div class="section-header">Top {len(top_bowlers)} Bowlers - {selected_league}</div>', unsafe_allow_html=True)
                        
                        if len(top_bowlers) > 0:
                            for i, bowler in top_bowlers.iterrows():
                                self.display_player_card(bowler, 'bowling', i+1)
                        else:
                            st.info("No bowlers found for the selected filters")
                    
                    with tab2:
                        st.markdown(f'<div class="section-header">Top {len(top_batsmen)} Batsmen - {selected_league}</div>', unsafe_allow_html=True)
                        
                        if len(top_batsmen) > 0:
                            for i, batsman in top_batsmen.iterrows():
                                self.display_player_card(batsman, 'batting', i+1)
                        else:
                            st.info("No batsmen found for the selected filters")
                    
                    # Summary statistics
                    if len(top_bowlers) > 0 or len(top_batsmen) > 0:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Players Analyzed", len(top_bowlers) + len(top_batsmen))
                        with col2:
                            if len(top_bowlers) > 0:
                                st.metric("Top Bowler Score", f"{top_bowlers.iloc[0]['predicted_score']:.2f}")
                            else:
                                st.metric("Top Bowler Score", "N/A")
                        with col3:
                            if len(top_batsmen) > 0:
                                st.metric("Top Batsman Score", f"{top_batsmen.iloc[0]['predicted_score']:.2f}")
                            else:
                                st.metric("Top Batsman Score", "N/A")
            
            else:
                # Welcome screen after data loaded and models trained
                st.markdown("""
                ## 🎯 Ready for Analysis!
                
                Your data is loaded and AI models are trained. 
                
                **Configure your scouting filters in the sidebar and click "Run Talent Analysis" to discover top performers!**
                
                ### 📊 What You'll Get:
                - **Top Bowlers & Batsmen** predictions
                - **Performance scores** for next 5 matches  
                - **Detailed player analysis** with trends
                - **AI-powered insights** and recommendations
                
                Get started by setting your filters and running the analysis!
                """)
        
        elif st.session_state.data_loaded and not st.session_state.models_trained:
            # Data loaded but models not trained
            st.markdown("""
            ## 📊 Data Loaded Successfully!
            
            Your cricket data has been loaded from Supabase.
            
            **Next step: Train the AI models to enable talent prediction.**
            
            Click the **"Train AI Models"** button in the sidebar to continue.
            
            This will train machine learning models to predict player performance 
            for the next 5 matches based on historical data.
            """)
        
        else:
            # Initial state - no data loaded
            st.markdown("""
            ## 🏏 Welcome to Cricket Talent Scout Pro!
            
            This AI-powered platform helps cricket teams and scouts identify top talent 
            across global T20 leagues with advanced machine learning predictions.
            
            ### 🚀 Getting Started:
            1. Click **"Load Data from Supabase"** in the sidebar
            2. Wait for data to load (this may take a few moments)
            3. Train the AI models
            4. Configure your scouting filters
            5. Run the talent analysis
            
            ### 📈 Features:
            - **Real-time data** from Supabase database
            - **AI performance predictions** for next 5 matches
            - **Comprehensive player analysis** with visual trends
            - **Multi-league support** (IPL, PSL, BBL, CPL, SA20, and more)
            - **Venue-specific insights**
            
            Ready to discover cricket's next superstars? Load your data to begin!
            """)

def main():
    app = CricketScoutApp()
    app.run()

if __name__ == "__main__":
    main()
