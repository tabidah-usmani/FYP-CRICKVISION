# talent_identifier.py
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rapidfuzz import process, fuzz
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class TalentIdentifier:
    def __init__(self, model=None, random_state=42):
        """
        model: sklearn regressor. If None, a LGBMRegressor is created.
        """
        self.random_state = random_state
        self.model = LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
        self.pipeline = None
        self.fitted = False
        self.metadata = {}  # store unique choices for fuzzy matching
        self.player_type = None  # 'batting' or 'bowling'

        # Define leagues and their teams
        self.leagues = {
        # --- IPL ---
        "IPL": [
            "Mumbai Indians", "Chennai Super Kings", "Kolkata Knight Riders", "Delhi Capitals",
            "Royal Challengers Bengaluru", "Punjab Kings", "Rajasthan Royals", "Gujarat Titans",
            "Lucknow Super Giants", "Sunrisers Hyderabad", "Deccan Chargers", "Kochi Tuskers Kerala",
            "Pune Warriors", "Rising Pune Supergiant", "Rising Pune Supergiants"
        ],

        # --- PSL ---
        "PSL": [
            "Islamabad United", "Karachi Kings", "Lahore Qalandars", "Multan Sultans",
            "Peshawar Zalmi", "Quetta Gladiators"
        ],

        # --- BBL (Australia) ---
        "BBL": [
            "Sydney Sixers", "Sydney Thunder", "Melbourne Stars", "Melbourne Renegades",
            "Adelaide Strikers", "Perth Scorchers", "Brisbane Heat", "Hobart Hurricanes"
        ],

        # --- CPL ---
        "CPL": [
            "Trinbago Knight Riders", "Jamaica Tallawahs", "Guyana Amazon Warriors",
            "St Lucia Kings", "St Lucia Zouks", "Barbados Royals", "Barbados Tridents",
            "St Kitts and Nevis Patriots", "Antigua Hawksbills", "Trinidad & Tobago Red Steel"
        ],

        # --- SA20 (South Africa) ---
        "SA20": [
            "MI Cape Town", "Durban's Super Giants", "Sunrisers Eastern Cape",
            "Paarl Royals", "Pretoria Capitals", "Joburg Super Kings"
        ],

        # --- ILT20 (UAE) ---
        "ILT20": [
            "Abu Dhabi Knight Riders", "Dubai Capitals", "Gulf Giants", "Sharjah Warriors",
            "MI Emirates", "Desert Vipers"
        ],

        # --- BPL (Bangladesh) ---
        "BPL": [
            "Comilla Victorians", "Dhaka Dynamites", "Khulna Tigers", "Chittagong Vikings",
            "Sylhet Strikers", "Fortune Barishal", "Rangpur Riders", "Rajshahi Royals",
            "Khulna Titans", "Dhaka Gladiators", "Minister Group Dhaka", "Duronto Rajshahi",
            "Dhaka Dominators", "Barisal Bulls", "Barisal Burners", "Sylhet Sixers",
            "Sylhet Thunder", "Cumilla Warriors", "Dhaka Platoon", "Dhaka Capital"
        ],

        # --- LPL (Sri Lanka) ---
        "LPL": [
            "Colombo Strikers", "Jaffna Kings", "Galle Titans", "Dambulla Sixers", "Kandy Falcons",
            "Kandy Tuskers", "Dambulla Aura", "Colombo Kings", "Galle Gladiators",
            "B-Love Kandy", "Galle Marvels"
        ],

        # --- MLC (USA) ---
        "MLC": [
            "MI New York", "Washington Freedom", "Los Angeles Knight Riders",
            "Seattle Orcas", "Texas Super Kings", "San Francisco Unicorns"
        ],

        # --- The Hundred (England) ---
        "The Hundred": [
            "Trent Rockets", "Birmingham Phoenix", "London Spirit",
            "Southern Brave", "Oval Invincibles", "Manchester Originals", "Welsh Fire",
            "Northern Superchargers"
        ],

        # --- Nepal T20 ---
        "Nepal T20": [
            "Chitwan Rhinos", "Kathmandu Gurkhas", "Lumbini Lions",
            "Pokhara Avengers", "Janakpur Bolts", "Sudur Paschim Royals",
            "Karnali Yaks"
        ]
        }

    # ---------------------------
    # Utility functions
    # ---------------------------
    @staticmethod
    def clean_text(s):
        if pd.isna(s):
            return ""
        s = str(s).lower().strip()
        s = re.sub(r'[^\w\s]', '', s)
        s = re.sub(r'\s+', ' ', s)
        return s

    def build_metadata(self, df):
        """Build sets of unique cleaned values for fuzzy matching."""
        self.metadata['venues'] = df['venue'].dropna().unique().tolist() if 'venue' in df.columns else []
        self.metadata['Teams'] = df['Team'].dropna().unique().tolist() if 'Team' in df.columns else []
        self.metadata['match_types'] = df['match_type'].dropna().unique().tolist() if 'match_type' in df.columns else []

    def fuzzy_match(self, text, choices, label, min_score=10):
        """
        Return best match from choices for text using rapidfuzz. If score < min_score, return None.
        """
        if not text:
            return None
        if not choices:
            return None
        match_tuple = process.extractOne(text, choices, scorer=fuzz.WRatio)
        if match_tuple is None:
            return None
        match, score, _ = match_tuple
        return match if score >= min_score else match

    # ---------------------------
    # Target creation for both batting and bowling
    # ---------------------------
    def make_performance_score(self, df, player_type, weights=None):
        """
        Improved performance score using the better logic from both approaches
        """
        if player_type == 'bowling':
            # Use default weights if none provided
            w = dict(wickets=0.6, economy=0.3, dot_balls=0.1)
            if weights:
                w.update(weights)

            wickets = df.get('Wickets Taken', pd.Series(0, index=df.index)).fillna(0)
            economy = df.get('Economy Rate', pd.Series(0, index=df.index)).fillna(0)
            dot_balls = df.get('Dot Ball %', pd.Series(0, index=df.index)).fillna(0)

            # Normalize and weight components
            wicket_score = wickets * w['wickets']
            economy_score = (100 - economy) * w['economy']  # Lower economy = higher score
            dot_ball_score = (dot_balls / 100) * w['dot_balls']  # Normalize percentage

            score = wicket_score + economy_score + dot_ball_score

        else:  # batting
            w = dict(runs_per_inn=0.5, strike_rate=0.3, boundaries=0.2)
            if weights:
                w.update(weights)

            runs = df.get('Total Runs', pd.Series(0, index=df.index)).fillna(0)
            innings = df.get('Innings', pd.Series(1, index=df.index)).fillna(1)  # Avoid division by zero
            strike_rate = df.get('Strike Rate', pd.Series(0, index=df.index)).fillna(0)
            boundaries = df.get('Total Boundaries', pd.Series(0, index=df.index)).fillna(0)

            # Calculate components
            runs_per_inn = (runs / innings) * w['runs_per_inn']
            strike_rate_score = (strike_rate / 100) * w['strike_rate']  # Normalize
            boundary_score = boundaries * w['boundaries']

            score = runs_per_inn + strike_rate_score + boundary_score

        return score

    def create_future_target(self, df, player_type, horizon_games=5):
        """
        Create target: average performance in next X games
        """
        df = df.copy()

        # Sort by player and match_id (assuming match_id is chronological)
        df = df.sort_values(['player_name', 'match_id'])

        future_targets = []

        for player in df['player_name'].unique():
            player_data = df[df['player_name'] == player].reset_index(drop=True)

            # For each game, look ahead to next horizon_games
            for i in range(len(player_data) - horizon_games):
                current_match = player_data.iloc[i]
                future_matches = player_data.iloc[i+1:i+horizon_games+1]

                # Calculate future performance based on player type
                if player_type == 'bowling':
                    future_perf = future_matches['Wickets Taken'].mean()
                else:  # batting
                    future_perf = future_matches['Total Runs'].mean()

                future_targets.append({
                    'player_name': player,
                    'match_id': current_match['match_id'],
                    'future_performance': future_perf
                })

        return pd.DataFrame(future_targets)

    def create_rolling_features(self, df, feature_cols, window=3):
        """
        Create rolling averages to capture recent form
        """
        df = df.copy().sort_values(['player_name', 'match_id'])

        for col in feature_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                rolling_col = f'{col}_last_{window}'
                df[rolling_col] = df.groupby('player_name')[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )

        return df

    def prepare_future_data(self, df, player_type, feature_cols=None, horizon=5, form_window=3):
        """
        Main method to prepare data for future performance prediction
        """
        # 1. Create future targets
        print("Creating future performance targets...")
        future_targets = self.create_future_target(df, player_type, horizon_games=horizon)

        # 2. Create rolling features for current form
        print("Creating rolling features for current form...")
        if feature_cols is None:
            feature_cols = self.get_default_features(player_type)

        df_with_rolling = self.create_rolling_features(df, feature_cols, window=form_window)

        # 3. Merge targets with features
        print("Merging features with targets...")
        merged_df = df_with_rolling.merge(
            future_targets,
            on=['player_name', 'match_id'],
            how='inner'
        )

        # 4. Filter out players with insufficient data
        player_match_counts = merged_df['player_name'].value_counts()
        sufficient_players = player_match_counts[player_match_counts >= horizon + form_window].index
        merged_df = merged_df[merged_df['player_name'].isin(sufficient_players)]

        print(f"Final dataset: {len(merged_df)} rows for {len(sufficient_players)} players")
        return merged_df

    def get_default_features(self, player_type):
        """
        Get feature columns including rolling features
        """
        if player_type == 'bowling':
            base_features = [
                'match_type', 'Team', 'Venue', 'Season',
                'Balls Bowled', 'Overs Bowled', 'Runs Conceded', 'Wickets Taken',
                'Maidens', 'Economy Rate', 'Dot Balls', 'Dot Ball %',
                'Fours Conceded', 'Sixes Conceded', 'Total Boundaries Conceded',
                'Powerplay Runs Conceded', 'Middle Overs Runs Conceded', 'Death Overs Runs Conceded',
                'Powerplay Wickets', 'Middle Overs Wickets', 'Death Overs Wickets'
            ]
        else:  # batting
            base_features = [
                'match_type', 'Team', 'venue', 'season',
                'Runs (Batter)', 'Total Runs', 'Balls Faced', 'Dot Balls',
                'Fours', 'Sixes', 'Total Boundaries', 'Singles', 'Twos', 'Threes',
                'Strike Rate', 'Boundary %', 'Dot Ball %', 'Scoring Shots %',
                'Avg Runs per Scoring Shot', 'Runs in Powerplay', 'Runs in Middle Overs',
                'Runs in Death Overs'
            ]

        return base_features

    def fit(self, df, player_type='bowling', feature_cols=None, horizon=5, form_window=3,
            test_size=0.2, random_state=None, verbose=True):
        """
        Train model to predict FUTURE performance
        """
        rs = random_state or self.random_state
        self.player_type = player_type

        # 1. Filter available players
        df = df.copy()
        if 'availability' in df.columns:
            df = df[df['availability'].astype(str).str.lower() != 'retired']

        # 2. Prepare future performance data
        prepared_df = self.prepare_future_data(
            df, player_type, feature_cols, horizon, form_window
        )

        if len(prepared_df) == 0:
            raise ValueError("No sufficient data for future performance prediction")

        # 3. Clean string columns
        for col in prepared_df.select_dtypes(include=['object']).columns:
            prepared_df[col] = prepared_df[col].astype(str).apply(self.clean_text)

        # 4. Build metadata
        self.build_metadata(prepared_df)

        # 5. Set target and features
        if feature_cols is None:
            feature_cols = self.get_default_features(player_type)

        # Add rolling feature columns
        rolling_features = [f'{col}_last_{form_window}' for col in feature_cols
                          if col in prepared_df.columns and pd.api.types.is_numeric_dtype(prepared_df[col])]
        all_features = feature_cols + rolling_features

        # Use future performance as target
        X = prepared_df[all_features]
        y = prepared_df['future_performance']

        if verbose:
            print(f"\nTraining to predict next {horizon} games performance")
            print(f"Target range: {y.min():.1f} to {y.max():.1f}")
            print(f"Features: {len(all_features)}")
            print(f"Dataset: {len(X)} matches")

        # 6. Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rs)

        # Check if we have enough data
        if len(X_train) < 2:
            raise ValueError(f"Not enough data after filtering. Only {len(X_train)} training rows remaining.")

        # 7. Identify numeric and categorical columns
        numeric_cols = [c for c in all_features if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
        categorical_cols = [c for c in all_features if c in X.columns and c not in numeric_cols]

        if verbose:
            print(f"Training {player_type} model...")
            print(f"Training rows: {len(X_train)}, Validation rows: {len(X_test)}")
            print("Numeric features:", numeric_cols)
            print("Categorical features:", categorical_cols)

        # 8. ColumnTransformer
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='drop'
        )

        # 9. Pipeline
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', self.model)
        ])

        # 10. Fit the model
        self.pipeline.fit(X_train, y_train)
        self.fitted = True

        # 11. Cross-validation
        try:
            scores = cross_val_score(self.pipeline, X_train, y_train, cv=3, scoring='r2', n_jobs=-1)
            if verbose:
                print("Cross-val R2 (3-fold) on train:", np.round(scores, 3))
        except Exception:
            if verbose:
                print("Cross-val failed (likely due to small data or heavy one-hot).")

        # 12. Validation metric
        val_score = self.pipeline.score(X_test, y_test)
        if verbose:
            print("Validation R2:", round(val_score, 4))

        # 13. Store metadata for later reference
        self.metadata['feature_cols'] = all_features  # Store ALL features including rolling ones
        self.metadata['player_type'] = player_type
        self.metadata['horizon'] = horizon
        self.metadata['form_window'] = form_window

        # 14. Calculate detailed metrics
        y_pred_train = self.pipeline.predict(X_train)
        y_pred_test = self.pipeline.predict(X_test)

        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

        if verbose:
            print(f"\n--- {player_type.title()} Future Performance Model ---")
            print(f"Training R²: {train_r2:.4f}, Validation R²: {test_r2:.4f}")
            print(f"Training MAE: {train_mae:.4f}, Validation MAE: {test_mae:.4f}")
            print(f"Training RMSE: {train_rmse:.4f}, Validation RMSE: {test_rmse:.4f}")
            print(f"Predicting: Average performance in next {horizon} games")

        # Return scores for further analysis if needed
        self.train_scores = {'r2': train_r2, 'mae': train_mae, 'rmse': train_rmse}
        self.test_scores = {'r2': test_r2, 'mae': test_mae, 'rmse': test_rmse}

        return self

    # ---------------------------
    # Player Analysis Methods
    # ---------------------------
    def get_player_recent_matches(self, df, player_name, num_matches=10):
        """
        Get the most recent matches for a player
        """
        player_matches = df[df['player_name'] == player_name].copy()
        
        # Sort by match_id (assuming it's chronological) or date if available
        if 'match_id' in player_matches.columns:
            player_matches = player_matches.sort_values('match_id', ascending=False)
        elif 'date' in player_matches.columns:
            player_matches = player_matches.sort_values('date', ascending=False)
        else:
            # If no date/match_id, just take the last rows
            player_matches = player_matches.tail(num_matches)
        
        return player_matches.head(num_matches)
    
    def plot_player_trends(self, df, player_name, player_type, num_matches=10):
        """
        Create line charts for player's recent performance
        """
        recent_matches = self.get_player_recent_matches(df, player_name, num_matches)
        
        if len(recent_matches) == 0:
            print(f"No recent matches found for {player_name}")
            return None
        
        # Reverse to show chronological order
        recent_matches = recent_matches.iloc[::-1].reset_index(drop=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Recent Performance Trends: {player_name}', fontsize=16, fontweight='bold')
        
        if player_type == 'bowling':
            # Bowling metrics
            if 'Wickets Taken' in recent_matches.columns:
                axes[0,0].plot(range(len(recent_matches)), recent_matches['Wickets Taken'], 
                              marker='o', linewidth=2, markersize=8, color='red')
                axes[0,0].set_title('Wickets Taken', fontweight='bold')
                axes[0,0].set_ylabel('Wickets')
                axes[0,0].grid(True, alpha=0.3)
            
            if 'Economy Rate' in recent_matches.columns:
                axes[0,1].plot(range(len(recent_matches)), recent_matches['Economy Rate'], 
                              marker='s', linewidth=2, markersize=8, color='blue')
                axes[0,1].set_title('Economy Rate', fontweight='bold')
                axes[0,1].set_ylabel('Economy')
                axes[0,1].grid(True, alpha=0.3)
            
            if 'Dot Ball %' in recent_matches.columns:
                axes[1,0].plot(range(len(recent_matches)), recent_matches['Dot Ball %'], 
                              marker='^', linewidth=2, markersize=8, color='green')
                axes[1,0].set_title('Dot Ball Percentage', fontweight='bold')
                axes[1,0].set_ylabel('Dot Ball %')
                axes[1,0].grid(True, alpha=0.3)
            
            # Phase-wise performance if available
            phase_cols = ['Powerplay Wickets', 'Middle Overs Wickets', 'Death Overs Wickets']
            available_phase_cols = [col for col in phase_cols if col in recent_matches.columns]
            if available_phase_cols:
                for col in available_phase_cols:
                    axes[1,1].plot(range(len(recent_matches)), recent_matches[col].fillna(0), 
                                  marker='d', linewidth=2, markersize=6, label=col)
                axes[1,1].set_title('Phase-wise Wickets', fontweight='bold')
                axes[1,1].set_ylabel('Wickets')
                axes[1,1].legend()
                axes[1,1].grid(True, alpha=0.3)
            else:
                # Fallback to other bowling metrics
                if 'Balls Bowled' in recent_matches.columns:
                    axes[1,1].plot(range(len(recent_matches)), recent_matches['Balls Bowled'], 
                                  marker='d', linewidth=2, markersize=6, color='orange')
                    axes[1,1].set_title('Balls Bowled', fontweight='bold')
                    axes[1,1].set_ylabel('Balls')
                    axes[1,1].grid(True, alpha=0.3)
            
        else:  # batting
            # Batting metrics
            if 'Total Runs' in recent_matches.columns:
                axes[0,0].plot(range(len(recent_matches)), recent_matches['Total Runs'], 
                              marker='o', linewidth=2, markersize=8, color='orange')
                axes[0,0].set_title('Runs Scored', fontweight='bold')
                axes[0,0].set_ylabel('Runs')
                axes[0,0].grid(True, alpha=0.3)
            
            if 'Strike Rate' in recent_matches.columns:
                axes[0,1].plot(range(len(recent_matches)), recent_matches['Strike Rate'], 
                              marker='s', linewidth=2, markersize=8, color='purple')
                axes[0,1].set_title('Strike Rate', fontweight='bold')
                axes[0,1].set_ylabel('Strike Rate')
                axes[0,1].grid(True, alpha=0.3)
            
            if 'Boundary %' in recent_matches.columns:
                axes[1,0].plot(range(len(recent_matches)), recent_matches['Boundary %'], 
                              marker='^', linewidth=2, markersize=8, color='teal')
                axes[1,0].set_title('Boundary Percentage', fontweight='bold')
                axes[1,0].set_ylabel('Boundary %')
                axes[1,0].grid(True, alpha=0.3)
            
            # Phase-wise runs if available
            phase_cols = ['Runs in Powerplay', 'Runs in Middle Overs', 'Runs in Death Overs']
            available_phase_cols = [col for col in phase_cols if col in recent_matches.columns]
            if available_phase_cols:
                for col in available_phase_cols:
                    axes[1,1].plot(range(len(recent_matches)), recent_matches[col].fillna(0), 
                                  marker='d', linewidth=2, markersize=6, label=col)
                axes[1,1].set_title('Phase-wise Runs', fontweight='bold')
                axes[1,1].set_ylabel('Runs')
                axes[1,1].legend()
                axes[1,1].grid(True, alpha=0.3)
            else:
                # Fallback to other batting metrics
                if 'Balls Faced' in recent_matches.columns:
                    axes[1,1].plot(range(len(recent_matches)), recent_matches['Balls Faced'], 
                                  marker='d', linewidth=2, markersize=6, color='brown')
                    axes[1,1].set_title('Balls Faced', fontweight='bold')
                    axes[1,1].set_ylabel('Balls')
                    axes[1,1].grid(True, alpha=0.3)
        
        # Set x-axis labels
        for ax in axes.flat:
            ax.set_xlabel('Recent Matches (Oldest to Latest)')
            ax.set_xticks(range(len(recent_matches)))
        
        plt.tight_layout()
        return fig

    def generate_detailed_scouting_report(self, df, player_name, player_type):
        """
        Generate comprehensive scouting report with trends and analysis
        """
        print(f"\n{'='*60}")
        print(f"SCOUTING REPORT: {player_name.upper()}")
        print(f"{'='*60}")
        
        # Get recent matches data
        recent_matches = self.get_player_recent_matches(df, player_name, 10)
        
        if len(recent_matches) == 0:
            print("No recent match data available")
            return
        
        # Calculate recent form averages
        if player_type == 'bowling':
            avg_wickets = recent_matches['Wickets Taken'].mean()
            avg_economy = recent_matches['Economy Rate'].mean()
            avg_dot_percentage = recent_matches['Dot Ball %'].mean() if 'Dot Ball %' in recent_matches.columns else 0
            
            print(f"\nRECENT FORM (Last {len(recent_matches)} matches):")
            print(f"• Average Wickets: {avg_wickets:.2f}")
            print(f"• Average Economy: {avg_economy:.2f}")
            print(f"• Average Dot Ball %: {avg_dot_percentage:.2f}%")
            
        else:  # batting
            avg_runs = recent_matches['Total Runs'].mean()
            avg_strike_rate = recent_matches['Strike Rate'].mean()
            avg_boundary_percentage = recent_matches['Boundary %'].mean() if 'Boundary %' in recent_matches.columns else 0
            
            print(f"\nRECENT FORM (Last {len(recent_matches)} matches):")
            print(f"• Average Runs: {avg_runs:.2f}")
            print(f"• Average Strike Rate: {avg_strike_rate:.2f}")
            print(f"• Average Boundary %: {avg_boundary_percentage:.2f}%")
        
        # Plot trends
        fig = self.plot_player_trends(df, player_name, player_type)
        if fig:
            plt.show()
        
        return {
            'player_name': player_name,
            'recent_matches': recent_matches
        }

    # ---------------------------
    # Candidate selection + prediction
    # ---------------------------
    def _prepare_candidates(self, df, team, venue, match_type, min_candidates=30):
        """
        Returns a candidate DataFrame (rows) to run predictions on.
        """
        df2 = df.copy()

        # ensure cleaned columns exist
        for col in ['venue', 'Team', 'match_type']:
            if col in df2.columns:
                df2[col] = df2[col].astype(str).apply(self.clean_text)

        # progressive filters
        filters = [
            (df2['Team'] == team) & (df2['venue'] == venue) & (df2['match_type'] == match_type),
            (df2['Team'] == team) & (df2['match_type'] == match_type),
            (df2['Team'] == team),
            (pd.Series([True]*len(df2)))  # all domestic
        ]

        candidate_df = pd.DataFrame()
        for filt in filters:
            subset = df2[filt].copy()
            if len(subset) >= min_candidates:
                candidate_df = subset
                break
            # accumulate progressively (take union) but prefer narrower data first
            if candidate_df.empty:
                candidate_df = subset
            else:
                candidate_df = pd.concat([candidate_df, subset]).drop_duplicates().reset_index(drop=True)
        if candidate_df.empty:
            candidate_df = df2.copy()
        return candidate_df

    def compute_phase_features(self, df):
        """
        Creates phase-level metrics for scouting insights based on player type.
        """
        df = df.copy()
        player_type = self.metadata.get('player_type', 'bowling')

        if player_type == 'bowling':
            # Bowling phase features
            for col in [
                "Powerplay Wickets", "Powerplay Runs Conceded",
                "Middle Overs Wickets", "Middle Overs Runs Conceded",
                "Death Overs Wickets", "Death Overs Runs Conceded",
                "Economy Rate", "Dot Ball %"
            ]:
                if col not in df.columns:
                    df[col] = 0

            df["PP_Efficiency"] = df["Powerplay Wickets"] / (df["Powerplay Runs Conceded"] + 1e-6)
            df["MO_Efficiency"] = df["Middle Overs Wickets"] / (df["Middle Overs Runs Conceded"] + 1e-6)
            df["DO_Efficiency"] = df["Death Overs Wickets"] / (df["Death Overs Runs Conceded"] + 1e-6)
            df["Overall_Economy"] = df["Economy Rate"]
            df["Dot_Control"] = df["Dot Ball %"]

            # Identify dominant phase
            df["Best_Phase"] = df[["PP_Efficiency", "MO_Efficiency", "DO_Efficiency"]].idxmax(axis=1)
            df["Best_Phase"] = df["Best_Phase"].map({
                "PP_Efficiency": "Powerplay",
                "MO_Efficiency": "Middle Overs",
                "DO_Efficiency": "Death Overs"
            })

        else:
            # Batting phase features
            for col in [
                "Runs in Powerplay", "Runs in Middle Overs", "Runs in Death Overs",
                "Strike Rate", "Boundary %"
            ]:
                if col not in df.columns:
                    df[col] = 0

            # Calculate phase strike rates and efficiencies
            df["PP_StrikeRate"] = df["Runs in Powerplay"] / 6.0  # Approximate strike rate for powerplay
            df["MO_StrikeRate"] = df["Runs in Middle Overs"] / 9.0  # Approximate for middle overs
            df["DO_StrikeRate"] = df["Runs in Death Overs"] / 5.0  # Approximate for death overs

            df["Overall_StrikeRate"] = df["Strike Rate"]
            df["Boundary_Proficiency"] = df["Boundary %"]

            # Identify dominant batting phase
            df["Best_Phase"] = df[["PP_StrikeRate", "MO_StrikeRate", "DO_StrikeRate"]].idxmax(axis=1)
            df["Best_Phase"] = df["Best_Phase"].map({
                "PP_StrikeRate": "Powerplay",
                "MO_StrikeRate": "Middle Overs",
                "DO_StrikeRate": "Death Overs"
            })

        return df

    def train_ai_advisor(self, df, n_estimators=80, max_depth=6, random_state=42, verbose=True):
        """
        Train an  advisor (classifier) that predicts the 'Best_Phase' from phase efficiencies.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report

        df2 = self.compute_phase_features(df)
        player_type = self.metadata.get('player_type', 'bowling')

        # features and target based on player type
        if player_type == 'bowling':
            X = df2[["PP_Efficiency", "MO_Efficiency", "DO_Efficiency", "Overall_Economy", "Dot_Control"]].fillna(0)
        else:
            X = df2[["PP_StrikeRate", "MO_StrikeRate", "DO_StrikeRate", "Overall_StrikeRate", "Boundary_Proficiency"]].fillna(0)

        y = df2["Best_Phase"].fillna("Unknown")

        # sanity check
        if y.nunique() <= 1:
            if verbose:
                print("Not enough class diversity to train advisor (Need >1 distinct Best_Phase).")
            return None

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

        advisor = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1)
        advisor.fit(X_train, y_train)

        if verbose:
            print(f"\n--- {player_type.title()}  Advisor Model (phase classifier) ---")
            print(classification_report(y_test, advisor.predict(X_test), zero_division=0))

        # store advisor so you can reuse later
        self.metadata['advisor_trained'] = True
        self.metadata['advisor_features'] = X.columns.tolist()
        self.advisor_model = advisor

        return advisor

    def generate_scouting_report(self, df, top_players, advisor_model=None):
        """
        Enhanced version: -based tactical suggestions using learned patterns.
        """
        import difflib
        import numpy as np

        # Normalize player names for comparison
        df["player_name_clean"] = df["player_name"].astype(str).str.strip().str.lower()
        top_players_clean = [p.strip().lower() for p in top_players]

        # Direct match first
        df_sub = df[df["player_name_clean"].isin(top_players_clean)].copy()

        # If no exact matches, use fuzzy matching
        if df_sub.empty:
            print(" No exact matches found — trying fuzzy name matching...")
            matched_names = []
            for p in top_players_clean:
                closest = difflib.get_close_matches(
                    p, df["player_name_clean"].unique(), n=1, cutoff=0.6
                )
                if closest:
                    matched_names.append(closest[0])

            if matched_names:
                df_sub = df[df["player_name_clean"].isin(matched_names)].copy()

        # Final safety check
        if df_sub.empty:
            raise ValueError(
                "No matching rows for the provided top_players in the dataframe (even after fuzzy matching)."
            )
        else:
            print(f" Matched {len(df_sub['player_name_clean'].unique())} player(s) after fuzzy matching.")

        # Compute phase-level features
        df_feat = self.compute_phase_features(df_sub)
        player_type = self.metadata.get('player_type', 'bowling')

        # Aggregate stats for summary
        if player_type == 'bowling':
            summary = (
                df_feat.groupby("player_name")
                .agg({
                    "Wickets Taken": "mean",
                    "Overall_Economy": "mean",
                    "Dot_Control": "mean",
                    "Powerplay Wickets": "mean",
                    "Middle Overs Wickets": "mean",
                    "Death Overs Wickets": "mean",
                    "PP_Efficiency": "mean",
                    "MO_Efficiency": "mean",
                    "DO_Efficiency": "mean",
                })
                .reset_index()
            )
        else:
            summary = (
                df_feat.groupby("player_name")
                .agg({
                    "Total Runs": "mean",
                    "Overall_StrikeRate": "mean",
                    "Boundary_Proficiency": "mean",
                    "Runs in Powerplay": "mean",
                    "Runs in Middle Overs": "mean",
                    "Runs in Death Overs": "mean",
                    "PP_StrikeRate": "mean",
                    "MO_StrikeRate": "mean",
                    "DO_StrikeRate": "mean",
                    "Fours": "mean",
                    "Sixes": "mean"
                })
                .reset_index()
            )

        # Select advisor model
        adv = advisor_model or getattr(self, "advisor_model", None)

        # Feature alignment for prediction
        if adv is not None:
            expected_cols = self.metadata.get('advisor_features', [])
            for col in expected_cols:
                if col not in summary.columns:
                    summary[col] = 0.0

            X_adv = summary[expected_cols].fillna(0)
            phase_preds = adv.predict(X_adv)
        else:
            phase_preds = ["Unknown"] * len(summary)

        # Generate -style suggestions based on player type
        suggestions = []
        for i, row in summary.iterrows():
            player = row.get("player_name", None) or row.iloc[0]
            phase = phase_preds[i]

            if player_type == 'bowling':
                eco = row.get("Overall_Economy", float("nan"))
                dot = row.get("Dot_Control", float("nan"))

                if phase == "Powerplay":
                    text = f"{player} excels in Powerplay — great control with new ball, builds early pressure."
                elif phase == "Middle Overs":
                    text = f"{player} strong in Middle Overs — uses variations to restrict scoring."
                elif phase == "Death Overs":
                    text = f"{player} is a Death Overs specialist — key weapon for closing games."
                else:
                    text = f"{player} is balanced across phases — dependable all-rounder."

                if not np.isnan(eco):
                    if eco < 6.5:
                        text += f" Maintains tight economy ({eco:.1f})."
                    elif eco > 8.0:
                        text += f" Slightly expensive ({eco:.1f}); use with attacking fields."

                if not np.isnan(dot):
                    if dot > 60:
                        text += f" Excellent dot-ball rate ({dot:.1f}%) keeps batsmen in check."
                    elif dot < 40:
                        text += f" Needs better control; low dot-ball rate ({dot:.1f}%)."

            else:  # batting suggestions
                strike_rate = row.get("Overall_StrikeRate", float("nan"))
                boundary_pct = row.get("Boundary_Proficiency", float("nan"))

                if phase == "Powerplay":
                    text = f"{player} excels in Powerplay — aggressive starter, capitalizes on field restrictions."
                elif phase == "Middle Overs":
                    text = f"{player} strong in Middle Overs — rotates strike well, builds innings."
                elif phase == "Death Overs":
                    text = f"{player} is a Death Overs specialist — finishes games with power hitting."
                else:
                    text = f"{player} is adaptable across phases — versatile batsman."

                if not np.isnan(strike_rate):
                    if strike_rate > 140:
                        text += f" Explosive striker ({strike_rate:.1f} SR)."
                    elif strike_rate < 110:
                        text += f" Accumulator type ({strike_rate:.1f} SR)."

                if not np.isnan(boundary_pct):
                    if boundary_pct > 20:
                        text += f" High boundary frequency ({boundary_pct:.1f}%)."
                    elif boundary_pct < 10:
                        text += f" More reliance on running ({boundary_pct:.1f}% boundaries)."

            suggestions.append(text)

        summary[" Suggestion"] = suggestions
        summary["Best Phase ()"] = phase_preds

        # Create display version with user-friendly column names
        display_summary = summary.copy()
        if player_type == 'bowling':
            rename_map = {
                "Overall_Economy": "Economy Rate",
                "Dot_Control": "Dot Ball %",
                "Wickets Taken": "Avg Wickets"
            }
        else:
            rename_map = {
                "Overall_StrikeRate": "Strike Rate",
                "Boundary_Proficiency": "Boundary %",
                "Total Runs": "Avg Runs"
            }

        display_summary.rename(columns=rename_map, inplace=True, errors="ignore")

        return display_summary

    def predict_top_n(self, df, top_n=5, venue_input="", team_input="", match_type_input="", min_candidates=30):
        """
        Given raw DataFrame and user inputs, return top_n unique player names ranked by predicted performance.
        """
        assert self.fitted, "Model not fitted. Call fit() first."

        # clean df local copy
        df_local = df.copy()
        for col in df_local.select_dtypes(include=['object']).columns:
            df_local[col] = df_local[col].astype(str).apply(self.clean_text)

        # map user inputs via fuzzy matching to dataset unique values
        venue_choice = self.fuzzy_match(self.clean_text(venue_input), self.metadata.get('venues', []), 'venue')
        team_choice = self.fuzzy_match(self.clean_text(team_input), self.metadata.get('Teams', []), 'team')
        match_type_choice = self.fuzzy_match(self.clean_text(match_type_input), self.metadata.get('match_types', []), 'match_type')

        # fallback defaults if fuzzy returns None
        venue_choice = venue_choice or ""
        team_choice = team_choice or ""
        match_type_choice = match_type_choice or ""

        # candidate selection
        candidates = self._prepare_candidates(
            df_local, team_choice, venue_choice, match_type_choice, min_candidates=min_candidates
        )

        # features to predict on
        feature_cols = self.metadata['feature_cols']
        X_cand = candidates[feature_cols].copy()

        # make predictions
        preds = self.pipeline.predict(X_cand)
        candidates = candidates.assign(predicted_score=preds)

        # aggregate by player_name (average predicted score)
        if 'player_name' not in candidates.columns:
            raise ValueError("DataFrame must contain 'player_name' column.")
        agg = candidates.groupby('player_name', as_index=False)['predicted_score'].mean()
        agg = agg.sort_values('predicted_score', ascending=False).reset_index(drop=True)

        # return top N players
        return agg.head(top_n)

    def league_scouting(self, df, league, format_type="all", venue=None, top_n=10, verbose=True):
        """
        Main league-based scouting method for both batting and bowling - FIXED VERSION
        """
        assert self.fitted, "Model not fitted. Call fit() first."

        # 1. Get the form window from metadata
        form_window = self.metadata.get('form_window', 3)

        # 2. Prepare the data with rolling features FIRST
        df_local = df.copy()

        # Get base features (without rolling suffixes)
        base_features = [col for col in self.metadata['feature_cols'] if not col.endswith(f'_last_{form_window}')]

        # Create rolling features for prediction
        df_local = self.create_rolling_features(df_local, base_features, window=form_window)

        # Clean team names for comparison
        df_local['Team_clean'] = df_local['Team'].astype(str).apply(self.clean_text)

        # 3. Get teams for the selected league
        if league not in self.leagues:
            raise ValueError(f"League '{league}' not found. Available leagues: {list(self.leagues.keys())}")

        league_teams = self.leagues[league]

        if verbose:
            print(f"Looking for {league} teams: {league_teams}")

        # Use fuzzy matching to find teams in the data
        matched_teams = []
        for league_team in league_teams:
            cleaned_league_team = self.clean_text(league_team)
            # Try exact match first
            if cleaned_league_team in df_local['Team_clean'].values:
                matched_teams.append(cleaned_league_team)
            else:
                # Try fuzzy matching
                match = self.fuzzy_match(cleaned_league_team, df_local['Team_clean'].unique(), f'team_{league_team}')
                if match:
                    matched_teams.append(match)

        if verbose:
            print(f"Matched teams: {matched_teams}")

        # Filter by matched teams
        if matched_teams:
            league_players = df_local[df_local['Team_clean'].isin(matched_teams)].copy()
        else:
            # If no matches, use the original filtering (case-insensitive contains)
            pattern = '|'.join([self.clean_text(team) for team in league_teams])
            league_players = df_local[df_local['Team_clean'].str.contains(pattern, case=False, na=False)].copy()

        if verbose:
            print(f"Found {len(league_players)} rows for {league}")
            if len(league_players) > 0:
                print(f"Teams found: {league_players['Team_clean'].unique()}")

        if len(league_players) == 0:
            available_teams = df_local['Team_clean'].unique()
            print(f"No players found for league '{league}'. Available teams in your data:")
            for team in available_teams[:20]:
                print(f"  - {team}")
            raise ValueError(f"No players found for league '{league}'. Check team names above.")

        # 4. Apply format filter - USE "all" for now to avoid issues
        if format_type and format_type.lower() != "all":
            # For now, let's skip format filtering to get results
            if verbose:
                print(f"Format filtering temporarily disabled. Using all formats.")
            # You can add format filtering back later once we get it working

        if venue and venue.lower() != "all venues":
            venue_clean = self.fuzzy_match(self.clean_text(venue), self.metadata.get('venues', []), 'venue') or ""
            if venue_clean:
                before_venue = len(league_players)
                league_players = league_players[league_players['venue'] == venue_clean]
                if verbose:
                    print(f"Venue filter: {before_venue} -> {len(league_players)} rows")

        # Final check before prediction
        if len(league_players) == 0:
            raise ValueError("No data remaining after applying filters. Try broader filters.")

        # 5. Prepare features and predict
        feature_cols = self.metadata['feature_cols']

        # Check if all required features are present
        missing_features = [col for col in feature_cols if col not in league_players.columns]
        if missing_features:
            print(f"Warning: Missing features {missing_features}. Using available features.")
            feature_cols = [col for col in feature_cols if col in league_players.columns]

        X_league = league_players[feature_cols].copy()

        if verbose:
            print(f"Making predictions on {len(X_league)} samples with {len(feature_cols)} features")

        # Make predictions
        preds = self.pipeline.predict(X_league)
        league_players = league_players.assign(predicted_score=preds)

        # 6. Aggregate by player and get top N
        if 'player_name' not in league_players.columns:
            raise ValueError("DataFrame must contain 'player_name' column.")

        # Aggregate by player (average predicted score)
        player_agg = league_players.groupby('player_name', as_index=False).agg({
            'predicted_score': 'mean',
            'Team': 'first',
            'match_type': 'first'
        })

        # Sort and get top N
        top_players = player_agg.sort_values('predicted_score', ascending=False).head(top_n)

        if verbose:
            print(f"Found {len(top_players)} unique players")

        return top_players

    def enhanced_league_report(self, df, league, format_type, venue=None, top_n=10):
        """
        Comprehensive league scouting with  insights for both batting and bowling
        """
        # Get top players with verbose output
        top_players_df = self.league_scouting(df, league, format_type, venue, top_n, verbose=True)

        if len(top_players_df) == 0:
            print("No players found for the given filters.")
            return pd.DataFrame()

        # Generate scouting report
        try:
            scouting_report = self.generate_scouting_report(
                df,
                top_players_df['player_name'].tolist()
            )

            # Merge with prediction scores
            final_report = scouting_report.merge(
                top_players_df[['player_name', 'predicted_score', 'Team']],
                on='player_name',
                how='left'
            )

            # Reorder columns for better presentation
            player_type = self.metadata.get('player_type', 'bowling')
            if player_type == 'bowling':
                column_order = ['player_name', 'Team', 'predicted_score', 'Best Phase ()',
                              'Avg Wickets', 'Economy Rate', 'Dot Ball %', ' Suggestion']
            else:
                column_order = ['player_name', 'Team', 'predicted_score', 'Best Phase ()',
                              'Avg Runs', 'Strike Rate', 'Boundary %', ' Suggestion']

            # Only include columns that exist
            final_columns = [col for col in column_order if col in final_report.columns]
            final_report = final_report[final_columns]

            return final_report.sort_values('predicted_score', ascending=False)

        except Exception as e:
            print(f"Error generating scouting report: {e}")
            return top_players_df

    def get_available_leagues(self):
        """Return list of available leagues"""
        return list(self.leagues.keys())

    def get_league_teams(self, league):
        """Return teams for a specific league"""
        return self.leagues.get(league, [])

    def get_available_filters(self, df):
        """Return available options for filters"""
        return {
            'formats': df['match_type'].unique().tolist(),
            'venues': df['venue'].unique().tolist() if 'venue' in df.columns else [],
            'player_types': ['batting', 'bowling']
        }

    # ---------------------------
    # Persistence
    # ---------------------------
    def save(self, path):
        joblib.dump({'pipeline': self.pipeline, 'metadata': self.metadata}, path)

    def load(self, path):
        data = joblib.load(path)
        self.pipeline = data['pipeline']
        self.metadata = data['metadata']
        self.fitted = True
        self.player_type = self.metadata.get('player_type', 'bowling')
        return self
