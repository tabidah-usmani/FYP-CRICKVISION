import os
import re
import json
import math
import hashlib
import zipfile
import requests
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from supabase import create_client
from datetime import datetime, timezone


# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(r"C:\Users\dilaw\Downloads\cric_pipeline")
ZIP_URL = "https://cricsheet.org/downloads/all_json.zip"

ZIP_PATH = BASE_DIR / "all_json.zip"
EXTRACT_FOLDER = BASE_DIR / "extracted"
MASTER_CSV = BASE_DIR / "master_matches.csv"

METADATA_JSON = BASE_DIR / "metadata.json"
FILE_METADATA_JSON = BASE_DIR / "file_metadata.json"

BATTERS_CSV = BASE_DIR / "batters_dataset.csv"
BOWLERS_CSV = BASE_DIR / "bowlers_dataset.csv"

BASE_DIR.mkdir(exist_ok=True)
EXTRACT_FOLDER.mkdir(exist_ok=True)


# ============================================================
# HELPERS
# ============================================================
def load_json_file(path, default={}):
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return default


def save_json_file(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def get_sha1(filepath):
    """Compute SHA-1 hash of a file"""
    h = hashlib.sha1()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


# ============================================================
# STEP 1 — LOAD STATE FILE (metadata.json)
# ============================================================
metadata = load_json_file(METADATA_JSON, default={
    "last_download_etag": None,
    "last_download_modified": None,
    "last_sync_time": None
})

file_metadata = load_json_file(FILE_METADATA_JSON, default={})


# ============================================================
# STEP 2 — CHECK REMOTE ZIP VERSION (HEAD request)
# ============================================================
print("\nChecking remote Cricsheet ZIP version...")

head = requests.head(ZIP_URL)

remote_etag = head.headers.get("ETag")
remote_modified = head.headers.get("Last-Modified")

should_download = False

if remote_etag != metadata.get("last_download_etag") or remote_modified != metadata.get("last_download_modified"):
    print("🟢 Update detected — new ZIP available.")
    should_download = True
else:
    print("🟡 No new update on Cricsheet — skipping download.")


# ============================================================
# If new ZIP, download it
# ============================================================
if should_download:
    print("\nDownloading updated ZIP...")
    data = requests.get(ZIP_URL)
    with open(ZIP_PATH, "wb") as f:
        f.write(data.content)
    print("Download completed.")

    # Update metadata state
    metadata["last_download_etag"] = remote_etag
    metadata["last_download_modified"] = remote_modified
    metadata["last_sync_time"] = datetime.utcnow().isoformat()

    save_json_file(METADATA_JSON, metadata)
else:
    if not ZIP_PATH.exists():
        print("❌ ZIP does not exist locally. Downloading anyway...")
        data = requests.get(ZIP_URL)
        with open(ZIP_PATH, "wb") as f:
            f.write(data.content)
    else:
        print("Using existing ZIP file.")


# ============================================================
# STEP 3 — EXTRACT ZIP + DETECT NEW/UPDATED MATCHES
# ============================================================
print("\nExtracting ZIP...")

new_or_updated_files = []

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    z.extractall(EXTRACT_FOLDER)

# Compare hashes of each file
for file_name in os.listdir(EXTRACT_FOLDER):
    if not file_name.endswith(".json"):
        continue

    full_path = EXTRACT_FOLDER / file_name
    file_hash = get_sha1(full_path)

    old_hash = file_metadata.get(file_name)

    if old_hash != file_hash:
        print(f" → New or updated match: {file_name}")
        new_or_updated_files.append(full_path)
        file_metadata[file_name] = file_hash

save_json_file(FILE_METADATA_JSON, file_metadata)

if not new_or_updated_files:
    print("\n🟡 No changed files. Pipeline finished.")
    exit(0)

# ============================================================
# STEP 4 — PROCESS ONLY NEW/UPDATED JSON FILES INTO CSV
# ============================================================
def load_json(fp):
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_rows(data, match_id):
    rows = []

    info = data.get('info', {}) or {}
    event = info.get('event', {}) or {}
    outcome = info.get('outcome', {}) or {}
    by = outcome.get('by', {}) or {}

    match_meta = {
        "match_id": match_id,
        "match_type": info.get("match_type"),
        "team_type": info.get("team_type"),
        "season": info.get("season"),
        "city": info.get("city"),
        "venue": info.get("venue"),
        "team_1": (info.get("teams") or [None, None])[0],
        "team_2": (info.get("teams") or [None, None])[1] if info.get("teams") else None,
        "toss_winner": info.get('toss', {}).get('winner', None),
        "toss_decision": info.get('toss', {}).get('decision', None),
        "winner": outcome.get("winner"),
        "winner_margin": by.get("runs") or by.get("wickets"),
        "player_of_match": ", ".join(info.get('player_of_match', []) if isinstance(info.get('player_of_match', []), list) else [info.get('player_of_match', [])]) if info.get('player_of_match') else None
    }

    for i, inn in enumerate(data.get("innings", []) or [], start=1):
        team = inn.get("team")
        for over in inn.get("overs") or []:
            over_num = over.get('over', None)
            for delivery in over.get('deliveries', []) or []:
                runs_batter = int(delivery.get('runs', {}).get('batter', 0) or 0)
                runs_extras = int(delivery.get('runs', {}).get('extras', 0) or 0)
                runs_total = int(delivery.get('runs', {}).get('total', 0) or 0)
                wide_or_noball = 1 if (runs_batter == 0 and runs_extras == 1) else 0

                row = {
                    **match_meta,
                    "inning": i,
                    "team": team,
                    "over": over_num,
                    "batter": delivery.get('batter', None),
                    "bowler": delivery.get('bowler', None),
                    "non_striker": delivery.get('non_striker', None),
                    "runs_batter": runs_batter,
                    "runs_extras": runs_extras,
                    "runs_total": runs_total,
                    "wide_or_noball": wide_or_noball,
                    "wicket_kind": None,
                    "player_out": None,
                    "fielder": None
                }

                wickets = delivery.get('wickets', []) or []
                if wickets:
                    wicket = wickets[0]
                    row['wicket_kind'] = wicket.get('kind', None)
                    row['player_out'] = wicket.get('player_out', None)
                    fielders = wicket.get('fielders', []) or []
                    row['fielder'] = ", ".join([f.get('name') for f in fielders if f.get('name')])
                rows.append(row)
    return rows

print("\nProcessing updated files...")

all_rows = []
match_ids_to_update = set()

for fpath in new_or_updated_files:
    match_id = int(hashlib.md5(str(fpath).encode()).hexdigest(), 16) % 10**9
    match_ids_to_update.add(match_id)
    data = load_json(fpath)
    rows = extract_rows(data, match_id)
    all_rows.extend(rows)

df_new = pd.DataFrame(all_rows)

# Append to master CSV
if MASTER_CSV.exists():
    df_old = pd.read_csv(MASTER_CSV)

    # --- CRITICAL CORRECTION: Remove old rows for updated matches ---
    if match_ids_to_update:
        df_old = df_old[~df_old['match_id'].isin(match_ids_to_update)]

    df_final = pd.concat([df_old, df_new], ignore_index=True)
else:
    df_final = df_new

df_final.to_csv(MASTER_CSV, index=False)

print("\n✅ MASTER CSV updated:", MASTER_CSV)
print("Added rows:", len(df_new))
print("Total rows:", len(df_final))
print("\n🎉 Incremental pipeline complete.")


# ============================================================
# STEP 5 — PREPROCESS THE MASTER CSV
# ============================================================

print("\nPreprocessing MASTER CSV...")


data = pd.read_csv(MASTER_CSV)

# Sort properly to ensure correct ordering
data = data.sort_values(by=['match_id', 'inning', 'over']).reset_index(drop=True)

# Add a sequential ball number within each match, inning, and over
data['ball_num'] = data.groupby(['match_id', 'inning', 'over']).cumcount() + 1


# Ensure fielder is only valid if wicket_kind exists
data.loc[data['wicket_kind'].isna(), 'fielder'] = np.nan

# Fill missing values with meaningful placeholders
fill_values = {
    'player_out': 'Not Out',
    'fielder': 'no fielder',
    'city': 'Unknown',
    'winner': 'Draw',
    'winner_margin': 0,
    'wicket_kind': 'No Wicket'
}
data = data.fillna(value=fill_values)

# Handle player_of_match safely (only replace invalid entries)
data['player_of_match'] = data['player_of_match'].replace(
    [np.nan, 'nan', 'NaN', '', None], 'no player of match'
)


# String cleaning for player names, teams,and fielders
data['batter'] = data['batter'].astype(str).str.strip().str.title()
data['bowler'] = data['bowler'].astype(str).str.strip().str.title()
data['fielder'] = data['fielder'].astype(str).str.strip().str.title()

# fix season formating
data['season'] = data['season'].astype(str).str.replace('/', '-')

# convert all relevant columns to category
categorical_cols = [
    'team', 'batter', 'bowler', 'non_striker', 'wicket_kind', 'fielder','season',
    'player_out', 'city', 'venue', 'team_1', 'team_2', 
    'toss_winner', 'toss_decision', 'winner', 'player_of_match', 'match_type'
]

for col in categorical_cols:
    data[col] = data[col].astype('category')

print(data.isnull().sum())

# runs should be non-negative
assert (data['runs_batter'] >= 0).all(), "Negative runs_batter found!"
assert (data['runs_extras'] >= 0).all(), "Negative runs_extras found!"
assert (data['runs_total'] >= 0).all(), "Negative runs_total found!"

# runs_total should equal runs_batter + runs_extras
mismatch = data[data['runs_total'] != (data['runs_batter'] + data['runs_extras'])]
print(f"Number of rows where runs_total != runs_batter + runs_extras: {len(mismatch)}")

# check over and ball ranges
invalid_over = data[data['over'] < 0]
invalid_ball = data[~data['ball_num'].between(1,6)]
print(f"Invalid overs: {len(invalid_over)}, Invalid balls: {len(invalid_ball)}")

# Check if balls per over exceed 6 for any match
check = data.groupby(['match_id', 'over'])['ball_num'].max().reset_index()
print("Number of overs where ball > 6:", (check['ball_num'] > 6).sum())

# Filter rows where ball > 6
invalid_balls = data[data['ball_num'] > 6]

# Check which match types these belong to
print(invalid_balls['match_type'].value_counts())

data.to_csv(MASTER_CSV, index=False)
print("\n✅ MASTER CSV preprocessed and saved.")

# ============================================================
# STEP 6 — SEPARATING BATTERS DATASET
# ============================================================

print("\nCreating BATTERS dataset...")

df = pd.read_csv(MASTER_CSV)

def calculate_batter_stats(df):

    for col in ['runs_batter', 'runs_extras', 'over']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)


    runs_batter = df['runs_batter'].sum()
    runs_extras = df['runs_extras'].sum()
    total_runs = runs_batter + runs_extras
    wide_no_ball = df['wide_or_noball'].sum()
    balls = len(df)
    dots = (df['runs_batter'] == 0).sum()
    fours = (df['runs_batter'] == 4).sum()
    sixes = (df['runs_batter'] == 6).sum()
    singles = (df['runs_batter'] == 1).sum()
    twos = (df['runs_batter'] == 2).sum()
    threes = (df['runs_batter'] == 3).sum()

    boundaries = fours + sixes

    strike_rate = round((runs_batter / balls * 100), 2) if balls > 0 else 0
    boundary_pct = round((boundaries / balls * 100), 2) if balls > 0 else 0
    dot_pct = round((dots / balls * 100), 2) if balls > 0 else 0
    scoring_pct = round(((balls - dots) / balls * 100), 2) if balls > 0 else 0
    avg_per_scoring = round((runs_batter / (balls - dots)), 2) if (balls - dots) > 0 else 0

    overs_batted = (balls // 6) + (balls % 6) / 10
    overs_batted_decimal = balls / 6

    # Identify dismissal row
    dismiss_rows = df[df['wicket_kind'].notna()]

    if dismiss_rows.empty:
        dismissal_type = "Not Out"
    else:
        dismissal_type = dismiss_rows['wicket_kind'].iloc[0]

    # fielder if dismissal is caught
    def get_fielder(df):
        dism = df[df['wicket_kind'].notna()]

        if dism.empty:
            return None
        
        kind = dism['wicket_kind'].iloc[0]

        if kind == "caught":
            return dism['fielder'].iloc[0]

        if kind == "caught and bowled":
            return dism['bowler'].iloc[0]
        
        if kind in ["stumped", "run out", "obstructing the field"]:
            return dism['fielder'].iloc[0]
        
        return None

    fielder_name = get_fielder(df)

    # Define match phases
    match_type = df['match_type'].iloc[0]

    phase_defs = {
        "T20": {"pp": (1, 6), "middle": (7, 15), "death": (16, 20)},
        "IT20": {"pp": (1, 6),  "middle": (7, 15), "death": (16, 20)},
        "ODI": {"pp": (1, 10), "middle": (11, 40), "death": (41, 50)},
        "ODM":  {"pp": (1, 10), "middle": (11, 40), "death": (41, 50)},
        "Test": {"pp": None, "middle": None, "death": None},
        "MDM": {"pp": None, "middle": None, "death": None},
    }

    if match_type == "Test":
        phases = phase_defs["Test"]
    else:
        phases = phase_defs.get(match_type, phase_defs["T20"])


    def runs_in_phase(phase):
        if phases[phase] is None:
            return 0
        start, end = phases[phase]
        return df[(df['over'] >= start) & (df['over'] <= end)][['runs_batter', 'runs_extras']].sum().sum()

    pp_runs = runs_in_phase("pp")
    middle_runs = runs_in_phase("middle")
    death_runs = runs_in_phase("death")


    batter_name = df['batter'].iloc[0]
    batter_team = (
        df['batting_team'].iloc[0]
        if 'batting_team' in df.columns
        else (df['team'].iloc[0] if 'team' in df.columns else None)
    )

    team_type = df['team_type'].iloc[0] if 'team_type' in df.columns else None


    # List of overs in which the batter played
    overs_played = sorted(df['over'].dropna().unique().tolist())
    overs_played_str = ", ".join(str(o) for o in overs_played)

    return pd.Series({
        "match_id": int(df['match_id'].iloc[0]) if pd.notna(df['match_id'].iloc[0]) else None,
        "match_type": match_type,
        "player_name": batter_name,
        "Team": batter_team,
        "Team Type": team_type,
        "Runs (Batter)": runs_batter,
        "Runs (Extras)": runs_extras,
        "Total Runs": total_runs,
        "Balls Faced": balls,
        "Dot Balls": dots,
        "Wide or No-Ball": wide_no_ball,
        "Number of Overs Batted": overs_batted,
        "Over Number Played": overs_played_str,
        "Fours": fours,
        "Sixes": sixes,
        "Total Boundaries": boundaries,
        "Singles": singles,
        "Twos": twos,
        "Threes": threes,
        "Strike Rate": strike_rate,
        "Boundary %": boundary_pct,
        "Dot Ball %": dot_pct,
        "Scoring Shots %": scoring_pct,
        "Avg Runs per Scoring Shot": avg_per_scoring,
        "Runs in Powerplay": pp_runs,
        "Runs in Middle Overs": middle_runs,
        "Runs in Death Overs": death_runs,
        "Dismissal Type": dismissal_type,
        "fielder (if caught)": fielder_name,
        # Metadata
        "season": df['season'].iloc[0],
        "city": df['city'].iloc[0],
        "venue": df['venue'].iloc[0],
        "team_1": df['team_1'].iloc[0],
        "team_2": df['team_2'].iloc[0],
        "toss_winner": df['toss_winner'].iloc[0],
        "toss_decision": df['toss_decision'].iloc[0],
        "winner_margin": df['winner_margin'].iloc[0],
        "winner": df['winner'].iloc[0]
    })


# Iterate over all unique bowlers
all_batter_data = []
players = df['batter'].dropna().unique()

for player in players:
    player_df = df[df['batter'] == player]

    if player_df.empty:
        continue

    # Compress to per-match stats
    compressed = (
        player_df.groupby('match_id', group_keys=False)
        .apply(calculate_batter_stats)
        .reset_index(drop=True)
    )

    all_batter_data.append(compressed)
    print(f"Processed batter: {player}")


final_batters_df = pd.concat(all_batter_data, ignore_index=True)
final_batters_df.to_csv(BATTERS_CSV, index=False)

print("\nAll batter stats successfully saved to 'all_batters_stats.csv'")


# ============================================================
# STEP 7 — PREPROCESSING BATTERS DATASET
# ============================================================

print("\nPreprocesing BATTERS dataset...")

df_batters = pd.read_csv(BATTERS_CSV)


# Check for missing or extra columns
# List of expected columns (from your batter stats code)
expected_columns = [
    "match_id", "match_type", "player_name", "Team", "Team Type", "Runs (Batter)", 
    "Runs (Extras)", "Total Runs", "Balls Faced", "Wide or No-Ball", "Dot Balls", "Fours", 
    "Sixes", "Total Boundaries", "Singles", "Twos", "Threes", "Strike Rate", 
    "Boundary %", "Dot Ball %", "Scoring Shots %", "Avg Runs per Scoring Shot",
    "Runs in Powerplay", "Runs in Middle Overs", "Runs in Death Overs",
    "Number of Overs Batted", "Dismissal Type", "fielder (if caught)", 
    "season", "city", "venue", "team_1", "team_2", "toss_winner", 
    "toss_decision", "winner_margin", "winner", "season_year"
]

# Check for missing or extra columns
missing_cols = set(expected_columns) - set(df_batters.columns)
extra_cols = set(df_batters.columns) - set(expected_columns)

print("Missing columns:", missing_cols)
print("Extra columns:", extra_cols)

print(df_batters.dtypes)

# Check for non-numeric values in numeric columns
numeric_cols = ["Runs (Batter)", "Runs (Extras)", "Total Runs", "Balls Faced", "Dot Balls",
                "Wide or No-Ball", "Fours", "Sixes", "Total Boundaries", "Singles", "Twos", "Threes",
                "Strike Rate", "Boundary %", "Dot Ball %", "Scoring Shots %", 
                "Avg Runs per Scoring Shot", "Runs in Powerplay", "Runs in Middle Overs",
                "Runs in Death Overs", "Number of Overs Batted", "winner_margin"]

for col in numeric_cols:
    non_numeric = df_batters[pd.to_numeric(df_batters[col], errors='coerce').isna()]
    if not non_numeric.empty:
        print(f"Non-numeric values found in {col}:")
        print(non_numeric[[col, "player_name", "match_id"]])


# Check for missing values and handle them
missing_values = df_batters.isnull().sum()
print("Missing values per column:\n", missing_values)

df_batters['fielder (if caught)'] = df_batters['fielder (if caught)'].fillna('No Fielder')

# Validate key statistics
errors = df_batters[df_batters["Total Runs"] != df_batters["Runs (Batter)"] + df_batters["Runs (Extras)"]]
print("Rows where Total Runs ≠ Runs (Batter) + Runs (Extras):")
print(errors[["player_name", "match_id", "Runs (Batter)", "Runs (Extras)", "Total Runs"]])

# Total Runs check
assert (df_batters['Total Runs'] == df_batters['Runs (Batter)'] + df_batters['Runs (Extras)']).all(), "Total Runs mismatch!"

# Boundaries
assert (df_batters['Total Boundaries'] == df_batters['Fours'] + df_batters['Sixes']).all(), "Boundary count mismatch!"

# Dot Ball percentage
assert ((df_batters['Dot Ball %'] >= 0) & (df_batters['Dot Ball %'] <= 100)).all(), "Dot Ball % out of range!"

# Runs in phases
assert ((df_batters['Runs in Powerplay'] + df_batters['Runs in Middle Overs'] + df_batters['Runs in Death Overs']) <= df_batters['Total Runs']).all(), "Phase runs exceed total runs!"

# Check for duplicate player-match entries
duplicates = df_batters.duplicated(subset=["match_id", "player_name"])
print("Duplicate player-match rows:", duplicates.sum())


# Recalculate derived stats
df_batters['Strike_Rate_Calc'] = (df_batters['Total Runs'] / df_batters['Balls Faced']) * 100
df_batters['Boundary_Percent_Calc'] = (df_batters['Total Boundaries'] / df_batters['Balls Faced']) * 100
df_batters['Dot_Ball_Percent_Calc'] = (df_batters['Dot Balls'] / df_batters['Balls Faced']) * 100
df_batters['Scoring_Shots_Percent_Calc'] = ((df_batters['Balls Faced'] - df_batters['Dot Balls']) / df_batters['Balls Faced']) * 100
df_batters['Avg_Runs_Per_Scoring_Calc'] = df_batters['Total Runs'] / (df_batters['Balls Faced'] - df_batters['Dot Balls']).replace(0, np.nan)

# Compare original vs calculated
derived_checks = {
    "Strike Rate": np.abs(df_batters['Strike Rate'] - df_batters['Strike_Rate_Calc']) > 0.01,
    "Boundary %": np.abs(df_batters['Boundary %'] - df_batters['Boundary_Percent_Calc']) > 0.01,
    "Dot Ball %": np.abs(df_batters['Dot Ball %'] - df_batters['Dot_Ball_Percent_Calc']) > 0.01,
    "Scoring Shots %": np.abs(df_batters['Scoring Shots %'] - df_batters['Scoring_Shots_Percent_Calc']) > 0.01,
    "Avg Runs per Scoring Shot": np.abs(df_batters['Avg Runs per Scoring Shot'] - df_batters['Avg_Runs_Per_Scoring_Calc']) > 0.01
}

for stat, mask in derived_checks.items():
    print(f"{stat} mismatches:", mask.sum())

df_batters.drop(columns=[
    'Strike_Rate_Calc',
    'Boundary_Percent_Calc',
    'Dot_Ball_Percent_Calc',
    'Scoring_Shots_Percent_Calc',
    'Avg_Runs_Per_Scoring_Calc'
], inplace=True)


# Numeric columns should be >= 0
numeric_cols = ["Balls Faced", "Wide or No-Ball", "Total Runs", "Runs (Batter)", "Runs (Extras)",
                "Dot Balls", "Fours", "Sixes", "Total Boundaries",
                "Singles", "Twos", "Threes", "Runs in Powerplay",
                "Runs in Middle Overs", "Runs in Death Overs"]

for col in numeric_cols:
    invalid = (df_batters[col] < 0).sum()
    if invalid > 0:
        print(f"{col} has {invalid} negative values!")

# Percentages should be 0–100
percent_cols = ["Boundary %", "Dot Ball %", "Scoring Shots %"]
for col in percent_cols:
    out_of_range = ((df_batters[col] < 0) | (df_batters[col] > 100)).sum()
    if out_of_range > 0:
        print(f"{col} has {out_of_range} values out of 0-100 range!")

# Runs in phases should not exceed Total Runs
phase_runs_invalid = (df_batters["Runs in Powerplay"] + 
                      df_batters["Runs in Middle Overs"] + 
                      df_batters["Runs in Death Overs"] > df_batters["Total Runs"]).sum()
print("Rows where phase runs exceed total runs:", phase_runs_invalid)

# Number of Overs Batted check (Balls Faced / 6 >= Number of Overs Batted)
invalid_overs = df_batters[df_batters['Number of Overs Batted'] > (df_batters['Balls Faced'] / 6)]
print("Rows where Number of Overs Batted > Balls Faced / 6:", len(invalid_overs))

# Identify invalid overs
invalid_overs = df_batters[df_batters['Number of Overs Batted'] > (df_batters['Balls Faced'] / 6)]

# Check match types for those rows
invalid_match_types = invalid_overs['match_type'].unique()
print("Unique match types with invalid overs:", invalid_match_types)

# verify how many invalid rows are non-Test
non_test_invalid = invalid_overs[~invalid_overs['match_type'].isin(['Test', 'MDM'])]
print("Non-Test invalid overs rows:", len(non_test_invalid))


# Find rows where overs and balls faced do not match exactly
df_batters['Calculated Overs'] = (
    df_batters['Balls Faced'] // 6
    + (df_batters['Balls Faced'] % 6) / 10
)

# Compare with rounding to avoid floating-point errors
mismatch_rows = df_batters[
    df_batters['Number of Overs Batted'].round(1) != df_batters['Calculated Overs'].round(1)
]

print("Number of incorrect overs:", len(mismatch_rows))
print(mismatch_rows[['player_name','Balls Faced','Number of Overs Batted','Calculated Overs']].head(20))
df_batters.drop(columns=['Calculated Overs'], inplace=True)

# Determine player availability based on last season played
CURRENT_YEAR = datetime.now(timezone.utc).year
metadata["last_sync_time"] = datetime.now(timezone.utc).isoformat()

def extract_year(season):
    if pd.isna(season):
        return None   
    season = str(season).strip()   
    # Handle "2023/24", "2023-24"
    if "/" in season or "-" in season:
        season = season.replace("/", "-")
        return int(season.split("-")[0])
    
    # Handle single years
    try:
        return int(season)
    except:
        return None

df_batters['season_year'] = df_batters['season'].apply(extract_year)


player_years = (
    df_batters.groupby('player_name')['season_year']
    .agg(['min', 'max'])
    .reset_index()
)

def check_availability(last_year):
    if pd.isna(last_year):
        return "Unknown"
    if last_year >= CURRENT_YEAR - 1:
        return "Available"
    return "Retired"

player_years['availability'] = player_years['max'].apply(check_availability)

df_batters = df_batters.merge(player_years[['player_name', 'availability']], on='player_name', how='left')

df_batters.drop(columns=['season_year'], inplace=True)

print(df_batters[['player_name', 'season', 'availability']].tail(100))
print(f"\nFinal dataset shape: {df_batters.shape}")
print(df_batters['availability'].value_counts())


df_batters.to_csv(BATTERS_CSV, index=False)
print("\n✅ BATTERS CSV preprocessed and saved.")

# ============================================================
# STEP 8 — SEPARATING BOWLERS DATASET
# ============================================================

df = pd.read_csv(MASTER_CSV)

print("\nCreating BOWLERS dataset...")

def calculate_bowler_stats(df):
    
    if df.empty:
        return pd.Series({})

    # Ensure numeric columns
    for col in ['runs_total', 'runs_batter', 'runs_extras', 'over']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0

    # Basic counts
    balls_bowled = len(df)
    legal_balls = len(df[df['runs_extras'] == 0])
    overs_bowled = (balls_bowled // 6) + (balls_bowled % 6) / 10
    overs_bowled_decimal = balls_bowled / 6
    runs_conceded = df['runs_total'].sum()
    wide_no_balls = df['runs_extras'].sum()
    wide_no_balls_pct = (wide_no_balls / balls_bowled * 100) if balls_bowled > 0 else 0.0

    # Wickets
    valid_wickets = {'caught', 'bowled', 'lbw', 'stumped', 'caught and bowled', 'hit wicket'}
    wickets = df['wicket_kind'].isin(valid_wickets).sum()

    # Maidens
    maiden_overs = int(df.groupby('over')['runs_total'].sum().eq(0).sum()) if 'over' in df.columns else 0

    # Economy
    economy_rate = (runs_conceded / overs_bowled_decimal) if overs_bowled_decimal > 0 else 0.0

    # Dot balls (legal deliveries)
    dot_balls = int((df[df['runs_extras'] == 0]['runs_batter'] == 0).sum())
    dot_ball_pct = (dot_balls / legal_balls * 100) if legal_balls > 0 else 0.0

    # Boundaries
    fours_conceded = int((df['runs_batter'] == 4).sum())
    sixes_conceded = int((df['runs_batter'] == 6).sum())
    total_boundaries = fours_conceded + sixes_conceded

    # Fielder for caught wickets
    fielder_name = None
    caught = df[df['wicket_kind'] == 'caught']
    if not caught.empty:
        fielder_list = caught['fielder'].dropna().unique()
        if len(fielder_list) > 0:
            fielder_name = ', '.join(fielder_list)

    # Teams
    batting_team = df['team'].dropna().unique()[0] if len(df['team'].dropna().unique()) > 0 else None
    team_1 = df['team_1'].iloc[0] if df['team_1'].notna().any() else None
    team_2 = df['team_2'].iloc[0] if df['team_2'].notna().any() else None
    bowler_team = team_2 if batting_team == team_1 else (team_1 if batting_team == team_2 else None)
    team_type = df['team_type'].iloc[0] if 'team_type' in df.columns else None

    # Match type
    match_type = df['match_type'].iloc[0] if 'match_type' in df.columns else None

    # Phase definitions
    phase_defs = {
        "T20": {"pp": (1, 6), "middle": (7, 15), "death": (16, 20)},
        "IT20": {"pp": (1, 6), "middle": (7, 15), "death": (16, 20)},
        "ODI": {"pp": (1, 10), "middle": (11, 40), "death": (41, 50)},
        "ODM": {"pp": (1, 10), "middle": (11, 40), "death": (41, 50)},
        "Test": {"pp": None, "middle": None, "death": None},
        "MDM": {"pp": None, "middle": None, "death": None},
    }
    phases = phase_defs.get(match_type, phase_defs["T20"])

    def runs_in_phase(phase):
        if phases[phase] is None:
            return 0
        start, end = phases[phase]
        return df[(df['over'] >= start) & (df['over'] <= end)]['runs_total'].sum()

    def wickets_in_phase(phase):
        if phases[phase] is None:
            return 0
        start, end = phases[phase]
        return df[(df['over'] >= start) & (df['over'] <= end) & (df['wicket_kind'].isin(valid_wickets))].shape[0]

    # Phase-wise stats
    pp_runs = runs_in_phase("pp")
    mid_runs = runs_in_phase("middle")
    death_runs = runs_in_phase("death")

    pp_wkts = wickets_in_phase("pp")
    mid_wkts = wickets_in_phase("middle")
    death_wkts = wickets_in_phase("death")

    # List of overs in which the batter played
    overs_played = sorted(df['over'].dropna().unique().tolist())
    overs_played_str = ", ".join(str(o) for o in overs_played)


    # Return Series with descriptive column names
    return pd.Series({
        "match_id": int(df['match_id'].iloc[0]) if pd.notna(df['match_id'].iloc[0]) else None,
        "match_type": match_type,
        "player_name": df['bowler'].iloc[0],
        "Team": bowler_team,
        "Team type": team_type,
        "Batting Team": batting_team,
        "Balls Bowled": int(legal_balls),
        "Overs Bowled": round(overs_bowled, 2),
        "Over Number Bowled": overs_played_str,
        "Runs Conceded": int(runs_conceded),
        "Wide/No Balls": int(wide_no_balls),
        "Wide/No Ball %": round(wide_no_balls_pct, 2),
        "Wickets Taken": int(wickets),
        "Fielder (if caught)": fielder_name,
        "Maidens": int(maiden_overs),
        "Economy Rate": round(economy_rate, 2),
        "Dot Balls": int(dot_balls),
        "Dot Ball %": round(dot_ball_pct, 2),
        "Fours Conceded": int(fours_conceded),
        "Sixes Conceded": int(sixes_conceded),
        "Total Boundaries Conceded": int(total_boundaries),
        "Powerplay Runs Conceded": pp_runs,
        "Middle Overs Runs Conceded": mid_runs,
        "Death Overs Runs Conceded": death_runs,
        "Powerplay Wickets": pp_wkts,
        "Middle Overs Wickets": mid_wkts,
        "Death Overs Wickets": death_wkts,
        "Season": df['season'].iloc[0] if 'season' in df.columns else None,
        "City": df['city'].iloc[0] if 'city' in df.columns else None,
        "Venue": df['venue'].iloc[0] if 'venue' in df.columns else None,
        "Team 1": team_1,
        "Team 2": team_2,
        "Toss Winner": df['toss_winner'].iloc[0] if 'toss_winner' in df.columns else None,
        "Toss Decision": df['toss_decision'].iloc[0] if 'toss_decision' in df.columns else None,
        "Winner": df['winner'].iloc[0] if 'winner' in df.columns else None,
        "Winner Margin": df['winner_margin'].iloc[0] if 'winner_margin' in df.columns else None
    })


# ---- Iterate over all unique bowlers ----
all_bowler_data = []
players = df['bowler'].dropna().unique()

for player in players:
    player_df = df[df['bowler'] == player]

    if player_df.empty:
        continue

    # Compress to per-match stats (fixed duplication issue)
    compressed = (
        player_df.groupby('match_id', group_keys=False)
        .apply(calculate_bowler_stats)
        .reset_index(drop=True)
    )

    all_bowler_data.append(compressed)
    print(f"Processed bowler: {player}")

# Combine all bowler stats into a single DataFrame
final_bowlers_df = pd.concat(all_bowler_data, ignore_index=True)

# Save to CSV
final_bowlers_df.to_csv(BOWLERS_CSV, index=False)

print(f"\nAll bowler stats successfully saved to '{BOWLERS_CSV}'")


# ============================================================
# STEP 9 — PREPROCESSING BOWLERS DATASET
# ============================================================

print("\nPreprocesing BOWLERS dataset...")

df_bowlers = pd.read_csv(BOWLERS_CSV)

# Check for missing or extra columns
expected_columns = [
    "match_id", "match_type", "player_name", "Team", "Team type", "Wide/No Ball %", "Batting Team",
    "Balls Bowled", "Overs Bowled", "Runs Conceded", "Wide/No Balls",
    "Wickets Taken", "Maidens", "Economy Rate", "Dot Balls", "Dot Ball %",
    "Fours Conceded", "Sixes Conceded", "Total Boundaries Conceded",
    "Fielder (if caught)",
    "Powerplay Runs Conceded", "Middle Overs Runs Conceded", "Death Overs Runs Conceded",
    "Powerplay Wickets", "Middle Overs Wickets", "Death Overs Wickets",
    "Season", "City", "Venue", "Team 1", "Team 2", "Toss Winner",
    "Toss Decision", "Winner", "Winner Margin"
]

# Compare
missing_cols = [c for c in expected_columns if c not in df_bowlers.columns]
extra_cols = [c for c in df_bowlers.columns if c not in expected_columns]

print("Missing Columns:", missing_cols if missing_cols else "None")
print("Extra Columns:", extra_cols if extra_cols else "None")

# Check economy rate consistency (Economy = Runs / Overs)
inconsistent_economy = df_bowlers[
    (df_bowlers["Overs Bowled"] > 0) &
    (abs(df_bowlers["Economy Rate"] - (df_bowlers["Runs Conceded"] / df_bowlers["Overs Bowled"])) > 0.5)
]

print(f"Economy mismatches: {len(inconsistent_economy)}")

# Check total boundaries
incorrect_boundaries = df_bowlers[
    df_bowlers["Total Boundaries Conceded"] != (
        df_bowlers["Fours Conceded"] + df_bowlers["Sixes Conceded"]
    )
]
print(f"Boundary mismatches: {len(incorrect_boundaries)}")

# Check for negative values and unrealistic stats
print("Negative balls:", (df_bowlers["Balls Bowled"] < 0).sum())
print("Economy rate > 50 check:", (df_bowlers["Economy Rate"] > 50).sum())


tests = ["Test", "MDM"]
overs_invalid_test = df_bowlers[
    (df_bowlers["match_type"].isin(tests)) &
    (df_bowlers["Overs Bowled"] > 100)
]
print("Overs > 100 (Tests):", len(overs_invalid_test))


limited_overs = ["T20", "IT20", "ODI", "ODM"]
overs_invalid_limited = df_bowlers[
    (df_bowlers["match_type"].isin(limited_overs)) &
    (df_bowlers["Overs Bowled"] > 20)
]
print("Overs > 20 (T20 etc.):", len(overs_invalid_limited))

# Check wickets > 10 only for limited-overs formats
invalid_wickets = df_bowlers[
    (df_bowlers["match_type"].isin(limited_overs)) &
    (df_bowlers["Wickets Taken"] > 10)
]
print("Invalid wickets (limited overs):", len(invalid_wickets))

# Check wickets > 20 for Tests/MDM
invalid_test_wickets = df_bowlers[
    (df_bowlers["match_type"].isin(tests)) &
    (df_bowlers["Wickets Taken"] > 20)
]
print("Invalid wickets (Test/MDM):", len(invalid_test_wickets))

# Check Dot Ball % consistency
dot_percent_invalid = df_bowlers[
    np.abs(df_bowlers["Dot Ball %"] - 
           (df_bowlers["Dot Balls"] / df_bowlers["Balls Bowled"]) * 100) > 1
]
print("Dot Ball % mismatches:", len(dot_percent_invalid))

# Check for missing values
print("\nMissing values per column:")
print(df_bowlers.isnull().sum())

print("\nData types:")
print(df_bowlers.dtypes)

df_bowlers['Fielder (if caught)'] = df_bowlers['Fielder (if caught)'].fillna('No Fielder')

df_bowlers.isnull().sum()

# Check for duplicate player-match entries
duplicates = df_bowlers.duplicated(subset=["match_id", "player_name"])
print("Duplicate rows:", duplicates.sum())

# Phase runs and wickets consistency
phase_runs_mismatches = df_bowlers[
    (df_bowlers["Powerplay Runs Conceded"] +
     df_bowlers["Middle Overs Runs Conceded"] +
     df_bowlers["Death Overs Runs Conceded"]) > df_bowlers["Runs Conceded"]
]
print("Phase runs exceed total runs:", len(phase_runs_mismatches))

phase_wickets_mismatches = df_bowlers[
    (df_bowlers["Powerplay Wickets"] +
     df_bowlers["Middle Overs Wickets"] +
     df_bowlers["Death Overs Wickets"]) > df_bowlers["Wickets Taken"]
]
print("Phase wickets exceed total wickets:", len(phase_wickets_mismatches))

# mismatched rows
df_bowlers['Calculated Overs'] = (df_bowlers['Balls Bowled']//6) + (df_bowlers['Balls Bowled'] % 6) / 10
mismatched_overs = df_bowlers[
    df_bowlers['Overs Bowled'].round(1) != df_bowlers['Calculated Overs'].round(1)
]
print("Mismatched overs rows:", len(mismatched_overs))
print(mismatched_overs[['player_name','Balls Bowled','Overs Bowled','Calculated Overs']].head(20))
df_bowlers.drop(columns=['Calculated Overs'], inplace=True)

df_bowlers.to_csv(BOWLERS_CSV, index=False, na_rep='None')

# Determine player availability based on last season played
CURRENT_YEAR = datetime.now(timezone.utc).year
metadata["last_sync_time"] = datetime.now(timezone.utc).isoformat()

def extract_year(season):
    if pd.isna(season):
        return None   
    season = str(season).strip()   
    # Handle "2023/24", "2023-24"
    if "/" in season or "-" in season:
        season = season.replace("/", "-")
        return int(season.split("-")[0])
    
    # Handle single years
    try:
        return int(season)
    except:
        return None

df_bowlers['season_year'] = df_bowlers['Season'].apply(extract_year)

player_years = (
    df_bowlers.groupby('player_name')['season_year']
    .agg(['min', 'max'])
    .reset_index()
)

def check_availability(last_year):
    if pd.isna(last_year):
        return "Unknown"
    if last_year >= CURRENT_YEAR - 1:
        return "Available"
    return "Retired"

player_years['availability'] = player_years['max'].apply(check_availability)

df_bowlers = df_bowlers.merge(player_years[['player_name', 'availability']], on='player_name', how='left')

df_bowlers.drop(columns=['season_year'], inplace=True)

print(df_bowlers[['player_name', 'Season', 'availability']].tail(100))
print(f"\nFinal dataset shape: {df_bowlers.shape}")
print(df_bowlers['availability'].value_counts())

df_bowlers.to_csv(BOWLERS_CSV, index=False)
print("\n✅ BOWLERS CSV preprocessed and saved.")

# ============================================================
# STEP 10 — UPLOAD ONTO SUPABASE
# ============================================================

# Only upload if there are new or updated files
if new_or_updated_files:

    SUPABASE_URL = "https://dxudswxuovyeglvzczpy.supabase.co"
    SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImR4dWRzd3h1b3Z5ZWdsdnpjenB5Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NzY2ODM5MiwiZXhwIjoyMDczMjQ0MzkyfQ.o9KnwdGAsgAmRYriNW9HFilA94T0-5UggYCwpvwpjPY"

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    def upload_table(table_name, df, chunk_size=1000):
        for i in range(0, len(df), chunk_size):
            chunk = df[i:i+chunk_size].to_dict('records')
            response = supabase.table(table_name).upsert(chunk).execute()
            print(f"{table_name} rows {i}–{i+len(chunk)} uploaded. Status: {response.status_code if hasattr(response, 'status_code') else 'Done'}")

    # Upload Batters
    print("\nUploading BATTERS dataset to Supabase...")
    upload_table("batting_stats", df_batters)

    # Upload Bowlers
    print("\nUploading BOWLERS dataset to Supabase...")
    upload_table("bowlers_stats", df_bowlers)

    print("\n✅ Both CSVs uploaded successfully!")
else:
    print("\nNo new or updated files detected, skipping Supabase upload.")
