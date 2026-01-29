import pandas as pd
import os
import re
import numpy as np # Added for easier labeling

def rapid_api_sentiment_spike(df):
    # --- 1. CONFIG ---
    DATE_COL = "review_date"
    SENTIMENT_COL = "sentiment_label"
    CATEGORY_COL = "category"
    SPIKE_THRESHOLD = 0.20      
    TREND_SHIFT_THRESHOLD = 0.20 
    LABEL_MAP = {"Positive": 1, "Neutral": 0, "Negative": -1}

    # --- 2. DATA PREP ---
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce').dt.tz_localize(None)
    df = df.dropna(subset=[DATE_COL, CATEGORY_COL])
    if SENTIMENT_COL not in df.columns:
        df[SENTIMENT_COL] = df['rating'].apply(lambda x: "Positive" if x >= 4 else ("Negative" if x <= 2 else "Neutral"))
    df["sentiment_score"] = df[SENTIMENT_COL].map(LABEL_MAP)

    # --- 3. WEEKLY AGGREGATION ---
    df['week_period'] = [d.to_period('W') for d in df[DATE_COL]]
    weekly_cat = (
        df.groupby([CATEGORY_COL, 'week_period'])["sentiment_score"]
        .mean()
        .reset_index(name="weekly_sentiment")
    )
    weekly_cat[DATE_COL] = [p.start_time for p in weekly_cat['week_period']]
    weekly_cat = weekly_cat.sort_values([CATEGORY_COL, DATE_COL])

    # --- 4. ROLLING WINDOWS & DELTA ---
    weekly_cat["rolling_avg"] = weekly_cat.groupby(CATEGORY_COL)["weekly_sentiment"] \
                                        .rolling(2, min_periods=1).mean().values
    weekly_cat["prev_rolling_avg"] = weekly_cat.groupby(CATEGORY_COL)["rolling_avg"].shift(1)
    weekly_cat["delta"] = weekly_cat["rolling_avg"] - weekly_cat["prev_rolling_avg"]
    weekly_cat["prev_delta"] = weekly_cat.groupby(CATEGORY_COL)["delta"].shift(1)

    # ============================================================
    # 5. CREATE THE 'ALERT' COLUMN (NEW)
    # ============================================================
    def determine_alert(row):
        if pd.isna(row["delta"]):
            return "Normal"
        
        # 🔄 Trend Shift Logic (Direction Flip)
        if (pd.notna(row["prev_delta"]) and 
            (row["delta"] * row["prev_delta"] < 0) and 
            abs(row["delta"]) >= TREND_SHIFT_THRESHOLD):
            return "TREND SHIFT"
        
        # 🔴 Negative Spike
        if row["delta"] <= -SPIKE_THRESHOLD:
            return "NEGATIVE SPIKE"
        
        # 🟢 Positive Spike
        if row["delta"] >= SPIKE_THRESHOLD:
            return "POSITIVE SPIKE"
        
        return "Normal"

    # Apply the logic to create the column
    weekly_cat["alert"] = weekly_cat.apply(determine_alert, axis=1)

    # --- 6. OUTPUT ---
    # We only keep the columns required by the Homework
    final_report = weekly_cat[[DATE_COL, CATEGORY_COL, "weekly_sentiment", "rolling_avg", "delta", "alert"]]
    return final_report.round(3)

if __name__ == "__main__":
    # Path to your existing data
    file_path = "final data/category_wise_lda_output_with_topic_labels.csv"
    
    if os.path.exists(file_path):
        df_rapid = pd.read_csv(file_path)
        report_df = rapid_api_sentiment_spike(df_rapid)
        
        # Save the single combined report
        report_df.to_csv("rapid_api_sentiment_report.csv", index=False)
        
        print("\n" + "="*40)
        print(f"✅ SUCCESS: Alert Column Added!")
        print(f"📂 Saved to: rapid_api_sentiment_report.csv")
        print("="*40)
        print(report_df[report_df['alert'] != 'Normal'].head()) # Show only the spikes
    else:
        print("❌ Error: Could not find your data file.")