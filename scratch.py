import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout='wide')

@st.cache_data
def load_quartal_for_instrument(instrument: str, period: str = "1H") -> pd.DataFrame:
    """
    Load the 1-minute quartal file for a single instrument.
    period must be "1H" or "3H".
    """
    base = "https://raw.githubusercontent.com/TuckerArrants/hourly_quarters/main"
    if period == "1H":
        fname = f"{instrument}_Hourly_Quartal_1min_Processed_from_2008_downcast.csv"
    elif period == "3H":
        fname = f"{instrument}_3H_Quartal_1min_Processed_from_2016.csv"
    else:
        raise ValueError("period must be '1H' or '3H'")
    url = f"{base}/{fname}"
    try:
        return pd.read_csv(url)
    except Exception:
        # fallback to empty DF if file not found or network hiccup
        return pd.DataFrame()


# ↓ in your sidebar:
instrument_options = ["ES", "NQ", "YM", "CL", "GC", "NG", "SI", "E6", "FDAX"]
selected_instrument = st.sidebar.selectbox("Instrument", instrument_options)

# ↓ now pull exactly one file per timeframe:
df_1h = load_quartal_for_instrument(selected_instrument, period="1H")

# ✅ Store username-password pairs
USER_CREDENTIALS = {
    "badboyz": "bangbang",
    "dreamteam" : "strike",
}

# ✅ Initialize session state for authentication
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None

# ✅ Login form (only shown if not authenticated)
if not st.session_state["authenticated"]:
    st.title("Login to Database")

    # Username and password fields
    username = st.text_input("Username:")
    password = st.text_input("Password:", type="password")

    # Submit button
    if st.button("Login"):
        if username in USER_CREDENTIALS and password == USER_CREDENTIALS[username]:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username  # Store the username
            # ← Clear *all* @st.cache_data caches here:
            st.cache_data.clear()

            st.success(f"Welcome, {username}! Loading fresh data…")
            st.rerun()
        else:
            st.error("Incorrect username or password. Please try again.")

    # Stop execution if user is not authenticated
    st.stop()

# ✅ If authenticated, show the full app
st.title("Quartal Database")

for col in [
    'Instrument','Q1_direction','Q2_direction','Q3_direction','Q4_direction',
    'Q1_direction_from_open','Q2_direction_from_open','Q3_direction_from_open','Q4_direction_from_open'
    '0_5_ORB_direction','0_5_ORB_valid', '0_5_ORB_conf_bucket',
    '5_10_ORB_direction','5_10_ORB_valid', '5_10_ORB_conf_bucket',
    'hour_direction',
    'day_of_week','phh_hit_bucket','phl_hit_bucket',
    'low_bucket','high_bucket'
]:
    if col in df_1h:
        df_1h[col] = df_1h[col].astype('category')

if df_1h is not None:

    hour_options = ['All'] + list(range(0, 24))
    hour_options.remove(17)
    three_hour_options = ['All'] + [0, 3, 6, 9, 12, 15, 18, 21]
    selected_hour = st.sidebar.selectbox("Select Hour", hour_options)
    day_options = ['All'] + ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    selected_day = st.sidebar.selectbox("Day of Week", day_options)
    selected_quarter_measurement = st.sidebar.selectbox("Measure Quarter From", ["Hourly Open", "Quarterly Open"])


    #Filters
    SIZE_BINS_0_5 = {
    '0% to 25%': (0.00, 0.25),
    '25% to 50%': (0.25, 0.50),
    '50% to 75%': (0.50, 0.75),
    '75% to 100%': (0.75, 1.00),
    }
    # — Row 1 —
    row1_cols = st.columns([1, 1, 1, 1, 1, 1, 1, 1])
    with row1_cols[0]:
        q1_filter = st.radio(
            "Q1",
            options=["All"] + sorted(df_1h["Q1_direction"].dropna().unique().tolist()),
            horizontal=False
        )
    with row1_cols[1]:
        q2_filter = st.radio(
            "Q2",
            options=["All"] + sorted(df_1h["Q2_direction"].dropna().unique().tolist()),
            horizontal=False
        )
    with row1_cols[2]:
        q3_filter = st.radio(
            "Q3",
            options=["All"] + sorted(df_1h["Q3_direction"].dropna().unique().tolist()),
            horizontal=False
        )
    with row1_cols[3]:
        q4_filter = st.radio(
            "Q4",
            options=["All"] + sorted(df_1h["Q4_direction"].dropna().unique().tolist()),
            horizontal=False
        )
    with row1_cols[4]:
        hourly_open_position = st.radio(
            "Hourly Open Position",
            options=["All"] + list(SIZE_BINS_0_5.keys()),
            horizontal=False
        )
    with row1_cols[5]:
        phh_hit_time_filter = st.radio(
            "PHH Hit Time",
            options=["All"] + sorted(df_1h["phh_hit_bucket"].dropna().unique().tolist()),
            horizontal=False
        )
    with row1_cols[6]:
        phl_hit_time_filter = st.radio(
            "PHL Hit Time",
            options=["All"] + sorted(df_1h["phl_hit_bucket"].dropna().unique().tolist()),
            horizontal=False
        )
    with row1_cols[7]:
        low_filter = st.multiselect(
            "Low Exclusion",
            options=sorted(df_1h["low_bucket"].dropna().unique().tolist())
        )
        high_filter = st.multiselect(
            "High Exclusion",
            options=sorted(df_1h["high_bucket"].dropna().unique().tolist())
        )

    
    # — Row 2 —
    row2_cols = st.columns([1, 1, 1, 1, 1, 1, 1, 1])
    with row2_cols[0]:
        orb_filter = st.radio(
            "0-5 ORB Direction",
            options=["All"] + sorted(df_1h["0_5_ORB_direction"].dropna().unique().tolist()),
            horizontal=False
        )
    with row2_cols[1]:
        orb_true_filter = st.radio(
            "0-5 ORB True/False",
            options=["All"] + [True, False],
            horizontal=False
        )
    with row2_cols[2]:
        orb_conf_filter = st.radio(
            "0-5 ORB Conf. Quarter",
            options=["All"] + ["Q1", "Q2", "Q3", "Q4"],
            horizontal=False
        )
    with row2_cols[3]:
        orb_size_filter = st.radio(
            "0-5 ORB Body / Wicks",
            options=["All"] + list(SIZE_BINS_0_5.keys()),
            horizontal=False
        )
    with row2_cols[4]:
        orb_filter_5_10 = st.radio(
            "5-10 ORB Direction",
            options=["All"] + sorted(df_1h["5_10_ORB_direction"].dropna().unique().tolist()),
            horizontal=False
        )
    with row2_cols[5]:
        orb_true_filter_5_10 = st.radio(
            "5-10 ORB True/False",
            options=["All"] + [True, False],
            horizontal=False
        )
    with row2_cols[6]:
        orb_conf_filter_5_10 = st.radio(
            "5-10 ORB Conf. Quarter",
            options=["All"] + ["Q1", "Q2", "Q3", "Q4"],
            horizontal=False
        )
    with row2_cols[7]:
        orb_size_filter_5_10 = st.radio(
            "5-10 ORB Body / Wicks",
            options=["All"] + list(SIZE_BINS_0_5.keys()),
            horizontal=False
        )

    ###  Apply Filters
    filtered_df_1h = df_1h[df_1h['Instrument'] == selected_instrument]

    # Optional: Apply hour filter (if it's not "All")
    if selected_hour != 'All':
        # Assumes you have a column like 'Hour' as int. If not, adapt accordingly.
        filtered_df_1h = filtered_df_1h[filtered_df_1h['hour'] == selected_hour]

    # Optional: Apply day filter (if it's not "All")
    if selected_day != 'All':
        # Assumes you have a column like 'Day' with string values like 'Monday'
        filtered_df_1h = filtered_df_1h[filtered_df_1h['day_of_week'] == selected_day]

    # Filter by Q directions
    quarter_col_label = 'direction' if selected_quarter_measurement=="Quarterly Open" else "direction_from_open"
    
    if q1_filter != "All":
        filtered_df_1h = filtered_df_1h[filtered_df_1h[f'Q1_{quarter_col_label}'] == q1_filter]
    if q2_filter != "All":
        filtered_df_1h = filtered_df_1h[filtered_df_1h[f'Q2_{quarter_col_label}'] == q2_filter]
    if q3_filter != "All":
        filtered_df_1h = filtered_df_1h[filtered_df_1h[f'Q3_{quarter_col_label}'] == q3_filter]
    if q4_filter != "All":
        filtered_df_1h = filtered_df_1h[filtered_df_1h[f'Q4_{quarter_col_label}'] == q4_filter]
    if orb_filter != 'All':
        filtered_df_1h = filtered_df_1h[filtered_df_1h['0_5_ORB_direction'] == orb_filter] 
    if orb_true_filter != 'All':
        filtered_df_1h = filtered_df_1h[filtered_df_1h['0_5_ORB_valid'] == orb_true_filter] 
    if orb_conf_filter != 'All':
        filtered_df_1h = filtered_df_1h[filtered_df_1h['0_5_ORB_conf_bucket'] == orb_conf_filter] 

    if orb_size_filter != 'All':
            low, high = SIZE_BINS_0_5[orb_size_filter]
            # filter on the absolute value
            filtered_df_1h = filtered_df_1h[
                filtered_df_1h['0_5_ORB_body_size'].abs().between(low, high, inclusive='left')
            ]
    
    if orb_filter_5_10 != 'All':
        filtered_df_1h = filtered_df_1h[filtered_df_1h['5_10_ORB_direction'] == orb_filter_5_10] 
    if orb_true_filter_5_10 != 'All':
        filtered_df_1h = filtered_df_1h[filtered_df_1h['5_10_ORB_valid'] == orb_true_filter_5_10] 
    if orb_conf_filter_5_10 != 'All':
        filtered_df_1h = filtered_df_1h[filtered_df_1h['5_10_ORB_conf_bucket'] == orb_conf_filter_5_10] 

    if orb_size_filter_5_10 != 'All':
        low, high = SIZE_BINS_0_5[orb_size_filter_5_10]
        # filter on the absolute value
        filtered_df_1h = filtered_df_1h[
            filtered_df_1h['5_10_ORB_body_size'].abs().between(low, high, inclusive='left')
        ]
        
    if hourly_open_position != 'All':
        low, high = SIZE_BINS_0_5[hourly_open_position]
        # inclusive on left, exclusive on right
        filtered_df_1h = filtered_df_1h[
            filtered_df_1h['hourly_open_position']
                         .between(low, high, inclusive='left')
        ]

    if phh_hit_time_filter != 'All':
        filtered_df_1h = filtered_df_1h[filtered_df_1h['phh_hit_bucket'] == phh_hit_time_filter] 
    if phl_hit_time_filter != 'All':
        filtered_df_1h = filtered_df_1h[filtered_df_1h['phl_hit_bucket'] == phl_hit_time_filter] 
        
    if low_filter:
        filtered_df_1h = filtered_df_1h[~filtered_df_1h['low_bucket'].isin(low_filter)]
    if high_filter:
        filtered_df_1h = filtered_df_1h[~filtered_df_1h['high_bucket'].isin(high_filter)]

    # Create two side-by-side columns
    col0, col1, col2, col3, _ = st.columns([1.5, 3, 1.5, 3, 5])
    
    # 0–5 ORB True Rate
    if '0_5_ORB_valid' in filtered_df_1h.columns and not filtered_df_1h.empty:
        orb0_5 = filtered_df_1h['0_5_ORB_valid'].value_counts(normalize=True)
        rate0_5 = orb0_5.get(True, 0)
        col0.metric(
            label="0–5 ORB True Rate",
            value=f"{rate0_5:.2%}"
        )
    # 0–5 ORB Return To Hourly Open
    if '0_5_ORB_retrace_to_hourly_open' in filtered_df_1h.columns and not filtered_df_1h.empty:
        orb0_5_hourly_hit = filtered_df_1h['0_5_ORB_retrace_to_hourly_open'].value_counts(normalize=True)
        rateorb0_5_hourly_hit = orb0_5_hourly_hit.get(True, 0)
        col1.metric(
            label="0-5 Retrace to Hourly Open After Conf.",
            value=f"{rateorb0_5_hourly_hit:.2%}"
        )
    
    # 5–10 ORB True Rate
    if '5_10_ORB_valid' in filtered_df_1h.columns and not filtered_df_1h.empty:
        orb5_10 = filtered_df_1h['5_10_ORB_valid'].value_counts(normalize=True)
        rate5_10 = orb5_10.get(True, 0)
        col2.metric(
            label="5–10 ORB True Rate",
            value=f"{rate5_10:.2%}"
        )

    # 0–5 ORB Return To Hourly Open
    if '5_10_ORB_retrace_to_hourly_open' in filtered_df_1h.columns and not filtered_df_1h.empty:
        orb5_10_hourly_hit = filtered_df_1h['5_10_ORB_retrace_to_hourly_open'].value_counts(normalize=True)
        rate5_10_hourly_hit = orb5_10_hourly_hit.get(True, 0)
        col3.metric(
            label="5-10 Retrace to Hourly Open After Conf.",
            value=f"{rate5_10_hourly_hit:.2%}"
        )

    # Calculate probability distributions for "low bucket" and "high bucket"
    low_counts = filtered_df_1h["low_bucket"].value_counts(normalize=True).reset_index()
    low_counts.columns = ["value", "probability"]

    high_counts = filtered_df_1h["high_bucket"].value_counts(normalize=True).reset_index()
    high_counts.columns = ["value", "probability"]

    # Create a bar chart for "low bucket" probabilities with text annotations
    desired_order = ["Q1", "Q2", "Q3", "Q4"]
    fig_low = px.bar(
        low_counts,
        x="value",
        y="probability",
        title="Low of Hour Bucket",
        labels={"value": "Low Bucket", "probability": "Probability"},
        # Format the probability as a percentage (e.g., "12.34%")
        text=low_counts["probability"].apply(lambda x: f"{x:.2%}")
    )
    # Position the text annotations outside the bars
    fig_low.update_traces(textposition="outside")
    fig_low.update_layout(
        title={'x': 0.5, 'xanchor': 'center'},
        xaxis=dict(
        categoryorder='array',
        categoryarray=desired_order
    )
)

    # Create a bar chart for "high bucket" probabilities with text annotations
    fig_high = px.bar(
        high_counts,
        x="value",
        y="probability",
        title="High of Hour Bucket",
        labels={"value": "High Bucket", "probability": "Probability"},
        text=high_counts["probability"].apply(lambda x: f"{x:.2%}")
    )
    fig_high.update_traces(textposition="outside")
    fig_high.update_layout(
        title={'x': 0.5, 'xanchor': 'center'},
        xaxis=dict(
        categoryorder='array',
        categoryarray=desired_order
    )
)

    # Here, the proportion (mean) of True values in a boolean series represents the percentage hit.
    phh_hit_pct = filtered_df_1h['phh_hit'].mean()
    phl_hit_pct = filtered_df_1h['phl_hit'].mean()
    pmid_hit_pct = filtered_df_1h['pmid_hit'].mean()
    
    # Create a DataFrame for plotting
    hit_pct_df = pd.DataFrame({
        'Hit Type': ['PHH Hit', 'PHL Hit', 'PHM Hit'],
        'Percentage': [phh_hit_pct, phl_hit_pct, pmid_hit_pct]
    })
    
    # Create a bar chart for hit percentages
    fig_hits = px.bar(
        hit_pct_df,
        x="Hit Type",
        y="Percentage",
        title="PHH / PHL / PHM Hit Rate",
        labels={"Hit Type": "Hit Type", "Percentage": "Hit Percentage"},
        text=hit_pct_df["Percentage"].apply(lambda x: f"{x:.2%}")
    )
    fig_hits.update_layout(title={'x': 0.5, 'xanchor': 'center'})

    fig_hits.update_traces(textposition='outside')
    fig_hits.update_yaxes(range=[0, 1])

    # Display the two charts side by side using st.columns
    st.markdown("### High / Low of Hour and Previous Hourly Levels")
    col1, col2, col3 = st.columns((1, 2, 2))
    col1.plotly_chart(fig_hits, use_container_width=True)
    col2.plotly_chart(fig_low, use_container_width=True)
    col3.plotly_chart(fig_high, use_container_width=True)

    # 1) Define your core 0.1-wide bins between 0 and 1
    width = 0.1
    core_edges = np.arange(0, 1 + width, width)   # [0.0, 0.1, …, 1.0]
    
    # 2) Prepend -inf and append +inf so that anything <0 or >1 falls into the end buckets
    bins  = np.concatenate(([-np.inf], core_edges, [np.inf]))
    # Build matching labels: "<0.0", "0.0–0.1", …, "0.9–1.0", ">1.0"
    labels = (
        [f"<{core_edges[0]:.1f}"] +
        [f"{core_edges[i]:.1f}–{core_edges[i+1]:.1f}" for i in range(len(core_edges)-1)] +
        [f">{core_edges[-1]:.1f}"]
    )
    
    def bucket_and_count(series):
        cat = pd.cut(
            series.dropna(),      # drop NaN
            bins=bins,
            labels=labels,
            include_lowest=True
        )
        cnt = cat.value_counts().reindex(labels, fill_value=0).reset_index()
        cnt.columns = ["bucket", "count"]
        cnt["cum_pct"] = cnt["count"].cumsum() / cnt["count"].sum()
        return cnt
    
    # 3) Apply to your filtered DataFrame
    cnt_0_5  = bucket_and_count(filtered_df_1h["0_5_ORB_max_retracement"])
    cnt_5_10 = bucket_and_count(filtered_df_1h["5_10_ORB_max_retracement"])
    
    # 4) Plot side-by-side
    st.markdown("### ORB Max Retracement Distribution")
    
    col_ret0, col_ret1 = st.columns(2)
    
    with col_ret0:
        fig_ret0 = px.bar(
            cnt_0_5,
            x="bucket",
            y="count",                    # <-- use "count"
            title="0–5 ORB Max Retracement",
            labels={"bucket":"Retracement Interval","count":"Count"},
            text=cnt_0_5['cum_pct'].apply(lambda x: f"{x:.1%}"),
            color_discrete_sequence=['#3366cc']
        )
        fig_ret0.update_traces(textposition="outside")
        fig_ret0.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_ret0, use_container_width=True)
    
    with col_ret1:
        fig_ret1 = px.bar(
            cnt_5_10,
            x="bucket",
            y="count",                    # <-- also just "count"
            title="5–10 ORB Max Retracement",
            labels={"bucket":"Retracement Interval","count":"Count"},
            text=cnt_5_10['cum_pct'].apply(lambda x: f"{x:.1%}"),
            color_discrete_sequence=['#3366cc']
        )
        fig_ret1.update_traces(textposition="outside")
        fig_ret1.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_ret1, use_container_width=True)

    # Calculate distribution of hour_direction in the filtered data
    # Normalize direction values
    filtered_df_1h['hour_direction'] = filtered_df_1h['hour_direction'].str.strip().str.title()
    
    # Recalculate counts
    hour_direction_counts = filtered_df_1h['hour_direction'].value_counts().reset_index()
    hour_direction_counts.columns = ['direction', 'count']
    
    direction_order = ["Long", "Short", "Neutral"]
    direction_colors = {
        "Long": "#2ecc71",       # Green
        "Short": "#e74c3c",     # Red
        "Neutral": "#5d6d7e"   # Gray
    }
    
    
    st.markdown("### Quarter and Hourly Direction")
    
    quartals = ["Q1_direction", "Q2_direction", "Q3_direction", "Q4_direction", "hour_direction"]
    quartal_titles = ["Q1 Direction", "Q2 Direction", "Q3 Direction", "Q4 Direction", "Hour Direction"]
    
    q_cols = st.columns(5)
    
    for i, q_col in enumerate(quartals):
        # Normalize and count values
        filtered_df_1h[q_col] = filtered_df_1h[q_col].str.strip().str.title()
        q_counts = filtered_df_1h[q_col].value_counts().reset_index()
        q_counts.columns = ['direction', 'count']
    
        # Build pie chart
        fig_q = px.pie(
            q_counts,
            names='direction',
            values='count',
            color='direction',
            title=quartal_titles[i],
            hole=0.3,
            category_orders={'direction': direction_order},
            color_discrete_map=direction_colors
        )
        fig_q.update_traces(textinfo='percent+label')
        q_cols[i].plotly_chart(fig_q, use_container_width=True)

    st.caption(f"Sample size: {len(filtered_df_1h):,} rows")
