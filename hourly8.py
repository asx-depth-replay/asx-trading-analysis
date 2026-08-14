import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import datetime
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

# --- GOOGLE DRIVE HELPERS ---
def get_gdrive_service():
    """Authenticates using Streamlit Secrets."""
    creds = service_account.Credentials.from_service_account_info(st.secrets["gdrive_service_account"])
    return build('drive', 'v3', credentials=creds)

def list_drive_folders(parent_id):
    """Lists subfolders (Tickers) in the root folder."""
    service = get_gdrive_service()
    query = f"'{parent_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    return {f['name']: f['id'] for f in results.get('files', [])}

def list_files_in_folder(folder_id):
    """Lists all parquet/csv files in a specific ticker's folder."""
    service = get_gdrive_service()
    query = f"'{folder_id}' in parents and (name contains '.parquet' or name contains '.csv') and trashed = false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    return {f['name']: f['id'] for f in results.get('files', [])}

def download_from_gdrive(file_id):
    """Downloads file from Drive into memory."""
    service = get_gdrive_service()
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)
    return fh

# --- App Configuration ---
st.set_page_config(layout="wide", page_title="Trading Session Review")

# --- Data Loading and Caching ---
@st.cache_data
def load_sales_data(uploaded_file, trade_date, filename=None):
    """
    Loads and processes the Course of Sales data, aligning it with the trade date from the depth file.
    """
    try:
        # Resolve filename: prefer explicit param, fall back to attribute on the file object
        _name = filename or getattr(uploaded_file, 'name', '')
        # Check if it's Parquet first (Parquet never has encoding errors!)
        if _name.endswith('.parquet'):
            df = pd.read_parquet(uploaded_file)
        else:
            # TRY CP1252 FIRST (Standard for ASX/Windows CSVs)
            try:
                df = pd.read_csv(uploaded_file, encoding='cp1252')
            except UnicodeDecodeError:
                # FALLBACK TO LATIN1 if CP1252 fails
                df = pd.read_csv(uploaded_file, encoding='latin1')
        df.rename(columns={'Price $': 'Price', 'Volume Traded': 'Volume', 'Volume': 'Volume'}, inplace=True)
        
        # Clean numeric columns by removing commas
        for col in ['Price', 'Volume']:
             if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')

        # Combine the trade date with the time from the sales file
        time_series = pd.to_datetime(df['Time'], format='%I:%M:%S %p', errors='coerce').dt.time
        df['datetime'] = time_series.apply(lambda t: datetime.datetime.combine(trade_date, t) if pd.notnull(t) else pd.NaT)

        df.dropna(subset=['datetime', 'Price', 'Volume'], inplace=True)
        
        # --- NEW: Fill blank conditions with a readable label ---
        if 'Condition' in df.columns:
            df['Condition'] = df['Condition'].fillna('Lit Order Book')
        else:
            df['Condition'] = 'Lit Order Book'
            
        if 'Market' not in df.columns:
             df['Market'] = 'Unknown'
        # --------------------------------------------------------
        
        return df
    except Exception as e:
        st.error(f"Error loading Course of Sales data: {e}")
        return None

@st.cache_data
def load_depth_data(uploaded_file, filename=None):
    """
    Loads and processes the Market Depth data with robust cleaning.
    """
    try:
        # Resolve filename: prefer explicit param, fall back to attribute on the file object
        _name = filename or getattr(uploaded_file, 'name', '')
        # Check if it's a parquet file first
        if _name.endswith('.parquet'):
            df = pd.read_parquet(uploaded_file)
        else:
            # Existing CSV logic
            df = pd.read_csv(uploaded_file)
            df.columns = ['Date', 'Time', 'Ticker', 'Type', 'Price', 'Volume', 'Number_of_Orders']
        
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d %b %Y %I:%M:%S %p', errors='coerce')
        
        for col in ['Price', 'Volume', 'Number_of_Orders']:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')

        df.dropna(subset=['datetime', 'Price', 'Volume', 'Type'], inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading Market Depth data: {e}")
        return None

@st.fragment
def order_book_explorer_fragment(df_depth, df_sales):
    st.markdown("Drill down into the exact order book state at any given second.")
    available_times = sorted(df_depth['datetime'].unique())

    if not available_times:
        st.error("No valid time entries found.")
        return

    # 1. Initialize the slider's state if it doesn't exist
    # We use the actual time value as the state, not the index number
    if "snapshot_slider" not in st.session_state:
        st.session_state["snapshot_slider"] = available_times[0]

    # 2. Define Callbacks to shift the time
    def move_snap(direction):
        current_val = st.session_state["snapshot_slider"]
        current_idx = available_times.index(current_val)
        
        if direction == "next" and current_idx < len(available_times) - 1:
            st.session_state["snapshot_slider"] = available_times[current_idx + 1]
        elif direction == "prev" and current_idx > 0:
            st.session_state["snapshot_slider"] = available_times[current_idx - 1]

    # 3. Layout: Navigation
    col_prev, col_slide, col_next = st.columns([1, 8, 1])
    
    with col_prev:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("⬅️", on_click=move_snap, args=("prev",), use_container_width=True)

    with col_next:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("➡️", on_click=move_snap, args=("next",), use_container_width=True)

    with col_slide:
        # 4. The Slider - 'value' is omitted because 'key' handles it automatically
        snapshot_time = st.select_slider(
            "Select Snapshot Time:", 
            options=available_times, 
            format_func=lambda x: pd.to_datetime(x).strftime('%I:%M:%S %p'),
            key="snapshot_slider" 
        )

    # 5. Depth Selection
    depth_opt = st.radio("Order Book Depth Display:", ['Top 10', 'Top 20', 'Full Book'], index=2, horizontal=True)

    # Fast filtering logic
    snapshot_df = df_depth[df_depth['datetime'] == snapshot_time]
    
    # 1. Filter and sort Bids
    bids = snapshot_df[snapshot_df['Type'] == 'BUY'].sort_values('Price', ascending=False)
    # Select only the columns you want, in the order of the first screenshot
    bids_display = bids[['Number_of_Orders', 'Volume', 'Price']].reset_index(drop=True)

    # 2. Filter and sort Asks
    asks = snapshot_df[snapshot_df['Type'] == 'SELL'].sort_values('Price', ascending=True)
    # Select only the columns you want
    asks_display = asks[['Price', 'Volume', 'Number_of_Orders']].reset_index(drop=True)

    # 3. Handle the "Depth" radio button logic
    if depth_opt != 'Full Book':
        depth_val = int(depth_opt.split(' ')[1])
        bids_display = bids_display.head(depth_val)
        asks_display = asks_display.head(depth_val)

    # 4. Render the clean tables
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Buyers (Bids)")
        # use hide_index=True to remove the row numbers on the far left
        st.dataframe(bids_display, use_container_width=True, hide_index=False)
    with c2:
        st.subheader("Sellers (Asks)")
        st.dataframe(asks_display, use_container_width=True, hide_index=False)
        
@st.cache_data
def generate_footprint_data(sales_df, depth_df, timeframe_str):
    # 1. Extract Bids and Asks
    bids = depth_df[depth_df['Type'] == 'BUY'].groupby('datetime')['Price'].max().rename('best_bid')
    asks = depth_df[depth_df['Type'] == 'SELL'].groupby('datetime')['Price'].min().rename('best_ask')
    quotes = pd.concat([bids, asks], axis=1).reset_index().sort_values('datetime')
    quotes.dropna(inplace=True)
    
    # 2. Merge Quotes with Sales
    sales = sales_df.sort_values('datetime').copy()
    sales = pd.merge_asof(sales, quotes, on='datetime', direction='backward')
    sales['best_bid'] = sales['best_bid'].fillna(sales['Price'])
    sales['best_ask'] = sales['best_ask'].fillna(sales['Price'])
    
    # 3. Classify Bid vs Ask Volume
    sales['BidVolume'] = 0.0
    sales['AskVolume'] = 0.0
    sales['prev_price'] = sales['Price'].shift(1)
    sales['tick_dir'] = np.where(sales['Price'] > sales['prev_price'], 1, np.where(sales['Price'] < sales['prev_price'], -1, 0))
    sales['tick_dir'] = sales['tick_dir'].replace(0, np.nan).ffill().fillna(1)
    
    is_ask_hit = (sales['Price'] >= sales['best_ask'])
    is_bid_hit = (sales['Price'] <= sales['best_bid'])
    is_mid = ~(is_ask_hit | is_bid_hit)
    
    sales.loc[is_ask_hit, 'AskVolume'] = sales.loc[is_ask_hit, 'Volume']
    sales.loc[is_bid_hit, 'BidVolume'] = sales.loc[is_bid_hit, 'Volume']
    sales.loc[is_mid & (sales['tick_dir'] == 1), 'AskVolume'] = sales.loc[is_mid & (sales['tick_dir'] == 1), 'Volume']
    sales.loc[is_mid & (sales['tick_dir'] == -1), 'BidVolume'] = sales.loc[is_mid & (sales['tick_dir'] == -1), 'Volume']
    
    # --- Calculate Tick-Level Data for Excel Export & VWAP ---
    sales['Delta'] = sales['AskVolume'] - sales['BidVolume']
    sales['CVD'] = sales['Delta'].cumsum()
    sales['Liquidity_Sign'] = np.where(sales['AskVolume'] > 0, 'ASK', np.where(sales['BidVolume'] > 0, 'BID', 'NONE'))
    
    # Session VWAP Calculation (Tick-by-Tick)
    sales['Cum_Vol'] = sales['Volume'].cumsum()
    sales['Cum_Vol_Price'] = (sales['Price'] * sales['Volume']).cumsum()
    sales['VWAP'] = sales['Cum_Vol_Price'] / sales['Cum_Vol']
    
    # 4. Group by Timeframe
    sales.set_index('datetime', inplace=True)
    sales['candle_time'] = sales.index.floor(timeframe_str)
    
    # Generate the Footprint nodes first so we can find the max volume node
    footprint = sales.groupby(['candle_time', 'Price'])[['BidVolume', 'AskVolume']].sum().reset_index()
    footprint = footprint[(footprint['BidVolume'] > 0) | (footprint['AskVolume'] > 0)]
    footprint['Total_Node_Vol'] = footprint['BidVolume'] + footprint['AskVolume']
    
    # --- Calculate the Candle POC ---
    # Find the index of the highest volume node for each candle, then map the price
    poc_idx = footprint.groupby('candle_time')['Total_Node_Vol'].idxmax()
    candle_poc = footprint.loc[poc_idx, ['candle_time', 'Price']].set_index('candle_time')['Price']
    
    # Aggregate OHLC and grab the closing VWAP AND the new Candle POC
    ohlc = sales.groupby('candle_time').agg(
        Open=('Price', 'first'),
        High=('Price', 'max'),
        Low=('Price', 'min'),
        Close=('Price', 'last'),
        VWAP=('VWAP', 'last')
    )
    ohlc['POC'] = candle_poc # Add the POC column to our master OHLC dataframe
    
    # 5. Full Day Stats & CVD Calculation (For the chart)
    stats = footprint.groupby('candle_time').agg(
        Total_Vol=('BidVolume', lambda x: x.sum() + footprint.loc[x.index, 'AskVolume'].sum()),
        Delta=('AskVolume', lambda x: x.sum() - footprint.loc[x.index, 'BidVolume'].sum())
    )
    stats = stats.reindex(ohlc.index).fillna(0)
    stats['CVD'] = stats['Delta'].cumsum()
    
    return footprint, ohlc, stats, sales.reset_index()

# --- Analysis & Visualisation Functions ---

def create_continuous_footprint_chart(ohlc_df, footprint_df, stats_df, tf_mins, show_footprints=False, bin_size=0.0):

    # --- Anchor CVD to the start of the selected viewing window ---
    stats_df = stats_df.copy() # Prevent Pandas SettingWithCopy warnings
    stats_df['CVD'] = stats_df['Delta'].cumsum()
    
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
    )

    # Combine stats into OHLC for rich hover text data
    ohlc_hover_df = ohlc_df.join(stats_df)

    # 1.Calculate VAH, VAL, and POC *first* so we can inject them into the hover text
    window_vah, window_val, window_poc = None, None, None
    if not footprint_df.empty:
        footprint_copy = footprint_df.copy()
        footprint_copy['Total_Node_Vol'] = footprint_copy['BidVolume'] + footprint_copy['AskVolume']

        # --- Dynamically Calculate Anchored VWAP ---
        # Calculate volume * price for each candle, then cumsum for the window
        vol_price = footprint_copy.groupby('candle_time').apply(lambda x: (x['Price'] * x['Total_Node_Vol']).sum())
        vol = footprint_copy.groupby('candle_time')['Total_Node_Vol'].sum()
        dynamic_vwap = (vol_price.cumsum() / vol.cumsum()).rename('Dynamic_VWAP')
        ohlc_hover_df = ohlc_hover_df.join(dynamic_vwap)
        ohlc_hover_df['VWAP'] = ohlc_hover_df['Dynamic_VWAP'].ffill() # Overwrite the old VWAP

        # --- Apply Global Bin Size to Footprint Value Area ---
        if bin_size > 0:
            min_price, max_price = footprint_copy['Price'].min(), footprint_copy['Price'].max()
            min_bin = np.floor(min_price / bin_size) * bin_size
            max_bin = np.ceil(max_price / bin_size) * bin_size
            bins = np.arange(min_bin, max_bin + (bin_size * 1.5), bin_size) 
            footprint_copy['price_bin'] = pd.cut(footprint_copy['Price'], bins=bins, right=False, labels=bins[:-1])
            profile = footprint_copy.groupby('price_bin', observed=False)[['BidVolume', 'AskVolume']].sum()
        else:
            profile = footprint_copy.groupby('Price')[['BidVolume', 'AskVolume']].sum()
            
        profile['Total_Vol'] = profile['BidVolume'] + profile['AskVolume']
        total_vol = profile['Total_Vol'].sum()
        
        if total_vol > 0:
            window_poc = profile['Total_Vol'].idxmax()
            va_target = total_vol * 0.70
            
            # --- Greedy Expansion Algorithm ---
            prices = np.sort(profile.index.astype(float).values)
            vp_dict = {float(k): float(v) for k, v in profile['Total_Vol'].items()}
            poc_val = float(window_poc)
            
            poc_idx = np.where(prices == poc_val)[0][0]
            high_idx = poc_idx
            low_idx = poc_idx
            current_vol = vp_dict[poc_val]

            # Expand up or down based on which adjacent price level has more volume
            while current_vol < va_target:
                up_vol = vp_dict[prices[high_idx + 1]] if high_idx + 1 < len(prices) else -1
                down_vol = vp_dict[prices[low_idx - 1]] if low_idx - 1 >= 0 else -1

                if up_vol == -1 and down_vol == -1:
                    break # Reached the edges

                if up_vol >= down_vol:
                    high_idx += 1
                    current_vol += up_vol
                else:
                    low_idx -= 1
                    current_vol += down_vol

            window_val = prices[low_idx]
            window_vah = prices[high_idx]

    # 2. Pre-format the HTML hover text
    hover_text = []
    for index, row in ohlc_hover_df.iterrows():
        time_str = index.strftime('%H:%M') 
        
        # Format the Value Area string if data exists
        va_text = ""
        if window_poc is not None:
            va_text = f"<b>VAH:</b> {window_vah:.2f} | <b>POC:</b> {window_poc:.2f} | <b>VAL:</b> {window_val:.2f}<br>"
            
        txt = (
            f"<b>Time:</b> {time_str}<br>"
            f"<b>O:</b> {row['Open']:.2f} <b>H:</b> {row['High']:.2f}<br>"
            f"<b>L:</b> {row['Low']:.2f} <b>C:</b> {row['Close']:.2f}<br>"
            f"<b>VWAP:</b> {row['VWAP']:.2f}<br>"
            f"{va_text}"
            f"---<br>"
            f"<b>Total Vol:</b> {row['Total_Vol']:,.0f}<br>"
            f"<b>Delta:</b> {row['Delta']:,.0f}<br>"
            f"<b>CVD:</b> {row['CVD']:,.0f}"
        )
        hover_text.append(txt)

    # 3. Main Candlesticks with Information Box Hover
    fig.add_trace(go.Candlestick(
        x=ohlc_hover_df.index,
        open=ohlc_hover_df['Open'], high=ohlc_hover_df['High'],
        low=ohlc_hover_df['Low'], close=ohlc_hover_df['Close'],
        increasing_line_color='mediumseagreen', increasing_fillcolor='rgba(60, 179, 113, 0.2)',
        decreasing_line_color='lightcoral', decreasing_fillcolor='rgba(240, 128, 128, 0.2)',
        name="Price",
        text=hover_text,        
        hoverinfo='text'        
    ), row=1, col=1)

    # 4. Draw the VWAP Line
    fig.add_trace(go.Scatter(
        x=ohlc_hover_df.index,
        y=ohlc_hover_df['VWAP'],
        mode='lines',
        name='VWAP',
        line=dict(color='darkorange', width=2, dash='dot'),
        hoverinfo='skip' 
    ), row=1, col=1)

    # 5. Plot VAH, POC, and VAL as Scatter Traces
    if window_poc is not None:
        start_x = ohlc_hover_df.index.min()
        end_x = ohlc_hover_df.index.max()
        
        # VAH
        fig.add_trace(go.Scatter(
            x=[start_x, end_x], y=[window_vah, window_vah],
            mode='lines+text', name='VAH', text=['', 'VAH'],
            textposition='bottom left', textfont=dict(color='purple', size=11, family="Arial Black"),
            line=dict(color='purple', width=1.5, dash='dot'), hoverinfo='skip'
        ), row=1, col=1)
        
        # POC
        fig.add_trace(go.Scatter(
            x=[start_x, end_x], y=[window_poc, window_poc],
            mode='lines+text', name='POC', text=['', 'POC'],
            textposition='top left', textfont=dict(color='red', size=11, family="Arial Black"),
            line=dict(color='red', width=2, dash='dash'), hoverinfo='skip'
        ), row=1, col=1)

        # VAL
        fig.add_trace(go.Scatter(
            x=[start_x, end_x], y=[window_val, window_val],
            mode='lines+text', name='VAL', text=['', 'VAL'],
            textposition='top left', textfont=dict(color='purple', size=11, family="Arial Black"),
            line=dict(color='purple', width=1.5, dash='dot'), hoverinfo='skip'
        ), row=1, col=1)

    # 6. Draw the Profile Shapes & Text (ONLY if toggle is ON)
    if show_footprints and not footprint_df.empty:
        footprint_df['Total_Node_Vol'] = footprint_df['BidVolume'] + footprint_df['AskVolume']
        footprint_df['Node_Delta'] = footprint_df['AskVolume'] - footprint_df['BidVolume']
        max_vol = footprint_df['Total_Node_Vol'].max() if not footprint_df.empty else 1
        
        max_width_td = pd.Timedelta(minutes=tf_mins * 0.8)
        offset_td = pd.Timedelta(minutes=tf_mins * 0.15)
        
        prices = sorted(footprint_df['Price'].unique())
        tick_size = min(np.diff(prices)) if len(prices) > 1 else 0.01
        half_tick = tick_size / 2

        text_x, text_y, text_labels = [], [], []

        for _, row in footprint_df.iterrows():
            base_time = row['candle_time']
            price = row['Price']
            
            x0 = base_time + offset_td
            width_ratio = row['Total_Node_Vol'] / max_vol
            x1 = x0 + (width_ratio * max_width_td)
            
            if row['Node_Delta'] > 0:
                fill_color, line_color = 'rgba(60, 179, 113, 0.3)', 'rgba(60, 179, 113, 0.8)'
            elif row['Node_Delta'] < 0:
                fill_color, line_color = 'rgba(240, 128, 128, 0.3)', 'rgba(240, 128, 128, 0.8)'
            else:
                fill_color, line_color = 'rgba(128, 128, 128, 0.3)', 'rgba(128, 128, 128, 0.8)'

            fig.add_shape(
                type="rect",
                x0=x0, y0=price - half_tick, x1=x1, y1=price + half_tick,
                fillcolor=fill_color, line=dict(color=line_color, width=1),
                layer="below", row=1, col=1
            )
            
            text_x.append(x0 + (max_width_td * 0.4))
            text_y.append(price)
            text_labels.append(f"{int(row['BidVolume'])} x {int(row['AskVolume'])}")

        fig.add_trace(go.Scatter(
            x=text_x, y=text_y, mode='text', text=text_labels,
            textfont=dict(size=9, color='black', family='Courier New'),
            hoverinfo='skip', showlegend=False
        ), row=1, col=1)

    # 7. Lower Pane: Delta Bars
    colors = ['mediumseagreen' if val >= 0 else 'lightcoral' for val in stats_df['Delta']]
    fig.add_trace(go.Bar(
        x=stats_df.index, y=stats_df['Delta'],
        marker_color=colors, name="Delta",
        hovertemplate='<b>Delta:</b> %{y:,.0f}<extra></extra>' 
    ), row=2, col=1, secondary_y=False)

    # 8. Lower Pane: CVD Line
    fig.add_trace(go.Scatter(
        x=stats_df.index, y=stats_df['CVD'],
        mode='lines', name='CVD',
        line=dict(color='dodgerblue', width=2),
        hovertemplate='<b>CVD:</b> %{y:,.0f}<extra></extra>'
    ), row=2, col=1, secondary_y=True)

    fig.update_layout(
        title=f"Profile Footprint Chart ({tf_mins}-Minute Candles)",
        xaxis_rangeslider_visible=False,
        height=850, margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified", plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    fig.update_yaxes(title_text="Price", tickformat=".2f", row=1, col=1, autorange=True)
    fig.update_yaxes(title_text="Delta", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="CVD", row=2, col=1, secondary_y=True, showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)', row=1, col=1)
    
    return fig

def create_volume_profile_fig(df, bin_size, close_price, selected_markets, selected_conditions, start_time, end_time):
    if df is None or df.empty:
        st.warning("No sales data available for the selected time range to generate Volume Profile.")
        return None
        
    fig, ax = plt.subplots(figsize=(10, 8))
    min_price, max_price = df['Price'].min(), df['Price'].max()
    
    df_copy = df.copy()
    
    # --- Dollar Size Binning ---
    if bin_size > 0:
        # Create exact dollar-value boundaries
        min_bin = np.floor(min_price / bin_size) * bin_size
        max_bin = np.ceil(max_price / bin_size) * bin_size
        bins = np.arange(min_bin, max_bin + (bin_size * 1.5), bin_size) 
        
        df_copy['price_bin'] = pd.cut(df_copy['Price'], bins=bins, right=False, labels=bins[:-1])
        volume_profile = df_copy.groupby('price_bin', observed=False)['Volume'].sum()
        bar_height = bin_size * 0.90
    else:
        # No grouping (Exact traded pennies)
        volume_profile = df_copy.groupby('Price')['Volume'].sum()
        prices = np.sort(df_copy['Price'].unique())
        tick_size = np.min(np.diff(prices)) if len(prices) > 1 else 0.01
        bar_height = tick_size * 0.90
        
    # Remove any empty bins so they don't mess up the Value Area math
    volume_profile = volume_profile[volume_profile > 0]
    
    # --- Key Profile Metrics ---
    total_volume = volume_profile.sum()
    poc = volume_profile.idxmax()

    # --- Calculate Session VWAP ---
    vwap = (df_copy['Price'] * df_copy['Volume']).sum() / total_volume
    
    # --- Value Area Calculation (Greedy Expansion Algorithm) ---
    va_target = total_volume * 0.70
    
    # Ensure prices are standard floats and sorted
    prices = np.sort(volume_profile.index.astype(float).values)
    
    # Convert profile to a standard dict for easy, safe lookups
    vp_dict = {float(k): float(v) for k, v in volume_profile.items()}
    poc_val = float(poc)
    
    poc_idx = np.where(prices == poc_val)[0][0]
    high_idx = poc_idx
    low_idx = poc_idx
    current_vol = vp_dict[poc_val]

    # Expand up or down based on which adjacent price level has more volume
    while current_vol < va_target:
        up_vol = vp_dict[prices[high_idx + 1]] if high_idx + 1 < len(prices) else -1
        down_vol = vp_dict[prices[low_idx - 1]] if low_idx - 1 >= 0 else -1

        if up_vol == -1 and down_vol == -1:
            break # Reached the absolute edges of the traded range

        if up_vol >= down_vol:
            high_idx += 1
            current_vol += up_vol
        else:
            low_idx -= 1
            current_vol += down_vol

    val = prices[low_idx]
    vah = prices[high_idx]

    # 1. Plot ALL bars as a faded gray background (Outside Value Area)
    # Changed align='center' so the lines cut perfectly through the middle of the bars
    ax.barh(volume_profile.index, volume_profile.values, height=bar_height, align='center', color='lightgray', edgecolor='darkgray')
    
    # 2. Overwrite the Value Area bars with vibrant color (Inside Value Area)
    va_profile = volume_profile.loc[val:vah]
    ax.barh(va_profile.index, va_profile.values, height=bar_height, align='center', color='deepskyblue', edgecolor='black')

    # --- Chart Formatting ---
    ax.set_title('Volume Profile', fontsize=16, fontweight='bold')
    ax.set_xlabel('Total Volume')
    ax.set_ylabel('Price Level')
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.2f}'))
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    # --- Lines & Labels ---
    # Total Volume
    ax.plot([], [], ' ', label=f'Total Vol: {total_volume:,.0f}')
    # POC VAH VAL Lines
    ax.axhline(y=vah, color='purple', linestyle=':', linewidth=2, label=f'VAH: ${vah:,.2f}')
    ax.axhline(y=val, color='purple', linestyle=':', linewidth=2, label=f'VAL: ${val:,.2f}')
    ax.axhline(y=poc, color='red', linestyle='--', linewidth=2, label=f'POC: ${poc:,.2f}')
    
    if close_price is not None:
        ax.axhline(y=close_price, color='darkorange', linestyle='-', linewidth=2, label=f'Close Price: ${close_price:,.2f}')

    # --- VWAP Line ---
    ax.axhline(y=vwap, color='black', linestyle='--', linewidth=2, label=f'VWAP: ${vwap:,.2f}')

    # --- Dynamic Filter Footnote (Bottom Left) ---
    # Format the timeframe and lists safely
    time_str = f"{start_time.strftime('%H:%M:%S')} to {end_time.strftime('%H:%M:%S')}"
    markets_str = ", ".join(selected_markets) if selected_markets else "None"
    conditions_str = ", ".join(selected_conditions) if selected_conditions else "None"
    
    footnote_text = (
        f"Data Parameters:\n"
        f"• Time Range: {time_str}\n"
        f"• Markets: {markets_str}\n"
        f"• Trade Conditions: {conditions_str}"
    )
    
    fig.text(x=0.01, y=0.01, s=footnote_text, fontsize=9, color='dimgray', family='sans-serif',
             verticalalignment='bottom', horizontalalignment='left', multialignment='left')

    ax.legend()
    plt.tight_layout(rect=[0, 0.08, 1, 1]) 
    
    return fig

def create_hourly_pivot(df, price_grouping):
    if df is None or df.empty: return pd.DataFrame(), pd.DataFrame()
    df_copy = df.copy()
    df_copy['Hour'] = df_copy['datetime'].dt.hour
    
    if price_grouping > 0:
        # Calculate the bin and immediately round to 2 decimal places
        df_copy['price_bin'] = ((df_copy['Price'] / price_grouping).apply(np.floor) * price_grouping).round(2)
    else:
        df_copy['price_bin'] = df_copy['Price'].round(2)
        
    pivot = pd.pivot_table(df_copy, values='Volume', index='price_bin', columns='Hour', aggfunc='sum', fill_value=0, margins=True, margins_name='Grand Total')
    
    grand_total_row = pivot.loc[['Grand Total']]
    data_rows = pivot.drop('Grand Total')
    sorted_data_rows = data_rows.sort_index(ascending=False)
    pivot = pd.concat([sorted_data_rows, grand_total_row])
    
    return pivot, df_copy

def create_hourly_distribution_fig(df):
    if df is None or df.empty: return None
    df_copy = df.copy()
    df_copy['Hour'] = df_copy['datetime'].dt.hour
    hourly_volume = df_copy.groupby('Hour')['Volume'].sum()
    total_volume = hourly_volume.sum()
    if total_volume == 0: return None
    hourly_percentage = (hourly_volume / total_volume) * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(hourly_percentage.index, hourly_percentage.values, color='mediumseagreen', edgecolor='black')
    ax.set_title('Hourly Volume Distribution', fontsize=16, fontweight='bold')
    ax.set_xlabel('Hour of the Day'), ax.set_ylabel('Percentage of Total Volume (%)')
    ax.set_xticks(hourly_percentage.index), ax.set_xticklabels([f'{h}:00' for h in hourly_percentage.index])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}%'))
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f}%', va='bottom', ha='center')
    plt.tight_layout()
    return fig

def create_heatmap_fig(depth_df, sales_df):
    """Generates the market liquidity heatmap with price overlay."""
    if depth_df is None or sales_df is None:
        st.warning("Heatmap requires both Market Depth and Course of Sales files.")
        return None

    # Filter data to the main trading session
    trade_date = depth_df['datetime'].iloc[0].date()
    SESSION_START = pd.to_datetime(f"{trade_date} 10:00:00").time()
    SESSION_END = pd.to_datetime(f"{trade_date} 16:00:00").time()

    depth_df_session = depth_df[(depth_df['datetime'].dt.time >= SESSION_START) & (depth_df['datetime'].dt.time < SESSION_END)]
    sales_df_session = sales_df[(sales_df['datetime'].dt.time >= SESSION_START) & (sales_df['datetime'].dt.time < SESSION_END)]

    if depth_df_session.empty or sales_df_session.empty:
        st.warning("No data found within the main trading session (10:00 AM - 4:00 PM).")
        return None

    # --- CLOUD-PROOFING START ---
    # Ensure Volume is a float and strip any hidden formatting/strings
    def clean_vol(series):
        if series.dtype == 'object':
            return pd.to_numeric(series.str.replace(',', ''), errors='coerce').fillna(0)
        return pd.to_numeric(series, errors='coerce').fillna(0)

    clean_depth_vol = clean_vol(depth_df_session['Volume'])
    # --- CLOUD-PROOFING END ---

    # Prepare Data for Heatmap
    depth_df_session['SignedVolume'] = np.where(
        depth_df_session['Type'] == 'BUY', 
        clean_depth_vol, 
        -clean_depth_vol
    )
    heatmap_pivot = depth_df_session.pivot_table(index='Price', columns='datetime', values='SignedVolume', aggfunc='sum').fillna(0)
    
    min_price, max_price = sales_df_session['Price'].min(), sales_df_session['Price'].max()
    price_bins = np.arange(np.floor(min_price), np.ceil(max_price) + 0.05, 0.05)
    binned_heatmap = heatmap_pivot.groupby(pd.cut(heatmap_pivot.index, bins=price_bins, right=False), observed=False).sum()

    # Create Plotly Figure
    fig = go.Figure()
    clip_level = np.percentile(np.abs(binned_heatmap.values[binned_heatmap.values != 0]), 95)

    fig.add_trace(go.Heatmap(
        x=binned_heatmap.columns, y=[interval.left for interval in binned_heatmap.index], z=binned_heatmap.values,
        colorscale='RdBu', zmid=0, zmin=-clip_level, zmax=clip_level,
        name='Net Liquidity', hoverinfo='none', colorbar=dict(x=1.0, title='Net Liquidity')
    ))
    fig.add_trace(go.Scatter(
        x=sales_df_session['datetime'], y=sales_df_session['Price'],
        customdata=np.stack((sales_df_session['Volume'],), axis=-1),
        mode='lines', name='Trade Price', line=dict(color='rgba(0, 0, 0, 0.8)', width=2),
        hovertemplate=('<b>Time:</b> %{x|%H:%M:%S}<br><b>Price:</b> $%{y:.3f}<br><b>Volume:</b> %{customdata[0]:,}<extra></extra>')
    ))
    fig.update_layout(height=650, title_text='Market Heatmap with Price Overlay', yaxis_title='Price Level')

    return fig

# --- Backtesting Engine (CVD-Regime / OU-Theta Strategy) ---

def prep_session_frame(df, start='10:00:00', end='16:00:00'):
    """Restrict a loaded depth/sales frame to the standard trading session and sort it."""
    if df is None or df.empty:
        return df
    return df.set_index('datetime').between_time(start, end).reset_index().sort_values('datetime')

def apply_smoothing(series, alg, s_win, thresh=5):
    """Shared velocity-smoothing step — matches the live heatmap's _smooth_vel exactly."""
    if alg == 'EMA':
        return series.ewm(span=s_win, adjust=False).mean()
    elif alg == 'SMA':
        return series.rolling(window=s_win, min_periods=1).mean()
    elif alg == 'DEMA':
        e1 = series.ewm(span=s_win, adjust=False).mean()
        e2 = e1.ewm(span=s_win, adjust=False).mean()
        return (2 * e1) - e2
    else:  # Threshold
        gated = series.apply(lambda x: x if abs(x) >= thresh else 0.0)
        return gated.ewm(span=3, adjust=False).mean()

def compute_signal_frame(cos_df, md_df, base_span, vel_lb, smooth_win, ou_win,
                          smooth_alg='DEMA', smooth_thresh=5,
                          include_spatial=False, tick_size=0.01,
                          touch_ticks=5, core_ticks=15, deep_ticks=35):
    """
    Builds the per-tick signal frame: order-book level velocity, CVD, and OU theta/Z-score regime.
    With include_spatial=True, also adds Touch/Core/Deep zone density and the Core+Deep
    "structural" spatial CVD (Spatial_CVD_BUY/SELL) used by the Signal Explorer.

    The total-book pipeline (through Z_Theta/Regime_Change) is numerically identical to the
    original Backtesting-only implementation when called with defaults — existing backtest
    results are unaffected.
    """
    lc = md_df.groupby(['datetime', 'Type'])['Price'].nunique().reset_index()
    pivot = lc.pivot(index='datetime', columns='Type', values='Price').fillna(0)
    for col in ['BUY', 'SELL']:
        if col not in pivot.columns:
            pivot[col] = 0

    b_ema = pivot['BUY'].ewm(span=base_span, adjust=False).mean()
    s_ema = pivot['SELL'].ewm(span=base_span, adjust=False).mean()
    pivot['BUY_EMA'] = b_ema
    pivot['SELL_EMA'] = s_ema
    b_vel_raw = b_ema.diff(periods=vel_lb).fillna(0)
    s_vel_raw = s_ema.diff(periods=vel_lb).fillna(0)

    pivot['BUY_Vel'] = apply_smoothing(b_vel_raw, smooth_alg, smooth_win, smooth_thresh)
    pivot['CVD_BUY'] = pivot['BUY_Vel'].cumsum()

    pivot['SELL_Vel'] = apply_smoothing(s_vel_raw, smooth_alg, smooth_win, smooth_thresh)
    pivot['CVD_SELL'] = pivot['SELL_Vel'].cumsum()

    # OU theta + Z-score regime filter
    pivot['Imbalance'] = pivot['BUY'] - pivot['SELL']
    x_prev = pivot['Imbalance'].shift(1)
    dx = pivot['Imbalance'].diff()
    roll_cov = x_prev.rolling(window=ou_win).cov(dx)
    roll_var = x_prev.rolling(window=ou_win).var()
    theta = (-roll_cov / roll_var).fillna(0).clip(lower=0)

    z_win = ou_win * 2
    t_mean = theta.rolling(window=z_win, min_periods=ou_win // 2).mean()
    t_std = theta.rolling(window=z_win, min_periods=ou_win // 2).std()
    z_theta = ((theta - t_mean) / t_std.replace(0, 1e-5)).fillna(0)

    if include_spatial:
        bids = md_df[md_df['Type'] == 'BUY'].groupby('datetime')['Price'].max().rename('Best_Bid')
        asks = md_df[md_df['Type'] == 'SELL'].groupby('datetime')['Price'].min().rename('Best_Ask')
        mids = ((bids + asks) / 2).rename('Mid_Price')

        md_spatial = md_df.merge(mids, on='datetime', how='left')
        md_spatial['Tick_Distance'] = (
            (abs(md_spatial['Price'] - md_spatial['Mid_Price']) / tick_size).round().fillna(0).astype(int)
        )
        md_spatial['Zone'] = pd.cut(
            md_spatial['Tick_Distance'], bins=[-1, touch_ticks, core_ticks, deep_ticks],
            labels=['Touch', 'Core', 'Deep']
        )

        spatial_counts = (
            md_spatial.groupby(['datetime', 'Type', 'Zone'], observed=False)['Price']
            .nunique().unstack(level=['Type', 'Zone'])
        )
        spatial_counts.columns = [f"{side}_{zone}_Density" for side, zone in spatial_counts.columns]

        pivot = pivot.join(spatial_counts, how='left').fillna(0)
        for side in ['BUY', 'SELL']:
            for zone in ['Touch', 'Core', 'Deep']:
                col = f"{side}_{zone}_Density"
                if col not in pivot.columns:
                    pivot[col] = 0

        pivot['BUY_Structural'] = pivot['BUY_Core_Density'] + pivot['BUY_Deep_Density']
        pivot['SELL_Structural'] = pivot['SELL_Core_Density'] + pivot['SELL_Deep_Density']

        b_struct_ema = pivot['BUY_Structural'].ewm(span=base_span, adjust=False).mean()
        s_struct_ema = pivot['SELL_Structural'].ewm(span=base_span, adjust=False).mean()
        b_struct_vel_raw = b_struct_ema.diff(periods=vel_lb).fillna(0)
        s_struct_vel_raw = s_struct_ema.diff(periods=vel_lb).fillna(0)

        pivot['BUY_Spatial_Vel'] = apply_smoothing(b_struct_vel_raw, smooth_alg, smooth_win, smooth_thresh)
        pivot['SELL_Spatial_Vel'] = apply_smoothing(s_struct_vel_raw, smooth_alg, smooth_win, smooth_thresh)
        pivot['Spatial_CVD_BUY'] = pivot['BUY_Spatial_Vel'].cumsum()
        pivot['Spatial_CVD_SELL'] = pivot['SELL_Spatial_Vel'].cumsum()

    # Sync signals with traded price
    sync_df = pd.merge_asof(
        pivot.reset_index().sort_values('datetime'),
        cos_df[['datetime', 'Price']].sort_values('datetime'),
        on='datetime', direction='backward',
    ).set_index('datetime')
    sync_df['Price'] = sync_df['Price'].ffill()
    sync_df['OU_Theta'] = theta.values
    sync_df['Z_Theta'] = z_theta.values

    sync_df['CVD_Regime'] = np.where(sync_df['CVD_BUY'] > sync_df['CVD_SELL'], 1, -1)
    sync_df['Regime_Change'] = sync_df['CVD_Regime'].diff()

    return sync_df

def find_regime_flip_events(sync_df, z_thresh, regime):
    """
    Every CVD regime flip in the session, classified exactly like simulate_session_trades'
    own entry gate: theta-accepted (would enter, direction included) vs theta-rejected.
    Used to overlay "signal fired but was filtered out" markers alongside real trades.
    """
    events = []
    flips = sync_df.index[sync_df['Regime_Change'].isin([2, -2])]
    for t in flips:
        row = sync_df.loc[t]
        z = row['Z_Theta']
        is_low_theta = z < z_thresh
        accepted = is_low_theta if regime == 'Trend (Low Theta)' else not is_low_theta

        direction = None
        if accepted:
            if regime == 'Trend (Low Theta)':
                direction = 'Long' if row['Regime_Change'] == 2 else 'Short'
            else:
                direction = 'Short' if row['Regime_Change'] == 2 else 'Long'

        events.append({'time': t, 'direction': direction, 'z_theta': z, 'accepted': accepted})

    return events

def simulate_session_trades(sync_df, z_thresh, regime, theta_exit, vel_exit, max_ticks, hysteresis_mult, starting_equity, slippage):
    """Runs the CVD-regime-change entry / theta-spike-or-momentum exit loop for a single session."""
    trade_log = []
    current_equity = starting_equity
    n = len(sync_df)
    idx = 0

    while idx < n:
        row = sync_df.iloc[idx]
        if row['Regime_Change'] in (2, -2):
            is_low_theta = row['Z_Theta'] < z_thresh
            triggered = is_low_theta if regime == 'Trend (Low Theta)' else not is_low_theta

            if triggered:
                if regime == 'Trend (Low Theta)':
                    direction = 'Long' if row['Regime_Change'] == 2 else 'Short'
                else:
                    direction = 'Short' if row['Regime_Change'] == 2 else 'Long'

                entry_price = row['Price']
                entry_time = sync_df.index[idx]
                exit_idx = min(idx + max_ticks, n - 1)
                exit_reason = f"Max Time ({max_ticks} ticks)"

                lookback_start = max(0, idx - 60)
                recent_noise = sync_df.iloc[lookback_start:idx]['BUY_Vel'].std()
                if pd.isna(recent_noise):
                    recent_noise = 0
                deadband = recent_noise * hysteresis_mult

                for j in range(idx + 1, min(idx + max_ticks + 1, n)):
                    er = sync_df.iloc[j]
                    if theta_exit and er['Z_Theta'] >= z_thresh:
                        exit_idx = j
                        exit_reason = "Theta Spike"
                        break
                    if vel_exit:
                        if direction == 'Long' and er['SELL_Vel'] > (er['BUY_Vel'] + deadband):
                            exit_idx = j
                            exit_reason = "Momentum Exhaustion"
                            break
                        if direction == 'Short' and er['BUY_Vel'] > (er['SELL_Vel'] + deadband):
                            exit_idx = j
                            exit_reason = "Momentum Exhaustion"
                            break

                exit_price = sync_df.iloc[exit_idx]['Price']
                exit_time = sync_df.index[exit_idx]
                shares = current_equity / entry_price
                gross = ((exit_price - entry_price) if direction == 'Long' else (entry_price - exit_price)) * shares
                net = gross - (shares * slippage * 2)
                current_equity += net
                ret_pct = (net / current_equity) * 100

                trade_log.append({
                    'Entry_Time': entry_time,
                    'Exit_Time': exit_time,
                    'Direction': direction,
                    'Hold_Ticks': exit_idx - idx,
                    'Entry_Price': round(entry_price, 3),
                    'Exit_Price': round(exit_price, 3),
                    'Shares': shares,
                    'Net_Profit': round(net, 2),
                    'Return_%': round(ret_pct, 4),
                    'Exit_Reason': exit_reason,
                    'Equity_After': round(current_equity, 2),
                })
                idx = exit_idx
        idx += 1

    return trade_log, current_equity

def run_backtest(session_specs, params, progress_callback=None):
    """Iterates paired Depth+Sales sessions, compounding equity across sessions in chronological order."""
    trade_log = []
    current_equity = params['capital']
    sessions_run = 0
    total = len(session_specs)

    for i, spec in enumerate(session_specs):
        if progress_callback:
            progress_callback(i, total, spec['label'])
        try:
            if spec['source'] == 'drive':
                depth_bytes = download_from_gdrive(spec['depth_id'])
                sales_bytes = download_from_gdrive(spec['sales_id'])
            else:
                depth_bytes = spec['depth_file']
                sales_bytes = spec['sales_file']

            md_df = load_depth_data(depth_bytes, spec['depth_name'])
            if md_df is None or md_df.empty:
                continue
            trade_date = md_df['datetime'].iloc[0].date()
            cos_df = load_sales_data(sales_bytes, trade_date, spec['sales_name'])
            if cos_df is None or cos_df.empty:
                continue

            md_df = prep_session_frame(md_df)
            cos_df = prep_session_frame(cos_df)
            if md_df.empty or cos_df.empty:
                continue

            sync_df = compute_signal_frame(
                cos_df, md_df,
                params['base_span'], params['vel_lb'], params['smooth_win'], params['ou_win']
            )
            session_trades, current_equity = simulate_session_trades(
                sync_df, params['z_thresh'], params['regime'],
                params['theta_exit'], params['vel_exit'],
                params['max_ticks'], params['hysteresis'],
                current_equity, params['slippage']
            )
            for t in session_trades:
                t['Session'] = spec['label']
            trade_log.extend(session_trades)
            sessions_run += 1
        except Exception as e:
            st.warning(f"Skipped session {spec['label']}: {e}")

    trades_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    return trades_df, sessions_run

def render_backtest_results(trades_df, capital, sessions_run):
    total = len(trades_df)
    wins = (trades_df['Net_Profit'] > 0).sum()
    win_rate = wins / total * 100
    gp = trades_df[trades_df['Net_Profit'] > 0]['Net_Profit'].sum()
    gl = abs(trades_df[trades_df['Net_Profit'] < 0]['Net_Profit'].sum())
    pf = gp / gl if gl else float('inf')
    avg_hold = trades_df['Hold_Ticks'].mean()
    final_equity = trades_df['Equity_After'].iloc[-1]
    net_ret = (final_equity - capital) / capital * 100

    eq = [capital] + trades_df['Equity_After'].tolist()
    peak = eq[0]
    max_dd = 0.0
    for v in eq:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd

    st.subheader("📊 Tearsheet")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sessions", sessions_run)
    c1.metric("Trades", total)
    c2.metric("Win Rate", f"{win_rate:.1f}%")
    c2.metric("Profit Factor", f"{pf:.2f}" if pf != float('inf') else "∞")
    c3.metric("Avg Hold", f"{avg_hold:.0f} ticks")
    c3.metric("Max Drawdown", f"{max_dd*100:.2f}%")
    c4.metric("Final Equity", f"${final_equity:,.2f}")
    c4.metric("Net Return", f"{net_ret:+.2f}%")

    ec = trades_df['Exit_Reason'].value_counts()
    st.caption("Exit Reasons — " + "  ·  ".join(f"{r}: {c}" for r, c in ec.items()))

    eq_fig = go.Figure()
    eq_fig.add_trace(go.Scatter(
        x=list(range(len(eq))), y=eq, mode='lines', name='Equity',
        line=dict(color='dodgerblue', width=2)
    ))
    eq_fig.add_hline(y=capital, line=dict(color='gray', width=1, dash='dot'))
    eq_fig.update_layout(title="Equity Curve", xaxis_title="Trade #", yaxis_title="Equity ($)", height=350, margin=dict(l=40, r=40, t=60, b=40))
    st.plotly_chart(eq_fig, use_container_width=True)

    st.subheader("📜 Trade Log")
    display_df = trades_df.copy()
    display_df['Entry_Time'] = display_df['Entry_Time'].astype(str).str[:19]
    display_df['Exit_Time'] = display_df['Exit_Time'].astype(str).str[:19]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv_bytes = trades_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Download Trade Log (CSV)", data=csv_bytes,
        file_name="Backtest_Trade_Log.csv", mime="text/csv"
    )

def render_backtesting_tab():
    st.header("🧪 Signal Backtesting Engine")
    st.markdown("Replay the CVD-regime / OU-theta strategy across one or more historical sessions.")

    session_specs = []

    with st.expander("📂 Backtest Data Source", expanded=True):
        bt_source = st.radio("Data Source", ["Google Drive Sessions", "Manual File Upload"], horizontal=True)

        if bt_source == "Google Drive Sessions":
            depth_files = st.session_state.get('depth_files', {})
            sales_files = st.session_state.get('sales_files', {})
            ticker_name = st.session_state.get('bt_ticker_name', 'Unknown')

            if not depth_files or not sales_files:
                st.warning("Select a Ticker with both Depth and Sales files in the 'Session Database' sidebar panel first.")
            else:
                valid_dates = []
                for fname in depth_files.keys():
                    m = re.match(r'^(\d{8})', fname)
                    if m:
                        date = m.group(1)
                        if date not in valid_dates and any(f.startswith(date) for f in sales_files.keys()):
                            valid_dates.append(date)
                valid_dates = sorted(valid_dates, reverse=True)

                if not valid_dates:
                    st.warning(f"No paired Depth + Sales sessions found for {ticker_name}.")
                else:
                    selected_dates = st.multiselect(
                        f"Sessions to backtest ({ticker_name}, {len(valid_dates)} available):",
                        options=valid_dates, default=valid_dates,
                        format_func=lambda d: f"{d[:4]}-{d[4:6]}-{d[6:]}"
                    )
                    # Process chronologically (oldest first) so equity compounds in session order,
                    # independent of the newest-first display/selection order above.
                    for date in sorted(selected_dates):
                        depth_name = next(f for f in depth_files if f.startswith(date))
                        sales_name = next(f for f in sales_files if f.startswith(date))
                        session_specs.append({
                            'label': date, 'source': 'drive',
                            'depth_id': depth_files[depth_name], 'depth_name': depth_name,
                            'sales_id': sales_files[sales_name], 'sales_name': sales_name,
                        })
        else:
            bt_depth_uploads = st.file_uploader("Upload Market Depth files", type=["csv", "parquet"], accept_multiple_files=True, key="bt_depth_up")
            bt_sales_uploads = st.file_uploader("Upload Course of Sales files", type=["csv", "parquet"], accept_multiple_files=True, key="bt_sales_up")

            if bt_depth_uploads and bt_sales_uploads:
                depth_by_date = {}
                for f in bt_depth_uploads:
                    m = re.match(r'^(\d{8})', f.name)
                    depth_by_date[m.group(1) if m else f.name] = f
                sales_by_date = {}
                for f in bt_sales_uploads:
                    m = re.match(r'^(\d{8})', f.name)
                    sales_by_date[m.group(1) if m else f.name] = f

                common_dates = sorted(set(depth_by_date) & set(sales_by_date), reverse=True)
                if not common_dates:
                    st.warning("Could not match uploaded Depth/Sales files by a common YYYYMMDD date prefix. Rename files accordingly.")
                else:
                    selected_dates = st.multiselect("Sessions to backtest:", options=common_dates, default=common_dates)
                    # Process chronologically (oldest first) so equity compounds in session order.
                    for date in sorted(selected_dates):
                        session_specs.append({
                            'label': date, 'source': 'upload',
                            'depth_file': depth_by_date[date], 'depth_name': depth_by_date[date].name,
                            'sales_file': sales_by_date[date], 'sales_name': sales_by_date[date].name,
                        })

    st.markdown("#### ⚙️ Strategy Parameters")
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("**Signal Parameters**")
        bt_base_span = st.number_input("Base EMA Span", 1, 200, 15, key="bt_base_span")
        bt_vel_lb = st.number_input("Vel Lookback", 1, 60, 5, key="bt_vel_lb")
        bt_smooth_win = st.number_input("Smooth Window", 1, 60, 10, key="bt_smooth_win")
    with p2:
        st.markdown("**Regime Filter**")
        bt_ou_win = st.number_input("OU Window", 10, 720, 120, step=10, key="bt_ou_win")
        bt_z_thresh = st.number_input("Z Threshold", 0.1, 5.0, 1.0, step=0.1, key="bt_z_thresh")
        bt_regime = st.selectbox("Trade Regime", ["Trend (Low Theta)", "Fade Trap (High Theta)"], key="bt_regime")

    st.markdown("**Exit Logic**")
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        bt_theta_exit = st.checkbox("Exit on Theta Spike", value=True, key="bt_theta_exit")
    with e2:
        bt_vel_exit = st.checkbox("Exit on Vel Cross", value=False, key="bt_vel_exit")
    with e3:
        bt_max_ticks = st.number_input("Max Hold Ticks", 1, 500, 40, key="bt_max_ticks")
    with e4:
        bt_hysteresis = st.number_input("Hysteresis", 0.0, 2.0, 0.6, step=0.1, key="bt_hysteresis")

    st.markdown("**Portfolio**")
    po1, po2 = st.columns(2)
    with po1:
        bt_capital = st.number_input("Initial Capital ($)", 1000, 10_000_000, 10000, step=1000, key="bt_capital")
    with po2:
        bt_slippage = st.number_input("Slippage / Share ($)", 0.0, 0.1, 0.005, step=0.001, format="%.3f", key="bt_slippage")

    st.markdown("---")
    run_clicked = st.button("▶️ Run Backtest", type="primary", disabled=not session_specs)

    if run_clicked:
        params = {
            'base_span': bt_base_span, 'vel_lb': bt_vel_lb, 'smooth_win': bt_smooth_win,
            'ou_win': bt_ou_win, 'z_thresh': bt_z_thresh, 'regime': bt_regime,
            'theta_exit': bt_theta_exit, 'vel_exit': bt_vel_exit,
            'max_ticks': bt_max_ticks, 'hysteresis': bt_hysteresis,
            'capital': bt_capital, 'slippage': bt_slippage,
        }
        progress_bar = st.progress(0, text="Starting backtest...")

        def _progress(i, total, label):
            progress_bar.progress((i + 1) / total, text=f"Processing {label} ({i + 1}/{total})...")

        with st.spinner("Running backtest..."):
            trades_df, sessions_run = run_backtest(session_specs, params, progress_callback=_progress)
        progress_bar.empty()

        st.session_state['bt_trades_df'] = trades_df
        st.session_state['bt_sessions_run'] = sessions_run
        st.session_state['bt_capital_used'] = bt_capital

        if trades_df.empty:
            st.info(f"Done — {sessions_run} session(s) processed, but no trades were generated with these parameters.")

    trades_df = st.session_state.get('bt_trades_df')
    if trades_df is not None and not trades_df.empty:
        render_backtest_results(trades_df, st.session_state.get('bt_capital_used', bt_capital), st.session_state.get('bt_sessions_run', 0))

# --- Signal Explorer (Single-Session Signal Replay) ---

def render_signal_explorer_tab(df_depth, df_sales):
    st.header("🔬 Signal Explorer")
    st.markdown("Replay this session's signals tick-by-tick and see exactly what triggered each position open/close.")

    st.markdown("#### ⚙️ Strategy Parameters")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("**Signal Parameters**")
        se_base_span = st.number_input("Base EMA Span", 1, 200, 15, key="se_base_span")
        se_vel_lb = st.number_input("Vel Lookback", 1, 60, 5, key="se_vel_lb")
        se_smooth_alg = st.selectbox("Smoothing Algorithm", ["DEMA", "EMA", "SMA", "Threshold"], key="se_smooth_alg")
        se_smooth_win = st.number_input("Smoothing Window", 1, 60, 10, key="se_smooth_win")
        se_thresh = st.number_input("Shock Threshold", 1, 50, 5, key="se_thresh", disabled=(se_smooth_alg != "Threshold"))
    with p2:
        st.markdown("**Regime Filter**")
        se_ou_win = st.number_input("OU Window (ticks)", 10, 720, 120, step=10, key="se_ou_win")
        se_z_thresh = st.number_input("Z Threshold", 0.1, 5.0, 1.0, step=0.1, key="se_z_thresh")
        se_regime = st.selectbox("Trade Regime", ["Trend (Low Theta)", "Fade Trap (High Theta)"], key="se_regime")
    with p3:
        st.markdown("**Spatial Zones**")
        se_tick_size = st.number_input("Tick Size ($)", 0.001, 1.0, 0.01, step=0.001, format="%.3f", key="se_tick_size")
        se_touch_ticks = st.number_input("Touch Zone Max", 1, 20, 5, key="se_touch_ticks")
        se_core_ticks = st.number_input("Core Zone Max", 5, 60, 15, key="se_core_ticks")
        se_deep_ticks = st.number_input("Deep Zone Max", 15, 100, 35, key="se_deep_ticks")

    st.markdown("**Exit Logic** (defines the shaded trade windows below)")
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        se_theta_exit = st.checkbox("Exit on Theta Spike", value=True, key="se_theta_exit")
    with e2:
        se_vel_exit = st.checkbox("Exit on Vel Cross", value=False, key="se_vel_exit")
    with e3:
        se_max_ticks = st.number_input("Max Hold Ticks", 1, 500, 40, key="se_max_ticks")
    with e4:
        se_hysteresis = st.number_input("Hysteresis", 0.0, 2.0, 0.6, step=0.1, key="se_hysteresis")

    st.markdown("**Portfolio**")
    po1, po2 = st.columns(2)
    with po1:
        se_capital = st.number_input("Initial Capital ($)", 1000, 10_000_000, 10000, step=1000, key="se_capital")
    with po2:
        se_slippage = st.number_input("Slippage / Share ($)", 0.0, 0.1, 0.005, step=0.001, format="%.3f", key="se_slippage")

    md_df = prep_session_frame(df_depth)
    cos_df = prep_session_frame(df_sales)
    if md_df is None or md_df.empty or cos_df is None or cos_df.empty:
        st.warning("No data in the standard 10:00–16:00 session window for this file.")
        return

    with st.spinner("Computing signals..."):
        sync_df = compute_signal_frame(
            cos_df, md_df, se_base_span, se_vel_lb, se_smooth_win, se_ou_win,
            smooth_alg=se_smooth_alg, smooth_thresh=se_thresh,
            include_spatial=True, tick_size=se_tick_size,
            touch_ticks=se_touch_ticks, core_ticks=se_core_ticks, deep_ticks=se_deep_ticks,
        )

        # Session-cumulative VWAP, synced onto the same tick grid as the signals
        cos_vwap = cos_df.sort_values('datetime').copy()
        cos_vwap['Cum_Vol'] = cos_vwap['Volume'].cumsum()
        cos_vwap['Cum_PV'] = (cos_vwap['Price'] * cos_vwap['Volume']).cumsum()
        cos_vwap['VWAP'] = cos_vwap['Cum_PV'] / cos_vwap['Cum_Vol'].replace(0, 1)
        sync_df = pd.merge_asof(
            sync_df.reset_index().sort_values('datetime'),
            cos_vwap[['datetime', 'VWAP']],
            on='datetime', direction='backward',
        ).set_index('datetime')
        sync_df['VWAP'] = sync_df['VWAP'].ffill()

        trades, final_equity = simulate_session_trades(
            sync_df, se_z_thresh, se_regime, se_theta_exit, se_vel_exit,
            se_max_ticks, se_hysteresis, se_capital, se_slippage
        )
        trades_df = pd.DataFrame(trades)
        events = find_regime_flip_events(sync_df, se_z_thresh, se_regime)

    executed_entry_times = set(trades_df['Entry_Time']) if not trades_df.empty else set()

    st.markdown("---")
    total_trades = len(trades_df)
    if total_trades:
        wins = (trades_df['Net_Profit'] > 0).sum()
        net_pnl = trades_df['Net_Profit'].sum()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trades", total_trades)
        c2.metric("Win Rate", f"{wins / total_trades * 100:.1f}%")
        c3.metric("Net P/L", f"${net_pnl:+,.2f}")
        c4.metric("Final Equity", f"${final_equity:,.2f}")
    else:
        st.info("No trades were generated for this session with these parameters.")

    rejected_n = sum(1 for e in events if not e['accepted'])
    skipped_n = sum(1 for e in events if e['accepted'] and e['time'] not in executed_entry_times)
    st.caption(
        f"Regime flips this session: {len(events)}  ·  Executed: {total_trades}  ·  "
        f"Filtered by theta gate: {rejected_n}  ·  Accepted but already in a position: {skipped_n}"
    )

    se_pnl_mode = st.radio(
        "Cumulative P&L Display", ["Mark-to-Market", "Realized Only"],
        index=0, horizontal=True, key="se_pnl_mode",
        help="Mark-to-Market values any open position at the current tick's price. "
             "Realized Only steps only when a trade actually closes."
    )

    # --- Multi-panel figure ---
    fig = make_subplots(
        rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        subplot_titles=(
            "Traded Price",
            "Smoothed Level Counts (Total Book)",
            "Spatial CVD — Core + Deep Structural Thrust",
            f"Standard CVD ({se_smooth_alg}) — Entries / Rejections",
            "Regime Classifier: OU Decay Rate (θ) & Z-Score",
            f"Cumulative P&L — {se_pnl_mode}",
        ),
        row_heights=[0.20, 0.12, 0.14, 0.18, 0.18, 0.18],
        specs=[[{}], [{}], [{}], [{}], [{"secondary_y": True}], [{}]],
    )

    # Row 1: Price + VWAP + trade markers + trade-window shading
    fig.add_trace(go.Scatter(x=sync_df.index, y=sync_df['Price'], name="Price",
                              line=dict(color='black', width=1.3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=sync_df.index, y=sync_df['VWAP'], name="VWAP",
                              line=dict(color='darkorange', width=1.6, dash='dot')), row=1, col=1)

    exit_colors = {"Theta Spike": "#8e44ad", "Momentum Exhaustion": "#e67e22"}
    for tr in trades:
        color = "#2ecc71" if tr['Direction'] == 'Long' else "#e74c3c"
        symbol = "triangle-up" if tr['Direction'] == 'Long' else "triangle-down"
        fig.add_trace(go.Scatter(
            x=[tr['Entry_Time']], y=[tr['Entry_Price']], mode='markers',
            marker=dict(symbol=symbol, size=11, color=color, line=dict(color='black', width=0.5)),
            name=f"{tr['Direction']} Entry", showlegend=False,
            hovertext=f"{tr['Direction']} Entry<br>{tr['Entry_Time']}<br>${tr['Entry_Price']:.3f}",
            hoverinfo='text',
        ), row=1, col=1)
        exit_reason_key = tr['Exit_Reason'].split(' (')[0]
        exit_color = exit_colors.get(exit_reason_key, "#7f8c8d")
        fig.add_trace(go.Scatter(
            x=[tr['Exit_Time']], y=[tr['Exit_Price']], mode='markers',
            marker=dict(symbol='x', size=9, color=exit_color, line=dict(width=1.5)),
            name="Exit", showlegend=False,
            hovertext=f"Exit — {tr['Exit_Reason']}<br>{tr['Exit_Time']}<br>${tr['Exit_Price']:.3f}<br>Net: ${tr['Net_Profit']:+,.2f}",
            hoverinfo='text',
        ), row=1, col=1)
        fig.add_vrect(x0=tr['Entry_Time'], x1=tr['Exit_Time'], fillcolor=color, opacity=0.08, line_width=0, row=1, col=1)
        fig.add_vrect(x0=tr['Entry_Time'], x1=tr['Exit_Time'], fillcolor=color, opacity=0.08, line_width=0, row=4, col=1)

    # Row 2: Smoothed level counts
    fig.add_trace(go.Scatter(x=sync_df.index, y=sync_df['BUY_EMA'], name="BUY Levels",
                              line=dict(color='#5e81ac', width=1.2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=sync_df.index, y=sync_df['SELL_EMA'], name="SELL Levels",
                              line=dict(color='#bf616a', width=1.2)), row=2, col=1)

    # Row 3: Spatial CVD (Core + Deep)
    fig.add_trace(go.Scatter(x=sync_df.index, y=sync_df['Spatial_CVD_BUY'], name="Spatial CVD BUY",
                              line=dict(color='#88c0d0', width=1.4)), row=3, col=1)
    fig.add_trace(go.Scatter(x=sync_df.index, y=sync_df['Spatial_CVD_SELL'], name="Spatial CVD SELL",
                              line=dict(color='#d08770', width=1.4, dash='dot')), row=3, col=1)
    fig.add_hline(y=0, line=dict(color='gray', width=0.7), row=3, col=1)

    # Row 4: Standard CVD + entry/rejection markers
    fig.add_trace(go.Scatter(x=sync_df.index, y=sync_df['CVD_BUY'], name="CVD BUY",
                              line=dict(color='#a3be8c', width=1.2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=sync_df.index, y=sync_df['CVD_SELL'], name="CVD SELL",
                              line=dict(color='#d08770', width=1.2)), row=4, col=1)
    fig.add_hline(y=0, line=dict(color='gray', width=0.7), row=4, col=1)

    cvd_mid = (sync_df['CVD_BUY'] + sync_df['CVD_SELL']) / 2
    long_x, long_y, short_x, short_y = [], [], [], []
    rejected_x, rejected_y, skipped_x, skipped_y = [], [], [], []
    for e in events:
        t = e['time']
        if t not in cvd_mid.index:
            continue
        y = cvd_mid.loc[t]
        if not e['accepted']:
            rejected_x.append(t); rejected_y.append(y)
        elif t in executed_entry_times:
            if e['direction'] == 'Long':
                long_x.append(t); long_y.append(y)
            else:
                short_x.append(t); short_y.append(y)
        else:
            skipped_x.append(t); skipped_y.append(y)

    if long_x:
        fig.add_trace(go.Scatter(x=long_x, y=long_y, mode='markers', name='Long entry (executed)',
                                  marker=dict(symbol='triangle-up', size=10, color='#a3be8c')), row=4, col=1)
    if short_x:
        fig.add_trace(go.Scatter(x=short_x, y=short_y, mode='markers', name='Short entry (executed)',
                                  marker=dict(symbol='triangle-down', size=10, color='#bf616a')), row=4, col=1)
    if skipped_x:
        fig.add_trace(go.Scatter(x=skipped_x, y=skipped_y, mode='markers', name='Accepted (already in position)',
                                  marker=dict(symbol='circle-open', size=8, color='#4c566a')), row=4, col=1)
    if rejected_x:
        fig.add_trace(go.Scatter(x=rejected_x, y=rejected_y, mode='markers', name='Rejected (theta gate)',
                                  marker=dict(symbol='x', size=7, color='#4c566a')), row=4, col=1)

    # Row 5: OU theta (primary) + Z-theta (secondary)
    fig.add_trace(go.Scatter(x=sync_df.index, y=sync_df['OU_Theta'], name="OU θ (raw)",
                              line=dict(color='#b48ead', width=1.0), fill='tozeroy',
                              fillcolor='rgba(180, 142, 173, 0.2)'), row=5, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=sync_df.index, y=sync_df['Z_Theta'], name="Z-θ",
                              line=dict(color='#2e3440', width=1.0, dash='dot')), row=5, col=1, secondary_y=True)
    fig.add_hline(y=se_z_thresh, line=dict(color='#bf616a', width=1, dash='dash'),
                  row=5, col=1, secondary_y=True)

    # Row 6: Cumulative P&L — Mark-to-Market (continuous, values open positions each tick)
    # or Realized Only (step chart, one step per trade exit)
    session_start = sync_df.index[0]
    session_end = sync_df.index[-1]

    if se_pnl_mode == "Mark-to-Market":
        realized = pd.Series(0.0, index=sync_df.index)
        floating = pd.Series(0.0, index=sync_df.index)
        running_realized = 0.0
        for tr in trades:
            sign = 1 if tr['Direction'] == 'Long' else -1
            hold_mask = (sync_df.index >= tr['Entry_Time']) & (sync_df.index <= tr['Exit_Time'])
            floating.loc[hold_mask] = (sync_df.loc[hold_mask, 'Price'] - tr['Entry_Price']) * tr['Shares'] * sign
            running_realized += tr['Net_Profit']
            realized.loc[sync_df.index > tr['Exit_Time']] = running_realized
        pnl_series = realized + floating
        pnl_times, pnl_values = pnl_series.index, pnl_series.values
        line_shape = 'linear'
        final_value = pnl_values[-1] if len(pnl_values) else 0.0
    else:
        pnl_times = [session_start]
        pnl_values = [0.0]
        running = 0.0
        for tr in trades:
            running += tr['Net_Profit']
            pnl_times.append(tr['Exit_Time'])
            pnl_values.append(running)
        pnl_times.append(session_end)
        pnl_values.append(running)
        line_shape = 'hv'
        final_value = running

    pnl_fill = 'rgba(163, 190, 140, 0.25)' if final_value >= 0 else 'rgba(191, 97, 106, 0.2)'
    fig.add_trace(go.Scatter(
        x=pnl_times, y=pnl_values, name=f"Cumulative P&L ({se_pnl_mode})", mode='lines',
        line=dict(color='#2e3440', width=1.6, shape=line_shape),
        fill='tozeroy', fillcolor=pnl_fill,
    ), row=6, col=1)
    fig.add_hline(y=0, line=dict(color='gray', width=0.7), row=6, col=1)

    fig.update_layout(
        height=1350, hovermode="x unified", template="plotly_white",
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Levels", row=2, col=1)
    fig.update_yaxes(title_text="Spatial CVD", row=3, col=1)
    fig.update_yaxes(title_text="CVD", row=4, col=1)
    fig.update_yaxes(title_text="OU θ", row=5, col=1, secondary_y=False)
    fig.update_yaxes(title_text=f"Z-θ (thresh={se_z_thresh:.1f})", row=5, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Cum P&L ($)", row=6, col=1)

    st.plotly_chart(fig, use_container_width=True, theme=None)

    if total_trades:
        st.subheader("📜 Trade Log (this session)")
        display_df = trades_df.copy()
        display_df['Entry_Time'] = display_df['Entry_Time'].astype(str).str[:19]
        display_df['Exit_Time'] = display_df['Exit_Time'].astype(str).str[:19]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        csv_bytes = trades_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Trade Log (CSV)", data=csv_bytes,
            file_name="Signal_Explorer_Trade_Log.csv", mime="text/csv", key="se_csv_dl"
        )

# --- Streamlit App UI ---
st.title("📈 Interactive Trade Session Analysis Tool")

# --- Sidebar for Controls ---
st.sidebar.header("⚙️ Controls")

# --- GOOGLE DRIVE SELECTOR ---
with st.sidebar.expander("Session Database", expanded=True):
    # Replace with your actual Google Drive Root Folder ID
    ROOT_FOLDER_ID = "11n_EsXh_a8HZo_f1tOHUepCSjAq4TvMn"

    try:
        # 1. Select Ticker
        tickers_dict = list_drive_folders(ROOT_FOLDER_ID)
        selected_ticker = st.selectbox("Select Ticker", options=sorted(tickers_dict.keys()))

        if selected_ticker:
            # 2. Select File from Ticker Folder
            ticker_id = tickers_dict[selected_ticker]
            files_dict = list_files_in_folder(ticker_id)

            # Filter for Depth vs Sales
            depth_files = {k: v for k, v in files_dict.items() if "Depth" in k}
            sales_files = {k: v for k, v in files_dict.items() if any(x in k for x in ["Sales", "sales"])}

            # Persist so other sections (e.g. Backtesting) can discover paired sessions
            st.session_state['depth_files'] = depth_files
            st.session_state['sales_files'] = sales_files
            st.session_state['bt_ticker_name'] = selected_ticker

            # Dropdowns for specific files
            depth_choice = st.selectbox("Market Depth File", options=["None"] + sorted(depth_files.keys(), reverse=True))
            sales_choice = st.selectbox("Course of Sales File", options=["None"] + sorted(sales_files.keys(), reverse=True))

            # Logic to trigger download
            if st.button("🚀 Load Drive Files"):
                if depth_choice != "None":
                    with st.spinner("Downloading Depth..."):
                        # 1. This actually fetches the bits from Google
                        depth_data = download_from_gdrive(depth_files[depth_choice])
                        # 2. Pass the filename separately so the loader can detect parquet/csv
                        #    without setting .name on the BytesIO (which makes PyArrow treat it
                        #    as a filesystem path and fail with FileNotFoundError)
                        st.session_state['df_depth'] = load_depth_data(depth_data, depth_choice)

                if sales_choice != "None":
                    with st.spinner("Downloading Sales..."):
                        t_date = datetime.date.today()
                        if 'df_depth' in st.session_state and st.session_state['df_depth'] is not None:
                            t_date = st.session_state['df_depth']['datetime'].iloc[0].date()

                        # 1. Fetch sales bits from Google
                        sales_data = download_from_gdrive(sales_files[sales_choice])
                        # 2. Load it into memory, passing filename for type detection
                        st.session_state['df_sales'] = load_sales_data(sales_data, t_date, sales_choice)

                st.rerun() # Refresh to show the charts immediately

    except Exception as e:
        st.error(f"Drive Connection Error: {e}")

st.sidebar.divider()

# Optional: Manual uploader
with st.sidebar.expander("📁 Manual Local Upload"):
    sales_file = st.file_uploader("Upload Sales", type=["csv", "parquet"])
    depth_file = st.file_uploader("Upload Depth", type=["csv", "parquet"])

# --- Main App Logic ---
# 1. Initialize data from session state (This is where Drive downloads live)
df_depth = st.session_state.get('df_depth')
df_sales = st.session_state.get('df_sales')

# 2. Check for Manual Local Uploads (These override Drive data if used)
if depth_file:
    with st.spinner('Reading Local Market Depth...'):
        # We pass the actual file object, not just the name
        df_depth = load_depth_data(depth_file)
        st.session_state['df_depth'] = df_depth

if sales_file:
    with st.spinner('Reading Local Course of Sales...'):
        # Align trades with depth date if available
        trade_date = df_depth['datetime'].iloc[0].date() if df_depth is not None else datetime.date.today()
        df_sales = load_sales_data(sales_file, trade_date)
        st.session_state['df_sales'] = df_sales

# 3. Setup metadata for filenames (Download buttons) - Safe Check
ticker = "Unknown"
date_str = "UnknownDate"

if df_depth is not None and not df_depth.empty:
    if 'Ticker' in df_depth.columns:
        ticker = df_depth['Ticker'].iloc[0]
    date_str = df_depth['datetime'].iloc[0].strftime('%Y%m%d')

st.sidebar.subheader("Analysis Options")
analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    ["Hourly Volume Analysis", "Volume Profile", "Hourly Volume Distribution", "Market Depth Explorer", "Backtesting", "Signal Explorer"]
)

# Global Price Bin Size (Consolidated)
global_bin_size = st.sidebar.selectbox(
    "Global Price Bin Size:", 
    options=[0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 1.00], 
    format_func=lambda x: f"${x:.2f}" if x > 0 else "No Grouping", 
    index=1 # Defaults to $0.05
)

# --- Global Trade Filtering (Collapsible) ---
# This block must start at the far left (no leading spaces)
if df_sales is not None and not df_sales.empty:
    with st.sidebar.expander("⚖️ Trade Filtering (Global)", expanded=True):
        
        # 1. Market Filter
        markets = df_sales['Market'].unique().tolist()
        selected_markets = st.multiselect("Include Markets:", options=markets, default=markets)
        
        # 2. Condition Filter
        conditions = df_sales['Condition'].unique().tolist()
        default_conds = ['Lit Order Book'] if 'Lit Order Book' in conditions else conditions
        selected_conditions = st.multiselect(
            "Include Trade Conditions:", 
            options=conditions, 
            default=default_conds,
            help="Uncheck special conditions to remove off-market block trades."
        )
        
    # Apply the filters to df_sales so it cascades to EVERY chart in the app
    df_sales = df_sales[
        (df_sales['Market'].isin(selected_markets)) & 
        (df_sales['Condition'].isin(selected_conditions))
    ]
    
    if df_sales.empty:
        st.sidebar.error("All trades filtered out! Please select at least one Market and Condition.")


if analysis_type == "Market Depth Explorer":
    st.header("🔍 Market Depth & Order Flow Explorer")
    st.markdown("Analyze market liquidity, order book snapshots, and executed footprint order flow.")

    if df_depth is not None and not df_depth.empty and df_sales is not None and not df_sales.empty:
        
        # ==========================================
        # 1. SESSION LIQUIDITY HEATMAP (Collapsible)
        # ==========================================
        with st.expander("🔥 Session Liquidity Heatmap", expanded=True):
            st.markdown("Spot massive liquidity walls resting in the book to identify key support and resistance areas.")
            with st.spinner("Rendering Heatmap..."):
                heatmap_fig = create_heatmap_fig(df_depth, df_sales)
                if heatmap_fig:
                    st.plotly_chart(
                        heatmap_fig, 
                        use_container_width=True,
                        config={
                            'toImageButtonOptions': {
                                'format': 'png', # one of png, svg, jpeg, webp
                                'filename': f'{date_str}_Heatmap_{ticker}',
                                'height': 700,
                                'width': 1200,
                                'scale': 2 # Increases resolution for better quality
                            }
                        }
                    )

        # ==========================================
        # 2. ORDER BOOK SNAPSHOT EXPLORER (Collapsible)
        # ==========================================
        with st.expander("📸 Order Book Snapshot Explorer", expanded=False):
            # Call the fragment function we defined above
            order_book_explorer_fragment(df_depth, df_sales)

        # ==========================================
        # 3. FOOTPRINT CHART (Collapsible)
        # ==========================================
        with st.expander("👣 Continuous Footprint Chart", expanded=False):
            st.markdown("Analyze aggressive market orders hitting the bid and ask to see who is truly in control.")
            
            # The controls for the footprint chart live safely inside this expander
            col1, col2 = st.columns([1, 4])
            with col1:
                tf_mins = st.selectbox("Candle Timeframe", [1, 2, 5, 15, 30], index=0)
                timeframe_str = f"{tf_mins}min"
                
            with st.spinner("Compiling Order Flow Data..."):
                footprint_df, ohlc_df, full_stats_df, raw_sales_df = generate_footprint_data(df_sales, df_depth, timeframe_str)
                
            if not footprint_df.empty:
                min_time = ohlc_df.index.min().to_pydatetime()
                max_time = ohlc_df.index.max().to_pydatetime()
                fp_trade_date = min_time.date()
                
                # 1. Define the standard 10:00 to 16:00 window bounds
                target_start = datetime.datetime.combine(fp_trade_date, datetime.time(10, 0, 0))
                target_end = datetime.datetime.combine(fp_trade_date, datetime.time(16, 0, 0))
                std_start = max(min_time, target_start)
                std_end = min(max_time, target_end)
                
                # 2. Initialize the slider's state in memory on first load
                if "fp_window" not in st.session_state:
                    st.session_state.fp_window = (std_start, std_end)

                # 3. Layout: Put the jump button right next to the slider
                col_slider, col_btn = st.columns([5, 1])
                
                with col_btn:
                    st.markdown("<br>", unsafe_allow_html=True) # Adds a little space to align vertically with the slider
                    if st.button("⏪ Reset to 10-16", help="Jump back to the standard trading session", use_container_width=True):
                        st.session_state.fp_window = (std_start, std_end)
                
                with col_slider:
                    window_start, window_end = st.slider(
                        "Select Viewing Window",
                        min_value=min_time,
                        max_value=max_time,
                        key="fp_window", # This KEY is the magic that connects the slider to the button!
                        step=datetime.timedelta(minutes=tf_mins),
                        format="HH:mm"
                    )
                
                # Filter the dataframes based on whatever the slider currently says
                view_ohlc = ohlc_df[(ohlc_df.index >= window_start) & (ohlc_df.index <= window_end)]
                view_footprint = footprint_df[(footprint_df['candle_time'] >= window_start) & (footprint_df['candle_time'] <= window_end)]
                view_stats = full_stats_df[(full_stats_df.index >= window_start) & (full_stats_df.index <= window_end)]
                
                if not view_ohlc.empty:
                    show_footprints = st.toggle("👣 Render Micro-Volume Profiles (Footprints)", value=False, help="Turn this on to see Bid/Ask volumes inside the candles.")
                    
                    fig = create_continuous_footprint_chart(view_ohlc, view_footprint, view_stats, tf_mins, show_footprints, global_bin_size)
                    st.plotly_chart(fig, use_container_width=True, theme=None)
                    
                    # Export Data Button
                    st.markdown("---")
                    @st.cache_data
                    def convert_df_to_excel(df):
                        import io
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            df.to_excel(writer, index=False, sheet_name='Enriched_Sales')
                        return output.getvalue()
                    
                    export_cols = ['datetime', 'Time', 'Price', 'Volume', 'best_bid', 'best_ask', 'Liquidity_Sign', 'Delta', 'CVD']
                    available_cols = [c for c in export_cols if c in raw_sales_df.columns]
                    excel_data = convert_df_to_excel(raw_sales_df[available_cols])
                    
                    st.download_button(
                        label="⬇️ Download Course of Sales Tick Data (.xlsx)",
                        data=excel_data,
                        file_name=f"Enriched_Sales_{fp_trade_date}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.info("No trades occurred in the selected time window.")

    else:
        st.warning("Please upload BOTH a Market Depth CSV and Course of Sales CSV to unlock the Market Depth Explorer features.")

elif analysis_type == "Backtesting":
    render_backtesting_tab()

elif analysis_type == "Signal Explorer":
    if df_depth is not None and not df_depth.empty and df_sales is not None and not df_sales.empty:
        render_signal_explorer_tab(df_depth, df_sales)
    else:
        st.warning("Please upload BOTH a Market Depth CSV and Course of Sales CSV to unlock the Signal Explorer.")

elif df_sales is not None:
    st.sidebar.subheader("⏳ Time Filter (Sales Data)")
    
    min_time = df_sales['datetime'].min().to_pydatetime()
    max_time = df_sales['datetime'].max().to_pydatetime()
    trade_date = min_time.date()

    # --- NEW: Regular Session Toggle ---
    regular_session = st.sidebar.checkbox("Standard Session Only (10:00 - 16:00)", value=True, help="Automatically snap to continuous trading hours to exclude auction matches.")

    if regular_session:
        target_start = datetime.datetime.combine(trade_date, datetime.time(10, 0, 0))
        target_end = datetime.datetime.combine(trade_date, datetime.time(16, 0, 0))
        
        # Clamp the defaults so we don't accidentally ask the slider to go out of bounds
        default_start = max(min_time, target_start)
        default_end = min(max_time, target_end)
        
        # Failsafe: if the whole file is outside regular hours, revert to min/max
        if default_start > default_end:
            default_start, default_end = min_time, max_time
    else:
        default_start, default_end = min_time, max_time

    start_time, end_time = st.sidebar.slider(
        "Select Time Range:", 
        min_value=min_time, 
        max_value=max_time, 
        value=(default_start, default_end), 
        format="HH:mm:ss", 
        step=datetime.timedelta(minutes=1)
    )
    
    df_filtered = df_sales[(df_sales['datetime'] >= start_time) & (df_sales['datetime'] <= end_time)]
    
    if df_filtered.empty:
        st.warning("No trade data available for the selected time range.")
    else:
        if analysis_type == "Hourly Volume Analysis":
            st.header("Hourly Volume Analysis")
            
            # Use the new global_bin_size
            hourly_pivot, df_with_bins = create_hourly_pivot(df_filtered, global_bin_size)
            st.dataframe(hourly_pivot.style.format("{:,.0f}"))
            
            st.markdown("---")
            st.subheader("🔍 Drill Down into Individual Trades")
            price_bins_options = sorted([b for b in df_with_bins['price_bin'].unique() if b != 'Grand Total'], reverse=True)
            selected_bins = st.multiselect("Step 1: Select price levels:", options=price_bins_options, format_func=lambda x: f"${x:.2f}")
            if selected_bins:
                drill_down_df = df_with_bins[df_with_bins['price_bin'].isin(selected_bins)]
                hour_options = sorted(drill_down_df['Hour'].unique())
                selected_hours = st.multiselect("Step 2: (Optional) Select hours:", options=hour_options, format_func=lambda x: f"{x}:00")
                if selected_hours:
                    drill_down_df = drill_down_df[drill_down_df['Hour'].isin(selected_hours)]
                st.write(f"Displaying {len(drill_down_df)} trades:")
                st.dataframe(drill_down_df[['datetime', 'Price', 'Volume']].rename(columns={'datetime': 'Time'}).sort_values(by=['Time', 'Price']))

        elif analysis_type == "Volume Profile":
            st.header("Volume Profile")
            
            latest_trade = df_filtered.loc[df_filtered['datetime'].idxmax()]
            close_price = latest_trade['Price']
            
            # Pass the global_bin_size directly into the function
            volume_profile_fig = create_volume_profile_fig(
                df_filtered, 
                global_bin_size, 
                close_price, 
                selected_markets,      
                selected_conditions,   
                start_time,            
                end_time               
            )
            
            if volume_profile_fig: 
                st.pyplot(volume_profile_fig)
                
                # Save the plot to a buffer to enable downloading
                import io
                buf = io.BytesIO()
                volume_profile_fig.savefig(buf, format="png", bbox_inches='tight')
                
                st.download_button(
                    label="💾 Download Volume Profile as PNG",
                    data=buf.getvalue(),
                    file_name=f"{date_str}_VP_{ticker}.png",
                    mime="image/png"
                )
                plt.close(volume_profile_fig)

        elif analysis_type == "Hourly Volume Distribution":
            st.header("Hourly Volume Distribution")
            distribution_fig = create_hourly_distribution_fig(df_filtered)
            if distribution_fig: 
                st.pyplot(distribution_fig)
                plt.close(distribution_fig)
else:
    st.info("Awaiting for a Course of Sales and/or Market Depth CSV file to be uploaded.")
