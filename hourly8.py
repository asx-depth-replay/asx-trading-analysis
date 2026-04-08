import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- App Configuration ---
st.set_page_config(layout="wide", page_title="Trading Session Review")

# --- Data Loading and Caching ---
@st.cache_data
def load_sales_data(uploaded_file, trade_date):
    """
    Loads and processes the Course of Sales data, aligning it with the trade date from the depth file.
    """
    try:
        df = pd.read_csv(uploaded_file)
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
def load_depth_data(uploaded_file):
    """
    Loads and processes the Market Depth data with robust cleaning.
    """
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = ['Date', 'Time', 'Ticker', 'Type', 'Price', 'Volume', 'Number_of_Orders']
        
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
        
        for col in ['Price', 'Volume', 'Number_of_Orders']:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')

        df.dropna(subset=['datetime', 'Price', 'Volume', 'Type'], inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading Market Depth data: {e}")
        return None

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
        hovermode="x unified", plot_bgcolor='white', paper_bgcolor='white'
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
        df_copy['price_bin'] = (df_copy['Price'] / price_grouping).apply(np.floor) * price_grouping
    else:
        df_copy['price_bin'] = df_copy['Price']
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

    # Prepare Data for Heatmap
    depth_df_session['SignedVolume'] = np.where(depth_df_session['Type'] == 'BUY', depth_df_session['Volume'], -depth_df_session['Volume'])
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

# --- Streamlit App UI ---
st.title("📈 Interactive Trade Session Analysis Tool")

# --- Sidebar for Controls ---
st.sidebar.header("⚙️ Controls")

# --- Persistent File Uploaders ---
with st.sidebar.expander("📂 Data Files Upload", expanded=True):
    st.markdown("Upload your CSV files here. You can collapse this menu once loaded.")
    sales_file = st.file_uploader("1. Upload Course of Sales", type=["csv"])
    depth_file = st.file_uploader("2. Upload Market Depth", type=["csv"])
st.sidebar.markdown("---")

# --- Main App Logic ---
df_sales = None
df_depth = None

if depth_file:
    df_depth = load_depth_data(depth_file)

if sales_file:
    trade_date = df_depth['datetime'].iloc[0].date() if df_depth is not None and not df_depth.empty else datetime.date.today()
    df_sales = load_sales_data(sales_file, trade_date)
    
    # --- Global Trade Filtering (Collapsible) ---
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
            
            # 3. Global Price Bin Size (Consolidated)
            global_bin_size = st.selectbox(
                "Global Price Bin Size:", 
                options=[0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 1.00], 
                format_func=lambda x: f"${x:.2f}" if x > 0 else "No Grouping", 
                index=1 # Defaults to $0.05
            )
            
        # Apply the filters to df_sales so it cascades to EVERY chart in the app
        df_sales = df_sales[
            (df_sales['Market'].isin(selected_markets)) & 
            (df_sales['Condition'].isin(selected_conditions))
        ]
        
        if df_sales.empty:
            st.sidebar.error("All trades filtered out! Please select at least one Market and Condition.")

st.sidebar.subheader("Analysis Options")
analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    ["Hourly Volume Analysis", "Volume Profile", "Hourly Volume Distribution", "Market Depth Explorer"]
)

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
                    st.plotly_chart(heatmap_fig, use_container_width=True)

        # ==========================================
        # 2. ORDER BOOK SNAPSHOT EXPLORER (Collapsible)
        # ==========================================
        with st.expander("📸 Order Book Snapshot Explorer", expanded=False):
            st.markdown("Drill down into the exact order book state at any given second.")
            available_times = sorted(df_depth['datetime'].unique())

            if not available_times:
                st.error("Could not find any valid time entries in the Market Depth file.")
            else:
                col1, col2 = st.columns([3, 2])
                with col1:
                    snapshot_time = st.select_slider("Select a specific time to view the order book:", options=available_times, format_func=lambda x: pd.to_datetime(x).strftime('%I:%M:%S %p'))
                with col2:
                    depth_display_option = st.radio("Order Book Depth:", options=['Top 10', 'Top 20', 'Top 30', 'Full Book'], index=3, horizontal=True)

                st.markdown(f"**Showing order book for snapshot at {pd.to_datetime(snapshot_time).strftime('%I:%M:%S %p')}.**")
                
                snapshot_df = df_depth[df_depth['datetime'] == snapshot_time]
                bids_table = snapshot_df[snapshot_df['Type'] == 'BUY'][['Number_of_Orders', 'Volume', 'Price']].sort_values('Price', ascending=False).reset_index(drop=True)
                asks_table = snapshot_df[snapshot_df['Type'] == 'SELL'][['Price', 'Volume', 'Number_of_Orders']].sort_values('Price', ascending=True).reset_index(drop=True)

                if depth_display_option != 'Full Book':
                    depth = int(depth_display_option.split(' ')[1])
                    bids_table = bids_table.head(depth)
                    asks_table = asks_table.head(depth)

                col_buy, col_sell = st.columns(2)
                with col_buy:
                    st.subheader("Buyers (Bids)")
                    st.dataframe(bids_table, use_container_width=True)
                with col_sell:
                    st.subheader("Sellers (Asks)")
                    st.dataframe(asks_table, use_container_width=True)
                    
                st.markdown("---")
                compare_trades = st.toggle("Compare with Executed Trades", value=True)

                if compare_trades:
                    minutes_after = st.number_input("Show trades for how many minutes after the snapshot?", min_value=1, max_value=60, value=5)
                    end_time_dt = pd.to_datetime(snapshot_time) + datetime.timedelta(minutes=minutes_after)
                    comparison_df = df_sales[(df_sales['datetime'] > pd.to_datetime(snapshot_time)) & (df_sales['datetime'] <= end_time_dt)]
                    st.markdown(f"**Showing {len(comparison_df)} executed trades between {pd.to_datetime(snapshot_time).strftime('%I:%M:%S %p')} and {end_time_dt.strftime('%I:%M:%S %p')}:**")
                    st.dataframe(comparison_df[['datetime', 'Price', 'Volume']].rename(columns={'datetime': 'Time'}))

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
                        file_name=f"Enriched_Sales_{trade_date}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.info("No trades occurred in the selected time window.")

    else:
        st.warning("Please upload BOTH a Market Depth CSV and Course of Sales CSV to unlock the Market Depth Explorer features.")

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
                plt.close(volume_profile_fig)

        elif analysis_type == "Hourly Volume Distribution":
            st.header("Hourly Volume Distribution")
            distribution_fig = create_hourly_distribution_fig(df_filtered)
            if distribution_fig: 
                st.pyplot(distribution_fig)
                plt.close(distribution_fig)
else:
    st.info("Awaiting for a Course of Sales and/or Market Depth CSV file to be uploaded.")
