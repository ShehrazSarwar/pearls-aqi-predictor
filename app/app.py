import os
import streamlit as st
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pymongo import MongoClient
from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore')

# Load Environment Variables
load_dotenv()

# --- CONFIGURATION ---
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", "")

MODEL_NAME = "AQI_MultiOutput_Predictor"
ALIAS = "champion"

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "aqi_predictor")
RAW_COLLECTION = "raw_data"
FEATURE_COLLECTION = "feature_store"

# --- PAGE SETTINGS ---
st.set_page_config(
    page_title="AQI Forecast Dashboard",
    layout="wide",
    page_icon="🌬️",
    initial_sidebar_state="expanded"
)

# --- MODERN CSS STYLING ---
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
    }

    /* Dark theme support */
    @media (prefers-color-scheme: dark) {
        .main {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        }
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }

    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }

    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }

    /* Metric cards - Equal sizing and theme support */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.85rem !important;
    }

    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        border-left: 4px solid #667eea;
        transition: transform 0.2s, box-shadow 0.2s;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    /* Dark theme metric cards */
    @media (prefers-color-scheme: dark) {
        div[data-testid="metric-container"] {
            background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
    }

    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: white;
        border-radius: 8px;
        padding: 0 24px;
        font-weight: 500;
        border: 2px solid transparent;
        transition: all 0.2s;
    }

    /* Dark theme tabs */
    @media (prefers-color-scheme: dark) {
        .stTabs [data-baseweb="tab"] {
            background-color: #2d3748;
            color: #e2e8f0;
        }
    }

    .stTabs [data-baseweb="tab"]:hover {
        border-color: #667eea;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }

    /* Alert boxes */
    .alert-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* Dark theme alert cards */
    @media (prefers-color-scheme: dark) {
        .alert-card {
            background: #2d3748;
            color: #e2e8f0;
        }
    }

    /* Info boxes */
    .stInfo, .stSuccess, .stWarning, .stError {
        border-radius: 10px;
        border-left-width: 5px;
    }

    /* Data table */
    .dataframe {
        border: none !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .dataframe thead tr th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 600;
        padding: 12px !important;
    }

    .dataframe tbody tr:hover {
        background-color: #f8f9fa !important;
    }

    /* Dark theme table */
    @media (prefers-color-scheme: dark) {
        .dataframe tbody tr:hover {
            background-color: #2d3748 !important;
        }
    }
    </style>
""", unsafe_allow_html=True)


# --- CACHED MODEL LOADING WITH AUTO-UPDATE ---
@st.cache_resource(show_spinner=False)
def load_champion_model():
    """Loads the model from MLflow Registry with error handling."""
    try:
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            raise ValueError("MLflow tracking URI not configured")

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        model_uri = f"models:/{MODEL_NAME}@{ALIAS}"

        model = mlflow.pyfunc.load_model(model_uri)
        client = mlflow.tracking.MlflowClient()
        model_ver = client.get_model_version_by_alias(MODEL_NAME, ALIAS)

        return model, model_ver.version, None
    except Exception as e:
        return None, None, str(e)


def check_for_model_updates(current_version):
    """
    Check if a new champion model version is available in the registry.
    Returns (has_update, new_version, error)
    """
    try:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        client = mlflow.tracking.MlflowClient()

        # Get the latest champion model version
        latest_model = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        latest_version = latest_model.version

        # Check if there's a new version
        if latest_version != current_version:
            return True, latest_version, None

        return False, current_version, None

    except Exception as e:
        return False, current_version, str(e)


# --- UTILITY FUNCTIONS ---
def calculate_aqi(pm25):
    """Calculate AQI from PM2.5 concentration using EPA formula."""
    if pm25 < 0:
        return 0
    if pm25 <= 12.0:
        return round(((50 - 0) / (12.0 - 0)) * (pm25 - 0) + 0)
    elif pm25 <= 35.4:
        return round(((100 - 51) / (35.4 - 12.1)) * (pm25 - 12.1) + 51)
    elif pm25 <= 55.4:
        return round(((150 - 101) / (55.4 - 35.5)) * (pm25 - 35.5) + 101)
    elif pm25 <= 150.4:
        return round(((200 - 151) / (150.4 - 55.5)) * (pm25 - 55.5) + 151)
    elif pm25 <= 250.4:
        return round(((300 - 201) / (250.4 - 150.5)) * (pm25 - 150.5) + 201)
    else:
        return 500


def get_aqi_info(aqi_val):
    """Get AQI category, color, and health recommendation."""
    if aqi_val <= 50:
        return ("Good", "#00e400", "🌿 Air quality is satisfactory. Ideal for all outdoor activities!")
    elif aqi_val <= 100:
        return ("Moderate", "#ffff00",
                "⚠️ Air quality is acceptable. Sensitive groups should limit prolonged outdoor exertion.")
    elif aqi_val <= 150:
        return ("Unhealthy for Sensitive Groups", "#ff7e00",
                "😷 Members of sensitive groups may experience health effects. Wear a mask if needed.")
    elif aqi_val <= 200:
        return (
        "Unhealthy", "#ff0000", "🏠 Everyone may experience health effects. Reduce prolonged outdoor activities.")
    elif aqi_val <= 300:
        return ("Very Unhealthy", "#8f3f97", "🚨 Health alert: Everyone should avoid all outdoor physical activity.")
    else:
        return ("Hazardous", "#7e0023", "☢️ Emergency conditions. Stay indoors with air purification systems.")


def create_aqi_chart(plot_dates, aqi_values, types):
    """Create a beautiful AQI visualization chart."""
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 7), facecolor='white')

    # Background AQI bands
    ax.axhspan(0, 50, color='#00e400', alpha=0.08, label='Good')
    ax.axhspan(51, 100, color='#ffff00', alpha=0.08, label='Moderate')
    ax.axhspan(101, 150, color='#ff7e00', alpha=0.08, label='Unhealthy (Sensitive)')
    ax.axhspan(151, 200, color='#ff0000', alpha=0.08, label='Unhealthy')
    ax.axhspan(201, 300, color='#8f3f97', alpha=0.08, label='Very Unhealthy')
    ax.axhspan(301, 500, color='#7e0023', alpha=0.08, label='Hazardous')

    # Vertical line separating observed and predicted
    separator_idx = types.index("Predicted") if "Predicted" in types else len(types)
    if separator_idx < len(plot_dates):
        ax.axvline(x=plot_dates[separator_idx], color='gray', linestyle='--',
                   linewidth=2, alpha=0.5, label='Forecast Boundary')

    # Main trend line
    ax.plot(plot_dates, aqi_values, color='#2c3e50', linewidth=2.5,
            alpha=0.7, zorder=3)

    # Data points with category colors
    for i, (date, aqi, dtype) in enumerate(zip(plot_dates, aqi_values, types)):
        _, color, _ = get_aqi_info(aqi)
        marker = 'o' if dtype == "Observed" else 'D'
        size = 200 if dtype == "Observed" else 250

        ax.scatter(date, aqi, color=color, s=size, edgecolors='white',
                   linewidth=2, zorder=5, marker=marker, alpha=0.9)

        # Value labels
        ax.annotate(f'{aqi}', (date, aqi), xytext=(0, 15),
                    textcoords='offset points', ha='center',
                    fontweight='bold', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='gray', alpha=0.8))

    # Formatting
    ax.set_xlabel('Date', fontsize=12, fontweight='600', color='#2c3e50')
    ax.set_ylabel('AQI Index', fontsize=12, fontweight='600', color='#2c3e50')
    ax.set_title('Air Quality Index - Historical & Forecast Trend',
                 fontsize=16, fontweight='700', color='#2c3e50', pad=20)

    # X-axis formatting
    ax.set_xticks(plot_dates)
    ax.set_xticklabels([d.strftime('%b %d\n%a') for d in plot_dates],
                       fontsize=10, fontweight='500')

    # Y-axis formatting
    ax.set_ylim(0, max(aqi_values) + 50)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.xaxis.grid(False)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2c3e50',
                   markersize=10, label='Observed', markeredgecolor='white', markeredgewidth=2),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#2c3e50',
                   markersize=10, label='Predicted', markeredgecolor='white', markeredgewidth=2)
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10,
              framealpha=0.9, edgecolor='gray')

    plt.tight_layout()
    return fig


# --- MAIN APP ---
def main():
    # Initialize session state for update tracking
    if 'update_checked' not in st.session_state:
        st.session_state.update_checked = False
        st.session_state.update_available = False
        st.session_state.new_version = None

    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🌬️ Air Quality Intelligence Dashboard</h1>
            <p>Real-time AQI Monitoring & AI-Powered 3-Day Forecast</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        # st.image("https://img.icons8.com/clouds/200/air-quality.png", width=150)
        st.markdown("### 🎯 System Status")

        # Load model
        with st.spinner("Loading AI model..."):
            model, version, error = load_champion_model()

        if error:
            st.error(f"❌ Model Loading Failed\n\n{error}")
            st.stop()
        else:
            st.success(f"✅ Model Active")
            st.info(f"**Model:** {MODEL_NAME}")
            st.info(f"**Version:** v{version}")
            st.info(f"**Registry:** DagsHub MLflow")

            # Auto-check for model updates on every app load (fast, just version comparison)
            if not st.session_state.update_checked:
                has_update, new_version, update_error = check_for_model_updates(version)
                st.session_state.update_checked = True
                st.session_state.update_available = has_update
                st.session_state.new_version = new_version

            # Display update notification if available
            if st.session_state.update_available:
                st.warning(f"🆕 New model available: v{st.session_state.new_version}")
                col1, col2 = st.columns(2)

                with col1:
                    if st.button("⬆️ Update Now", use_container_width=True, type="primary"):
                        with st.spinner("Updating model..."):
                            st.cache_resource.clear()
                            st.session_state.update_available = False
                            st.session_state.update_checked = False
                            st.rerun()

                with col2:
                    if st.button("⏭️ Skip", use_container_width=True):
                        st.session_state.update_available = False
                        st.rerun()
            else:
                st.caption("✓ Model up to date")

        st.markdown("---")

        # Manual refresh button
        if st.button("🔄 Refresh Model", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

        st.markdown("---")
        st.markdown("### 📊 About")
        st.markdown("""
        This dashboard provides:
        - **Real-time** air quality monitoring
        - **AI-powered** 3-day forecasts
        - **Auto-update** model detection on every load
        - **Health recommendations** based on AQI levels
        - **Historical trend** analysis
        """)

        st.markdown("---")
        st.caption("Developed by Shehraz Sarwar khan")
        st.caption("(a.k.a Data Scientist)")

    # Main content
    st.markdown("### 📍 Latest Air Quality Analysis")

    # Prediction button
    if st.button("🚀 Generate AI Forecast", use_container_width=True, type="primary"):

        with st.spinner("🔄 Fetching data from MongoDB..."):
            try:
                # Connect to MongoDB
                client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
                db = client[DB_NAME]

                # Fetch historical data
                history = list(db[RAW_COLLECTION].find().sort("datetime", -1).limit(96))
                if not history:
                    st.error("❌ No historical data found in database")
                    client.close()
                    st.stop()

                hist_df = pd.DataFrame(history).sort_values("datetime")
                hist_df['datetime'] = pd.to_datetime(hist_df['datetime'])
                present_time = hist_df['datetime'].iloc[-1]

                # Fetch latest features
                latest_feat_doc = list(db[FEATURE_COLLECTION].find().sort("datetime", -1).limit(1))
                if not latest_feat_doc:
                    st.error("❌ No feature data found in database")
                    client.close()
                    st.stop()

                latest_feat_doc = latest_feat_doc[0]
                client.close()

            except Exception as e:
                st.error(f"❌ Database Error: {str(e)}")
                st.stop()

        with st.spinner("🤖 Running AI prediction model..."):
            try:
                # Prepare features
                feat_df = pd.DataFrame([latest_feat_doc]).drop(
                    columns=['_id', 'datetime', 'target_h24', 'target_h48', 'target_h72'],
                    errors='ignore'
                )

                # Make predictions
                forecast_pm25 = np.maximum(model.predict(feat_df).flatten(), 0)

            except Exception as e:
                st.error(f"❌ Prediction Error: {str(e)}")
                st.stop()

        # Process results
        plot_dates, aqi_values, types = [], [], []

        # Past 4 days (Observed)
        for d in [3, 2, 1, 0]:
            t_date = (present_time - timedelta(days=d)).date()
            day_data = hist_df[hist_df['datetime'].dt.date == t_date]
            avg_pm = day_data['pm2_5'].mean() if not day_data.empty else 0
            aqi = calculate_aqi(avg_pm)
            plot_dates.append(datetime.combine(t_date, datetime.min.time()))
            aqi_values.append(aqi)
            types.append("Observed")

        # Next 3 days (Predicted)
        for i, h in enumerate([24, 48, 72]):
            f_dt = present_time + timedelta(hours=h)
            aqi = calculate_aqi(forecast_pm25[i])
            plot_dates.append(f_dt)
            aqi_values.append(aqi)
            types.append("Predicted")

        # --- DISPLAY RESULTS ---
        st.success("✅ Analysis Complete!")

        # Current status card
        cur_aqi = aqi_values[3]  # Today's observed
        cat_name, cat_color, rec = get_aqi_info(cur_aqi)

        st.markdown(f"""
            <div class="alert-card" style="border-left-color: {cat_color};">
                <h3 style="color: {cat_color}; margin-top: 0;">
                    Current Air Quality: {cat_name} (AQI {cur_aqi})
                </h3>
                <p style="font-size: 1.1rem; margin-bottom: 0;">{rec}</p>
            </div>
        """, unsafe_allow_html=True)

        # Key metrics
        st.markdown("### 📊 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            cur_aqi_cat = get_aqi_info(cur_aqi)[0]
            st.metric(
                label="Current AQI",
                value=f"{cur_aqi}",
                delta=cur_aqi_cat
            )

        with col2:
            forecast_24h = aqi_values[4]
            forecast_24h_cat = get_aqi_info(forecast_24h)[0]
            st.metric(
                label="24h Forecast",
                value=f"{forecast_24h}",
                delta=forecast_24h_cat
            )

        with col3:
            forecast_48h = aqi_values[5]
            forecast_48h_cat = get_aqi_info(forecast_48h)[0]
            st.metric(
                label="48h Forecast",
                value=f"{forecast_48h}",
                delta=forecast_48h_cat
            )

        with col4:
            forecast_72h = aqi_values[6]
            forecast_72h_cat = get_aqi_info(forecast_72h)[0]
            st.metric(
                label="72h Forecast",
                value=f"{forecast_72h}",
                delta=forecast_72h_cat
            )

        # Visualization
        st.markdown("### 📈 Trend Analysis")
        fig = create_aqi_chart(plot_dates, aqi_values, types)
        st.pyplot(fig, use_container_width=True)

        # Detailed information tabs
        tab1, tab2, tab3 = st.tabs(["📅 Forecast Table", "🏥 Health Guidance", "📊 Data Insights"])

        with tab1:
            st.markdown("#### Complete 7-Day AQI Report")
            report_df = pd.DataFrame({
                "Date": [d.strftime('%Y-%m-%d') for d in plot_dates],
                "Day": [d.strftime('%A') for d in plot_dates],
                "AQI": aqi_values,
                "Category": [get_aqi_info(a)[0] for a in aqi_values],
                "Type": types
            })

            # Display the dataframe without styling
            st.dataframe(report_df, use_container_width=True, hide_index=True)

        with tab2:
            st.markdown("#### Personalized Health Recommendations")

            for i in range(4, 7):  # Predicted days
                d_name = plot_dates[i].strftime('%A, %B %d')
                c_name, c_color, c_rec = get_aqi_info(aqi_values[i])

                st.markdown(f"""
                    <div class="alert-card" style="border-left-color: {c_color};">
                        <h4 style="color: {c_color}; margin-top: 0;">{d_name}</h4>
                        <p style="margin: 0;"><strong>Forecast:</strong> {c_name} (AQI {aqi_values[i]})</p>
                        <p style="margin: 0.5rem 0 0 0;">{c_rec}</p>
                    </div>
                """, unsafe_allow_html=True)

        with tab3:
            st.markdown("#### Statistical Summary")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Historical Data (Past 4 Days)**")
                hist_aqi = aqi_values[:4]
                st.write(f"• Average AQI: {np.mean(hist_aqi):.1f}")
                st.write(f"• Max AQI: {max(hist_aqi)}")
                st.write(f"• Min AQI: {min(hist_aqi)}")
                st.write(f"• Trend: {'Improving ↓' if hist_aqi[-1] < hist_aqi[0] else 'Worsening ↑'}")

            with col2:
                st.markdown("**Forecast (Next 3 Days)**")
                pred_aqi = aqi_values[4:]
                st.write(f"• Average AQI: {np.mean(pred_aqi):.1f}")
                st.write(f"• Max AQI: {max(pred_aqi)}")
                st.write(f"• Min AQI: {min(pred_aqi)}")
                st.write(f"• Overall Outlook: {get_aqi_info(int(np.mean(pred_aqi)))[0]}")

            st.markdown("---")
            st.info(f"📅 Last Updated: {present_time.strftime('%Y-%m-%d %H:%M:%S')}")

    else:
        # Initial state - show instructions
        st.info("👆 Click the button above to generate the latest AI-powered air quality forecast")

        # Show example info
        st.markdown("### 🎯 What You'll Get")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
                **📊 Real-time Analysis**
                - Current AQI status
                - PM2.5 measurements
                - 24-hour trends
            """)

        with col2:
            st.markdown("""
                **🤖 AI Predictions**
                - 24h forecast
                - 48h forecast
                - 72h forecast
            """)

        with col3:
            st.markdown("""
                **🏥 Health Guidance**
                - Category-based advice
                - Activity recommendations
                - Risk assessments
            """)


if __name__ == "__main__":
    main()