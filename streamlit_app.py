
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Brand Reputation Monitor | CulturalAI Insights",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }

    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }

    .alert-high {
        background: #ff4757;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #ff3742;
    }

    .alert-medium {
        background: #ffa502;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #ff9500;
    }

    .alert-low {
        background: #2ed573;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #00d2d3;
    }

    .insight-box {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_demo_data():
    """Load or generate demo data"""
    try:
        df = pd.read_csv('multilingual_brand_data.csv')
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except FileNotFoundError:
        return generate_demo_data()

def generate_demo_data():
    """Generate demo data for showcase"""
    np.random.seed(42)

    brands = ['nike', 'adidas', 'samsung', 'apple', 'tesla', 'mcdonalds', 'cocacola']
    languages = ['en', 'de', 'hi', 'es']
    regions = ['USA', 'Germany', 'India', 'Spain']
    sentiments = ['positive', 'neutral', 'negative']
    platforms = ['twitter', 'facebook', 'instagram', 'reddit']

    # Text templates for different cultures and sentiments
    text_templates = {
        'USA': {
            'positive': ["{brand} is absolutely amazing! Best quality ever 🔥",
                        "Love my new {brand} purchase! Totally worth it",
                        "{brand} innovation is incredible, leading the industry"],
            'negative': ["{brand} quality is disappointing for the price",
                        "Had terrible experience with {brand} customer service",
                        "{brand} is overpriced, not worth the money"],
            'neutral': ["{brand} products are okay, nothing special",
                       "Average experience with {brand}, could be better",
                       "{brand} is fine but there are better alternatives"]
        },
        'Germany': {
            'positive': ["{brand} Produkte haben ausgezeichnete Qualität",
                        "Sehr zufrieden mit {brand}, gute Verarbeitung",
                        "{brand} bietet nachhaltige Lösungen"],
            'negative': ["{brand} Datenschutz ist unzureichend",
                        "Überteuerte {brand} Produkte",
                        "{brand} berücksichtigt Umwelt nicht genug"],
            'neutral': ["{brand} Produkte sind akzeptabel",
                       "Durchschnittliche {brand} Qualität",
                       "{brand} ist okay, nichts Besonderes"]
        },
        'India': {
            'positive': ["{brand} के products family के लिए perfect हैं",
                        "{brand} की quality बहुत अच्छी है",
                        "{brand} great value for money है"],
            'negative': ["{brand} बहुत महंगा है Indian market के लिए",
                        "{brand} की service India में अच्छी नहीं है",
                        "{brand} Indian consumers को consider नहीं करते"],
            'neutral': ["{brand} okay है but कुछ खास नहीं",
                       "{brand} average quality है",
                       "{brand} ठीक है लेकिन better options हैं"]
        },
        'Spain': {
            'positive': ["{brand} es increíble para toda la familia",
                        "Encantado con {brand}, excelente calidad",
                        "{brand} perfecta para nuestra comunidad"],
            'negative': ["{brand} no considera el mercado español",
                        "Decepcionado con {brand}, esperaba más",
                        "{brand} demasiado caro para lo que ofrece"],
            'neutral': ["{brand} está bien pero hay mejores opciones",
                       "Experiencia normal con {brand}",
                       "{brand} aceptable para el precio"]
        }
    }

    data = []
    for i in range(1000):
        brand = np.random.choice(brands)
        region = np.random.choice(regions)
        language = {'USA': 'en', 'Germany': 'de', 'India': 'hi', 'Spain': 'es'}[region]

        # Cultural sentiment bias
        if region == 'USA':
            sentiment = np.random.choice(sentiments, p=[0.4, 0.4, 0.2])
        elif region == 'Germany':
            sentiment = np.random.choice(sentiments, p=[0.3, 0.5, 0.2])
        elif region == 'India':
            sentiment = np.random.choice(sentiments, p=[0.5, 0.3, 0.2])
        else:  # Spain
            sentiment = np.random.choice(sentiments, p=[0.45, 0.35, 0.2])

        # Generate text
        template = np.random.choice(text_templates[region][sentiment])
        text = template.format(brand=brand.title())

        # Calculate cultural risk
        base_risk = {'positive': 0.1, 'neutral': 0.3, 'negative': 0.5}[sentiment]
        cultural_risk = np.clip(base_risk + np.random.uniform(-0.1, 0.2), 0.0, 1.0)

        data.append({
            'text': text,
            'brand': brand,
            'language': language,
            'region': region,
            'sentiment': sentiment,
            'cultural_risk_score': cultural_risk,
            'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 30)),
            'platform': np.random.choice(platforms),
            'engagement': np.random.randint(10, 1000),
            'emotion_intensity': np.random.uniform(0.2, 0.9),
            'formality_score': np.random.uniform(0.3, 0.8)
        })

    return pd.DataFrame(data)

# =============================================================================
# CULTURAL ANALYSIS FUNCTIONS
# =============================================================================

def analyze_cultural_appropriateness(text, region):
    """Analyze cultural appropriateness"""
    risk_keywords = {
        'USA': {'high': ['political', 'controversial'], 'medium': ['expensive', 'cheap']},
        'Germany': {'high': ['data', 'privacy', 'tracking'], 'medium': ['american', 'fast']},
        'India': {'high': ['beef', 'cow', 'religious'], 'medium': ['western', 'expensive']},
        'Spain': {'high': ['separatist', 'independence'], 'medium': ['crisis', 'unemployment']}
    }

    text_lower = text.lower()
    risk_score = 0.2
    issues = []

    if region in risk_keywords:
        for word in risk_keywords[region].get('high', []):
            if word in text_lower:
                risk_score += 0.3
                issues.append(f"High-risk keyword: '{word}'")

        for word in risk_keywords[region].get('medium', []):
            if word in text_lower:
                risk_score += 0.15
                issues.append(f"Medium-risk keyword: '{word}'")

    # Formality check
    if region == 'Germany' and any(word in text_lower for word in ['fire', 'lit', 'awesome']):
        risk_score += 0.2
        issues.append("Informal language inappropriate for German market")

    risk_score = min(1.0, risk_score)

    # Generate recommendations
    recommendations = []
    if risk_score > 0.6:
        recommendations.append("⚠️ High cultural risk - major revisions needed")
    elif risk_score > 0.3:
        recommendations.append("⚠️ Moderate risk - consider cultural adaptation")
    else:
        recommendations.append("✅ Content appears culturally appropriate")

    if region == 'Germany' and 'data' in text_lower:
        recommendations.append("🇩🇪 Add privacy assurances for German market")
    elif region == 'India' and not any(word in text_lower for word in ['family', 'value']):
        recommendations.append("🇮🇳 Consider adding family-oriented themes")

    return {
        'risk_score': risk_score,
        'issues': issues,
        'recommendations': recommendations
    }

def detect_crisis(df, hours=24):
    """Detect potential crisis situations"""
    cutoff_time = datetime.now() - timedelta(hours=hours)
    recent_df = df[df['timestamp'] > cutoff_time]

    if len(recent_df) == 0:
        return []

    crisis_alerts = []

    for brand in recent_df['brand'].unique():
        for region in recent_df['region'].unique():
            brand_region_data = recent_df[
                (recent_df['brand'] == brand) &
                (recent_df['region'] == region)
            ]

            if len(brand_region_data) < 5:
                continue

            negative_ratio = (brand_region_data['sentiment'] == 'negative').mean()
            avg_risk = brand_region_data['cultural_risk_score'].mean()

            alert_level = None
            if negative_ratio >= 0.4:
                alert_level = 'HIGH'
            elif negative_ratio >= 0.25:
                alert_level = 'MEDIUM'
            elif negative_ratio >= 0.15:
                alert_level = 'LOW'

            if alert_level:
                crisis_alerts.append({
                    'brand': brand,
                    'region': region,
                    'alert_level': alert_level,
                    'negative_ratio': negative_ratio,
                    'cultural_risk': avg_risk,
                    'sample_size': len(brand_region_data)
                })

    return crisis_alerts

# =============================================================================
# MAIN DASHBOARD
# =============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ Brand Reputation Monitor</h1>', unsafe_allow_html=True)
    st.markdown("### Multilingual Brand Monitoring with Cultural Context Analysis")
    st.markdown("**Team:** CulturalAI Insights | **Project:** Advanced NLP with Cultural Intelligence")

    # Load data
    df = load_demo_data()

    # Sidebar controls
    st.sidebar.header("🎛️ Dashboard Controls")

    available_brands = df['brand'].unique()
    selected_brands = st.sidebar.multiselect(
        "Select Brands:",
        available_brands,
        default=available_brands[:3]
    )

    available_regions = df['region'].unique()
    selected_regions = st.sidebar.multiselect(
        "Select Regions:",
        available_regions,
        default=available_regions
    )

    time_filter = st.sidebar.selectbox(
        "Time Period:",
        ["Last 24 hours", "Last 7 days", "Last 30 days", "All time"]
    )

    # Filter data
    filtered_df = df[
        (df['brand'].isin(selected_brands)) &
        (df['region'].isin(selected_regions))
    ]

    if time_filter == "Last 24 hours":
        cutoff = datetime.now() - timedelta(hours=24)
        filtered_df = filtered_df[filtered_df['timestamp'] > cutoff]
    elif time_filter == "Last 7 days":
        cutoff = datetime.now() - timedelta(days=7)
        filtered_df = filtered_df[filtered_df['timestamp'] > cutoff]
    elif time_filter == "Last 30 days":
        cutoff = datetime.now() - timedelta(days=30)
        filtered_df = filtered_df[filtered_df['timestamp'] > cutoff]

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Live Monitor",
        "🌍 Cultural Analysis",
        "🚨 Crisis Alerts",
        "🏢 Competitor Analysis",
        "🔍 Content Tester"
    ])

    # =============================================================================
    # TAB 1: LIVE MONITOR
    # =============================================================================

    with tab1:
        st.header("📊 Real-Time Brand Monitoring")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_mentions = len(filtered_df)
            st.metric("Total Mentions", f"{total_mentions:,}")

        with col2:
            if len(filtered_df) > 0:
                positive_ratio = (filtered_df['sentiment'] == 'positive').mean()
                st.metric("Positive Sentiment", f"{positive_ratio:.1%}")
            else:
                st.metric("Positive Sentiment", "N/A")

        with col3:
            if len(filtered_df) > 0:
                avg_risk = filtered_df['cultural_risk_score'].mean()
                st.metric("Avg Cultural Risk", f"{avg_risk:.3f}")
            else:
                st.metric("Avg Cultural Risk", "N/A")

        with col4:
            languages = filtered_df['language'].nunique()
            st.metric("Languages", languages)

        if len(filtered_df) > 0:
            # Sentiment trends
            st.subheader("📈 Sentiment Trends")
            daily_sentiment = filtered_df.groupby([
                filtered_df['timestamp'].dt.date, 'sentiment'
            ]).size().reset_index(name='count')

            if len(daily_sentiment) > 0:
                fig = px.line(daily_sentiment, x='timestamp', y='count', color='sentiment',
                            title="Daily Sentiment Trends",
                            color_discrete_map={'positive': '#2ed573', 'neutral': '#57606f', 'negative': '#ff4757'})
                st.plotly_chart(fig, use_container_width=True)

            # Regional and platform analysis
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🌍 Regional Distribution")
                region_data = filtered_df['region'].value_counts()
                fig = px.pie(values=region_data.values, names=region_data.index, title="Mentions by Region")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("📱 Platform Distribution")
                platform_data = filtered_df['platform'].value_counts()
                fig = px.bar(x=platform_data.index, y=platform_data.values, title="Mentions by Platform")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for selected filters")

    # =============================================================================
    # TAB 2: CULTURAL ANALYSIS
    # =============================================================================

    with tab2:
        st.header("🌍 Cultural Context Analysis")

        if len(filtered_df) > 0:
            # Cultural risk heatmap
            st.subheader("🔥 Cultural Risk Heatmap")
            risk_matrix = filtered_df.groupby(['brand', 'region'])['cultural_risk_score'].mean().reset_index()
            risk_pivot = risk_matrix.pivot(index='brand', columns='region', values='cultural_risk_score')

            if not risk_pivot.empty:
                fig = px.imshow(risk_pivot.values, x=risk_pivot.columns, y=risk_pivot.index,
                              color_continuous_scale='Reds', title="Cultural Risk Scores by Brand and Region",
                              labels=dict(color="Risk Score"))
                st.plotly_chart(fig, use_container_width=True)

            # Regional insights
            st.subheader("📋 Regional Cultural Insights")

            for region in selected_regions:
                with st.expander(f"🌍 {region} Cultural Profile"):
                    region_data = filtered_df[filtered_df['region'] == region]

                    if len(region_data) > 0:
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("**Communication Patterns:**")
                            avg_formality = region_data['formality_score'].mean()
                            avg_emotion = region_data['emotion_intensity'].mean()

                            st.write(f"- Formality Score: {avg_formality:.2f}/1.0")
                            st.write(f"- Emotion Intensity: {avg_emotion:.2f}/1.0")
                            st.write(f"- Total Mentions: {len(region_data):,}")

                        with col2:
                            st.write("**Sentiment Distribution:**")
                            sentiment_dist = region_data['sentiment'].value_counts(normalize=True)
                            for sentiment, ratio in sentiment_dist.items():
                                st.write(f"- {sentiment.title()}: {ratio:.1%}")

                    # Cultural recommendations
                    if region == 'Germany':
                        st.info("🇩🇪 **German Market:** Focus on quality, reliability, and data privacy. Avoid overly casual language.")
                    elif region == 'India':
                        st.info("🇮🇳 **Indian Market:** Emphasize family values, respect, and value-for-money. Use warm, respectful tone.")
                    elif region == 'USA':
                        st.info("🇺🇸 **US Market:** Direct communication works well. Emphasize innovation and individual achievement.")
                    elif region == 'Spain':
                        st.info("🇪🇸 **Spanish Market:** Community-focused messaging. Warm, expressive communication is appreciated.")

    # =============================================================================
    # TAB 3: CRISIS ALERTS
    # =============================================================================

    with tab3:
        st.header("🚨 Crisis Detection & Alerts")

        crisis_alerts = detect_crisis(filtered_df)

        if crisis_alerts:
            st.warning(f"🚨 {len(crisis_alerts)} potential crisis situations detected!")

            for alert in crisis_alerts:
                alert_class = f"alert-{alert['alert_level'].lower()}"

                st.markdown(f"""
                <div class="{alert_class}">
                    <h4>🚨 {alert['alert_level']} PRIORITY ALERT</h4>
                    <p><strong>Brand:</strong> {alert['brand'].title()}</p>
                    <p><strong>Region:</strong> {alert['region']}</p>
                    <p><strong>Negative Sentiment:</strong> {alert['negative_ratio']:.1%}</p>
                    <p><strong>Cultural Risk:</strong> {alert['cultural_risk']:.3f}</p>
                    <p><strong>Sample Size:</strong> {alert['sample_size']} mentions</p>
                </div>
                """, unsafe_allow_html=True)

                if alert['alert_level'] == 'HIGH':
                    st.error("🔴 **IMMEDIATE ACTION REQUIRED** - Monitor channels closely and prepare crisis response")
                elif alert['alert_level'] == 'MEDIUM':
                    st.warning("🟡 **ELEVATED MONITORING** - Increase monitoring frequency and review recent activities")
                else:
                    st.info("🟢 **WATCH STATUS** - Continue regular monitoring")

                st.markdown("---")
        else:
            st.success("✅ No crisis alerts detected. All brands performing normally.")

        # Crisis trends
        if len(filtered_df) > 0:
            st.subheader("📊 Crisis Risk Trends")
            daily_risk = filtered_df.groupby([filtered_df['timestamp'].dt.date, 'brand'])['cultural_risk_score'].mean().reset_index()

            if len(daily_risk) > 0:
                fig = px.line(daily_risk, x='timestamp', y='cultural_risk_score', color='brand',
                            title="Daily Cultural Risk Trends by Brand")
                st.plotly_chart(fig, use_container_width=True)

    # =============================================================================
    # TAB 4: COMPETITOR ANALYSIS
    # =============================================================================

    with tab4:
        st.header("🏢 Competitor Analysis")

        if len(filtered_df) > 0:
            # Brand metrics
            st.subheader("📊 Brand Performance Comparison")
            brand_metrics = filtered_df.groupby('brand').agg({
                'sentiment': lambda x: (x == 'positive').mean(),
                'cultural_risk_score': 'mean',
                'engagement': 'mean'
            }).round(3)

            brand_metrics.columns = ['Positive Sentiment %', 'Cultural Risk Score', 'Avg Engagement']
            brand_metrics = brand_metrics.sort_values('Positive Sentiment %', ascending=False)

            st.dataframe(brand_metrics, use_container_width=True)

            # Comparison charts
            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(x=brand_metrics.index, y=brand_metrics['Positive Sentiment %'],
                           title="Positive Sentiment by Brand")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.bar(x=brand_metrics.index, y=brand_metrics['Cultural Risk Score'],
                           title="Cultural Risk by Brand")
                st.plotly_chart(fig, use_container_width=True)

    # =============================================================================
    # TAB 5: CONTENT TESTER
    # =============================================================================

    with tab5:
        st.header("🔍 Cultural Content Testing Tool")
        st.write("Test your marketing content for cultural appropriateness across different regions:")

        # Content input
        test_content = st.text_area(
            "Enter your marketing content:",
            value="Just do it! Our new shoes are fire! 🔥",
            height=100
        )

        # Quick examples
        st.write("**Quick Test Examples:**")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🇺🇸 US Style"):
                st.session_state.test_content = "Just do it! Our new shoes are absolutely fire! 🔥"

        with col2:
            if st.button("🇩🇪 German Style"):
                st.session_state.test_content = "Premium quality engineered for performance and reliability"

        with col3:
            if st.button("🇮🇳 Family Style"):
                st.session_state.test_content = "Perfect for families who value quality and tradition"

        # Use session state for test content if available
        if 'test_content' in st.session_state:
            test_content = st.session_state.test_content

        # Analysis
        if st.button("🔍 Analyze Cultural Impact"):
            if test_content.strip():
                st.markdown("---")
                st.subheader("📊 Cultural Impact Analysis Results")
                st.info(f"**Analyzed Content:** {test_content}")

                regions = ['USA', 'Germany', 'India', 'Spain']
                cols = st.columns(2)

                for i, region in enumerate(regions):
                    analysis = analyze_cultural_appropriateness(test_content, region)

                    with cols[i % 2]:
                        risk_score = analysis['risk_score']
                        if risk_score > 0.6:
                            risk_emoji = "🔴"
                            risk_level = "HIGH RISK"
                        elif risk_score > 0.3:
                            risk_emoji = "🟡"
                            risk_level = "MEDIUM RISK"
                        else:
                            risk_emoji = "🟢"
                            risk_level = "LOW RISK"

                        st.markdown(f"### {risk_emoji} {region}: {risk_level}")
                        st.write(f"**Risk Score:** {risk_score:.3f}")

                        if analysis['issues']:
                            st.write("**Issues:**")
                            for issue in analysis['issues']:
                                st.write(f"- {issue}")

                        st.write("**Recommendations:**")
                        for rec in analysis['recommendations']:
                            st.write(f"- {rec}")

                        st.markdown("---")

# =============================================================================
# RUN THE APP
# =============================================================================

if __name__ == "__main__":
    main()
