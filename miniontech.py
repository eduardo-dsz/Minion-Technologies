import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
import smtplib
from email.message import EmailMessage
from io import BytesIO

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="Minion Anomaly Detector", layout="wide")

# ------------------- STYLING -------------------
st.markdown("""
    <style>
    body {
        background-color: #7b8b82;
    }
    .main {
        background-color: #7b8b82;
    }
    .reportview-container .markdown-text-container {
        color: #f0f0f0;
    }
    .css-1aumxhk {
        background-color: #2e2e2e;
    }
    .block-container {
        padding-top: 2rem;
    }
    .stButton > button {
        color: white;
        background-color: #444;
        border-radius: 10px;
        padding: 0.5rem 1rem;
    }
    .stRadio > div {
        background-color: #444;
        color: white;
        border-radius: 10px;
        padding: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- SIDEBAR -------------------
with st.sidebar:
    st.title("☰ Minion Tech")
    st.markdown("🏠 **Home**")
    st.markdown("📊 **Menu**")
    st.markdown("📞 **Contact**")

# ------------------- HEADER -------------------
st.markdown("<h1 style='text-align:center; color:white;'>✨ Minion Technologies</h1>", unsafe_allow_html=True)

# ------------------- FILE UPLOAD -------------------
uploaded_file = st.file_uploader("📂 Upload wind turbine CSV", type="csv")

if uploaded_file:
    with st.spinner("🔍 Processing..."):
        df = pd.read_csv(uploaded_file)

        # ----- CLEANING -----
        df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%d %m %Y %H:%M')
        df.rename(columns={'Date/Time': 'date', 'LV ActivePower (kW)': 'output_kwh'}, inplace=True)
        df['day_of_year'] = df['date'].dt.dayofyear

        # ----- ANOMALY DETECTION -----
        model = IsolationForest(contamination=0.05)
        df['anomaly'] = model.fit_predict(df[['output_kwh']]) == -1

        # ----- REGRESSION -----
        lr = LinearRegression()
        lr.fit(df[['day_of_year']], df['output_kwh'])
        df['predicted'] = lr.predict(df[['day_of_year']])

        anomalies = df[df['anomaly']]

        # ----- PLOT -----
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(df['date'], df['output_kwh'], label="Actual", color="blue")
        ax.plot(df['date'], df['predicted'], label="Predicted", color="green")
        if not anomalies.empty:
            ax.scatter(anomalies['date'], anomalies['output_kwh'], color='red', label="Anomaly", s=60)
        ax.set_title("Output with Anomalies")
        ax.set_xlabel("Date")
        ax.set_ylabel("kW")
        ax.legend()
        ax.grid(True)

        # ----- DISPLAY -----
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Data Summary")
            st.write(f"• Avg Output: {df['output_kwh'].mean():.2f} kW")
            st.write(f"• Peak Output: {df['output_kwh'].max():.2f} kW")
            st.write(f"• Anomalies Found: {len(anomalies)}")
        with col2:
            st.subheader("📈 Output Chart")
            st.pyplot(fig)

        # ----- ALERT -----
        st.markdown("### 🚨 Anomaly Alert")
        if not anomalies.empty:
            st.info("🔔 Action Required: Anomalies detected.")
            st.dataframe(anomalies[['date', 'output_kwh']])
            st.radio("Export anomaly to Slack group?", ["Yes", "No"], index=1)

        # ----- EMAIL EXPORT -----
        st.markdown("### 📤 Export to Email")

        email_address = st.text_input("Enter recipient email address")
        if st.button("Send Email"):
            if email_address:
                # Save CSV to buffer
                buffer = BytesIO()
                df.to_csv(buffer, index=False)
                buffer.seek(0)

                # Compose email
                msg = EmailMessage()
                msg['Subject'] = 'Wind Turbine Report'
                msg['From'] = 'your_email@gmail.com'  # CHANGE THIS
                msg['To'] = email_address
                msg.set_content(f"""
                Hello,

                Please find the attached wind turbine anomaly report.
                - Average output: {df['output_kwh'].mean():.2f} kW
                - Anomalies: {len(anomalies)}

                Minion Technologies
                """)

                msg.add_attachment(buffer.read(), maintype='application', subtype='csv', filename="report.csv")

                try:
                    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                        smtp.login('your_email@gmail.com', 'your_app_password')  # CHANGE THIS
                        smtp.send_message(msg)
                    st.success("✅ Email sent successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to send email: {e}")
            else:
                st.warning("📧 Please enter a valid email address.")

        # ----- DOWNLOAD BUTTON -----
        st.download_button("⬇ Download Results", df.to_csv(index=False), "results.csv", "text/csv")
else:
    st.warning("📂 Please upload a CSV file.")
