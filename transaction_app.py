import streamlit as st
import pandas as pd
import pickle
import os

# Page Configuration
st.set_page_config(page_title="SafeCoin: Bitcoin Fraud Detector", layout="wide")

# --- DATA & MODEL LOADING ---
@st.cache_resource
def load_assets():
    origin_path = os.path.dirname(__file__)
    
    # Mapping friendly names back to original file names
    files = {
        'rf': 'rf_model.pkl',
        'xgb': 'xgboost.pkl',
        'lr': 'log_reg.pkl',
        'scaler': 'scaler.pkl'
    }
    
    loaded_tools = {}
    for key, filename in files.items():
        with open(os.path.join(origin_path, filename), 'rb') as f:
            loaded_tools[key] = pickle.load(f)
    
    test_data = pd.read_csv(os.path.join(origin_path, 'test_data.csv'))
    return loaded_tools['rf'], loaded_tools['xgb'], loaded_tools['lr'], loaded_tools['scaler'], test_data

# Initialize tools
try:
    rf, xgb, lr, scaler, test_data = load_assets()
except Exception as e:
    st.error(f"Error loading files: {e}. Please ensure .pkl and .csv files are in the same folder.")
    st.stop()

# --- SIDEBAR SETTINGS ---
st.sidebar.title("🛠️ Tools")
st.sidebar.info("Select a detection method to see its specific performance metrics.")
model_choice = st.sidebar.selectbox(
    "Choose a Detection Method", 
    ["Random Forest", "Xgboost", "Logistic Regression"]
)

# Map UI names back to model objects
model_map = {
    "Random Forest": rf, 
    "Xgboost": xgb, 
    "Logistic Regression": lr
}
selected_model = model_map[model_choice]

# --- MAIN UI ---
st.title("SafeCoin: Bitcoin Fraud Detector")
st.markdown("Checking transactions for **Safe (Licit)** vs **Suspicious (Illicit)** activity.")

# Note: Metrics customized for naive users
model_stats = {
    "Random Forest": {
        "precision": "99.50%",
        "recall": "88.42%",
        "f1": "93.63%",
        "accuracy": "99.1%"
    },
    "Xgboost": {
        "precision": "100.00%",
        "recall": "87.24%",
        "f1": "93.19%",
        "accuracy": "99.2%"
    },
    "Logistic Regression": {
        "precision": "44.23%",
        "recall": "92.23%",
        "f1": "59.79%",
        "accuracy": "90.1%"
    }
}

# 1. Dynamic Performance Section
st.header(f"Security Profile: {model_choice}")
stats = model_stats[model_choice]

col1, col2, col3, col4 = st.columns(4)

# Displaying the dynamic stats based on selection
col1.metric("Overall Success Rate", stats["accuracy"])
col2.metric("Alarm Accuracy (Precision)", stats["precision"])
col3.metric("Fraud Catch Rate (Recall)", stats["recall"], delta="Priority")
col4.metric("Reliability (F1-Score)", stats["f1"])

st.divider()

# 2. Test Set Prediction Section
st.subheader("🧪 Test a Live Transaction")
st.write("Pick a transaction ID from the test database to see how our AI handles it.")
row_idx = st.number_input("Enter Transaction ID to check", 0, len(test_data)-1, 0)

# Display specific transaction details
sample = test_data.iloc[[row_idx]]
actual_val = sample['label'].values[0]
actual_label = "🔴 Suspicious (Illicit)" if actual_val == 1 else "🟢 Safe (Licit)"

st.info(f"**Verified Record for this Transaction:** {actual_label}")

if st.button("🚀 Run Security Check"):
    # Prepare features
    X_raw = sample.drop(columns=['label'])
    
    # Scale features (Critical step)
    X_scaled = scaler.transform(X_raw)
    
    # Run predictions for comparison
    results = {
        "Detection Method": ["Random Forest (Model 1)", "Xgboost (Model 2)", "Logistic Regression (Model 3)"],
        "Verdict": [
            "🔴 Suspicious" if rf.predict(X_scaled)[0] == 1 else "🟢 Safe",
            "🔴 Suspicious" if xgb.predict(X_scaled)[0] == 1 else "🟢 Safe",
            "🔴 Suspicious" if lr.predict(X_scaled)[0] == 1 else "🟢 Safe"
        ]
    }
    
    st.table(pd.DataFrame(results))
    
    # Final Summary
    final_pred = selected_model.predict(X_scaled)[0]
    if final_pred == 1:
        st.warning(f"**AI Verdict:** This transaction is flagged as **Suspicious**.")
    else:
        st.info(f"**AI Verdict:** This transaction is flagged as **Safe**.")

    # 2. Check if the AI was correct compared to the verified record
    is_correct = (final_pred == actual_val)

    if is_correct:
        st.success(f"✅ **Correct Prediction:** The {model_choice} successfully matched the verified record!")
    else:
        st.error(f"❌ **Incorrect Prediction:** The {model_choice} failed to match the verified record.")

# --- HELP SECTION ---
with st.expander("❓ What do these labels mean?"):
    st.write("""
    - **(Random Forest):** Uses a group of decision trees to vote on fraud.
    - **(XGBoost):** An advanced scanner that learns from previous mistakes.
    - **(Logistic Regression):** Uses mathematical patterns to separate safe from suspicious.
    - **Fraud Catch Rate (Recall):** The percentage of all actual frauds that the model successfully found.
    """)