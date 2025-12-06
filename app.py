import streamlit as st
import pandas as pd
import os
from datetime import datetime

from backend.utils import authenticate_user, get_user_info, get_user_id, MERCHANT_TYPES, CHANNELS
from backend.model import fraud_model, log_transaction, get_user_transaction_history


st.set_page_config(
    page_title="PayU Fraud Detection System",
    page_icon="🔐",
    layout="wide"
)


def init_session_state():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'pending_transaction' not in st.session_state:
        st.session_state.pending_transaction = None
    if 'show_suspicious_dialog' not in st.session_state:
        st.session_state.show_suspicious_dialog = False


def login_page():
    st.title("PayU Financial Fraud Detection System")
    st.subheader("Secure Login")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("---")
        
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        
        if st.button("Login", type="primary", use_container_width=True):
            if authenticate_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid username or password. Please try again.")
        
        st.markdown("---")
        st.info("Demo accounts: ahmed/ahmed123, bilal/bilal123, sana/sana123, etc.")


def load_ml_models():
    if not fraud_model.is_loaded:
        models_exist = os.path.exists("models/base_rf.pkl") and \
                       os.path.exists("models/base_xgb.pkl") and \
                       os.path.exists("models/stacked_model.pkl")
        
        if models_exist:
            try:
                fraud_model.load_models()
                return True
            except Exception as e:
                st.error(f"Failed to load ML models. Please run the training pipeline first. Error: {str(e)}")
                return False
        else:
            return False
    return True


def dashboard_page():
    username = st.session_state.username
    user_info = get_user_info(username)
    user_id = get_user_id(username)
    
    st.sidebar.title(f"Welcome, {username.title()}")
    st.sidebar.markdown(f"**User ID:** {user_id}")
    st.sidebar.markdown(f"**Account Balance:** ${user_info['balance']:,.2f}")
    
    if st.sidebar.button("Logout", type="secondary"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.pending_transaction = None
        st.session_state.show_suspicious_dialog = False
        st.rerun()
    
    st.title("Transaction Dashboard")
    
    models_loaded = load_ml_models()
    
    if not models_loaded:
        st.warning("ML models not yet trained. The system will use default risk assessment. Please run the training pipeline first.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Account Summary")
        
        user_history = get_user_transaction_history(user_id)
        limits = fraud_model.calculate_user_limits(user_id, user_info['balance'])
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric("Current Balance", f"${user_info['balance']:,.2f}")
        
        with metric_col2:
            st.metric("ML Base Limit", f"${limits['base_limit']:,.2f}")
        
        with metric_col3:
            st.metric("Total Limit (with Leverage)", f"${limits['total_limit']:,.2f}")
        
        st.markdown("---")
        st.markdown(f"**Leverage Amount:** ${limits['leverage']:,.2f}")
        st.caption("Your transaction limit is calculated using ML-based risk assessment.")
    
    with col2:
        st.subheader("New Transaction")
        
        with st.form("transaction_form"):
            amount = st.number_input("Transaction Amount ($)", min_value=0.01, max_value=100000.0, value=100.0, step=10.0)
            merchant_type = st.selectbox("Merchant Type", MERCHANT_TYPES)
            channel = st.selectbox("Transaction Channel", CHANNELS)
            
            submit_button = st.form_submit_button("Process Transaction", type="primary", use_container_width=True)
            
            if submit_button:
                if amount > user_info['balance']:
                    st.error("Insufficient balance for this transaction.")
                else:
                    user_history = get_user_transaction_history(user_id)
                    
                    result = fraud_model.evaluate_transaction(
                        user_id=user_id,
                        amount=amount,
                        merchant_type=merchant_type,
                        channel=channel,
                        balance=user_info['balance'],
                        user_history=user_history if len(user_history) > 0 else None
                    )
                    
                    if result['status'] == "APPROVED":
                        log_transaction(result, "APPROVED")
                        st.success(f"Transaction Approved! Amount: ${amount:,.2f}")
                        st.info(f"Fraud Probability: {result['fraud_probability']*100:.2f}%")
                    
                    elif result['status'] == "REJECTED":
                        log_transaction(result, "REJECTED")
                        st.error(f"Transaction Rejected! {result['message']}")
                        st.warning(f"Fraud Probability: {result['fraud_probability']*100:.2f}%")
                    
                    elif result['status'] == "SUSPICIOUS":
                        st.session_state.pending_transaction = result
                        st.session_state.show_suspicious_dialog = True
                        st.rerun()
    
    if st.session_state.show_suspicious_dialog and st.session_state.pending_transaction:
        show_suspicious_dialog()
    
    st.markdown("---")
    display_transaction_history(user_id)


def show_suspicious_dialog():
    result = st.session_state.pending_transaction
    
    st.subheader("Suspicious Transaction Detected")
    
    warning_container = st.container()
    
    with warning_container:
        st.warning(f"""
        **Transaction flagged for review**
        
        - **Amount:** ${result['amount']:,.2f}
        - **Merchant:** {result['merchant_type']}
        - **Channel:** {result['channel']}
        - **Fraud Probability:** {result['fraud_probability']*100:.2f}%
        - **Risk Level:** {result['risk_level']}
        - **Reason:** {result['message']}
        """)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("Approve Transaction", type="primary", use_container_width=True):
                log_transaction(result, "APPROVED_BY_USER")
                st.session_state.pending_transaction = None
                st.session_state.show_suspicious_dialog = False
                st.success("Transaction approved by user.")
                st.rerun()
        
        with col2:
            if st.button("Reject Transaction", type="secondary", use_container_width=True):
                log_transaction(result, "REJECTED_BY_USER")
                st.session_state.pending_transaction = None
                st.session_state.show_suspicious_dialog = False
                st.info("Transaction rejected by user.")
                st.rerun()
        
        with col3:
            if st.button("Cancel", use_container_width=True):
                st.session_state.pending_transaction = None
                st.session_state.show_suspicious_dialog = False
                st.rerun()


def display_transaction_history(user_id: str):
    st.subheader("Transaction History")
    
    history_path = "data/transactions_history.csv"
    
    if os.path.exists(history_path):
        df = pd.read_csv(history_path)
        user_history = df[df['user_id'] == user_id].copy()
        
        if len(user_history) > 0:
            user_history = user_history.sort_values('timestamp', ascending=False)
            
            display_df = user_history[['timestamp', 'amount', 'merchant', 'channel', 'status', 'fraud_prob', 'risk_level']].copy()
            display_df.columns = ['Date/Time', 'Amount ($)', 'Merchant', 'Channel', 'Status', 'Fraud Prob', 'Risk Level']
            
            def color_status(val):
                if 'APPROVED' in str(val):
                    return 'background-color: #90EE90'
                elif 'REJECTED' in str(val):
                    return 'background-color: #FFB6C1'
                else:
                    return 'background-color: #FFD700'
            
            st.dataframe(
                display_df.style.applymap(color_status, subset=['Status']),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No transactions yet. Make your first transaction above!")
    else:
        st.info("No transaction history available. Make your first transaction above!")


def main():
    init_session_state()
    
    if st.session_state.logged_in:
        dashboard_page()
    else:
        login_page()


if __name__ == "__main__":
    main()
