# Set up and run this Streamlit App
import streamlit as st
import pandas as pd
# from helper_functions import llm
from logics.customer_query_handler import process_user_message
import random  
import hmac 

from helper_functions.utility import check_password

# Do not continue if check_password is not True.  
if not check_password():  
    st.stop()

# Set page configuration as the first Streamlit command
st.set_page_config(
    page_title="My Streamlit App",
    page_icon=":house:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the page titles
page_titles = ["Home / About This App", "Q&A", "View All Financial Assistance in NP", "Methodology"]

# Create a sidebar with the page titles as options
selected_page = st.sidebar.selectbox("Select a page", page_titles)

# endregion <--------- Streamlit App Configuration --------->
if selected_page == "Q&A":
    st.title("NP Financial Assistance Q&A")

    form = st.form(key="form")
    form.subheader("What do you want to know about financial assistance schemes in NP?")

    user_prompt = form.text_area("Suggested question: I am looking at tuition fee help. I am from a low income family. / What help can I get for laptop purchase?", height=200)

    if form.form_submit_button("Submit"):
        st.toast(f"User Input Submitted: {user_prompt}")

        st.divider()

        response, scheme_details = process_user_message(user_prompt)
        st.write(response)

        st.divider()

        if scheme_details:
            df = pd.DataFrame(scheme_details)
            st.dataframe(df)  # Display scheme details as a dataframe
        else:
            st.write("No relevant schemes found.")

elif selected_page == "View All Financial Assistance in NP":
    st.title("All Types of Financial Assistance in NP")
    
    # Read the CSV file
    df = pd.read_csv("2023FinancialAssistanceSchemes.csv")
    st.write(df)   


elif selected_page == "Home / About This App":
    st.title("Home / About This App")
    st.write("Understanding what this app does")

    with st.expander("What do we have here?"):
        st.write('''
            This site has all information about financial assistance schemes available to students in NP.
        ''')

    with st.expander("How to use this app?"):
        st.write('''
            1. Navigate to the Q&A page
            2. Enter your prompt in the box
            3. Click Submit
            4. Output generated
        ''')

    with st.expander("Use cases"):
        st.write('''
            1. Ask questions related to the financial assistance schemes.
            2. Show all financial assistance schemes at a glance. 
        ''')

    with st.expander("Disclaimer"):
        st.write('''
            IMPORTANT NOTICE: This web application is a prototype developed for educational purposes only. The information provided here is NOT intended for real-world usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.\
            Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.\
            Always consult with qualified professionals for accurate and personalized advice. 
        ''')


elif selected_page == "Methodology":
    st.title("Methodology flow for the LLM")
    
    flowchart = """
  Start
    |
    v
[User Input (Query)]
    |
    v
[Identify Relevant Schemes]
    |
    +-----------------------------+
    |                             |
    | Matches Found               | No Matches Found
    |                             |
    v                             |
[Retrieve Scheme Details]         |
    |                             |
    v                             |
[Generate Response]               |
    |                             |
    v                             |
[Return Response and Details] <---+
    |
    v
   End
"""

    # Display the flowchart
    st.code(flowchart, language='plaintext')

    with st.expander("Further explanations"):
        st.write('''
            1. Problem Statement: Difficulties in finding the suitable financial assistance schemes.
            2. Proposed Solution: Getting a chatbot with information on the financial assistance schemes to answer any queries in NLP.
        ''')
