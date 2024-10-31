# Set up and run this Streamlit App
import streamlit as st
import pandas as pd
# from helper_functions import llm
from logics.customer_query_handler import process_user_message
import random  
import hmac 
import os
import sqlite3

from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

from helper_functions.utility import check_password

# Do not continue if check_password is not True.  
if not check_password():  
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="NP Financial Assistance",
    page_icon=":moneybag:",
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

    user_prompt = form.text_area("Suggested question: Financial aid for Malay students. / I am looking at tuition fee help. I am from a low income family. / What help can I get for laptop purchase?", height=200)

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
    st.title("Ask questions on up-to-date financial assistance in NP")
    # Load environment variables
    load_dotenv()

    # Set up the OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")

    form = st.form(key="my_form")
    form.subheader("For updated information from the NP website, ask questions here:")
    url = "https://www.np.edu.sg/admissions-enrolment/guide-for-prospective-students/aid"
    query = form.text_area("Ask me!", height=200)

    submitted = form.form_submit_button("Submit") 

    if submitted:
        if url and query:
            st.info("Loading documents...")
            # Load the document
            loader = WebBaseLoader(url)
            documents = loader.load()

            st.info("Processing documents...")

            # Split and chunk the documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
            all_splits = text_splitter.split_documents(documents)

            # Store embeddings in vector database
            persist_directory = 'db'
            embedding = OpenAIEmbeddings()
            vectordb = Chroma.from_documents(
                documents=all_splits,
                embedding=embedding,
                persist_directory=persist_directory
            )
            vectordb.persist()

            st.success("Generating response...")

            # Retrieve relevant chunks based on query
            docs = vectordb.similarity_search(query)

            llm = OpenAI(temperature=0)

            # Use RetrievalQA instead of load_qa_with_sources_chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectordb.as_retriever(),
                chain_type="stuff",
                return_source_documents=True
            )

            # Run the QA chain
            result = qa_chain({"query": query})
            result_text = result["result"]

            # Split the answer into lines and take the first 3 for summary
            summary_lines = result_text.splitlines()[:3]

            # Display the summary
            st.subheader("Summary:")
            for line in summary_lines:  
                st.write(line)


    st.title("All Types of Financial Assistance in NP")
    
    # Read the CSV file
    df = pd.read_csv("2023FinancialAssistanceSchemes.csv")
    st.write(df)   




elif selected_page == "Home / About This App":
    st.title("Home / About This App")
    st.write("Understanding what this app does")

    with st.expander("What do we have here?"):
        st.write('''
            This site has all information about **_financial assistance schemes_** available to students in NP.
        ''')

    with st.expander("How to use this app?"):
        st.write('''
            1. Navigate to the different pages.
            2. Enter your prompts for the LLMs.
            3. Click Submit.
            4. Response generated!
        ''')

    with st.expander("Use cases"):
        st.write('''
            1. Ask questions related to the financial assistance schemes.
            2. Ask questions related to up-to-date information on website.
            3. Show all financial assistance schemes at a glance. 
        ''')

    with st.expander("Further Explanations"):
        st.write('''
            1. **Problem Statement**: Users face difficulties finding suitable financial assistance schemes.
            2. **Proposed Solution (a)**: Creating a chatbot that uses a pre-defined dataset of financial assistance schemes to answer questions via NLP.
            3. **Proposed Solution (b)**: Enabling the chatbot to answer questions using up-to-date information retrieved from the financial assistance schemes website.
        ''')

    with st.expander("Disclaimer"):
        st.write('''
            **IMPORTANT NOTICE:** This web application is a prototype developed for educational purposes only. The information provided here is NOT intended for real-world usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.\
            Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.\
            Always consult with qualified professionals for accurate and personalized advice. 
        ''')


elif selected_page == "Methodology":
    st.title("Methodology flow for the LLM for Q&A")
    
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

    # Another flowchart
    st.title("Methodology flow for the LLM for up-to-date web search")
    
    flowchart2 = """
  START
    |
    v
[Load CSV File]
    |
    v
[Set Up Environment Variables]
    |
    v
[User Input (Form: URL and Query)]
    |
    v
[Load Web Content]
    |
    v
[Split Text into Chunks]
    |
    v
[Store Embeddings in Vector Database]
    |
    +------------------------------------+
    |                                    |
    | Relevant Chunks Found              | No Relevant Chunks Found
    |                                    |
    v                                    |
[Generate Response]                      |
    |                                    |
    v                                    |
[Summarize and Display Response] <-------+
    |
    v
   END

"""

    # Display the flowchart
    st.code(flowchart2, language='plaintext')