import json
import pandas as pd
from helper_functions import llm

# Load the CSV file
filepath = './2023FinancialAssistanceSchemes.csv'
df = pd.read_csv(filepath)

# Convert CSV data into a dictionary format for easy lookup by scheme name
dict_of_schemes = df.set_index('Financial_Scheme').to_dict(orient='index')

# Predefined categories and schemes
category_n_scheme_name = {'Financial Aid for Tuition Fee': ['Post Secondary Education Account (PSEA)',
                                                          'CPF Education Loan Scheme',
                                                          'Tuition Fee Loan',
                                                          'Tertiary Tuition Fee Subsidy for Malay Students',
                                                          'Government Study Loan'],
                          'Grants': ['NP Emergency Grant'],
                          'Bursaries': ['Higher Education Community Bursary',
                                        'Higher Education Bursary',
                                        'Diploma Foundation Programme (DFP) Bursary',
                                        'Private Donor Bursaries & Grants',
                                        'Higher Education Bursary (for PTD students)'],
                          'Overseas Student Programme Schemes': ['Post-Secondary Education Account (PSEA)',
                                            'Overseas Programme Loan'],
                          'Mobile Computing Schemes': ['Interest-free Financing Loan for Purchase of Notebook',
                                                       'Interest-free Financing Loan for Purchase of iPad',
                                                       'Opportunity Fund Subsidy for Purchase of Notebook',
                                                       'Opportunity Fund Subsidy for Purchase of iPad',
                                                       'IMDA DigitalAccess@Home']}

def identify_category_and_schemes(user_message):
    delimiter = "####"

    system_message = f"""
    You will be provided with customer service queries. \
    The customer service query will be enclosed in
    the pair of {delimiter}.

    Decide if the query is relevant to any specific financial schemes
    in the Python dictionary below, where each key is a `category`
    and the value is a list of `financial_scheme` names.

    If there are any relevant scheme(s) found, output the pair(s) of a) `financial_scheme` and b) the associated `category` into a
    list of dictionary objects, where each item in the list is a relevant scheme
    and each scheme is a dictionary that contains two keys:
    1) category
    2) financial_scheme

    {category_n_scheme_name}

    If no relevant schemes are found, output an empty list.

    Ensure your response contains only the list of dictionary objects or an empty list, \
    without any enclosing tags or delimiters.
    """

    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': f"{delimiter}{user_message}{delimiter}"},
    ]
    category_and_product_response_str = llm.get_completion_by_messages(messages)
    category_and_product_response_str = category_and_product_response_str.replace("'", "\"")
    category_and_product_response = json.loads(category_and_product_response_str)
    return category_and_product_response

def get_scheme_details(list_of_relevant_category_n_scheme: list[dict]):
    scheme_details = []
    for scheme in list_of_relevant_category_n_scheme:
        scheme_name = scheme.get('financial_scheme')
        if scheme_name in dict_of_schemes:
            scheme_details.append(dict_of_schemes[scheme_name])
    return scheme_details

def generate_response_based_on_scheme_details(user_message, scheme_details):
    delimiter = "####"

    system_message = f"""
    Follow these steps to answer the customer queries.
    The customer query will be delimited with a pair {delimiter}.

    Step 1:{delimiter} If the user is asking about financial aid, \
    identify the relevant scheme(s) from the following list.
    All available schemes are shown in the data below:
    {scheme_details}

    Step 2:{delimiter} Use the information about the scheme to \
    generate the answer for the customer's query.
    You must only rely on the facts or information in the scheme information.
    Your response should be as detailed as possible and \
    include information that is useful for the customer to understand the scheme.

    Step 3:{delimiter}: Answer the customer in a friendly tone.
    Make sure the statements are factually accurate.
    Your response should be comprehensive and informative to help the \
    customer make their decision.
    Use Neural Linguistic Programming to construct your response.

    Use the following format:
    Step 1:{delimiter} <step 1 reasoning>
    Step 2:{delimiter} <step 2 reasoning>
    Step 3:{delimiter} <step 3 response to customer>

    Make sure to include {delimiter} to separate every step.
    """

    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': f"{delimiter}{user_message}{delimiter}"},
    ]

    response_to_customer = llm.get_completion_by_messages(messages)
    response_to_customer = response_to_customer.split(delimiter)[-1]
    return response_to_customer

def process_user_message(user_input):
    # Step 1: Identify relevant financial schemes
    relevant_schemes = identify_category_and_schemes(user_input)
    print("Relevant schemes:", relevant_schemes)

    # Step 2: Get the scheme details
    scheme_details = get_scheme_details(relevant_schemes)

    # Step 3: Generate response based on scheme details
    reply = generate_response_based_on_scheme_details(user_input, scheme_details)

    # Return both reply and scheme details for unpacking
    return reply, scheme_details