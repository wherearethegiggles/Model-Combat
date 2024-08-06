import os
import requests
import gradio as gr
import pandas as pd
import cohere
import random
import logging
import time
from datetime import datetime
from dotenv import load_dotenv
import gspread
from oauth2client.service_account import ServiceAccountCredentials

load_dotenv()

# Set up Google Sheets API
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_name('civic-matrix-427804-i8-73526c501530.json', scope)
gc = gspread.authorize(credentials)

# Open Google Sheets
questions_sheet = gc.open("cleaned_data").sheet1
cache_sheet = gc.open("cache").sheet1
votes_sheet = gc.open("votes").sheet1
remarks_sheet = gc.open("remarks").sheet1

# Load questions from CSV
questions_df = pd.DataFrame(questions_sheet.get_all_records())

# Extract unique categories for the filter
categories = questions_df['Category'].unique().tolist()

# Read API keys from environment variables
user_service_api_key = os.getenv('USER_SERVICE_API_KEY')
middleware_service_api_key = os.getenv('MIDDLEWARE_SERVICE_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')
cohere_api_key = os.getenv('COHERE_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
login = os.getenv('login_url')
lang = os.getenv('lang_url')
call = os.getenv('call_url')
id = os.getenv('userId')
number = os.getenv('mobileNumber')
# Initialize variables
access_token = ""
language_id = ""

def do_login():
    """
    Log in to the user service and obtain an access token.
    """
    global access_token
    login_url = login
    headers = {
        'accept': 'application/json',
        'x-api-key': user_service_api_key,
        'Content-Type': 'application/json'
    }
    body = {
        "data": {
            "userId": id,
            "mobileNumber": number,
            "platform": "WEB"
        }
    }
    
    response = requests.post(login_url, headers=headers, json=body)
    if response.status_code == 200:
        decoded_body = response.json()
        access_token = decoded_body["data"]["tokens"]["accessToken"]
        get_language()
    else:
        print("Login failed with status code:", response.status_code)

def get_language():
    """
    Get the language ID from the middleware service using the access token.
    """
    do_login()
    global language_id
    lang_url = lang
    headers = {
        "Content-Type": "application/json",
        "x-api-key": middleware_service_api_key,
        "x-access-token": access_token,
        "accept": "application/json"
    }
    
    response = requests.get(lang_url, headers=headers)
    if response.status_code == 200:
        decoded_body = response.json()
        if decoded_body["data"]["languages"]:
            language_id = decoded_body["data"]["languages"][0]["languageId"]
    else:
        print("Failed to get languages with status code:", response.status_code)

def call_chatgpt(input_text):
    """
    Call the OpenAI GPT-3.5 Turbo API to generate a response.
    
    Parameters:
        input_text (str): The input text prompt for the API.

    Returns:
        str: The generated response from the API.
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer ' + openai_api_key
    }
    body = {
        "model": "gpt-3.5-turbo-0125",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_text}
        ]
    }

    response = requests.post(url, headers=headers, json=body)

    if response.status_code == 200:
        decoded_body = response.json()
        return decoded_body["choices"][0]["message"]["content"]
    else:
        return "Failed to call ChatGPT"

def call_hanooman(input_text):
    """
    Call the Hanooman middleware service to generate a response.
    
    Parameters:
        input_text (str): The input text prompt for the API.

    Returns:
        str: The generated response from the API.
    """
    call_url = call
    headers = {
        "Content-Type": "application/json",
        "x-api-key": middleware_service_api_key,
        "x-access-token": access_token,
        "accept": "application/json"
    }
    body = {
        "data": {
            "languageId": language_id,
            "conversationId": "",
            "query": input_text,
            "platform": "WEB"
        }
    }

    response = requests.post(call_url, headers=headers, json=body)

    if response.status_code == 200:
        decoded_body = response.json()
        return decoded_body["data"]["response"]["chats"][0]["answers"][0]["answer"]
    else:
        return "Failed to call Hanooman"

def call_cohere(input_text):
    """
    Call the Cohere API to generate a response.
    
    Parameters:
        input_text (str): The input text prompt for the API.

    Returns:
        str: The generated response from the API.
    """
    co = cohere.Client(cohere_api_key)
    response = co.generate(
        prompt=input_text
    )
    return response.generations[0].text.strip()

def call_claude(input_text):
    """
    Call the Claude API to generate a response.

    Parameters:
        input_text (str): The input text prompt for the API.

    Returns:
        str: The generated response from the API.
    """
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        'Content-Type': 'application/json',
        'x-api-key': anthropic_api_key
    }
    body = {
        "prompt": input_text,
        "model": "Claude 3 Haiku",  
        "max_tokens_to_sample": 300  
    }

    response = requests.post(url, headers=headers, json=body)

    if response.status_code == 200:
        decoded_body = response.json()
        return decoded_body["completion"]
    else:
        return "Failed to call Claude"
    
def read_cache():
    """
    Read the cache from the Google Sheet into a DataFrame.
    
    Returns:
        pd.DataFrame: The cache DataFrame.
    """
    data = cache_sheet.get_all_values()
    headers = data.pop(0)
    return pd.DataFrame(data, columns=headers)

def write_cache(model, prompt, response):
    """
    Write a response to the cache CSV file.
    
    Parameters:
        model (str): The model used to generate the response.
        prompt (str): The input prompt for the model.
        response (str): The generated response from the model.
    """
    # Read the existing cache DataFrame
    cache_df = read_cache()
    
    # Create a new entry
    new_entry = {"model": model, "prompt": prompt, "response": response}
    
    # Append the new entry to the DataFrame
    cache_df = cache_df._append(new_entry, ignore_index=True)
    
    # Convert DataFrame to list of lists
    data = [cache_df.columns.values.tolist()] + cache_df.values.tolist()
    
    # Update the Google Sheet
    cache_sheet.update(data)

def format_text_with_openai(text):
    """
    Call the OpenAI GPT-3.5 Turbo API to format the given text.
    
    Parameters:
        text (str): The text to be formatted.

    Returns:
        str: The formatted text from the API.
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer ' + openai_api_key
    }
    body = {
        "model": "gpt-3.5-turbo-0125",
        "messages": [
            {"role": "system", "content": "Format the following text without changing its content"},
            {"role": "user", "content": text}
        ]
    }

    response = requests.post(url, headers=headers, json=body)

    if response.status_code == 200:
        decoded_body = response.json()
        return decoded_body["choices"][0]["message"]["content"]
    else:
        return text

def process_input(input_text, model):
    """
    Process the input text using the specified model, using cache if available.
    
    Parameters:
        input_text (str): The input text prompt for the model.
        model (str): The model to use ("ChatGPT", "Hanooman", or "Cohere").

    Returns:
        str: The generated response from the model.
    """
    cache_df = read_cache()
    cached_response = cache_df[(cache_df["model"] == model) & (cache_df["prompt"] == input_text)]
    
    if not cached_response.empty:
        return cached_response.iloc[0]["response"]
    
    if model == "ChatGPT":
        response = call_chatgpt(input_text)
    elif model == "Hanooman":
        response = call_hanooman(input_text)
    elif model == "Cohere":
        response = call_cohere(input_text)
    elif model == "Claude":
        response = call_claude(input_text)
    elif model == "Stack_response":
        raw_response = questions_df[questions_df["Title"] == input_text]["Accepted Answer"].values[0]
        response = format_text_with_openai(raw_response)
    
    write_cache(model, input_text, response)
    return response

def save_vote_to_google_sheet(vote_data):
    # Convert lists to strings for Google Sheets compatibility
    for key, value in vote_data.items():
        if isinstance(value, list):
            vote_data[key] = ", ".join(map(str, value))
    
    votes_df = pd.DataFrame([vote_data])
    votes_sheet.append_rows(votes_df.values.tolist(), value_input_option="RAW")

def save_remark_to_google_sheet(remark_data):
    # Convert lists to strings for Google Sheets compatibility
    for key, value in remark_data.items():
        if isinstance(value, list):
            remark_data[key] = ", ".join(map(str, value))
    
    remarks_df = pd.DataFrame([remark_data])
    remarks_sheet.append_rows(remarks_df.values.tolist(), value_input_option="RAW")


def vote_and_generate_new(model_a, model_b, vote_type, prompt, response_a, response_b, category):
    """
    Handle the voting and generate new responses from the models.
    
    Parameters:
        model_a (str): The first model being compared.
        model_b (str): The second model being compared.
        vote_type (str): The type of vote ("A is better", "B is better", "Tie", "Both are bad").
        prompt (str): The input prompt used for generating the responses.
        response_a (str): The response from model A.
        response_b (str): The response from model B.
        category (str): The category of the question.
        remark (str): The remark if both responses are considered bad.

    Returns:
        Tuple: Updated chatbot content, model names, responses, and Gradio components' states.
    """
    vote_data = {
        "models": [model_a, model_b],
        "vote_response": vote_type,
        "prompt": prompt,
        "response_a": response_a,
        "response_b": response_b,
        "category": category,  
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    save_vote_to_google_sheet(vote_data)
    return gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(visible=False), gr.update(interactive=False)

def freeze():
    """
    Show the remark input box.

    Returns:
        gr.update: Update object to make the remark input box visible.
    """
    return gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(visible=False), gr.update(interactive=False)

def Category_freeze():
    
    return gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(visible=False), gr.update(interactive=False)

def filter_questions(category):
    """
    Filter questions based on the selected category and randomly select tags for further filtering.
    
    Parameters:
        category (str): The category to filter questions.

    Returns:
        List: List of filtered questions.
    """
    filtered_df = questions_df.copy()
    if category:
        filtered_df = filtered_df[filtered_df["Category"] == category]

    # Randomly select 10 tags from the filtered questions
    tags = set(tag for sublist in filtered_df["Tags"].str.split(",").tolist() for tag in sublist if tag)
    tags_list = list(tags)
    if len(tags_list) > 10:
        selected_tags = random.sample(tags_list, 10)
    else:
        selected_tags = tags_list

    # Filter questions based on the selected tags
    if selected_tags:
        filtered_df = filtered_df[filtered_df["Tags"].apply(lambda x: any(tag in x for tag in selected_tags))]

    return filtered_df["Title"].tolist()

def skip(category):
    """
    Skip the current question and generate new responses from the models.
    
    Parameters:
        category (str): The category to filter questions.

    Returns:
        Tuple: Updated chatbot content, model names, responses, and Gradio components' states.
    """
    filtered_questions = filter_questions(category)
    prompt = random.choice(filtered_questions)
    model_a, model_b = random.sample(["ChatGPT", "Cohere", "Stack_response"], 2)
    response_a = process_input(prompt, model_a)
    response_b = process_input(prompt, model_b)
    unfreeze_buttons = [gr.update(interactive=True)] * 6
    show_remark_box = gr.update(visible=True)
    return [(prompt, response_a)], [(prompt, response_b)], model_a, model_b, prompt, response_a, response_b, unfreeze_buttons[0], unfreeze_buttons[1], unfreeze_buttons[2], unfreeze_buttons[3], unfreeze_buttons[4], show_remark_box, unfreeze_buttons[5]

def catagory_generate(category):
    """
    Skip the current question and generate new responses from the models.
    
    Parameters:
        category (str): The category to filter questions.

    Returns:
        Tuple: Updated chatbot content, model names, responses, and Gradio components' states.
    """
    filtered_questions = filter_questions(category)
    prompt = random.choice(filtered_questions)
    model_a, model_b = random.sample(["ChatGPT", "Cohere", "Stack_response"], 2)
    response_a = process_input(prompt, model_a)
    response_b = process_input(prompt, model_b)
    unfreeze_buttons = [gr.update(interactive=True)] * 6
    show_remark_box = gr.update(visible=True)
    return [(prompt, response_a)], [(prompt, response_b)], model_a, model_b, prompt, response_a, response_b, unfreeze_buttons[0], unfreeze_buttons[1], unfreeze_buttons[2], unfreeze_buttons[3], unfreeze_buttons[4], show_remark_box, unfreeze_buttons[5]

def submit_remark(model_a, model_b, prompt, response_a, response_b, category, remark):
    """
    Submit a remark for the given responses.

    Parameters:
        model_a (str): The first model being compared.
        model_b (str): The second model being compared.
        prompt (str): The input prompt used for generating the responses.
        response_a (str): The response from model A.
        response_b (str): The response from model B.
        category (str): The category of the question.
        remark (str): The remark for the responses.

    Returns:
        str: Empty string (used for clearing the input box).
    """
    remark_data = {
        "models": [model_a, model_b],
        "Category": category,
        "Prompt": prompt,
        "response_a": response_a,
        "response_b": response_b,
        "remark": remark,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    save_remark_to_google_sheet(remark_data)
    return ""


css = """
    .chatbox {
        border: 2px solid #e3e3e3;
        border-radius: 10px;
        overflow-y: scroll;  /* Enable manual scrolling */
        height: 550px;  /* Fixed height to avoid window size change */
    }
    .message {
        padding: 10px;
        margin: 5px;
        border-radius: 10px;
        display: inline-block;
        max-width: 80%;
    }
    .message.user {
        background-color: #d1f0ff;
        align-self: flex-end;
    }
    .message.bot {
        background-color: #f1f0f0;
        align-self: flex-start;
    }
    .chatbox::-webkit-scrollbar {
        width: 8px;
    }
    .chatbox::-webkit-scrollbar-thumb {
        background: #ccc;
        border-radius: 4px;
    }
    .chatbox::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
"""

js = """
function maintainScrollPosition() {
    let chatboxes = document.querySelectorAll('.chatbox');
    chatboxes.forEach(chatbox => {
        chatbox.scrollTop = 0;  // Ensure it stays at the top
    });
}

// Observe changes in the chatbox and apply scroll position maintenance
let observer = new MutationObserver(maintainScrollPosition);
document.querySelectorAll('.chatbox').forEach(chatbox => {
    observer.observe(chatbox, { childList: true });
});
"""

with gr.Blocks(css=css) as demo:
    notice_markdown = """
## ⚔️  Model Combat

## 📜 Rules
- **Select a Category that interests you.**
- **Vote for the response that you find more vaild or better than the other one**
- **Votes will be recorded to evaluate the models.**
- **Feel free to share any remarks on the responses**

## 👇 Chat now! """

    gr.Markdown(notice_markdown, elem_id="notice_markdown")
    with gr.Row():
        category_dr = gr.Dropdown(choices=categories, label="Category", value=None)

    with gr.Row():
        chatbot_a = gr.Chatbot(label="Model A", height=650, elem_classes=["chatbox"], show_copy_button=True,)
        chatbot_b = gr.Chatbot(label="Model B", height=650, elem_classes=["chatbox"], show_copy_button=True,)
    
    with gr.Accordion("Remarks", open=False):
        remark_text = gr.Textbox(lines=4, show_label=False, placeholder="👉 Enter remark here ", elem_id="input_box", label="Remark")
        save_remark_button = gr.Button(value="Submit Remark", variant="primary", scale=0, interactive=False)

    with gr.Row():
        leftvote_btn = gr.Button("👈  A is better", interactive=False)
        rightvote_btn = gr.Button("👉  B is better", interactive=False)
        tie_btn = gr.Button("🤝  Tie", interactive=False)
        bothbad_btn = gr.Button("👎  Both are bad", interactive=False)
        play_btn = gr.Button("⏭️  Skip", variant="primary", size="large", interactive=False)
    
    prompt = gr.State()
    response_a = gr.State()
    response_b = gr.State()
    model_a = gr.State()
    model_b = gr.State()

    category_dr.change(
        catagory_generate,
        inputs=[category_dr],
        outputs=[chatbot_a, chatbot_b, model_a, model_b, prompt, response_a, response_b, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, play_btn, category_dr, save_remark_button]
    )

    category_dr.change(
        Category_freeze,
        outputs = [leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, play_btn, category_dr, save_remark_button]
    )

    play_btn.click(
        skip,
        inputs=[category_dr],
        outputs=[chatbot_a, chatbot_b, model_a, model_b, prompt, response_a, response_b, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, play_btn, category_dr, save_remark_button]
    )

    play_btn.click(
        freeze,
        outputs=[leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, play_btn, category_dr, save_remark_button]
    )

    save_remark_button.click(
        submit_remark,
        inputs=[model_a, model_b, prompt, response_a, response_b, category_dr, remark_text],
        outputs=[remark_text]
    )

    leftvote_btn.click(
        vote_and_generate_new,
        inputs=[model_a, model_b, gr.Textbox(visible=False, value="A is better"), prompt, response_a, response_b, category_dr],
        outputs=[leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, play_btn, category_dr, save_remark_button]
    )

    leftvote_btn.click(
        skip,
        inputs=[category_dr],
        outputs=[chatbot_a, chatbot_b, model_a, model_b, prompt, response_a, response_b, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, play_btn, category_dr, save_remark_button]
    )
    
    rightvote_btn.click(
        vote_and_generate_new,
        inputs=[model_a, model_b, gr.Textbox(visible=False, value="B is better"), prompt, response_a, response_b, category_dr],
        outputs=[leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, play_btn, category_dr, save_remark_button]
    )

    rightvote_btn.click(
        skip,
        inputs=[category_dr],
        outputs=[chatbot_a, chatbot_b, model_a, model_b, prompt, response_a, response_b, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, play_btn, category_dr, save_remark_button]
    )

    tie_btn.click(
        vote_and_generate_new,
        inputs=[model_a, model_b, gr.Textbox(visible=False, value="Tie"), prompt, response_a, response_b, category_dr],
        outputs=[leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, play_btn, category_dr, save_remark_button]
    )
    
    tie_btn.click(
        skip,
        inputs=[category_dr],
        outputs=[chatbot_a, chatbot_b, model_a, model_b, prompt, response_a, response_b, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, play_btn, category_dr, save_remark_button]
    )

    bothbad_btn.click(
        vote_and_generate_new,
        inputs=[model_a, model_b, gr.Textbox(visible=False, value="Both are bad"), prompt, response_a, response_b, category_dr],
        outputs=[leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, play_btn, category_dr, save_remark_button]
    )

    bothbad_btn.click(
        skip,
        inputs=[category_dr],
        outputs=[chatbot_a, chatbot_b, model_a, model_b, prompt, response_a, response_b, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, play_btn, category_dr, save_remark_button]
    )

demo.launch(share=True, server_port=7999)
