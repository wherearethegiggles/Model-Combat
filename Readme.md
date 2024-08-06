# ⚔️ Model Combat

Model Combat is an application that compares responses from different AI models (ChatGPT, Hanooman, and Cohere) based on user inputs. Users can vote on which model gives a better response or remark on the responses. The results are saved in Google Sheets for further analysis.

## Features

- **Category Selection**: Users can filter questions based on specific categories.
- **Model Comparison**: Compare responses from ChatGPT, Hanooman, Cohere models and Stack Responses.
- **Voting System**: Vote on which model provides a better response.
- **Remarks**: Provide remarks on the responses given by the models.
- **Data Storage**: Votes and remarks are stored in Google Sheets.

## Installation

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up your environment variables. Create a `.env` file in the root directory and add the following:
    ```
    USER_SERVICE_API_KEY=<your-user-service-api-key>
    MIDDLEWARE_SERVICE_API_KEY=<your-middleware-service-api-key>
    OPENAI_API_KEY=<your-openai-api-key>
    COHERE_API_KEY=<your-cohere-api-key>
    ```

4. Set up Google Sheets API credentials:
    - Obtain your `credentials.json` file from Google Cloud Console.
    - Save the file in the root directory of your project.

## Usage

1. Run the application:
    ```bash
    python app.py
    ```

2. Open the provided URL in your web browser.

## Components

### Authentication
- **do_login**: Logs in to the user service and obtains an access token.
- **get_language**: Gets the language ID from the middleware service using the access token.

### API Calls
- **call_chatgpt**: Calls the OpenAI GPT-3.5 Turbo API to generate a response.
- **call_hanooman**: Calls the Hanooman middleware service to generate a response.
- **call_cohere**: Calls the Cohere API to generate a response.

### Google Sheets
- **read_cache**: Reads the cache from the Google Sheet into a DataFrame.
- **write_cache**: Writes a response to the cache in the Google Sheet.
- **save_vote_to_google_sheet**: Saves votes to the Google Sheet.
- **save_remark_to_google_sheet**: Saves remarks to the Google Sheet.

### User Interaction
- **process_input**: Processes the input text using the specified model, using cache if available.
- **vote_and_generate_new**: Handles the voting and generates new responses from the models.
- **filter_questions**: Filters questions based on the selected category and randomly selects tags for further filtering.
- **skip**: Skips the current question and generates new responses from the models.
- **submit_remark**: Submits a remark for the given responses.

### Gradio Interface
The application uses Gradio to create an interactive web interface with the following components:
- **Category Dropdown**: Allows users to select a category to filter questions.
- **Chatbots**: Displays responses from Model A and Model B.
- **Voting Buttons**: Allows users to vote on which model provides a better response.
- **Remarks**: Allows users to submit remarks for the responses.


