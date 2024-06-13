# PRD Analyzer for Abuse Cases 🚨

This is a Streamlit-based application that analyzes Product Requirement Documents (PRDs) to identify potential abuse cases and suggest appropriate control methods using OpenAI's GPT-3.5-turbo model.

## Features

- Upload PDF and Word documents for analysis
- Extract text from uploaded documents
- Use OpenAI's GPT-3.5-turbo to analyze the text for potential abuse cases
- Suggest appropriate control methods for identified abuse cases

## Installation

1. **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up environment variables:**
    - Create a `.env` file in the root directory of the project.
    - Add your OpenAI API key to the `.env` file as follows:
      ```env
      OPENAI_API_KEY=your_openai_api_key
      ```

4. **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

## Usage
1. Open the application in your web browser.
2. Upload a PDF or Word document using the file uploader.
3. The application will process the file, analyze the text, and display potential abuse cases along with suggested control methods.

## License

This project is licensed under the MIT License.
