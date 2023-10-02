
# Customized Mobile Packages Chatbot

## Description
This chatbot provides information and answers queries related to mobile packages. Users can ask questions about different mobile packages, plans, features, and other related topics. The chatbot uses an underlying model to generate relevant responses based on user input and past chat history.

## Installation
1. Ensure you have Python (version 3.7 or newer) installed on your system.
2. Install necessary libraries and dependencies using pip:
```bash
pip install streamlit sentence-transformers langchain
```
3. Clone the repository or download the project files to your local machine.
4. Ensure the data file `mobile_packages.csv` and the model file `llama-2-7b-chat.ggmlv3.q8_0.bin` are placed in the appropriate directories (`data/` and `models/` respectively).

## Usage
1. Navigate to the project directory in your terminal or command prompt.
2. Run the Streamlit application using the command:
```bash
streamlit run main.py
```
3. Open the provided link in your web browser to interact with the chatbot.
4. Enter your question in the text input field and click on the 'Ask the Chatbot' button to get a response.

## Acknowledgments
- The project utilizes the langchain library for chatbot modeling and Streamlit for the user interface.
- Special thanks to OpenAI for guidance and assistance in enhancing the project.

**Note**: For any issues or further enhancements, please open an issue or submit a pull request to the project repository.
