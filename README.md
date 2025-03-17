Here is an updated version of the README file with the GitHub repository link included:

---

# Personal AI Assistant with Chat History

This is a Streamlit-based web application that provides a personal AI assistant powered by OpenAI's GPT models, integrated with ChromaDB for conversation memory and user authentication. It allows users to ask questions, store conversations, and retrieve past chats.

The project is hosted on GitHub:

[Memory-Based AI Chatbot with GPT](https://github.com/Notmeher/Memory-Based-AI-Chatbot-With-GPT)

## Features
- **User Authentication**: Users can sign up or log in using a user ID and password. Passwords are securely hashed before being stored in ChromaDB.
- **AI Responses**: The assistant generates responses using OpenAI's GPT models (gpt-4 or gpt-3.5-turbo) and allows customization of the model and response temperature.
- **Conversation Memory**: All conversations are saved and indexed in ChromaDB for future retrieval. Users can review and load previous conversations.
- **Secure Storage**: User data and conversation history are stored in ChromaDB, with each message being saved along with metadata (role, timestamp, conversation ID).
- **Persistent Conversations**: Conversations are assigned unique IDs, making it easy to resume past conversations at any time.

## Requirements
- Python 3.8 or higher
- Streamlit
- OpenAI API Key
- ChromaDB (used for storing conversations)
- dotenv (for managing environment variables)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Notmeher/Memory-Based-AI-Chatbot-With-GPT.git
   cd Memory-Based-AI-Chatbot-With-GPT
   ```

2. **Install the required Python packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up the environment variables**:
   - Create a `.env` file in the root directory of the project.
   - Add your OpenAI API key to the `.env` file:
     ```bash
     OPENAI_API_KEY=your_openai_api_key
     ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

   This will start the Streamlit web application. Open your browser and navigate to the provided local URL to start using the AI assistant.

## Usage

- **Login/Signup**: In the sidebar, enter your user ID and password to log in or create a new account.
- **Start a New Conversation**: After logging in, you can start a new conversation, and the assistant will remember all your queries and responses.
- **Retrieve Past Conversations**: You can view your past conversations and load them by clicking on their titles in the "Past Conversations" section.
- **Settings**: Change the model or temperature for AI responses in the settings section.

## How It Works

- **User Authentication**: The user's credentials are securely hashed using SHA-256 and stored in ChromaDB. When a user logs in, the system verifies their password against the stored hash.
- **AI Responses**: When the user asks a question, the assistant sends the input to OpenAI’s API (either `gpt-4` or `gpt-3.5-turbo`), and the response is displayed in the app.
- **ChromaDB Integration**: All conversations are stored in ChromaDB for persistence. Messages are indexed, making it easy to search and load past conversations.

## File Structure

```plaintext
.
├── app.py                    # Main Streamlit application file
├── .env                      # Environment variables (OpenAI API key)
├── requirements.txt           # Required Python packages
└── chroma_db/                 # ChromaDB persistent storage folder
```

## Future Enhancements

- **Improved User Interface**: Design a more intuitive UI for managing conversations.
- **Better Security**: Implement more robust authentication mechanisms such as OAuth or multi-factor authentication (MFA).
- **AI Model Customization**: Allow users to fine-tune AI models for better personalized responses.
- **Analytics**: Provide insights into user conversations, including sentiment analysis and topic modeling.

## License

This project is open-source and available under the MIT License.

## Acknowledgements

- **OpenAI**: For providing the powerful GPT models.
- **ChromaDB**: For offering a reliable and scalable database solution for persistent storage.

---

![screencapture-localhost-8501-2025-03-17-19_20_37](https://github.com/user-attachments/assets/28f396a2-fc7c-4e0e-9e4d-69bc43f53041)
