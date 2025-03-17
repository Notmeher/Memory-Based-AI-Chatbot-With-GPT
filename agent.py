import streamlit as st
import openai
from dotenv import load_dotenv
import os
import hashlib
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction
import time
import uuid

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient("./chroma_db")

# Custom OpenAI embedding function compatible with older OpenAI API
class CustomOpenAIEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key, model_name="text-embedding-ada-002"):
        self.api_key = api_key
        self.model_name = model_name
        openai.api_key = api_key
        
    def __call__(self, texts: Documents) -> list:
        # Make sure inputs are strings
        texts = [str(text) for text in texts]
        
        # Call the old OpenAI API format
        response = openai.Embedding.create(
            model=self.model_name,
            input=texts
        )
        
        # Extract embeddings from the response
        embeddings = [data["embedding"] for data in response["data"]]
        return embeddings

# Initialize the custom embedding function
openai_ef = CustomOpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-ada-002"  # Using older model compatible with old API
)

# Simple password hashing function
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to get or create a collection for a user
def get_user_collection(user_id):
    try:
        # Try to get existing collection
        return chroma_client.get_collection(name=user_id, embedding_function=openai_ef)
    except:
        # Create new collection if it doesn't exist
        return chroma_client.create_collection(name=user_id, embedding_function=openai_ef)

# Function to get or create a collection for storing conversation metadata
def get_conversations_collection():
    try:
        # Try to get existing collection
        return chroma_client.get_collection(name="conversations", embedding_function=openai_ef)
    except:
        # Create new collection if it doesn't exist
        return chroma_client.create_collection(name="conversations", embedding_function=openai_ef)

# Function to save message to ChromaDB
def save_message_to_db(collection, message, message_id):
    collection.add(
        documents=[message["content"]],
        metadatas=[{
            "role": message["role"], 
            "timestamp": message.get("timestamp", ""),
            "conversation_id": st.session_state.get("current_conversation_id", "default")
        }],
        ids=[message_id]
    )

# Function to generate response using OpenAI API (older version)
def generate_response(prompt, model_name="gpt-4", temp=0.7):
    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant who provides accurate, concise, and useful information."},
                *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                {"role": "user", "content": prompt}
            ],
            temperature=temp,
            max_tokens=1000
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Function to load conversation history from ChromaDB
def load_conversation_history(collection):
    result = collection.get()
    messages = []
    for i, (doc, metadata) in enumerate(zip(result["documents"], result["metadatas"])):
        messages.append({
            "role": metadata["role"],
            "content": doc
        })
    return messages

# Function to load specific conversation by ID
def load_conversation_by_id(collection, conversation_id):
    result = collection.get(
        where={"conversation_id": conversation_id}
    )
    messages = []
    for i, (doc, metadata) in enumerate(zip(result["documents"], result["metadatas"])):
        messages.append({
            "role": metadata["role"],
            "content": doc
        })
    return messages

# Function to save a conversation to the conversations collection
def save_conversation(user_id, conversation_id, title=None):
    conv_collection = get_conversations_collection()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Get a title from the first few messages if not provided
    if not title and st.session_state.messages:
        first_msg = st.session_state.messages[0]["content"]
        title = first_msg[:30] + "..." if len(first_msg) > 30 else first_msg
    
    if not title:
        title = f"Conversation at {timestamp}"
        
    conv_collection.add(
        documents=[title],
        metadatas=[{
            "user_id": user_id,
            "timestamp": timestamp,
            "conversation_id": conversation_id
        }],
        ids=[f"conv_{conversation_id}"]
    )

# Function to get all conversations for a user
def get_user_conversations(user_id):
    conv_collection = get_conversations_collection()
    try:
        result = conv_collection.get(
            where={"user_id": user_id}
        )
        conversations = []
        for i, (doc, metadata, id) in enumerate(zip(result["documents"], result["metadatas"], result["ids"])):
            conversations.append({
                "id": metadata["conversation_id"],
                "title": doc,
                "timestamp": metadata["timestamp"]
            })
        # Sort by timestamp (newest first)
        conversations.sort(key=lambda x: x["timestamp"], reverse=True)
        return conversations
    except Exception as e:
        st.error(f"Error retrieving conversations: {str(e)}")
        return []

# App title and description
st.title("Your Personal AI Assistant")
st.subheader("Ask me anything, and I'll try to help!")

# Initialize session state for conversations if it doesn't exist
if "conversations" not in st.session_state:
    st.session_state.conversations = []

# Sidebar with authentication
with st.sidebar:
    st.header("User Authentication")
    
    # User login form
    user_id = st.text_input("User ID")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login/Sign Up")
    
    st.divider()
    
    # App settings (only shown when logged in)
    if "authenticated" in st.session_state and st.session_state.authenticated:
        st.header("Settings")
        
        # Option to start a new conversation
        if st.button("Start New Conversation"):
            # Save current conversation if there are messages
            if st.session_state.messages:
                # Generate a unique ID for the current conversation
                if "current_conversation_id" not in st.session_state:
                    st.session_state.current_conversation_id = str(uuid.uuid4())
                
                # Save the conversation metadata
                save_conversation(st.session_state.user_id, st.session_state.current_conversation_id)
                
                # Update the conversations list
                st.session_state.conversations = get_user_conversations(st.session_state.user_id)
            
            # Create a new conversation ID
            st.session_state.current_conversation_id = str(uuid.uuid4())
            st.session_state.messages = []  # Reset conversation history
            st.rerun()
        
        # Model selection
        model = st.selectbox(
            "Select Model",
            ["gpt-4", "gpt-3.5-turbo"],
            index=0
        )
        
        # Temperature slider
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        
        # Past conversations section
        st.divider()
        st.header("Past Conversations")
        
        if st.session_state.conversations:
            for conv in st.session_state.conversations:
                if st.button(f"{conv['timestamp']}: {conv['title']}", key=conv['id']):
                    # Load this conversation
                    st.session_state.current_conversation_id = conv['id']
                    st.session_state.messages = load_conversation_by_id(
                        st.session_state.user_collection, 
                        conv['id']
                    )
                    st.rerun()
        else:
            st.info("No past conversations")
        
        # Logout button
        st.divider()
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.user_id = None
            st.session_state.messages = []
            st.session_state.conversations = []
            if "current_conversation_id" in st.session_state:
                del st.session_state.current_conversation_id
            st.rerun()
    
    st.divider()
    st.markdown("### About")
    st.markdown("This is your personal AI assistant powered by GPT models with conversation memory.")

# Handle login/signup logic
if login_button and user_id and password:
    # Hash the password
    hashed_password = hash_password(password)
    
    # In a real app, you'd verify against a secure user database
    # Here we're just setting up a new user or accepting returning users
    
    # Create a users collection if it doesn't exist
    try:
        users_collection = chroma_client.get_collection(name="users", embedding_function=openai_ef)
    except:
        users_collection = chroma_client.create_collection(name="users", embedding_function=openai_ef)
    
    # Check if user exists
    try:
        user_result = users_collection.get(ids=[user_id])
        if user_result["ids"]:
            # User exists, verify password
            stored_password = user_result["documents"][0]
            if stored_password == hashed_password:
                st.session_state.authenticated = True
                st.session_state.user_id = user_id
                st.toast(f"Welcome back, {user_id}!")
            else:
                st.error("Incorrect password!")
        else:
            # New user, create account
            users_collection.add(
                documents=[hashed_password],
                ids=[user_id]
            )
            st.session_state.authenticated = True
            st.session_state.user_id = user_id
            st.toast(f"New account created for {user_id}!")
    except:
        # New user, create account
        users_collection.add(
            documents=[hashed_password],
            ids=[user_id]
        )
        st.session_state.authenticated = True
        st.session_state.user_id = user_id
        st.toast(f"New account created for {user_id}!")
    
    # Get or create user collection
    if "authenticated" in st.session_state and st.session_state.authenticated:
        st.session_state.user_collection = get_user_collection(user_id)
        
        # Initialize a new conversation ID
        st.session_state.current_conversation_id = str(uuid.uuid4())
        
        # Load previous conversations list
        st.session_state.conversations = get_user_conversations(user_id)
        
        # Initialize empty messages for new conversation
        st.session_state.messages = []
        st.rerun()

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main chat interface (only shown when authenticated)
if "authenticated" in st.session_state and st.session_state.authenticated:
    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Ensure we have a conversation ID
        if "current_conversation_id" not in st.session_state:
            st.session_state.current_conversation_id = str(uuid.uuid4())
        
        # Add user message to chat history
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)
        
        # Save to ChromaDB
        message_id = f"{st.session_state.user_id}_{int(time.time())}_{len(st.session_state.messages)}"
        save_message_to_db(st.session_state.user_collection, user_message, message_id)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get current model and temperature settings
        current_model = model if "model" in locals() else "gpt-4"
        current_temp = temperature if "temperature" in locals() else 0.7
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(prompt, current_model, current_temp)
                st.markdown(response)
        
        # Add assistant response to chat history
        assistant_message = {"role": "assistant", "content": response}
        st.session_state.messages.append(assistant_message)
        
        # Save to ChromaDB
        message_id = f"{st.session_state.user_id}_{int(time.time())}_{len(st.session_state.messages)}"
        save_message_to_db(st.session_state.user_collection, assistant_message, message_id)
        
        # If this is the first message in a new conversation, save the conversation metadata
        if len(st.session_state.messages) == 2:  # User message + assistant response
            save_conversation(st.session_state.user_id, st.session_state.current_conversation_id)
            # Update conversations list
            st.session_state.conversations = get_user_conversations(st.session_state.user_id)
else:
    # Show login prompt when not authenticated
    st.info("Please log in or sign up using the sidebar to start chatting.")