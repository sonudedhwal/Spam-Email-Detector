import random
import nltk
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data files (runs only the first time)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

lemmatizer = WordNetLemmatizer()

# Define patterns, intents, and possible responses
chatbot_knowledge = {
    "greetings": {
        "patterns": ["hello", "hi", "hey", "greetings", "sup"],
        "responses": ["Hello! How can I help you navigate NLP today?", "Hi there! What's on your mind?", "Greetings! Ready to talk code?"]
    },
    "how_are_you": {
        "patterns": ["how are you", "how do you do", "are you okay"],
        "responses": ["I'm just a bundle of Python code, but I'm fully operational!", "Doing great! Ready to process some language."]
    },
    "nlp_definition": {
        "patterns": ["what is nlp", "explain nlp", "natural language processing"],
        "responses": ["NLP stands for Natural Language Processing. It's how computers understand, analyze, and generate human language!"]
    },
    "goodbye": {
        "patterns": ["bye", "goodbye", "exit", "quit", "see you"],
        "responses": ["Goodbye! Keep coding!", "See you later! Feel free to come back with more questions."]
    }
}

def preprocess_and_tokenize(user_text):
    """Tokenizes and lemmatizes user input for better matching."""
    # Convert to lowercase and tokenize into words
    tokens = nltk.word_tokenize(user_text.lower())
    # Lemmatize each word to its base root form (e.g., 'running' -> 'run')
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return cleaned_tokens

def match_intent(user_tokens):
    """Finds which intent matches the user's input tokens best."""
    for intent, data in chatbot_knowledge.items():
        for pattern in data["patterns"]:
            # Check if any pattern word exists in the user's tokens
            if any(word in user_tokens for word in pattern.split()):
                return intent
    return None

print("--- NLP Smart Chatbot Initialized ---")
print("Type something to chat. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Chatbot: Goodbye! Keep exploring NLP.")
        break
        
    # Process user input using NLP fundamentals
    tokens = preprocess_and_tokenize(user_input)
    intent = match_intent(tokens)
    
    # Respond based on matched intent
    if intent and intent in chatbot_knowledge:
        reply = random.choice(chatbot_knowledge[intent]["responses"])
        print(f"Chatbot: {reply}")
    else:
        print("Chatbot: That's fascinating! As an NLP bot, I'm still learning. Could you rephrase that?")
