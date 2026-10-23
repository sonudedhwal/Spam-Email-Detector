from textblob import TextBlob

print("--- Welcome to your first NLP Test Bot! ---")
print("Type a sentence below. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye! Keep exploring NLP!")
        break
        
    # 1. Sentiment Analysis using TextBlob
    analysis = TextBlob(user_input)
    polarity = analysis.sentiment.polarity
    
    # Classify polarity score (-1.0 to 1.0)
    if polarity > 0.1:
        sentiment = "Positive 😊"
    elif polarity < -0.1:
        sentiment = "Negative 😞"
    else:
        sentiment = "Neutral 😐"
        
    print(f"   ↳ [AI Sentiment Analysis: {sentiment} (Polarity Score: {polarity:.2f})]")
    
    # 2. Simple Rule-Based Chatbot Responses
    text_lower = user_input.lower()
    if "hello" in text_lower or "hi" in text_lower:
        print("Chatbot: Hello there! How can I help you today?")
    elif "how are you" in text_lower:
        print("Chatbot: I'm just a beginner Python script, but I'm doing great!")
    elif "weather" in text_lower:
        print("Chatbot: I can't check the weather yet, but I hope it's nice outside!")
    else:
        print("Chatbot: That is interesting! Tell me more.")
