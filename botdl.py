import nltk
import json
import pickle
import numpy as np
import random
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import streamlit as st

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the documentation dataset
def load_dataset(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            dataset = [json.loads(line) for line in lines]
        return dataset
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        # Sample dataset if file not found
        return [
            {"input": "What is Project?", "output": "Admin > Setup > Project \nUnder the first menu item the user can set up project specific details such as the company name, \nassign a logo (that will appear on the main interface as well as all reports), and the project title."},
            {"input": "Explain regions", "output": "Admin > Setup > Regions \nA project can be divided into multiple regions, then into sub-regions. Regions do not represent a \ngeographic space though they can."}
        ]
    
# Process the dataset
def process_dataset(dataset):
    questions = []
    answers = {}
    all_words = []
    
    for item in dataset:
        # Get the question and answer
        question = item['input']
        answer = item['output']
        
        # Tokenize the question
        tokens = word_tokenize(question.lower())
        # Add to the list of tokens
        all_words.extend(tokens)
        # Add the question and its tokens to the questions list
        questions.append((tokens, question))
        # Store the answer
        answers[question] = answer
    
    # Lemmatize and remove duplicates from the word list
    all_words = [lemmatizer.lemmatize(word) for word in all_words]
    all_words = sorted(list(set(all_words)))
    
    # Store unique questions for the classification
    unique_questions = sorted(list(set([q for _, q in questions])))
    
    return all_words, questions, unique_questions, answers

# Create the training data
def create_training_data(all_words, questions, unique_questions):
    training = []
    output_empty = [0] * len(unique_questions)
    
    # Create a bag of words for each question
    for question_tokens, question in questions:
        bag = []
        # Lemmatize the tokens
        question_words = [lemmatizer.lemmatize(word) for word in question_tokens]
        
        # Create the bag of words
        for word in all_words:
            bag.append(1) if word in question_words else bag.append(0)
        
        # Create the output row with 1 for the current question
        output_row = list(output_empty)
        output_row[unique_questions.index(question)] = 1
        
        training.append([bag, output_row])
    
    # Shuffle the training data
    random.shuffle(training)
    training = np.array(training, dtype=object)
    
    # Split the features and labels
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])
    
    return train_x, train_y

# Create the model
def create_model(train_x, train_y):
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))
    
    # Compile the model
    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    return model

# Train and save the model
def train_model(train_x, train_y, model_path='documentation_model.h5'):
    model = create_model(train_x, train_y)
    
    # Train the model
    model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)
    
    # Save the model
    model.save(model_path)
    st.success(f"Model trained and saved to {model_path}")
    
    return model

# Process user input
def process_input(user_input, all_words):
    # Tokenize and lemmatize the input
    input_words = word_tokenize(user_input.lower())
    input_words = [lemmatizer.lemmatize(word) for word in input_words]
    
    # Create a bag of words
    bag = [0] * len(all_words)
    for word in input_words:
        for i, w in enumerate(all_words):
            if w == word:
                bag[i] = 1
    
    return np.array(bag)

# Get response from the model
def get_response(user_input, model, all_words, unique_questions, answers, threshold=0.2):
    # Process the input
    input_data = process_input(user_input, all_words)
    
    # Predict the question
    results = model.predict(np.array([input_data]))[0]
    
    # Get the index of the highest probability
    max_index = np.argmax(results)
    
    # If the probability is greater than the threshold, return the answer
    if results[max_index] > threshold:
        predicted_question = unique_questions[max_index]
        return answers[predicted_question]
    else:
        return "I'm not sure I understand. Could you please rephrase your question?"

# Load or train model
@st.cache_resource
def load_or_train_model(dataset_path):
    try:
        # Try to load the saved data and model
        with open('documentation_data.pkl', 'rb') as f:
            all_words, unique_questions, answers = pickle.load(f)
        model = load_model('documentation_model.h5')
        return model, all_words, unique_questions, answers
    except:
        # If not available, process the dataset and train the model
        dataset = load_dataset(dataset_path)
        all_words, questions, unique_questions, answers = process_dataset(dataset)
        train_x, train_y = create_training_data(all_words, questions, unique_questions)
        model = train_model(train_x, train_y)
        
        # Save the processed data
        with open('documentation_data.pkl', 'wb') as f:
            pickle.dump((all_words, unique_questions, answers), f)
            
        return model, all_words, unique_questions, answers

# Streamlit app
def main():
    st.title("Documentation Assistant")
    st.write("Ask me anything about the documentation!")
    
    # File path for the dataset
    dataset_path = r"C:\bot_misc\training_dataset.jsonl"
    
    # Initialize session state for chat history if it doesn't exist
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Welcome to the Documentation Assistant. How can I help you today?"}]
    
    # Load or train the model
    with st.spinner("Loading model..."):
        model, all_words, unique_questions, answers = load_or_train_model(dataset_path)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # User input
    user_input = st.chat_input("Type your question here...")
    
    # Generate and display response
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Get response from the model
        response = get_response(user_input, model, all_words, unique_questions, answers)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.write(response)

# Run the app
if __name__ == "__main__":
    main()