from transformers import TFAutoModelForQuestionAnswering, AutoTokenizer
import tensorflow as tf
import re

# Specify the model and tokenizer identifiers
model_identifier = "asafaya/bert-base-arabic"
tokenizer_identifier = "asafaya/bert-base-arabic"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(tokenizer_identifier)
model = TFAutoModelForQuestionAnswering.from_pretrained(model_identifier)

# Read the text content from the file
with open("demofile2.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Split the text into pages based on "page_number:"
pages = re.split(r'page_number:\d+', text)

# Define a question in Arabic
question = " ما هوالحج"

# Set a maximum length for tokenization
max_length = 512  # You can adjust this value as needed

with open("answers.txt", "w", encoding="utf-8") as writeFile:
    # Process each page of text and perform question-answering
    for i, page_text in enumerate(pages[1:], start=1):  # Skip the first empty page
        # Tokenize the page text and question
        inputs = tokenizer(
            page_text, 
            question, 
            return_tensors="tf", 
            padding=True, 
            truncation=True, 
            max_length=max_length
        )

        # Perform question-answering
        output = model(inputs)

        # Retrieve the answer
        answer_start = tf.argmax(output.start_logits, axis=1).numpy()[0]
        answer_end = tf.argmax(output.end_logits, axis=1).numpy()[0]
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"].numpy()[0][answer_start:answer_end + 1]))

        #print(f"Page {i} ")
        if answer:
            #print(f"Page {i} - Question:", question)
            #print("Answer:", answer)
            writeFile.write(f"Page {i} - Answer: " + answer + "\n")
        #else:
            #print(f"Page {i} - No answer found.")

