from transformers import TFAutoModelForQuestionAnswering, AutoTokenizer, AutoConfig, TFAutoModel

import tensorflow as tf
from langdetect import detect

# Define function to load the model and tokenizer based on language
def load_model_and_tokenizer_arabic():
    #config = AutoConfig.from_pretrained("asafaya/bert-base-arabic")
    tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
    model = TFAutoModelForQuestionAnswering.from_pretrained("asafaya/bert-base-arabic")

    return tokenizer, model

def load_model_and_tokenizer_persian():
    #config = AutoConfig.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
    tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
    model = TFAutoModelForQuestionAnswering.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")

    return tokenizer, model

# Set a maximum length for tokenization
max_length = 512  # You can adjust this value as needed

# Define a function to perform question-answering
def perform_question_answering(text, question, tokenizer, model):
    inputs = tokenizer(text, question, return_tensors="tf", padding=True, truncation=True, max_length=max_length)
    output = model(inputs)
    
    answer_start = tf.argmax(output.start_logits, axis=1).numpy()[0]
    answer_end = tf.argmax(output.end_logits, axis=1).numpy()[0]
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"].numpy()[0][answer_start:answer_end + 1]))

    if answer:
        return answer
    else:
        return "No answer found."

# Load models and tokenizers for Arabic and Persian
arabic_tokenizer, arabic_model = load_model_and_tokenizer_arabic()
persian_tokenizer, persian_model = load_model_and_tokenizer_persian()

# Load text from your file (Replace 'your_file.txt' with the actual file path)
with open('demofile2.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Specify your hard-coded question in either Arabic or Persian
question_arabic = "ما هو الحج"
question_persian = "حج چیست"

# Detect the language of the text
text_language = detect(text)

# Perform question-answering based on the detected language
if text_language == 'ar':
    answer = perform_question_answering(text, question_arabic, arabic_tokenizer, arabic_model)
elif text_language == 'fa':
    answer = perform_question_answering(text, question_persian, persian_tokenizer, persian_model)
else:
    answer = "Language not detected or unsupported."

print("Question:", question_arabic if text_language == 'ar' else question_persian)
print("Answer:", answer)
