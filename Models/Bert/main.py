from transformers import pipeline
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf


# Load pre-trained transformer model for question answering
qa_pipeline = pipeline("question-answering", model="deepset/bert-base-cased-squad2")

with open("data.txt", "r", encoding="utf-8") as file:
    context = file.read()


while True:
    def get_transformer_response(question):
        if question == 'bye':
            print("Byee")
            exit()
        result = qa_pipeline(question=question, context=context)
        return result["answer"]

    # Example usage
    reqest=input("Enter Your Query:")
    print(get_transformer_response(reqest))  # Output: Shaik Masthan Basha
