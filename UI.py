from transformers import pipeline
import gradio as gr

model = pipeline("text-classification",model="MahmoudMohamed/Amazon_rating_review_model")


def predict_rating(review):
    
    rating = model(review)
    print(rating)
    
    return rating[0]['label'] 

ui = gr.Interface(fn=predict_rating, inputs="textbox", outputs="label")

ui.launch() 