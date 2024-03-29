# Amazon-Reviews-Rating-Prediction
## Overview

Welcome to the Amazon Reviews Rating Prediction project! This repository contains a fine-tuned BERT-based model designed to predict ratings on a scale of 1 to 5 for Amazon product reviews. The model has been trained on a carefully curated subset of 500,000 reviews from the expansive Amazon Customer Reviews dataset. Leveraging the power of a pretrained BERT base model, our goal is to provide accurate and insightful predictions for your Amazon reviews.

Sample output from the Model
<img width="1672" alt="Screenshot 2024-01-02 at 7 35 51 PM" src="https://github.com/m-mahmoud-mohamed/Amazon-Reviews-Rating-Prediction/assets/78882792/14d778bc-1bb2-471b-802a-8c99ff64c381">


## Model Details

- **Architecture:** BERT Base
- **Fine-Tuning Data:** 500,000 Amazon reviews subset
- **Task:** Amazon Reviews Rating Prediction
- **Rating Scale:** 1 to 5

## Model Availability

Experience the power of our model effortlessly by accessing it on Hugging Face. Simply follow this [link](https://huggingface.co/MahmoudMohamed/Amazon_rating_review_model) to integrate the model into your projects.

## Docker Image

We understand the importance of ease in deployment. Therefore, we provide a Docker image on Docker Hub, allowing you to utilize the model with just a few simple commands. Whether you have a single text review or an entire CSV file of reviews, our Docker image has you covered.
[link](https://hub.docker.com/repository/docker/mahmoudmohamedmahmoud/amazon_review/general)

**Pull the Docker image:**

   ```bash
   docker pull mahmoudmohamedmahmoud/amazon_review:latest


