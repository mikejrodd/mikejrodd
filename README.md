
# Mike Rodden

[![LinkedIn](https://img.shields.io/badge/-LinkedIn-blue?style=flat&logo=linkedin&logoColor=white&link=https://www.linkedin.com/in/mike-rodden/)](https://www.linkedin.com/in/mike-rodden/)
[![Email](https://img.shields.io/badge/-Email-c14438?style=flat&logo=Gmail&logoColor=white&link=mailto:mike.rodden@esmt.berlin)](mailto:mike.rodden@esmt.berlin)

Welcome to my GitHub profile! I'm Mike, a Product Manager and hobby enthusiast with a deep interest in data science, machine learning, and software development. 

## 🚀 Recent Projects

### [Esca Image Classifier](https://github.com/mikejrodd/esca_image_classifier)
This project involves building and training a deep learning model to classify images of grapevine leaves affected by Esca disease. The model helps in early detection and management of the disease, which is crucial for maintaining healthy vineyards and optimizing grape yields.
- **Technologies Used:** Python, TensorFlow, Keras, OpenCV, Hugging Face
- **How It Works:**
  - **Data Loading and Preprocessing:**
    - Load grape leaf images from a dataset.
    - Preprocess the images by resizing, normalizing, and augmenting to enhance the training process.
  - **Model Training:**
    - Utilize a convolutional neural network (CNN) architecture for image classification.
    - Implement transfer learning with pre-trained models from Hugging Face to improve performance.
    - Train the model on labeled data to distinguish between healthy and infected leaves.
  - **Model Evaluation:**
    - Evaluate the model's performance using metrics such as accuracy, precision, recall, and F1-score.
    - Fine-tune the model to optimize its accuracy and reduce false positives/negatives.
  - **Deployment:**
    - Deploy the trained model to predict the presence of Esca in new images.
    - Integrate the model into a user-friendly application using Streamlit for real-time classification.

Check out the [Hugging Face model](https://huggingface.co/mikejrodd/esca_image_classifier) for more details.

### [Semantic Music Recommendations](https://github.com/mikejrodd/music_recommendations/tree/main/semantic-vdb)
This project focuses on creating a music recommendation system using semantic analysis. The goal is to provide personalized music recommendations based on user preferences and the semantic content of the music tracks.
- **Technologies Used:** Python, TensorFlow, Keras, Natural Language Processing (NLP), Semantic Analysis, React, Flask, FastAPI, Qdrant
- **How It Works:**
  - **Data Collection:**
    - Gather music track data, including metadata, lyrics, and audio features.
    - Generate a dataset by scraping lyrics from Genius.com and pre-process to clean and filter the data.
  - **Semantic Analysis:**
    - Use NLP techniques to analyze the semantic content of the lyrics and metadata.
    - Apply vector embeddings to represent the semantic meaning of the tracks using a BERT model fine-tuned on song lyrics.
  - **Model Training:**
    - Train a recommendation model using a hybrid approach of collaborative filtering and content-based filtering.
    - Utilize deep learning models to improve the accuracy of recommendations.
  - **Recommendation System:**
    - Implement the recommendation algorithm to suggest music tracks based on user preferences and the semantic content of the tracks.
    - Use a hybrid dense and sparse retrieval system powered by a vector database (Qdrant) and BM25 model.
  - **Deployment:**
    - **Back End:** Deploy the recommendation system using Flask and FastAPI to create an API that serves the recommendation results.
    - **Front End:** Build a user-friendly interface using React and Next.js for interactive exploration of music recommendations.
    - **Integration:** Connect the React front end with the Flask back end to provide real-time music recommendations.
    - **Hosting:** Host the application using cloud platforms like Heroku or AWS for the back end and Vercel or Netlify for the front end.

### [grapeleafGPT](https://github.com/mikejrodd/grapeleafGPT)
A wine industry specific fork of AnomalyGPT, a Large Vision-Language Model (LVLM) designed for Industrial Anomaly Detection (IAD). It detects and localizes anomalies in industrial images without the need for manual threshold adjustments. This project combines a pre-trained image encoder with a large language model to align visual and textual data, making it a powerful tool for anomaly detection.
- **Technologies Used:** Python, TensorFlow, Hugging Face Transformers, ImageBind, LLaMA
- **Key Features:**
  - Detects and localizes anomalies
  - Few-shot learning capabilities
  - Multi-turn dialogues for detailed analysis

### [Esca Detection App](https://github.com/mikejrodd/esca_detection_app)
This application uses a pre-trained deep learning model to detect Esca disease in grapevine leaves. Esca is a serious fungal disease that affects vineyards globally, and early detection is crucial for managing its spread.
- **Technologies Used:** Python, Streamlit, TensorFlow, Hugging Face Transformers
- **How It Works:**
  - Upload a grape leaf image
  - The app preprocesses the image and runs it through a Keras model
  - Displays the prediction result indicating the presence of Esca

### [Wine Price Prediction](https://github.com/mikejrodd/wine_price_model)
A machine learning project aimed at predicting wine prices based on various quality parameters. This project explores the relationships between wine characteristics and market pricing.
- **Technologies Used:** Python, Pandas, Scikit-learn, Jupyter Notebook
- **Highlights:**
  - Data preprocessing and feature engineering
  - Model training and evaluation
  - Insights into wine quality factors affecting price

### [Beer Recommendation System](https://github.com/mikejrodd/beer_app)
A recommendation system to help users discover new beers based on their preferences. This project involves data preprocessing, exploratory data analysis, and machine learning algorithms to provide personalized beer suggestions.
- **Technologies Used:** Python, Pandas, Scikit-learn, Streamlit
- **Features:**
  - User input for beer preferences
  - Machine learning-based recommendations
  - Interactive Streamlit app

## 🛠️ Skills

- **Product Management:** Jira, Agile, Scrum, Figma, Salesforce
- **Programming Languages:** Python, SQL
- **Data Analytics and Visualization:** Tableau, Google Analytics 4, Seaborn
- **Machine Learning Frameworks:** TensorFlow, Scikit-learn, Hugging Face Transformers
- **Web Development:** Streamlit, Flask

## 🌱 Interests

- **Wine and Technology:** Exploring the intersection of wine and tech through AI and analytics.
- **Home-brewing:** Crafting sustainable beer recipes with local ingredients.
- **Photography:** Capturing and promoting the beauty of various destinations, particularly [Lompoc, California](https://www.instagram.com/d_james_photography/).

Feel free to explore my repositories and reach out if you're interested in collaborating on exciting projects!

<!---
mikejrodd/mikejrodd is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
