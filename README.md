AskMate — AI FAQ Chatbot

An NLP-powered Frequently Asked Questions (FAQ) chatbot built using Python and Streamlit. AskMate matches user questions against a curated FAQ knowledge base and returns the most relevant available answer.

Features

Interactive chatbot interface.

NLP-based question matching using TF-IDF.

Cosine similarity for matching user questions.

Word and character-level text features.

Suggested questions and chat history.

FAQ dataset managed through CSV.

Similarity score and matched FAQ display.

Responsive Streamlit interface.

Technologies Used

Language: Python

Frontend: Streamlit and custom CSS

Data Processing: Pandas

NLP: Scikit-learn TF-IDF Vectorizer

Matching Algorithm: Cosine Similarity

Dataset: CSV

Project Structure

AI-FAQ-Chatbot/
├── app.py
├── faqs.csv
├── requirements.txt
├── README.md
└── .gitignore

Installation and Setup

Clone this repository:

git clone YOUR_GITHUB_REPOSITORY_URL
cd AI-FAQ-Chatbot

Create a virtual environment:

python -m venv .venv

Activate it on Windows PowerShell:

.\.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

Dataset Format

The faqs.csv file must contain two columns:

Question

Answer

Add or edit FAQ records in this file to customize the chatbot's knowledge base.

How It Works

Load FAQ questions and answers from the CSV dataset.

Convert FAQ questions into numerical TF-IDF vectors.

Process the user's question using word-level and character-level features.

Calculate cosine similarity scores to identify the closest matching FAQ.

Display the corresponding answer when a suitable match is found.

Note: AskMate retrieves answers from its FAQ dataset. It does not generate unrestricted answers like a large language model.

Future Improvements

Expand the FAQ dataset.

Add topic-based FAQ categories.

Improve spelling correction and semantic question matching.

Add user feedback for answer quality.

Deploy the application online.

Author

Developed as an internship project to demonstrate Python, NLP, information retrieval, and Streamlit application development.