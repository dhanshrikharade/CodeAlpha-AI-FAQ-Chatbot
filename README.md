# AskMate — AI FAQ Chatbot

An NLP-powered FAQ chatbot built with Python and Streamlit. AskMate matches user questions against a curated FAQ dataset and returns the most relevant answers using text similarity techniques.

## Features

- Interactive chatbot interface
- NLP-based question matching using TF-IDF
- Cosine similarity for finding relevant answers
- Word-level and character-level text analysis
- Suggested questions and chat history
- CSV-based FAQ dataset
- Similarity score and matched FAQ display
- Custom dark theme with orange accents

## Technologies Used

- **Language:** Python
- **Frontend:** Streamlit, HTML, CSS
- **Data Processing:** Pandas
- **NLP:** Scikit-learn TF-IDF Vectorizer
- **Matching Algorithm:** Cosine Similarity
- **Dataset:** CSV

## Project Structure

```text
AI-FAQ-Chatbot/
├── askmate.py
├── faqs.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/dhanshrikharade/AI-FAQ-Chatbot.git
   ```

2. Open the project directory:

   ```bash
   cd AI-FAQ-Chatbot
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the chatbot:

   ```bash
   streamlit run askmate.py
   ```

## How It Works

1. The chatbot loads FAQs from a CSV dataset.
2. TF-IDF converts questions into numerical text features.
3. Cosine similarity compares the user's question with stored FAQs.
4. The chatbot returns the most relevant available answer.

## Future Improvements

- Expand the FAQ knowledge base
- Improve intent recognition
- Add multilingual question support
- Deploy the chatbot online

## Author

**Dhanashri Kharade**

GitHub: https://github.com/dhanshrikharade

---

*Built as an internship project to explore Natural Language Processing and conversational AI.*
