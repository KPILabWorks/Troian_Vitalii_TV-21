import os
import googleapiclient.discovery
from textblob import TextBlob
import pandas as pd
import re

# --- 1. Налаштування API ---
API_KEY = "1111" # не справжнє API з метою зпхисту даних ;)  

# Створення клієнта YouTube API
youtube = googleapiclient.discovery.build(
    "youtube", "v3", developerKey=API_KEY)

# --- 2. Функції для роботи з API та даними ---
def clean_text(text):
    """Очищає текст від непотрібних символів."""
    text = re.sub(r'<[^>]+>', '', text) # Видалення HTML тегів
    text = re.sub(r'http\S+', '', text)  # Видалення URL
    text = re.sub(r'[^a-zA-Zа-яА-ЯіІїЇєЄґҐ\s\'.,!?-]', '', text) # Видалення символів, окрім літер, пробілів та деяких розділових знаків
    text = text.strip()
    return text

def get_video_comments(video_id, max_results=100):
    """Отримує коментарі до відео за його ID."""
    comments_data = []
    try:
        # Виконуємо запит для отримання коментарів
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(max_results, 100), # API дозволяє максимум 100 за раз
            textFormat="plainText" # Отримуємо простий текст
        )
        response = request.execute()

        total_comments_processed = 0
        while response and total_comments_processed < max_results:
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                cleaned_text = clean_text(comment['textDisplay'])
                if cleaned_text: # Якщо коментар не порожній після очищення
                    comments_data.append({
                        'comment_id': item['id'],
                        'author': comment['authorDisplayName'],
                        'text': cleaned_text,
                        'published_at': comment['publishedAt']
                    })
                    total_comments_processed += 1
                    if total_comments_processed >= max_results:
                        break # Досягли бажаної кількості коментарів

            # Перевіряємо, чи є наступна сторінка коментарів
            if 'nextPageToken' in response and total_comments_processed < max_results:
                next_page_token = response['nextPageToken']
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=min(100, max_results - total_comments_processed),
                    pageToken=next_page_token,
                    textFormat="plainText"
                )
                response = request.execute()
            else:
                break 

    except googleapiclient.errors.HttpError as e:
        print(f"Сталася помилка API: {e}")
    except Exception as e:
        print(f"Сталася інша помилка: {e}")

    return comments_data

def analyze_comment_sentiment(comments):
    """Аналізує емоційне забарвлення списку коментарів."""
    sentiments = []
    for comment in comments:
        blob = TextBlob(comment['text'])
        sentiment_score = blob.sentiment.polarity
        sentiments.append({
            'comment_id': comment['comment_id'],
            'text': comment['text'],
            'sentiment_score': sentiment_score,
            'sentiment_label': 'positive' if sentiment_score > 0 else ('negative' if sentiment_score < 0 else 'neutral')
        })
    return sentiments

def calculate_sentiment_summary(sentiment_results):
    """Обчислює зведену статистику настроїв."""
    df = pd.DataFrame(sentiment_results)
    if df.empty:
        return 0, 0, 0

    positive = (df['sentiment_score'] > 0).sum()
    negative = (df['sentiment_score'] < 0).sum()
    neutral = (df['sentiment_score'] == 0).sum()
    total = len(df)

    positive_percentage = positive / total if total > 0 else 0
    negative_percentage = negative / total if total > 0 else 0
    neutral_percentage = neutral / total if total > 0 else 0

    return positive_percentage, negative_percentage, neutral_percentage

# --- 3. Запуск аналізу ---

if __name__ == "__main__":
    video_id_to_analyze = "dQw4w9WgXcQ" # Rick Astley - Never Gonna Give You Up (для прикладу)
    max_comments_to_fetch = 2000 # Кількість коментарів

    print(f"Збір коментарів для відео ID: {video_id_to_analyze} (максимум {max_comments_to_fetch})")
    comments = get_video_comments(video_id_to_analyze, max_comments_to_fetch)

    if comments:
        print(f"Отримано {len(comments)} коментарів. Аналіз емоційного забарвлення...")
        sentiment_results = analyze_comment_sentiment(comments)

        positive_perc, negative_perc, neutral_perc = calculate_sentiment_summary(sentiment_results)

        print("\nРезультати аналізу настроїв:")
        print(f"  Позитивні: {positive_perc:.2%}")
        print(f"  Негативні: {negative_perc:.2%}")
        print(f"  Нейтральні: {neutral_perc:.2%}")

        # Створення DataFrame для перегляду результатів
        df_results = pd.DataFrame(sentiment_results)
        print("\nПриклад даних з аналізом настроїв:")
        print(df_results.head())

        # Збереження результатів у файл
        df_results.to_csv(f"youtube_sentiment_{video_id_to_analyze}.csv", index=False, encoding='utf-8-sig')
        print(f"\nРезультати збережено у файл youtube_sentiment_{video_id_to_analyze}.csv")

    else:
        print("Не вдалося отримати коментарі для цього відео.")