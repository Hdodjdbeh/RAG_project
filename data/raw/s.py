import pandas as pd
import json

# Загрузи данные (укажи свой путь к файлу)
df = pd.read_csv('balanced_dataset.csv')

# Посмотрим структуру
print(f"Всего отзывов: {len(df)}")
print("\nПримеры:")
print(df[['productValuation', 'text']].head(10))



# Сохраним в удобном формате
reviews = df['text'].tolist()
ratings = df['productValuation'].tolist()

# Сохрани в JSON для дальнейшей работы
with open('reviews_clean.json', 'w', encoding='utf-8') as f:
    json.dump([{'text': t, 'rating': r} for t, r in zip(reviews, ratings)],
              f, ensure_ascii=False, indent=2)