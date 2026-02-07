import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess_and_train():
    # 1. Загрузка данных
    df = pd.read_csv('russian_companies_data.csv')

    # 2. Feature Engineering (Финансовый анализ)
    # Создаем коэффициенты, которые реально используются в корпоративных финансах
    df['liquidity_ratio'] = df['total_assets'] / (df['short_term_debt'] + 1)  # Текущая ликвидность
    df['autonomy_ratio'] = df['equity'] / (df['total_assets'] + 1)  # Коэф. автономии
    df['roa'] = df['net_income'] / (df['total_assets'] + 1)  # Рентабельность активов
    df['debt_to_equity'] = df['short_term_debt'] / (df['equity'] + 1)  # Плечо рычага
    df['revenue_to_assets'] = df['revenue'] / (df['total_assets'] + 1)  # Оборачиваемость

    # Выбираем признаки для обучения
    features = [
        'total_assets', 'revenue', 'net_income', 'equity',
        'liquidity_ratio', 'autonomy_ratio', 'roa',
        'debt_to_equity', 'revenue_to_assets'
    ]
    X = df[features]
    y = df['bankrupt']

    # 3. Разделение на выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Инициализация CatBoost с поддержкой GPU
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function='Logloss',
        task_type='GPU',  # Использование твоей RTX 4060
        devices='0',  # Индекс видеокарты
        verbose=100,  # Вывод логов каждые 100 итераций
        early_stopping_rounds=50  # Остановка, если точность перестала расти
    )

    # 5. Обучение
    print("Начинаем обучение на GPU...")
    model.fit(X_train, y_train, eval_set=(X_test, y_test))

    # 6. Оценка модели
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("\n--- Отчет по метрикам ---")
    print(classification_report(y_test, preds))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")

    # 7. Визуализация важности признаков
    feature_importance = model.get_feature_importance()
    feature_names = features

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance, y=feature_names)
    plt.title('Какие финансовые показатели важнее всего для прогноза банкротства?')
    plt.xlabel('Важность')
    plt.ylabel('Показатель')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("\nГрафик важности признаков сохранен в 'feature_importance.png'")

    return model


if __name__ == "__main__":
    trained_model = preprocess_and_train()