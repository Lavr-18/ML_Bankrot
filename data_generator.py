import pandas as pd
import numpy as np


def generate_russian_finance_data(n_samples=1000):
    np.random.seed(42)

    data = []
    for i in range(n_samples):
        # 10% компаний будут банкротами (Label 1)
        is_bankrupt = 1 if np.random.random() < 0.10 else 0

        # Генерируем базовые показатели (в тыс. рублей)
        if is_bankrupt:
            assets = np.random.uniform(10000, 50000)
            revenue = np.random.uniform(5000, 20000)
            net_income = np.random.uniform(-10000, 0)  # Убыток
            short_term_debt = assets * np.random.uniform(0.7, 1.2)  # Долг больше активов
            equity = assets - short_term_debt  # Часто отрицательный
        else:
            assets = np.random.uniform(50000, 500000)
            revenue = assets * np.random.uniform(0.5, 2.0)
            net_income = revenue * np.random.uniform(0.01, 0.15)  # Прибыль 1-15%
            short_term_debt = assets * np.random.uniform(0.1, 0.5)
            equity = assets - short_term_debt

        data.append({
            'company_id': i,
            'total_assets': assets,
            'revenue': revenue,
            'net_income': net_income,
            'short_term_debt': short_term_debt,
            'equity': equity,
            'cash': assets * np.random.uniform(0.01, 0.1),
            'bankrupt': is_bankrupt
        })

    return pd.DataFrame(data)


if __name__ == "__main__":
    df = generate_russian_finance_data(2000)
    df.to_csv('russian_companies_data.csv', index=False)
    print("Dataset 'russian_companies_data.csv' generated.")