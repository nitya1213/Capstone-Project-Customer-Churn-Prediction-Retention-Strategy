import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from utils import load_csv, save_csv
from preprocessing import basic_cleaning, feature_engineering

os.makedirs('outputs/figures', exist_ok=True)

def plot_churn_distribution(df):
    plt.figure(figsize=(6,4))
    sns.countplot(x='Churn (Target Variable)', data=df)
    plt.title('Overall Churn Distribution')
    plt.savefig('outputs/figures/churn_distribution.png')
    plt.close()

def churn_by_age_gender_country(df):
    # Case 1: distribution across age groups, genders, countries
    df['age_bin'] = pd.cut(df['Age'], bins=[0,25,35,45,55,65,100])
    plt.figure(figsize=(10,6))
    sns.countplot(x='age_bin', hue='Churn (Target Variable)', data=df)
    plt.title('Churn by Age Group')
    plt.savefig('outputs/figures/churn_by_age.png')
    plt.close()

    plt.figure(figsize=(6,4))
    sns.countplot(x='Gender', hue='Churn (Target Variable)', data=df)
    plt.title('Churn by Gender')
    plt.savefig('outputs/figures/churn_by_gender.png')
    plt.close()

    plt.figure(figsize=(10,6))
    churn_country = df.groupby('Country')['Churn (Target Variable)'].mean().sort_values(ascending=False)
    churn_country.plot(kind='bar')
    plt.ylabel('Churn Rate')
    plt.title('Churn Rate by Country')
    plt.savefig('outputs/figures/churn_by_country.png')
    plt.close()

def membership_vs_churn(df):
    plt.figure(figsize=(6,4))
    sns.barplot(x='Membership Status', y='Churn (Target Variable)', data=df, estimator=lambda x: sum(x)/len(x))
    plt.title('Membership Status vs Churn Rate')
    plt.savefig('outputs/figures/membership_vs_churn.png')
    plt.close()

def product_preferences_churn(df):
    plt.figure(figsize=(10,5))
    prod = df.groupby('Product Purchased')['Churn (Target Variable)'].mean().sort_values(ascending=False)
    prod.plot(kind='bar')
    plt.title('Churn Rate by Product Purchased')
    plt.ylabel('Churn Rate')
    plt.savefig('outputs/figures/product_churn.png')
    plt.close()

def feedback_vs_churn(df):
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Churn (Target Variable)', y='Feedback Score', data=df)
    plt.title('Feedback Score by Churn')
    plt.savefig('outputs/figures/feedback_vs_churn.png')
    plt.close()

def support_calls_vs_churn(df):
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Churn (Target Variable)', y='Customer Support Calls', data=df)
    plt.title('Support Calls by Churn')
    plt.savefig('outputs/figures/support_calls_vs_churn.png')
    plt.close()

def recency_vs_churn(df):
    from preprocessing import feature_engineering
    df_r = feature_engineering(df.copy())
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Churn (Target Variable)', y='DaysSinceLastPurchase', data=df_r)
    plt.title('Days Since Last Purchase by Churn')
    plt.savefig('outputs/figures/recency_vs_churn.png')
    plt.close()

def login_freq_patterns(df):
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Churn (Target Variable)', y='Login Frequency', data=df)
    plt.title('Login Frequency by Churn')
    plt.savefig('outputs/figures/login_freq_vs_churn.png')
    plt.close()

def run_all_cases():
    df = load_csv('data/customer_churn.csv')
    df = basic_cleaning(df)

    plot_churn_distribution(df)
    churn_by_age_gender_country(df)
    membership_vs_churn(df)
    product_preferences_churn(df)
    feedback_vs_churn(df)
    support_calls_vs_churn(df)
    recency_vs_churn(df)
    login_freq_patterns(df)

    print("EDA plots saved to outputs/figures/")

if __name__ == '__main__':
    run_all_cases()
