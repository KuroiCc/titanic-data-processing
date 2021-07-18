#!/usr/bin/env python
import pandas as pd
import numpy as np
from typing import List

TRAIN_DATA_PATH = "train.csv"
TEST_DATA_PATH = "test.csv"


def substrings_in_string(big_string: str, substrings: List[str]):
    for substring in substrings:
        if big_string.find(substring) != -1:
            return substring
    # print(big_string)
    return np.nan


def replace_missing_fare_to_class_mean(df):
    df["Fare"] = df["Fare"].map(lambda x: np.nan if x == 0 else x)
    classmeans = pd.pivot_table(df, index='Pclass', values='Fare', aggfunc='mean')
    df['Fare'] = df.apply(
        lambda x: classmeans['Fare'][x['Pclass']] if pd.isnull(x['Fare']) else x['Fare'],
        axis=1
    )


def fill_cabin_embarked_to_na_unknown_mode(df):
    df['Cabin'] = df['Cabin'].fillna('Unknown')
    from scipy.stats import mode
    modeEmbarked = mode(df['Embarked'])[0][0]
    df['Embarked'] = df['Embarked'].fillna(modeEmbarked)


def replace_titles_to_4_class(df):
    title_list = [
        'Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev', 'Dr', 'Ms', 'Mlle', 'Col', 'Capt',
        'Mme', 'Countess', 'Don', 'Jonkheer'
    ]

    df['Title'] = df['Name'].map(lambda x: substrings_in_string(x, title_list))

    def replace_titles(x):
        title = x['Title']
        if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
            return 'Mr'
        elif title in ['Countess', 'Mme']:
            return 'Mrs'
        elif title in ['Mlle', 'Ms']:
            return 'Miss'
        elif title == 'Dr':
            if x['Sex'] == 'Male':
                return 'Mr'
            else:
                return 'Mrs'
        else:
            return title

    df['Title'] = df.apply(replace_titles, axis=1)


def turning_cabin_number_into_deck(df):
    cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
    df['Deck'] = df['Cabin'].map(lambda x: substrings_in_string(str(x), cabin_list))


def turning_fare_per_pclass(df):
    df['Fare_per_pclass'] = df.apply(lambda x: x['Fare'] / (4 - x['Pclass']), axis=1)


if __name__ == "__main__":
    path = TRAIN_DATA_PATH
    df = pd.read_csv(path)

    replace_missing_fare_to_class_mean(df)

    fill_cabin_embarked_to_na_unknown_mode(df)

    replace_titles_to_4_class(df)

    turning_cabin_number_into_deck(df)

    # Turning cabin number into Deck
    df['Family_Size'] = df['SibSp'] + df['Parch']

    turning_fare_per_pclass(df)

    print(df.head(), end="\n\n")

    df.to_csv(
        "processed_train_without_turning_age.csv",
        encoding="UTF-8",
        index=False
    )
