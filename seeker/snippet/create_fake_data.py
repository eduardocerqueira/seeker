#date: 2024-08-26T17:06:03Z
#url: https://api.github.com/gists/f81463f8230bc7f911bda4e4bf6708f0
#owner: https://api.github.com/users/alonsoir

import pandas as pd
from faker import Faker
import random

fake = Faker()


def generate_fake_data(num_entries=10):
    data = []

    for _ in range(num_entries):
        entry = {
            "Name": fake.name(),
            "Address": fake.address(),
            "Email": fake.email(),
            "Phone Number": fake.phone_number(),
            "Date of Birth": fake.date_of_birth(minimum_age=18, maximum_age=65).strftime("%Y-%m-%d"),
            "Random Number": random.randint(1, 100),
            "Job Title": fake.job(),
            "Company": fake.company(),
            "Lorem Ipsum Text": fake.text(),
        }
        data.append(entry)

    return pd.DataFrame(data)


if __name__ == "__main__":
    num_entries = 10  # You can adjust the number of entries you want to generate
    fake_data_df = generate_fake_data(num_entries)

    ## Dataframe with Fake Data
    print(fake_data_df)