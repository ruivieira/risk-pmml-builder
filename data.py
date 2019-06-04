import numpy as np
import pandas as pd


def disp_risk(holder, amount):
    if holder == "PLATINUM":
        if amount >= 200:
            return 4
        elif amount >= 100:
            return 2
        else:
            return 1
    elif holder == "GOLD":
        if amount >= 200:
            return 4
        elif amount >= 150:
            return 3
        elif amount >= 100:
            return 2
        else:
            return 1
    elif holder == "SILVER":
        if amount >= 200:
            return 5
        elif amount >= 150:
            return 3
        elif amount >= 100:
            return 2
        else:
            return 1
    else:
        if amount >= 200:
            return 5
        elif amount >= 150:
            return 4
        elif amount >= 100:
            return 3
        else:
            return 2


def card_holder_risk(holder, incident_count, age):
    if incident_count > 3:
        if holder == "PLATINUM":
            return 1
        else:
            return 3
    elif incident_count == 3:
        if holder == "STANDARD":
            return 4
        elif holder == "SILVER":
            return 3
        elif holder == "GOLD":
            return 2
        else:
            return 1
    else:
        if holder == "STANDARD":
            if age <= 25:
                return 2
            else:
                return 1
        if holder == "SILVER":
            if age <= 25:
                return 1
            else:
                return 0
        else:
            return 0


STATUS = ["STANDARD", "SILVER", "GOLD", "PLATINUM"]


def build_dataset(size):
    amounts_raw = np.random.normal(125, 50, size)
    holders_index = np.random.randint(0, 4, size)

    amounts = list(map(lambda x: round(max(10, x), 2), amounts_raw))
    holders = list(map(lambda x: STATUS[x], holders_index))
    ages = np.random.randint(18, 70, size)
    incidents = list(map(lambda x: int(x), np.random.exponential(1.5, size)))
    dispute_risk = list(map(lambda x, y: disp_risk(x, y), holders, amounts))
    holder_risk = list(map(lambda x, y, z: card_holder_risk(x, y, z), holders, incidents, ages))

    data = {'holder': holders, 'holder_index': holders_index, 'amount': amounts, 'dispute_risk': dispute_risk,
            'incidents': incidents, 'age': ages, 'holder_risk': holder_risk}

    return pd.DataFrame(data)
