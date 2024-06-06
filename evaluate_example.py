import numpy as np
import pandas as pd
import mrf_data
import mrf
import evaluate

def suite1(file):
    cols = ["room_type",
            "accommodates",
            "beds",
            "price"]
    df = pd.read_csv(file, usecols=cols)
    df.dropna(inplace=True)
    df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
    df['beds'] = df['beds'].astype(int)
    df['accommodates'] = df['accommodates'].astype(int)
    data = mrf_data.data(dataframe=df, drop_zero=True, clean=cols, cols=cols)

    loss_tvd = []
    loss_kl = []
    # q1
    p1 = data.marg("price", cond=lambda df: df[df["accommodates"] == 3])
    p2 = data.marg("price") # in practice you'd want to replace this with a MRF dist.
    loss_tvd.append(evaluate.total_variation_distance(np.array(p1), np.array(p2)))
    loss_kl.append(evaluate.kl_divergence(np.array(p1), np.array(p2)))

    # q2
    p1 = data.marg("price", cond=lambda df: df[df["beds"] < 6])
    p2 = data.marg("price") # in practice you'd want to replace this with a MRF dist.
    loss_tvd.append(evaluate.total_variation_distance(np.array(p1), np.array(p2)))
    loss_kl.append(evaluate.kl_divergence(np.array(p1), np.array(p2)))


    # q3
    p1 = data.marg("price", cond=lambda df: df[(df["beds"] < 3) & (df["room_type"] == "Private room")])
    p2 = data.marg("price") # in practice you'd want to replace this with a MRF dist.
    loss_tvd.append(evaluate.total_variation_distance(np.array(p1), np.array(p2)))
    loss_kl.append(evaluate.kl_divergence(np.array(p1), np.array(p2)))


    # q4
    p1 = data.marg("room_type", cond=lambda df: df[df["price"] > 100])
    p2 = data.marg("room_type") # in practice you'd want to replace this with a MRF dist.
    print(p1, p2)
    loss_tvd.append(evaluate.total_variation_distance(np.array(p1), np.array(p2)))
    loss_kl.append(evaluate.kl_divergence(np.array(p1), np.array(p2)))

    return {
        "tvd": loss_tvd, 
        "kl": loss_kl
    }

if __name__ == "__main__":
    """
    {
        "tvd": [
            0.13375000000000004,
            0.0180769230769231,
            0.23149572649572647,
            0.0
        ],
        "kl": [
            0.12174494224578239,
            0.003555734393945808,
            0.1862837628503838,
            0.0
        ]
    } 
    """
    import json
    print(json.dumps(suite1("Albany010624"), indent=4, ))
