import numpy as np
import pandas as pd

from equimotion.metrics import score_segment


def make_segment():
    time = np.arange(300) * 0.1
    fav = 10.0 * time
    lv = fav + 10.0
    return pd.DataFrame(
        {
            "Pos_FAV": fav,
            "Pos_LV": lv,
            "Spatial_Headway": np.full(300, 10.0),
            "Spatial_Gap": np.full(300, 5.5),
        }
    )


def test_perfect_constant_speed_prediction_scores_one():
    segment = make_segment()
    prediction = segment["Pos_FAV"].iloc[100:].to_numpy()
    metrics = score_segment(segment, prediction)
    assert metrics["accuracy"] == 1.0
    assert metrics["safety"] == 1.0
    assert metrics["comfort"] > 0.999999
    assert metrics["final"] > 0.999999
