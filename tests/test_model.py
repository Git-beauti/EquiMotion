import torch

from equimotion.training import (
    FutureConditionedStateSpace,
    StateSpaceConfig,
    make_speed_smoother,
)


def test_equimotion_architecture_forward_shapes():
    config = StateSpaceConfig(
        hidden_channels=16,
        layers=1,
        dropout=0.0,
        hybrid_direct=True,
        gap_position_head=True,
        identity_trajectory_bias=True,
    )
    model = FutureConditionedStateSpace(
        config, make_speed_smoother(config.speed_smooth_lambda)
    ).eval()
    features = torch.zeros(2, 300, 36)
    lv_future_speed = torch.zeros(2, 200)
    initial_speed = torch.zeros(2)
    initial_gap = torch.full((2,), 10.0)
    with torch.no_grad():
        position, speed, acceleration, gap = model(
            features, lv_future_speed, initial_speed, initial_gap
        )
    assert position.shape == (2, 200)
    assert speed.shape == (2, 200)
    assert acceleration.shape == (2, 200)
    assert gap.shape == (2, 200)
