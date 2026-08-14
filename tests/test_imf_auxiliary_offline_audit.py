import torch

from diffuser.scripts.audit_imf_auxiliary_offline import raw_actions


def test_raw_actions_inverts_train_range_normalization():
    normalized = torch.tensor([[-1.0, 0.0, 1.0], [2.0, -2.0, 0.5]])
    minimum = torch.tensor([-0.5, -1.0, 0.0])
    maximum = torch.tensor([0.5, 1.0, 2.0])

    raw = raw_actions(normalized, minimum, maximum)

    torch.testing.assert_close(raw[0], torch.tensor([-0.5, 0.0, 2.0]))
    torch.testing.assert_close(raw[1], torch.tensor([1.0, -2.0, 1.5]))
