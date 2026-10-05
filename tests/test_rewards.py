from wm.rewards import dec_reward

DEC = dict(a=0.2, b=1.0, p=0.5, p_irr=2.0, q=0.5, kappa=0.05)


def test_decision_reward_ordering():
    def r(correct, latency):
        return dec_reward(correct, latency, False, False, 1000.0, **DEC)

    assert r(True, 500) > r(True, 4000) > r(False, 500) > r(False, 4000)
    assert dec_reward(False, 500, False, True, 1000.0, **DEC) < r(False, 500)
    assert dec_reward(None, 500, False, False, 1000.0, **DEC) == 0.0
