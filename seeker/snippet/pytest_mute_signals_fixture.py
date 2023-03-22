#date: 2023-03-22T17:02:33Z
#url: https://api.github.com/gists/ffaf2c2c07b47fd2d441c172f28a3fbd
#owner: https://api.github.com/users/umarmughal824

@pytest.fixture(autouse=True)
def mute_signals(request):
    # Skip applying, if marked with `enabled_signals`
    if 'enable_signals' in request.keywords:
        return

    signals = [pre_save, post_save, pre_delete, post_delete, m2m_changed]
    restore = {}
    for signal in signals:
        restore[signal] = signal.receivers
        signal.receivers = []

    def restore_signals():
        for signal, receivers in restore.items():
            signal.sender_receivers_cache.clear()
            signal.receivers = receivers

    request.addfinalizer(restore_signals)