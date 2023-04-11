#date: 2023-04-11T17:08:42Z
#url: https://api.github.com/gists/61febfab051c37666e313d7478799c17
#owner: https://api.github.com/users/alecgunny

class Validator:
    def __init__(self, waveforms, sample_rate, kernel_length, spacing=3, **sky_params):
        self.sample_rate = sample_rate
        self.spacing = spacing

        waveforms = project(waveforms, **sky_params)

        # don't even keep anything close to the full waveform
        start = waveforms.shape[-1] // 2 - int(kernel_length * sample_rate)
        stop = start + int(1.5 * kernel_length * sample_rate)
        self.waveforms = waveforms[:, :, start: stop].numpy()

    def __call__(self, model, background):
        idx, shift, Tb = 0, 0, 0
        background_events, foreground_events = [], []
        while idx < len(self.waveforms):
            start = int(shift * self.sample_rate)
            X = np.stack([background[0, :-start], background[1, start:]])

            injection_times = calc_num_injections(X.shape[-1], spacing, sample_rate)
            num_waveforms = len(injection_times)
            waveforms = self.waveforms[idx: idx + num_waveforms]
            X_inj = inject(X, waveforms, injection_times)

            X = InMemoryDataset(X, coincident=True, shuffle=False, ...)
            X_inj = InMemoryDataset(X_inj, coincident=True, shuffle=False, ...)

            y, y_inj = [], []
            for x, x_inj in zip(X, X_inj):
                y.append(model(x)[:, 0])
                y_inj.append(model(x_inj)[:, 0])
            y = torch.cat(y)
            y_inj = torch.cat(y_inj)

            y, _ = integrate_and_cluster(y)
            y_inj, times = integrate_and_cluster(y_inj)
            y_inj = recover(y_inj, times, injection_times)
 
            background_events.append(y)
            foreground_events.append(y_inj)
            Tb += X.shape[-1] / sample_rate
            shift += 1
            idx += num_waveforms
        return background_events, foreground_events, Tb