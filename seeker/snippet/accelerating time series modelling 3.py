#date: 2023-06-30T16:52:46Z
#url: https://api.github.com/gists/abde192b830f2e680d12056e1c032d95
#owner: https://api.github.com/users/davidemastricci

fh = 2 #forecast horizon indicates how many points in the future we want to forecast
experiment = TSForecastingExperiment()
experiment.setup(data=air_pass, fh=fh, target='Passengers', session_id=42)