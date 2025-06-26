#date: 2025-06-26T16:57:57Z
#url: https://api.github.com/gists/40e8625a2ddb6a97a8187926abe12bfe
#owner: https://api.github.com/users/LSzubelak


motivation = np.random.normal(0, 1, n)
training_hours = 0.9 * motivation + np.random.normal(0, 2, n)
productivity = 5 * training_hours + 3 * motivation + np.random.normal(0, 1, n)

data1 = pd.DataFrame({
    'motivation': motivation,
    'training_hours': training_hours,
    'productivity': productivity
})