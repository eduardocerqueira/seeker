#date: 2025-06-26T16:56:53Z
#url: https://api.github.com/gists/9de5de2b9e88f02e8381a88e60e0b5c8
#owner: https://api.github.com/users/LSzubelak

draw_dag(
    edges=[
        ('Motivation', 'TrainingHours'),
        ('Motivation', 'Productivity')
    ],
    title='Productivity <- Motivation -> TrainingHours'
)