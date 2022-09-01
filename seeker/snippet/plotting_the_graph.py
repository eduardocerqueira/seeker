#date: 2022-09-01T17:04:42Z
#url: https://api.github.com/gists/fc9879d8bcfc957a57faee9417d91b16
#owner: https://api.github.com/users/RocketDav1d

with open("data.csv", 'w') as f:
            # create the csv writer
            writer = csv.writer(f)

            writer.writerow(["dates", "Portfolio Performance"])

            for i in range(len(portfolio_values_of_all_instruments)):
                writer.writerow([dates_dict["DE0007664039"][i], portfolio_values_of_all_instruments[i]])

        df = pd.read_csv('data.csv')
        print(df)

        fig = px.line(df, x='dates', y="Portfolio Performance")

        with placeholder_1.container():
            st.plotly_chart(figure_or_data=fig)