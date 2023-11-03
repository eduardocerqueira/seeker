#date: 2023-11-03T16:45:48Z
#url: https://api.github.com/gists/2614cb7b33920aa0765681c767977437
#owner: https://api.github.com/users/lostquix

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.


from dash import Dash, html, dcc, Input, Output, callback
import plotly.express as px
import pandas as pd

app = Dash(__name__)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options



df = pd.read_excel("Vendas.xlsx")


fig = px.bar(df, x="Produto", y="Quantidade", color="ID Loja", barmode="group")
opçoes = list(df['ID Loja'].unique())
opçoes.append("Todas as lojas")


app.layout = html.Div(children=[
    html.H1(children='Grafico de lojas'),
    html.H2(children='Grafico criado na aula do lira sobre dashboards'),
    html.Div(children='''
        obs: Esse grafico ira mostrar a quantidade de vendas de produtos e não o faturamento
    '''),

    dcc.Dropdown(opçoes, 'Todas as lojas', id='Lista_Lojas'),
    dcc.Graph(
        id='grafico_de_vendas',
        figure=fig
    )

])

@callback(
    Output('grafico_de_vendas', 'figure'),
    Input('Lista_Lojas', 'value')
)

def update_output(value):
    if value == 'Todas as lojas':
        fig = px.bar(df, x="Produto", y="Quantidade", color="ID Loja", barmode="group")

    return fig



if __name__ == '__main__':
    app.run(debug=True)