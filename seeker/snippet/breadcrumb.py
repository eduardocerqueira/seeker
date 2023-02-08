#date: 2023-02-08T17:04:28Z
#url: https://api.github.com/gists/4c2f5df447eb8addd794347d37aa983e
#owner: https://api.github.com/users/1cezar

#Adicione uma lista de tuplas como parte do contexto de cada página,
#que represente cada seção do breadcrumb.
#Cada tupla deve ter um nome de seção e uma URL.

@app.route('/home')
def home():
    breadcrumb = [("Home", "/home")]
    return render_template("home.html", breadcrumb=breadcrumb)

@app.route('/products')
def products():
    breadcrumb = [("Home", "/home"), ("Products", "/products")]
    return render_template("products.html", breadcrumb=breadcrumb)


#No arquivo HTML, você pode iterar sobre a lista de tuplas 
#e exibir o nome de cada seção como um link,
#exceto a última seção, que será exibida sem link.

<ol class="breadcrumb">
  {% for section, url in breadcrumb %}
    {% if loop.last %}
      <li class="breadcrumb-item active">{{ section }}</li>
    {% else %}
      <li class="breadcrumb-item"><a href="{{ url }}">{{ section }}</a></li>
    {% endif %}
  {% endfor %}
</ol>
