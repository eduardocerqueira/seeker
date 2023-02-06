#date: 2023-02-06T16:46:52Z
#url: https://api.github.com/gists/a9e1addb563fe55390e43d06f93e4636
#owner: https://api.github.com/users/eder-projetos-dev

from bs4 import BeautifulSoup

html = 'lt;p&gt;&lt;span style="font-weight: 400;"&gt;Você não teve o e-mail pessoal criado por falta de informações, para a criação você deve abrir um novo chamado com os seguintes dados:&lt;/span&gt;&lt;/p&gt;&lt;p&gt;&lt;strong&gt;CPF -&lt;/strong&gt;&lt;/p&gt;&lt;p&gt;&lt;strong&gt;NOME COMPLETO - &lt;/strong&gt;&lt;/p&gt;&lt;p&gt;&lt;strong&gt;MATRÍCULA -&lt;/strong&gt;&lt;/p&gt;&lt;p&gt;&lt;strong&gt;CARGO -&lt;/strong&gt;&lt;/p&gt;&lt;p&gt;&lt;strong&gt;TELEFONE -&lt;/strong&gt;&lt;/p&gt;&lt;p&gt;&lt;strong&gt;SETOR -&lt;/strong&gt;&lt;/p&gt;'

soup = BeautifulSoup(html, 'html5lib')

print(soup.get_text())
