#date: 2022-02-25T16:53:41Z
#url: https://api.github.com/gists/949fe43fdeca328ad79d76ed6afa022e
#owner: https://api.github.com/users/ssime-git

    #A Identification de tous les paramètres de requête fournis dans l’URL
    query_parameters = request.args
    
    #B Récupération des valeurs des paramètres dans des paramètres
    employeeid = query_parameters.get('EmployeeId')
    lastname = query_parameters.get('LastName')
    city = query_parameters.get('City')
    
    #C Définition des paramètres et de la liste des filtres
    query = "SELECT * FROM employees WHERE"
    to_filter = []