#date: 2021-09-30T17:13:02Z
#url: https://api.github.com/gists/99e98f7ea4859c4ac18bc674d2c60a43
#owner: https://api.github.com/users/fagoner


alumnos = [
    {
        'nombre': 'Alfa',
        'carnet': '2020202',
        'mensualidad': 100.00
    },
    {
        'nombre': 'Beta',
        'carnet': '2020203',
        'mensualidad': 1000.00
    },
    {
        'nombre': 'Gama',
        'carnet': '2020202',
        'mensualidad':999.90
    }
]

suma = 0.0

for alumno in alumnos:
    print("Alumno: " + alumno['nombre'] + " mensualidad: " + str(alumno.get('mensualidad')))
    suma += alumno.get('mensualidad')
print("ingresos generales: ", suma)