#date: 2022-08-01T17:12:45Z
#url: https://api.github.com/gists/dabf25692d43f83a1cd196e85d804731
#owner: https://api.github.com/users/amindadgar

from sklearn.preprocessing import StandardScaler


def find_manually(data, mean, var):
    """
    scale the data manually using mean(`mean`) and variance (`var`)
    """

    return (data - mean) / var
    
def start_program():
    """
    Start the process of the program
    """

    data = [[0, 0], [0, 0], [1, 1], [1, 1]]

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    print(f'Original data: {data}')
    print(f'Scaled Original data: {data_scaled}')
    print(f'mean: {scaler.mean_}, variance: {scaler.var_}')

    print('-'* 50)

    data_new = [[2, 2]]
    print(f'new data: {data_new}, the new data scaled using sklearn: {scaler.transform(data_new)}')

    print(f'new data: {data_new}, the new data scaled manually: {find_manually(data_new, scaler.mean_, scaler.var_)}')


if __name__ == '__main__':
    start_program()