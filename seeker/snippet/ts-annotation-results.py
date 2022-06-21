#date: 2022-06-21T17:00:53Z
#url: https://api.github.com/gists/11a80e9e341286ebaea60b3029ea646e
#owner: https://api.github.com/users/michaelhoarau

payload = {
    "id": task_id
}

response = requests.get(f'http://localhost:8080/api/tasks/{task_id}/annotations', headers=headers, data=json.dumps(payload))
annotations_df = pd.DataFrame([result['value'] for result in response.json()[0]['result']])[['start', 'end']]
annotations_df.to_csv('labels.csv', index=None, header=None)