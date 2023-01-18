#date: 2023-01-18T16:39:24Z
#url: https://api.github.com/gists/01c814ca205064506846e45f547835fe
#owner: https://api.github.com/users/dsakovych

# get all pods from namespace
kubectl get pods -n spark-ds

# container port forwarding
kubectl port-forward service/airflow-ds 8095:8080 -n spark-ds

# stream logs from container
kubectl -n spark-ds logs -f om-extraction-task.93d8e47352dc4777b600bfa7b41f2f8a

# connect to container
kubectl -n spark-ds exec -it airflow-ds-65bf6f85d6-f52c5 bash