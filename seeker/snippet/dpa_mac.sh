#date: 2022-05-02T17:14:16Z
#url: https://api.github.com/gists/4df95e6ffd9049b2f4b6fd9624def50f
#owner: https://api.github.com/users/ichux

dpa(){
  docker ps -a --format "\nID\t{{.ID}}\nIMAGE\t{{.Image}}\nCOMMAND\t{{.Command}}\nCREATED\t{{.RunningFor}}\nSTATUS\t\
> {{.Status}}\nPORTS\t{{.Ports}}\nNAMES\t{{.Names}}\n"
}