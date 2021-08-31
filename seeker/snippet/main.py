#date: 2021-08-31T13:15:39Z
#url: https://api.github.com/gists/875ffc94f0b87295b65d21411327bc70
#owner: https://api.github.com/users/mypy-play

class Class1:
    class Class2:
        a: str

        class Class3:
            a: str

        Class1.Class2.Class3

    prop: Class1.Class2
    
