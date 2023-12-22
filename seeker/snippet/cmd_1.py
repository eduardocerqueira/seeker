#date: 2023-12-22T16:47:21Z
#url: https://api.github.com/gists/bcb85323ff2e227ab297ba370b5aabf6
#owner: https://api.github.com/users/Krishnabhutada

import string,random
class admin:
    def __init__(self):
        print("****************************************************")
        uname=input("Enter User Name:")
        password=input("Enter Password: "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"u "**********"n "**********"a "**********"m "**********"e "**********"  "**********"= "**********"= "**********"  "**********"' "**********"a "**********"d "**********"m "**********"i "**********"n "**********"' "**********"  "**********"a "**********"n "**********"d "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"  "**********"= "**********"= "**********"  "**********"' "**********"a "**********"d "**********"m "**********"i "**********"n "**********"' "**********": "**********"
            while True:
                self.ids=[]
                self.data=("Employ Name","Employ Age","Employ Gender","Employ Contact","Employ Address")
                with open('save.txt') as f:
                    for i in f.readlines():
                        self.ids.append(i.split('|')[0])
                print("****************************************************")
                print('''\t\t1.Add record
                2.Edit record
                3.Delete record
                4.Search record
                5.Display record
                6.Project 
                7.Exit''')
                print("****************************************************")
                choice = input("Enter Choice:")
                options={'1':self.add,'2':self.modify,'3':self.delete,'4':self.search,'5':self.display,'6':self.project}
                if int(choice) <=6 and int(choice) >=1:
                    options.get(choice)()
                elif choice == "7":
                    break
                else:
                    print('Invalid Choice!')
        else:
            print('Invalid')
    def add(self):
        print('******* Add Employ Records *******')
        new_id=[]
        data=[]
        n=input('How many records to add:')
        for i in range(1,int(n)+1):
            print(f'\nEnter Employ {i} Record')
            with open('save.txt','a') as f:
                id=input('Enter Employ Id:')
                if id in self.ids or id in new_id:
                    print('id exist')
                    break
                data.append(id)
                for i in self.data:
                    data.append(input(f"Enter {i}:"))
                new_id.append(id)
                f.write(f'{"|".join(data)}\n')
                data.clear()
                self.password_generate(id)
 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"_ "**********"g "**********"e "**********"n "**********"e "**********"r "**********"a "**********"t "**********"e "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"u "**********"i "**********"d "**********") "**********": "**********"
        length=int(input('Enter the length of password: "**********"
        if length<=4:
            print('password too weak,choose greater length')
        else:    
            res=string.punctuation+string.ascii_letters+string.digits
            a=random.sample(res, length)
            password= "**********"
            print(f'Employ id: "**********":{password}')
            with open('credential.txt','a') as f:
                f.write(f'{uid}  {password}\n')
    def modify(self):
        print('******* Modify Employ Records *******')
        id=input('Enter Employ Id:')
        modify_value=[]
        data=[]
        if id in self.ids:
            with open('save.txt') as f:
                index=self.ids.index(id)
                for i,j in enumerate(f):
                        if index == i:
                            with open('modify.txt','a') as mf:
                                mf.write(j)
                                print(f'Found Employ Id:{id}')
                        else:
                            modify_value.append(j)
            while True:
                id=input('Enter new Employ Id:')
                if id in self.ids:
                    print('id exist')
                else:
                    data.append(id)
                    for i in self.data:
                        data.append(input(f"Enter {i}:"))
                    modify_value.append(f"{'|'.join(data)}\n")  
                    with open('save.txt','w') as f:
                        modify_value.sort()
                        for i in modify_value:
                            f.write(i)
                        print(f'Employ Id:{id} added successfully!')
                        break
        else:
            print('Employ Id not found!')
    def delete(self):
        print('******* Delete Employ Records *******')
        id=input('Enter Employ Id:')
        if id in self.ids:
            saves=[]
            with open('save.txt') as f:
                index=self.ids.index(id)
                for i,j in enumerate(f):
                    if index == i:
                        with open('delete.txt','a') as df:
                            df.write(j)
                        continue
                    else:
                        saves.append(j)
            with open('save.txt','w') as f:
                f.writelines(saves)
            print(f'Employ Id {id} Deleted successfully!')
        else:
            print('Employ Id not found!')
    def search(self):
        print('******* Search Employ Records *******')
        id=input('Enter Employ Id:')
        if id in self.ids:
            with open('save.txt') as f:
                index=self.ids.index(id)
                for i,j in enumerate(f):
                    if i == index:
                        print(f"Found Employ Id:{id}\n")
                        evalue=j.split('|')
                        print(f'Employ Id:{id}')
                        for z in range(len(self.data)):
                            print(f"{self.data[z]}:{evalue[z+1]}")               
        else:
            print('Employ Id not found!')
    def display(self):
        count=0
        values=[]
        print('******* Display Employ Records *******')
        with open('save.txt') as f:
            for i in f.readlines():
                values.append(i.split('|'))
        for i in values:
            count+=1
            print(f'Employ Id:{i[0]}')
            for j in range(len(self.data)):
                print(f"{self.data[j]}:{i[j+1]}")
        print(f'Total Records:{count}')
    def project(self):
        while True:
            print("""\t\t1.Add New project
                2.Completed Project
                3.Exit""")
            print("****************************************************")
            choice = input('Enter choice:')
            if choice == '1':
                self.project_add()
            elif choice == '2':
                self.project_complet()
            elif choice == '3':
                break
            else:
                print('Invalid choice!')
    def project_add(self):
        print("****************************************************")
        project_title=input('Enter Project Title:')
        start_project=input('project Start date:')
        end_project=input('Project End data:')
        group=[input('Enter Employ Id:') for i in range(int(input("Enter total group member:")))]
        print("****************************************************")
        with open('project.txt','a') as f:
            f.write(f'{project_title}|{start_project}|{end_project}|{group}\n')
    def project_complet(self):
        count=0
        with open('project_completed.txt') as f:
            for i in f.readlines():
                count+=1
                print(i)
        if count==0:
            print('No Project completed')
class employ:
    def __init__(self):
        self.eid=input('Enter Employ Id:')
        password=input('Enter Password: "**********"
        values=[]
        with open('credential.txt') as f:
            for i in f:
                a=i.split('  ')
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"e "**********"i "**********"d "**********"  "**********"= "**********"= "**********"  "**********"a "**********"[ "**********"0 "**********"] "**********"  "**********"a "**********"n "**********"d "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"  "**********"= "**********"= "**********"  "**********"a "**********"[ "**********"1 "**********"] "**********". "**********"s "**********"p "**********"l "**********"i "**********"t "**********"( "**********"' "**********"\ "**********"n "**********"' "**********") "**********"[ "**********"0 "**********"] "**********": "**********"
                    values.append(a)
                    print('found')
                    self.run()
            else:
                print('Invalid')
    def run(self):
        while True:
            print("****************************************************")
            print("""\t\t1.profile
                2.Display Project
                3.Project Complete
                4.Exit""")
            print("****************************************************")
            choice = input('Enter choice:')
            if choice == '1':
                self.profile()
            elif choice == '2':
                self.display()
            elif choice == '3':
                self.complete()
            elif choice == '4':
                break
            else:
                print('Invalid choice')
    def profile(self):
        with open('save.txt') as f:
            for i in f.readlines():
                a=i.split('|')
                if self.eid == a[0]:
                    print("****************************************************")
                    print(f"""\tEmploy Id:{a[0]}
                        Employ Name:{a[1]}
                        Employ Age:{a[2]}
                        Employ Gender:{a[3]}
                        Employ Contact:{a[4]}
                        Employ Address:{a[5]}""")
    def display(self):
        self.list1=[]
        count=0
        with open('project.txt') as f:
            for i in f.readlines():
                a=i.split('|')
                self.list1.append(a)
        for i in self.list1:
            if self.eid in i[-1]:
                print(f'Project Title:{i[0]}\nProject start date:{i[1]}\nProject end date:{i[2]}')
                count+=1
        if count == 0:
            print('No Project')
    def complete(self):
        save_val=[]
        count=0
        with open('project.txt') as f1:
            read=f1.readlines()
            if len(read) !=0:
                a=input('Enter Project Completed:')
                if a == 'y':
                    title=input('Enter Project Title:')
                for i in read:
                    val=i.split('|')
                    if title == val[0] and self.eid in val[-1]:
                        count+=1
                        with open("project_completed.txt",'a') as f2:
                            f2.write(str(val)+'\n')
                    elif title == val[0] and self.eid not in val[-1]:
                        print(f'Project {title} not found')
                    else:
                        save_val.append(i)
                with open('project.txt','w') as f3:
                    save_val.sort()
                    if len(save_val) !=0:
                        f3.write(*save_val)
                if count == 0:
                    print('Project not found')
            else:
                print('No project')
if __name__=="__main__":
    while True:
        print('************* Employ Management System *************')
        print('''\t\t1.Admin
                2.Employ
                3.Exit''')
        print("****************************************************")
        choice = input('Enter choice:')
        if choice == '1':
            admin()
        elif choice == '2':
            employ()
        elif choice == '3':
            break
        else:
            print('Invalid Choice!')