#date: 2023-12-22T16:47:21Z
#url: https://api.github.com/gists/bcb85323ff2e227ab297ba370b5aabf6
#owner: https://api.github.com/users/Krishnabhutada

import string,random,threading,datetime
MAIN="main.txt"
CREDENTIAL="credential.txt"
MODIFY="modify.txt"
DELETE="delete.txt"
PROJECT="project.txt"
PROJECT_COMPLETED="project_completed.txt"
PROJECT_INCOMPLETE="project_incomplete.txt"
class admin:
    def __init__(self):
        print("****************************************************")
        uname=input("Enter User Name:")
        password=input("Enter Password: "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"u "**********"n "**********"a "**********"m "**********"e "**********"  "**********"= "**********"= "**********"  "**********"' "**********"a "**********"d "**********"m "**********"i "**********"n "**********"' "**********"  "**********"a "**********"n "**********"d "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"  "**********"= "**********"= "**********"  "**********"' "**********"a "**********"d "**********"m "**********"i "**********"n "**********"' "**********": "**********"
            while True:
                self.ids=[]
                self.data=("Employ Name","Employ Age","Employ Gender","Employ Contact","Employ Address")
                with open(MAIN) as f:
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
                    options[choice]()
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
            with open(MAIN,'a') as f:
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
                with open(CREDENTIAL,'a') as f:
                    f.write(self.password_generate(id))
        sort(MAIN)
        sort(CREDENTIAL)
 "**********"def password_generate(self,uid): "**********"
        res=string.punctuation+string.ascii_letters+string.digits
        a=random.sample(res, 5)
        password= "**********"
        print(f'Employ id: "**********":{password}')
        return f'{uid}  {password}\n'
    def modify(self):
        print('******* Modify Employ Records *******')
        id=input('Enter Employ Id:')
        modify_value=[]
        data=[]
        list1=[]
        if id in self.ids:
            with open(MAIN) as f:
                for i,j in enumerate(f):
                        if self.ids.index(id) == i:
                            with open(MODIFY,'a') as mf:
                                mf.write(j)
                                print(f'Found Employ Id:{id}')
                        else:
                            modify_value.append(j)                        
            while True:
                data.append(id)
                for i in self.data:
                    data.append(input(f"Enter {i}:"))
                modify_value.append(f"{'|'.join(data)}\n")  
                with open(MAIN,'w') as f:
                    modify_value.sort()
                    for i in modify_value:
                        f.write(i)
                    print(f'Employ Id:{id} added successfully!')
                    break
            with open(CREDENTIAL) as pf:
                for i,j in enumerate(pf):
                    if self.ids.index(id) == i:
                        list1.append(self.password_generate(id))
                    else:
                        list1.append(j)
            with open(CREDENTIAL,"w") as pwf:
                for i in list1:
                    pwf.write(i)
        else:
            print('Employ Id not found!')
    def delete(self):
        print('******* Delete Employ Records *******')
        id=input('Enter Employ Id:')
        if id in self.ids:
            saves=[]
            with open(MAIN) as f:
                index=self.ids.index(id)
                for i,j in enumerate(f):
                    if index == i:
                        with open(DELETE,'a') as df:
                            df.write(j)
                        continue
                    else:
                        saves.append(j)
            with open(MAIN,"w") as f:
                f.writelines(saves)
            print(f'Employ Id {id} Deleted successfully!')
        else:
            print('Employ Id not found!')
    def search(self):
        print('******* Search Employ Records *******')
        id=input('Enter Employ Id:')
        if id in self.ids:
            with open(MAIN) as f:
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
        with open(MAIN) as f:
            for i in f.readlines():
                values.append(i.split('|'))
        for i in values:
            count+=1
            print(f'Employ Id:{i[0]}')
            for j in range(len(self.data)):
                print(f"{self.data[j]}:{i[j+1]}")
            print("\n")
        print(f'Total Records:{count}')
    def project(self):
        while True:
            print("****************************************************")
            print("""\t\t1.Add New project
                2.Completed Project
                3.Display Project
                4.Incompleted Project
                5.Exit""")
            print("****************************************************")
            choice = input('Enter choice:')
            if choice == '1':
                self.project_add()
            elif choice == '2':
                self.project_complet()
            elif choice == '3':
                self.display_project()
            elif choice == '4':
                self.project_incomplet()
            elif choice == "5":
                break
            else:
                print('Invalid choice!')
    def project_incomplet(self):
        print('******* Display Project Incomplete *******')
        with open(PROJECT_INCOMPLETE) as f:
            print(*f.readlines())
    def display_project(self):
        count=0
        print('******* Display Project Records *******')
        with open(PROJECT) as f:
            for i in f.readlines():
                a=i.split('|')
                print(f"Project Title:{a[0]}\nproject Start date:{a[1]}\nProject End data:{a[2]}\nTeam member:{a[3]}")
                count+=1
        print(f"Total Projects:{count}")
    def project_add(self):
        print("****************************************************")
        project_title=input('Enter Project Title:')
        start_project=input('project Start date:')
        end_project=input('Project End data:')
        group=[]
        n=int(input('Enter total Team Member:'))
        for _ in range(n):
            id=input('Enter Employ Id:')
            if id in self.ids:
                group.append(id)
            else:
                print(f"Employ id {id} do not exist\nData not saved in File!")
                break
        if len(group) == n:
            print("****************************************************")
            with open(PROJECT,'a') as f:
                f.write(f'{project_title}|{start_project}|{end_project}|{group}\n')
            print('Project Added Successfully!')
    def project_complet(self):
        count=0
        with open(PROJECT_COMPLETED) as f:
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
        with open(CREDENTIAL) as f:
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
                4.Project Incomplete
                5.Exit""")
            print("****************************************************")
            choice = input('Enter choice:')
            if choice == '1':
                self.profile()
            elif choice == '2':
                self.display()
            elif choice == '3':
                self.complete()
            elif choice == '4':
                self.project_incomplet()
            elif choice == "5":
                break
            else:
                print('Invalid choice')
    def profile(self):
        with open(MAIN) as f:
            for i in f.readlines():
                a=i.split('|')
                if self.eid == a[0]:
                    print("****************************************************")
                    print(f"""\t\tEmploy Id:{a[0]}
                        Employ Name:{a[1]}
                        Employ Age:{a[2]}
                        Employ Gender:{a[3]}
                        Employ Contact:{a[4]}
                        Employ Address:{a[5]}""")
    def project_incomplet(self):
        print('******* Display Project Incomplete *******')
        with open(PROJECT_INCOMPLETE) as f:
            print(*f.readlines())
    def display(self):
        self.list1=[]
        count=0
        with open(PROJECT) as f:
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
        with open(PROJECT) as f1:
            read=f1.readlines()
            if len(read) !=0:
                title=input('Enter Project Title:')
                for i in read:
                    val=i.split('|')
                    if title == val[0] and self.eid in val[-1]:
                        count+=1
                        with open(PROJECT_COMPLETED,'a') as f2:
                            f2.write(str(val)+'\n')
                    elif title == val[0] and self.eid not in val[-1]:
                        print(f'Project {title} not found')
                    else:
                        save_val.append(i)
                with open(PROJECT,'w') as f3:
                    save_val.sort()
                    if len(save_val) !=0:
                        f3.write(*save_val)
                if count == 0:
                    print('Project not found')
            else:
                print('No project')
def sort(file1):
    list1=[]
    with open(file1) as wf:
        for i in wf.readlines():
            list1.append(i)
    with open(file1,"w") as rf:
        for i in sorted(list1):
            rf.write(i)
def check():
    values=[]
    with open(PROJECT) as f1:
        cont=f1.readlines()
        if len(cont) != 0:
            for i in cont:
                if i.split("|")[2] < str(datetime.date.today()):
                    with open(PROJECT_INCOMPLETE,"a") as cf:
                        cf.write(i)
                else:
                    values.append(i)
            with open(PROJECT,"w") as f2:
                for i in sorted(values):
                    f2.write(i)
def main():
    while True:
        t1=threading.Thread(target=check)
        t1.start()
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
if __name__=="__main__":
    main()