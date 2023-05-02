#date: 2023-05-02T16:54:51Z
#url: https://api.github.com/gists/f46fdb211ca45b742471396dbcee1b96
#owner: https://api.github.com/users/TanjimReza

#Task10
class Student:
    def __init__(self,name,ID):
        self.name = name
        self.ID = ID
    def Details(self):
        return "Name: "+self.name+"\n"+"ID: "+self.ID+"\n"
#Write your code here
class CSEStudent(Student):
    def __init__(self,name,ID,session):
        super().__init__(name,ID)
        self.session=session
        # self.course_list=[]
        # self.marks_list=[]
        self.status = {}

    def Details(self):
        return(f'{super().Details()}Current semester:{self.session}')

    def addCourseWithMarks(self,*dets):
        self.dets=dets
        
        # print(self.dets)
        
        for i in range(0,len(dets),2):
            course = dets[i] 
            marks = dets[i+1]
            gpa = 0
            
            if marks>=85: 
                gpa = 4.0
            elif 80<= marks <=84: 
                gpa = 3.3 
            elif 70<=marks<=79:
                gpa = 3.0 
            elif 65<=marks<=69: 
                gpa = 2.3
            elif 57<=marks<=64:
                gpa = 2.0
            elif 55<=marks<=56:
                gpa = 1.3
            elif 50<=marks<=54:
                gpa = 1.0 
            elif marks >50:
                gpa = 0.0

            
            
            self.status[course] = gpa
        print(self.status)
    
    def showGPA(self):
        print(f"{self.name} has taken {len(self.status)} courses")
        for course in self.status:
            print(f"{course}: {self.status[course]}")
        print(f"Semester GPA: {sum(self.status.values())/len(self.status):.2f}")



Bob = CSEStudent("Bob","20301018",'Fall 2020')
Carol = CSEStudent("Carol","16301814",'Fall 2020')
Anny = CSEStudent("Anny","18201234",'Fall 2020')
print("#########################")
print(Bob.Details())
print("#########################")
print(Carol.Details())
print("#########################")
print(Anny.Details())
print("#########################")
Bob.addCourseWithMarks("CSE111",83.5,"CSE230",73.0,"CSE260",92.5)
Carol.addCourseWithMarks("CSE470",62.5,"CSE422",69.0,"CSE460",76.5,"CSE461",87.0)
Anny.addCourseWithMarks("CSE340",45.5,"CSE321",95.0,"CSE370",91.0)
print("----------------------------")
Bob.showGPA()
print("----------------------------")
Carol.showGPA()
print("----------------------------")
Anny.showGPA()