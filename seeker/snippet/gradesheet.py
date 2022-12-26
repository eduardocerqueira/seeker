#date: 2022-12-26T16:19:22Z
#url: https://api.github.com/gists/6625e805d8c80dc527ed602669994422
#owner: https://api.github.com/users/asmittiwari50-cmd

def initialdisplay():
   display='''Sunway Int'l Business School\n
                Maithidevi,Ktm'''
   return display
def inputinformation():
    StudentName=input("Enter the student name")
    StudentAddress=input("Enter the student address")
    StudentFaculty=input("Enter the student faculty name")
    StudentProgramme=input("Enter the student programme")
    StudentIntake= input("Enter the student intake")
    Python,Java,Flutter,Entreprenuership,Communication=input("Enter five subject 1.Python,2.Java,3.Flutter,4.Entreprenuership,5.Communication").split(",")
    a=int(Python)
    b=int(Java)
    c=int(Flutter)
    d=int(Entreprenuership)
    e=int(Communication)
    Sum=a+b+c+d+e
    print("All subject marks obtained by",StudentName,"is",Sum)
    Percentage=Sum/500*100
    print("Percentage of",StudentName,"is",Percentage,"%")
    return StudentName, StudentAddress,StudentProgramme,StudentFaculty,StudentIntake,Percentage
def calculation(Percentage):
    if(Percentage>90):
        print("The student grade is A+")
    elif(80<Percentage<90):
        print("The student grade is A")
    elif (70<Percentage < 80):
        print("The student grade is B+")
    elif(60<Percentage<70):
        print("The student grade is B-")
    elif(50<Percentage<60):
        print("The student grade is B")
    elif(40<Percentage<50):
        print("The student grade is C+")
    elif(30<Percentage<40):
         print("The student grade is D")
    else:
         print("The student grade is F")
    print("")

display=initialdisplay()
print(display)
StudentName, StudentAddress,StudentProgramme,StudentFaculty,StudentIntake,Percentage=inputinformation()
print("")
print("Student Name:",StudentName,"               ","Student Address:",StudentAddress)
print("Student Faculty:",StudentFaculty,"                ","Student Programme:",StudentProgramme)
print("Student Intake:",StudentIntake)
calculation(Percentage)


