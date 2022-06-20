#date: 2022-06-20T17:13:42Z
#url: https://api.github.com/gists/925675d74d0d8975e0bb72c87de84a0b
#owner: https://api.github.com/users/kunal1209

from tkinter import*
import tkinter as tk




def submitForm():
    strFile_1= optVariable_1.get()
    strFile_2= optVariable_2.get()
    strFile_3= optVariable_3.get()
    strFile_4= optVariable_4.get()
    strFile_5= optVariable_5.get()
    strFile_6= optVariable_6.get("1.0",'end-1c')
    lst=[strFile_1,strFile_2,strFile_3,strFile_4,strFile_5,strFile_6]
    print(lst)



root = Tk()
root.geometry("1005x650+0+0")
root.title("Demo Form ")
bg = PhotoImage(file="background.png")
photo_1 = PhotoImage(file = r"img3.png")
root.wm_attributes('-transparentcolor', 'grey')



#background image
label_1 = Label(root, image=bg)
label_1.place(x=-50, y=-130)


def open():
    top = Toplevel(root)

    # big lable

    label_2 = Label(root, text="SALARY PREDICTION", font=("Enriqueta", 36))
    label_2.place(x=400, y=25)
    label_7 = Label(root, text="Welcome user", bg='#A257EC', font=("Enriqueta", 20))
    label_7.place(x=400, y=65)

    # option 1

    label_3 = Label(root, text="NAAC GRADE ", width=20, font=("Enriqueta", 18))
    label_3.place(x=380, y=110)

    optVariable_1 = StringVar(root)
    optVariable_1.set("Choose Your Input")
    optFiles_1 = OptionMenu(root, optVariable_1, "A++", "A+", "A", "B++", "B+", "B", "C", "D", "No Grade")
    optFiles_1.pack()
    optFiles_1.place(x=430, y=145)

    # option 2

    label_4 = Label(root, text="UG COURSE ", width=20, font=("Enriqueta", 18))
    label_4.place(x=680, y=110)

    optVariable_2 = StringVar(root)
    optVariable_2.set("Choose Your Input")
    optFiles_2 = OptionMenu(root, optVariable_2, "BE in CS,Electronics,IT,Telecom", "BE in data science",
                            "BE in others", "BCOM", "BA", "BSc", "PG in HR", "PG in Marketing", "PG in finance",
                            "PG in Data science", "CA")
    optFiles_2.pack()
    optFiles_2.place(x=740, y=145)

    label_1 = Label(root, image=bg)
    label_1.place(x=-50, y=-130)
    # OPTION 4

    label_5 = Label(root, text="HIGHER STUDIES COURSE ", font=("Enriqueta", 15))
    label_5.place(x=435, y=190)

    optVariable_4 = StringVar(root)
    optVariable_4.set("Choose Your Input")
    optFiles_4 = OptionMenu(root, optVariable_4, "Engineering", "IT", "Management", "Finance", "Economics", "Others")
    optFiles_4.pack()
    optFiles_4.place(x=430, y=235)

    # OPTION 5

    label_6 = Label(root, text="COUNTRY ", font=("Enriqueta", 15))
    label_6.place(x=435, y=280)

    optVariable_5 = StringVar(root)
    optVariable_5.set("Choose Your Input")
    optFiles_5 = OptionMenu(root, optVariable_5, "US", "Australia", "Germany", "UK", "India", "Canada",
                            "Other Asian countries", "Other European countries")
    optFiles_5.pack()
    optFiles_5.place(x=430, y=320)

    # OPTION 6

    label_7 = Label(root, text="University Ranking ", font=("Enriqueta", 15))
    label_7.place(x=745, y=280)

    optVariable_6 = tk.Text(root, bg="#BBBEC3", height=2, width=20)
    optVariable_6.pack()
    optVariable_6.place(x=740, y=320)

    # submit button

    photo = PhotoImage(file=r"Screenshot (44).png")
    Button(root, text='Submit', command=submitForm, width=0, image=photo).place(x=580, y=485)


    top.mainloop()


btn = Button(root, text="open", command=open).place(x=580,y=585)




#big lable

label_2 = Label(root, text="SALARY PREDICTION",font=("Enriqueta", 36))
label_2.place(x=400, y=25)
label_7= Label(root, text="Welcome user", bg= '#A257EC',font=("Enriqueta", 20))
label_7.place(x=100, y=65)

#option 1

label_3 = Label(root, text="NAAC GRADE ",width=20,font=("Enriqueta", 18))
label_3.place(x=380,y=110)


optVariable_1 = StringVar(root)
optVariable_1.set("Choose Your Input")
optFiles_1 = OptionMenu(root,optVariable_1,"A++","A+","A","B++","B+","B","C","D","No Grade")
optFiles_1.pack()
optFiles_1.place(x=430,y=145)



#option 2





label_4 = Label(root, text="UG COURSE ",width=20,font=("Enriqueta", 18))
label_4.place(x=680,y=110)


optVariable_2 = StringVar(root)
optVariable_2.set("Choose Your Input")
optFiles_2 = OptionMenu(root,optVariable_2,"BE in CS,Electronics,IT,Telecom","BE in data science","BE in others","BCOM","BA","BSc","PG in HR","PG in Marketing","PG in finance","PG in Data science","CA")
optFiles_2.pack()
optFiles_2.place(x=740,y=145)

#OPTION 3



label_5 = Label(root, text="UPSKILL ",font=("Enriqueta", 18))
label_5.place(x=745,y=190)


optVariable_3 = StringVar(root)
optVariable_3.set("Choose Your Input")
optFiles_3 = OptionMenu(root,optVariable_3,"IT","Data Science","Finance","Management","Art")
optFiles_3.pack()
optFiles_3.place(x=740,y=235)






#submit button

photo = PhotoImage(file = r"Screenshot (44).png")
Button(root, text='Submit', command=submitForm,width=0,image=photo).place(x=580,y=485)


root.mainloop()