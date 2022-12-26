#date: 2022-12-26T16:20:29Z
#url: https://api.github.com/gists/60657dcc6f348d6a5186f1698524fe47
#owner: https://api.github.com/users/asmittiwari50-cmd

initialDisplay="Sunway Int'L Business School" \
               "Maithidevi,Ktm"
print(initialDisplay)
customerName=input("Enter the Customer Name")
customerAddress=input("Enter the customer address")
TotalUnit=int(input("Enter the total unit charge"))
if(TotalUnit <=100):
    TotalAmount=0
elif(101>=TotalUnit<=200):
    TotalAmount=(TotalUnit-100)*5
    print("bill",TotalAmount)
    print(f"CustomerName:{customerName}")
elif(201>=TotalUnit<=500):
    TotalAmount=100*5+(TotalUnit-200)*10
    print("bill",TotalAmount)
    print(f"CustomerName:{customerName}")
elif(TotalUnit>500):
    TotalAmount=100*5+(TotalUnit-400)*10
    DiscountAmount=TotalAmount*0.15
    AfterDiscountAmount=TotalAmount-DiscountAmount
    print("bill",AfterDiscountAmount)





