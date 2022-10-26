class Account:
    def __init__(self):
        self.__accID = ""
        self.__balance = 0
        self.__accNum = 0
        self.__cusName = ""

    def ShowMenu(self):
        print("""
        -----Menu----
        1. Make Account
        2. Deposit
        3. Withdraw
        4. Show All Account Info
        5. Exit
        """)
        pass

    def MakeAccount(self):
        print("[Make Account]")
        self.__accID = int(input("Account ID: "))
        self.__cusName = str(input("Name: "))
        self.__balance = int(input("Amount Money: "))

        self.__accNum += 1

    def DepositMoney(self):
        print("[Deposit]")
        id = int(input("Account ID: "))
        pass

    def WithdrawMoney(self):
        pass

    def ShowAllAccount(self):
        pass

if __name__ == "__main__":
    acc = Account()
    acc.ShowMenu()
    acc.MakeAccount()
