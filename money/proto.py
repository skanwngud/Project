a = input("입금이면 1, 출금이면 2 : ")

if int(a) == 1:
    print("입금을 선택하셨습니다.")
    b = input("금액을 입력하세요. : ")
    if int(b) > 0:
        print(f"입력하신 금액은 {b}원입니다.")
elif int(a) == 2:
    print("출금을 선택하셨습니다.")
    b = int(input("금액을 입력하세요. : "))
    if b > 0:
        print(f"입력하신 금액은 {b}원입니다.")