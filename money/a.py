import pandas as pd

df = pd.DataFrame(
    columns=["입금", "출금", "저축액", "잔고", "목표 저축액"],
    index=[f"2022.{i}" for i in range(1, 13)]
)

goal = 1000000
income = 10000
out = 100
save = 1000

print(df)