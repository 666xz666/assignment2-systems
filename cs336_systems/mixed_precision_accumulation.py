import torch

# 第一段：float32 累加1000次0.01
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float32)
print(s)

# 第二段：float16 直接累加1000次0.01
s = torch.tensor(0, dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(s)

# 第三段：float16转float32后再累加
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01, dtype=torch.float16)
    s += x.type(torch.float32)
print(s)

"""
python cs336_systems/mixed_precision_accumulation.py
tensor(10.0001)
tensor(9.9531, dtype=torch.float16)
tensor(10.0021)
"""
