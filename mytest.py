import torch

tensor_a = torch.randn((1, 4, 4))
tensor_b = torch.randn((1, 4, 4))
tensor_c = torch.randn((1, 4, 4))

result = torch.cat([tensor_a, tensor_b, tensor_c], dim=0)
result, _ = torch.max(result, dim=0, keepdim=True)

print(tensor_a)
print(tensor_b)
print(tensor_c)
print("========")
print(result)
