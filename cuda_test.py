import torch
print(torch.cuda.is_available())
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
