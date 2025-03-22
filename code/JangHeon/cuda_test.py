import torch
print(torch.__version__)                # 2.2.x 이상이면 OK
print(torch.cuda.is_available())        # True
print(torch.cuda.get_device_name(0))    # GPU 이름

