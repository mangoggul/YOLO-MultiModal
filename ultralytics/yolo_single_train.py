from ultralytics import YOLO
import torch
import os
import numpy as np
import random, requests
import time
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"


#api_url = "https://notify-api.line.me/api/notify"
#token = "LYy0yPmrqjMc3rmvdQR2WcbCCVZkmFlf6FZBZGEkpYQ"
#headers = {'Authorization':'Bearer '+token}


def seed_everything(seed):
    torch.manual_seed(seed) #torch를 거치는 모든 난수들 의 생성순서를 고정한다
    torch.cuda.manual_seed(seed) #cuda를 사용하는 메소드들의 난수시드는 따로 고정해줘야한다 
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True #딥러닝에 특화된 CuDNN의 난수시드도 고정 
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed) #numpy를 사용할 경우 고정
    random.seed(seed) #파이썬 자체 모듈 random 모듈의 시드 고정

seed_everything(42)

def convert_seconds(seconds):
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    
    return f"{hours}시간 {minutes}분 {seconds}초"

if __name__ == '__main__':

  def start_model(model_name,model_num,ex_name):
    model = YOLO(f"yolov{model_num}{model_name}.yaml")
    s = time.time()
    
    model_save_dir = f"EX/{ex_name}"
    model_save_name = f"yolov{model_num}{model_name}"
    
    model.train(data=f"yaml/{ex_name}.yaml", epochs=1000,patience=20, imgsz=640,device=0, project=model_save_dir+"/train",name=model_save_name, augment=False)  
    val_metrics = model.val(data=f"yaml/{ex_name}.yaml", project=model_save_dir+"/val", name=model_save_name+"_val") 
    test_metrics = model.val(data=f"yaml/{ex_name}_test.yaml" ,project=model_save_dir+"/test", name=model_save_name+"_test") 
    path = model.export(format="onnx") 
    e = time.time()
    finish = e-s
   
  model_list = ['n']
  model_9_list = ['c','e']
  model_num = ['8'] 
  ex_name = ["new_data"]
  
  for ex in ex_name:
    for num in model_num:
      if num == '9':
        model_list = model_9_list
      for model_name in model_list:
        start_model(model_name,num,ex)

