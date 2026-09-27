"""Run the isolated fixed-size CPU detector, leaving GPU service untouched."""
import argparse,json,sys,time,statistics
sys.path.insert(0,'/home/chihmin/src/OmniParser')
import torch
from util.yolov9 import YOLOv9Detector
from cpu_adapter import CPUForward
p=argparse.ArgumentParser();p.add_argument('image');p.add_argument('--size',type=int,choices=[800,1024,1280],default=1024);p.add_argument('--repeats',type=int,default=8)
a=p.parse_args()
if a.repeats<1:p.error('--repeats must be positive')
torch.set_num_threads(16);torch.set_num_interop_threads(1)
if torch.backends.cpu.get_cpu_capability()!='AVX512':raise RuntimeError('This tuning requires an AVX512 CPU')
torch.jit.enable_onednn_fusion(True)
t=time.perf_counter()
d=YOLOv9Detector('/home/chihmin/src/OmniParser/weights/icon_detect_v3/model.pt',device='cpu')
d.model=CPUForward(d.model,a.size)
for _ in range(6):d.predict(a.image,conf=.05,imgsz=a.size)
startup=time.perf_counter()-t
ts=[]
for _ in range(a.repeats):
 t=time.perf_counter();r=d.predict(a.image,conf=.05,imgsz=a.size)[0].boxes;ts.append(time.perf_counter()-t)
print(json.dumps({'torch':torch.__version__,'size':a.size,'startup_and_warmup_seconds':startup,'median_predict_seconds':statistics.median(ts),'times':ts,'boxes':r.xyxy.tolist(),'confidence':r.conf.tolist()}))
