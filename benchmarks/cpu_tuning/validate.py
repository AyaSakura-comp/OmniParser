import sys,time,json,statistics,hashlib
from pathlib import Path
sys.path.insert(0,'/home/chihmin/src/OmniParser')
import torch
from PIL import Image
from torchvision.ops import box_iou
from util.yolov9 import YOLOv9Detector
from cpu_adapter import prepare
T=int(sys.argv[1]) if len(sys.argv)>1 else 16
torch.set_num_threads(T);torch.set_num_interop_threads(1)
weight='/home/chihmin/src/OmniParser/weights/icon_detect_v3/model.pt'
b=YOLOv9Detector(weight,device='cpu');c=YOLOv9Detector(weight,device='cpu');c.model=prepare(c.model)
class Meter:
 def __init__(self,m,bf=False):self.m=m;self.bf=bf;self.last=None;self.seconds=0
 def __call__(self,x):
  t=time.perf_counter()
  with torch.jit.optimized_execution(self.bf):self.last=self.m(x.bfloat16() if self.bf else x)
  self.seconds=time.perf_counter()-t;return [v.float() for v in self.last] if self.bf else self.last
b.model=Meter(b.model);c.model=Meter(c.model,True)
torch.jit.enable_onednn_fusion(True)
paths=['/tmp/pi-nodriver-shot-tgbv3gsk/screenshot.jpg']+[str(Path('/home/chihmin/src/OmniParser/imgs')/n) for n in ['google_page.png','windows_multitab.png','mobile.png']]
records=[]
for size in ([int(sys.argv[2])] if len(sys.argv)>2 else (800,1024)):
 for path in paths:
  image=Image.open(path).convert('RGB')
  if path==paths[0]:image=image.crop((0,87,1280,720))
  for d in (b,c):
   for _ in range(6):d.predict(image,conf=.05,imgsz=size)
  times={'b':[],'c':[]};raw={'b':[],'c':[]}
  for i in range(8):
   for name,d in ([('b',b),('c',c)] if i%2==0 else [('c',c),('b',b)]):
    t=time.perf_counter();result=d.predict(image,conf=.05,imgsz=size);times[name].append(time.perf_counter()-t);raw[name].append(d.model.seconds)
    if name=='b':br=result[0].boxes
    else:cr=result[0].boxes
  cos=[];scorecos=[]
  for a,z in zip(b.model.last,c.model.last):
   a=a.float().flatten();z=z.float().flatten()
   assert a.shape==z.shape and torch.isfinite(z).all()
   cos.append(float(torch.nn.functional.cosine_similarity(a.double(),z.double(),dim=0)))
  for a,z in zip(b.model.last[::2],c.model.last[::2]):scorecos.append(float(torch.nn.functional.cosine_similarity(a.float().sigmoid().flatten().double(),z.float().sigmoid().flatten().double(),dim=0)))
  ious=box_iou(br.xyxy.float(),cr.xyxy.float())
  recall=float((ious.max(1).values>=.5).float().mean()) if len(br.xyxy) and len(cr.xyxy) else None
  rec={'image':Path(path).name,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'size':size,'times':times,'raw_times':raw,'speedup':statistics.median(times['b'])/statistics.median(times['c']),'raw_speedup':statistics.median(raw['b'])/statistics.median(raw['c']),'cosines':cos,'probability_cosines':scorecos,'baseline_boxes':len(br.xyxy),'candidate_boxes':len(cr.xyxy),'baseline_box_coverage_iou50':recall}
  records.append(rec);print(json.dumps(rec),flush=True)
  Path(f'/home/chihmin/omni-cpu-tuning/validation-fusion-{T}-{sys.argv[2] if len(sys.argv)>2 else 'both'}.json').write_text(json.dumps({'threads':T,'torch':torch.__version__,'records':records},indent=2))
