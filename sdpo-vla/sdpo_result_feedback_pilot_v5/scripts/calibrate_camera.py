from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from sdpo_result_feedback.env_adapter import PandaPushOutcomePilot
out=Path(__file__).resolve().parents[1]/'artifacts/calibration'
e=PandaPushOutcomePilot(); e.reset('smoke_0001')
canvas=Image.new('RGB',(448*4,478*2),'white'); draw=ImageDraw.Draw(canvas)
for r,pitch in enumerate([-45,-65]):
 for c,yaw in enumerate([0,90,180,270]):
  e.base.render_yaw=yaw;e.base.render_pitch=pitch
  im=np.asarray(e.env.render())
  canvas.paste(Image.fromarray(im),(c*448,r*478+30))
  draw.text((c*448+8,r*478+8),f'yaw={yaw}, pitch={pitch}',fill='black')
canvas.save(out/'camera_sweep.jpg');e.close()
