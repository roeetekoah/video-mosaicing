"""Original procedural 2D harbor scene. No downloaded images, fonts, or footage."""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def harbor_scene():
    """Draw a deliberately textured calibration scene, 1536x448 BGR uint8."""
    w, h = 1536, 448
    im = Image.new('RGB', (w*2, h*2), '#142e43')
    d = ImageDraw.Draw(im)
    def polygon(points, color): d.polygon([(x*2,y*2) for x,y in points], fill=color)
    def rect(box, color): d.rectangle(tuple(int(v*2) for v in box), fill=color)
    def line(points, color, width=1): d.line([(x*2,y*2) for x,y in points], fill=color, width=width*2)
    def ellipse(box, color): d.ellipse(tuple(int(v*2) for v in box), fill=color)
    def text(x,y,s,size=12,color='#e2d9bb'):
        d.text((x*2,y*2), s, font=ImageFont.load_default(size=size*2), fill=color)
    # Dusk sky, distant cliffs, and restrained golden lights.
    for y in range(255):
        t=y/255; rect((0,y,w,y+1),tuple(int(a*(1-t)+b*t) for a,b in zip((19,42,65),(91,127,136))))
    ellipse((1250,32,1303,85),'#e3ba79')
    rng=np.random.default_rng(47)
    for x,y in rng.integers((16,12),(1518,130),size=(80,2)):
        ellipse((x,y,x+1,y+1),'#acc3bc')
    polygon([(0,232),(0,190),(80,149),(160,186),(249,126),(330,197),(422,168),(535,223),
             (637,152),(747,186),(860,122),(963,179),(1076,136),(1217,203),(1370,138),(1536,178),(1536,270)],'#375568')
    polygon([(0,240),(120,201),(226,219),(370,187),(494,223),(611,196),(735,229),(932,196),
             (1120,222),(1300,192),(1536,227),(1536,290),(0,290)],'#294957')
    rect((0,300,w,h),'#193b4c')
    # Quay and masonry.
    rect((0,287,w,316),'#beaa88'); rect((0,316,w,333),'#8d826e')
    for row,y in enumerate(range(291,330,9)):
        line([(0,y),(w,y)],'#6e7267')
        for x in range(-20 if row%2 else 0,w,34): line([(x,y),(x,y+8)],'#6e7267')
    # Four different structures make landmarks easy to inspect across seams.
    def warehouse(x,y,width,height,color,roof,label):
        rect((x,y,x+width,y+height),color)
        polygon([(x-8,y),(x+width/2,y-40),(x+width+8,y)],roof)
        line([(x-8,y),(x+width/2,y-40),(x+width+8,y)],'#d4bf94',2)
        for xx in range(x+12,x+width-12,28):
            for yy in range(y+18,y+height-12,35):
                rect((xx,yy,xx+15,yy+22),'#193e4b')
                rect((xx+2,yy+3,xx+12,yy+17),'#e5bd7b')
                line([(xx+7,yy),(xx+7,yy+22)],'#5d6e6b')
                line([(xx,yy+11),(xx+15,yy+11)],'#5d6e6b')
        rect((x+12,y+height-24,x+width-12,y+height-5),'#254955')
        text(x+20,y+height-22,label,12)
    warehouse(150,203,215,84,'#bd795c','#354a51','COASTAL SURVEY / 01')
    warehouse(455,173,180,114,'#d3b68b','#4e6d71','MARINE LAB')
    warehouse(805,210,222,77,'#8aa29a','#315563','INSTRUMENTS / 03')
    warehouse(1110,178,195,109,'#a66852','#344d58','HARBOR WORKS')
    # Lighthouse, antenna, cranes, and connecting cables.
    polygon([(45,287),(57,161),(83,161),(97,287)],'#d3c6a4')
    polygon([(54,201),(87,201),(90,224),(51,224)],'#ae6752')
    rect((52,142,88,163),'#d8bf89'); rect((56,146,84,157),'#f2cc82')
    polygon([(45,142),(70,125),(95,142)],'#263e4d'); line([(70,125),(70,112)],'#d9c7a2')
    for x in (702,1386):
        line([(x,287),(x,114),(x+97,114)],'#d3b582',5)
        line([(x,114),(x+97,148),(x+97,114),(x+20,148),(x,114)],'#d3b582',2)
        line([(x+94,116),(x+94,245)],'#b2afa0'); ellipse((x+90,242,x+98,251),'#d3b582')
        line([(x,172),(x-38,287)],'#d3b582',2)
    for x in (390,1052):
        line([(x,287),(x,203)],'#d3c3a2',3);ellipse((x-6,196,x+6,208),'#eec783')
    # Harbor boat and reflections with deterministic nonperiodic surface detail.
    polygon([(590,355),(768,355),(738,385),(619,385)],'#ba7456')
    rect((631,328,705,355),'#c3bc9b');rect((639,333,695,345),'#244957')
    line([(675,328),(675,274)],'#c4c3a7',2);line([(675,276),(730,350)],'#a1aaa0')
    for x in range(610,750,24):ellipse((x,362,x+7,369),'#203f4c')
    for x,y,length in zip(rng.integers(0,w,380),rng.integers(338,h-5,380),rng.integers(3,28,380)):
        line([(x,y),(x+length,y)],['#315766','#446a73','#75918b'][int(rng.integers(3))])
    for x in (63,220,304,480,565,832,944,1146,1245):
        for y in range(337,434,10):line([(x-int(rng.integers(8)),y),(x+int(rng.integers(14)),y)],'#a89972')
    # Foreground rails and numbered mooring posts.
    for i,x in enumerate(range(15,w,115)):
        rect((x,278,x+10,302),'#294552');ellipse((x-3,276,x+13,283),'#294552')
        text(x+18,271,f'{i+1:02}',10,'#294552')
    text(1036,38,'NORTH QUAY',29,'#e4cda6')
    text(1038,76,'A PROCEDURAL REGISTRATION SCENE',11,'#a9c2bc')
    im=im.resize((w,h),Image.Resampling.LANCZOS)
    return cv2.cvtColor(np.array(im),cv2.COLOR_RGB2BGR)


def camera_sequence():
    """29 overlapping views; known x/y translation and in-plane rotation.

    World-to-frame transforms generate the pixels. Ground truth is returned
    for scoring only; the mosaicing pipeline never receives these transforms.
    """
    world=harbor_scene(); h,w=384,576
    frames, transforms=[],[]
    for i,x in enumerate(range(32,929,32)):
        y=32+3*np.sin(i*.24)
        theta=np.deg2rad(.45*np.sin(i*.21))
        c,s=np.cos(theta),np.sin(theta)
        center=np.array([(w-1)/2,(h-1)/2])
        r=np.array([[c,-s],[s,c]])
        matrix=np.eye(3);matrix[:2,:2]=r;matrix[:2,2]=center-r@center-r@np.array([x,y])
        frames.append(cv2.warpAffine(world,matrix[:2],(w,h),flags=cv2.INTER_LINEAR))
        transforms.append(matrix)
    return world,frames,transforms
