#!/usr/bin/env python
# coding: utf-8

# In[29]:


from moviepy.editor import *
loc = "./"
files = [str(loc)+'0.png',str(loc)+'1.png', str(loc)+'2.png', str(loc)+'3.png', str(loc)+'4.png',str(loc)+'5.png', str(loc)+'6.png', str(loc)+'7.png', str(loc)+'8.png',str(loc)+'9.png', str(loc)+'10.png', str(loc)+'11.png', str(loc)+'12.png',str(loc)+'13.png', str(loc)+'14.png', str(loc)+'15.png', str(loc)+'16.png',str(loc)+'17.png', str(loc)+'18.png', str(loc)+'19.png', str(loc)+'20.png',str(loc)+'21.png', str(loc)+'22.png', str(loc)+'23.png', str(loc)+'24.png',str(loc)+'25.png', str(loc)+'26.png', str(loc)+'27.png', str(loc)+'28.png',str(loc)+'29.png', str(loc)+'30.png', str(loc)+'31.png', str(loc)+'32.png',str(loc)+'33.png', str(loc)+'34.png', str(loc)+'35.png', str(loc)+'36.png',str(loc)+'37.png',str(loc)+'38.png',str(loc)+'39.png']
clip = ImageSequenceClip(files, fps = 4) 
clip.write_videofile("Contact_Pred_Unseen.mp4", fps = 30)

