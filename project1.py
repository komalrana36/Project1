


#! pip install deepface


# In[3]:


from deepface import DeepFace
import matplotlib.pyplot as plt
import cv2 as CV
import os


# In[6]:


img4_path='E:\photos\img4.jpg'


# In[7]:


img4=CV.imread(img4_path)


# In[8]:


plt.imshow(img4[:,:,::-1])
plt.show()


# In[12]:


img14_path='E:\img\img4.jpg'
img14=CV.imread(img14_path)


# In[13]:


plt.imshow(img14[:,:,::-1])
plt.show()


# In[14]:


obj=DeepFace.analyze(img_path="E:\img\img4.jpg", actions=['age','gender','race','emotion'])
print(obj["age"],"years old", obj["gender"],obj["dominant_race"],obj["dominant_emotion"])


# In[ ]:




