from PIL import Image
import numpy as np
from skimage.transform import resize
import scipy.misc
import h5py

def load_image(infilename):
  img = Image.open(infilename)
  img.load()
  data = np.asarray(img, dtype="int32")
  return data

def save_image(npdata, outfilename) :
  img = Image.fromarray(np.asarray(np.clip(npdata,0,255), dtype="uint8"), "RGB")
  img.save(outfilename)

bbox = np.loadtxt('list_bbox_celeba.txt', usecols = (1, 2, 3, 4))
bbox = bbox.astype(int)

def cent(x, cropH, cropW=None, resizeW=64):
    # crop the images to [cropH,cropW,3] then resize to [resizeH,resizeW,3]
    if cropW is None:
        cropW = cropH # the width and height after cropped
    h, w = x.shape[:2]
    j = int((h - cropH)/2.)
    i = int((w - cropW)/2.)
    return scipy.misc.imresize(x[j:j+cropH, i:i+cropW],
                               [resizeW, resizeW])
p = np.empty(shape=(12288, 50000))
p = p.astype(np.uint8)
#202599
for i in range(50000, 100000):
  # img = load_image('../../../Downloads/img_celeba/img_celeba.7z/img_celeba/'+(6 - len(str(i+1))) * '0' + str(i+1) + '.jpg')
  img = load_image('../../../Downloads/img_align_celeba/'+(6 - len(str(i+1))) * '0' + str(i+1) + '.jpg')
  # img = img[bbox[i, 1] : bbox[i, 1] + bbox[i, 3], bbox[i, 0] : bbox[i, 0] + bbox[i, 2]]
  # img = np.asarray(img, dtype="float64")
  # img = resize(img, (78, 64), anti_aliasing=True)
  # img = img[0:64, :]
  img = cent(img, 145)
  img = np.reshape(img, (12288), 'F')
  p[:, i - 50000] = img
  #save_image(img, 'celebA/' + str(i + 1) + '.jpg')
  if (i % 10000 == 0):
    print(str(i) + ' images done.')

p = p.astype(np.uint8)

with h5py.File('celebA_2.h5', 'w') as hf:
    hf.create_dataset("celebA_2",  data=p)

# img = load_image('202563.jpg')
# img = img[67:337, 42:237, :]
# print(img.shape)
# save_image(img, 'cropped.jpg')
# img = np.asarray(img, dtype="float64")
# print(img.shape)
# img = resize(img, (88, 64), anti_aliasing=True)
# save_image(img, 'croppedres.jpg')

# hf = h5py.File('celebA.h5', 'r')
# n1 = np.empty((12288, 202599), dtype=np.uint8)
# hf['celebA'].read_direct(n1)
# hf.close()

# print(n1.shape)
