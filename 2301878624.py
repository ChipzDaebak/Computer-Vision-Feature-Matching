import cv2
import os
import matplotlib.pyplot as plt

path = 'Dataset/Data'

img = cv2.imread('Dataset/Object.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

current_img = []
for i in os.listdir(path):
    file = i.split('.')
    if file[1] == 'jpg':
        img_path = cv2.imread(path + '/' + i)
        img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2GRAY)
        img_path = cv2.GaussianBlur(img_path, (3,3), 0)
        current_img.append(img_path)
        
SURF = cv2.xfeatures2d.SURF_create()

current_kp, current_desc = SURF.detectAndCompute(img, None)

flann_idx = 1
check_tree = 50

flann = cv2.FlannBasedMatcher(dict(algorithm=flann_idx), dict(checks=check_tree))

total_mask = []
current_idx = -1
total_matching = 0
current_keypoints = None
final_matching = None

for idx, i in enumerate(current_img):
    curr_kp, curr_desc = SURF.detectAndCompute(i, None)
    flannMatcher = flann.knnMatch(current_desc, curr_desc, 2)
    
    curr_mask = [[0,0] for j in range(0, len(flannMatcher))]
    count = 0
    
    for j, (firstMatch, secondMatch) in enumerate(flannMatcher):
        if firstMatch.distance < 0.7 * secondMatch.distance:
            curr_mask[j] = [1,0]
            count += 1
            
    total_mask.append(curr_mask)
    if total_matching < count:
        total_matching = count
        current_idx = idx
        current_keypoints = curr_kp
        final_matching = flannMatcher
        
result = cv2.drawMatchesKnn(
    img,
    current_kp,
    current_img[current_idx],
    current_keypoints,
    final_matching,
    None,
    matchColor=[0, 255, 0],
    singlePointColor=[0, 0, 255],
    matchesMask=total_mask[current_idx]
)

plt.imshow(result, cmap='gray')
plt.show()