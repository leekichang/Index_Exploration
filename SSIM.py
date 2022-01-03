from skimage.metrics import structural_similarity as compare_ssim
import imutils
import sys
import cv2 as cv
import os

def get_SSIM(imageA, imageB):
    grayA = cv.cvtColor(imageA, cv.COLOR_BGR2GRAY)
    grayB = cv.cvtColor(imageB, cv.COLOR_BGR2GRAY)
    score, diff = compare_ssim(grayA, grayB, full=True)

    diff = (diff * 255).astype('uint8')
    print(f"SSIM:{score:.5f}")

test_path = "./images/input"
zero_path = "./images/zero_feature"
rand_path = "./images/rand_feature"
vgg_path = "./images/vgg_output"
trained_vgg_path = "./images/trained_vgg_output"
vgg_trained_zero_path = "./images/trained_zero_feature"
vgg_trained_rand_path = "./images/trained_rand_feature"

test_images = [os.path.join(test_path, f) for f in os.listdir(test_path) if f.endswith('.png')]
zero_images = [os.path.join(zero_path, f) for f in os.listdir(zero_path) if f.endswith('.png')]
rand_images = [os.path.join(rand_path, f) for f in os.listdir(rand_path) if f.endswith('.png')]
vgg_images = [os.path.join(vgg_path, f) for f in os.listdir(vgg_path) if f.endswith('.png')]
trained_vgg_images = [os.path.join(trained_vgg_path, f) for f in os.listdir(trained_vgg_path) if f.endswith('.png')]
vgg_zero_images = [os.path.join(vgg_trained_zero_path, f) for f in os.listdir(vgg_trained_zero_path) if f.endswith('.png')]
vgg_rand_images = [os.path.join(vgg_trained_rand_path, f) for f in os.listdir(vgg_trained_rand_path) if f.endswith('.png')]

test_images.sort()
zero_images.sort()
rand_images.sort()
vgg_images.sort()
trained_vgg_images.sort()
vgg_zero_images.sort()
vgg_rand_images.sort()

# sys.stdout = open(f'./SSIM_result/results.txt', 'w')
for idx, image in enumerate(test_images):
    # print(image, zero_images[idx], rand_images[idx], vgg_images[idx], trained_vgg_images[idx],vgg_zero_images[idx], vgg_rand_images[idx])
    imageA = cv.imread(image)
    image_zero = cv.imread(zero_images[idx])
    image_rand = cv.imread(rand_images[idx])
    image_vgg = cv.imread(vgg_images[idx])
    image_trained_vgg = cv.imread(trained_vgg_images[idx])
    image_vgg_zero = cv.imread(vgg_zero_images[idx])
    image_vgg_rand = cv.imread(vgg_rand_images[idx])

    print(f"{idx}_image")
    print(f"SSIM with zero feature")
    get_SSIM(imageA, image_zero)
    print(f"SSIM with random feature")
    get_SSIM(imageA, image_rand)
    print(f"SSIM with vgg feature extractor(not trained)")
    get_SSIM(imageA, image_vgg)
    print(f"SSIM with vgg feature extractor(trained)")
    get_SSIM(imageA, image_trained_vgg)
    print(f"SSIM with vgg feature extractor and zero feature")
    get_SSIM(imageA, image_vgg_zero)
    print(f"SSIM with vgg feature extractor and random feature")
    get_SSIM(imageA, image_vgg_rand)

# sys.stdout.close()


















