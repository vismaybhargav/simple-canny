import cv2
import numpy as np

# Done
def gaussian_blur(img, kernel_size=5, sigma=1.4):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

# Done
def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    Ix = cv2.filter2D(img, -1, Kx)
    Iy = cv2.filter2D(img, -1, Ky)

    magnitude = np.hypot(Ix, Iy)
    magnitude = magnitude / magnitude.max() * 255
    angle = np.arctan2(Iy, Ix)
    return magnitude, angle

def non_max_suppression(mag, angle):
    M, N = mag.shape
    output = np.zeros((M, N), dtype=np.uint8)
    angle = angle * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            q = 255
            r = 255

            # Angle 0
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = mag[i, j+1]
                r = mag[i, j-1]
            # Angle 45
            elif (22.5 <= angle[i,j] < 67.5):
                q = mag[i+1, j-1]
                r = mag[i-1, j+1]
            # Angle 90
            elif (67.5 <= angle[i,j] < 112.5):
                q = mag[i+1, j]
                r = mag[i-1, j]
            # Angle 135
            elif (112.5 <= angle[i,j] < 157.5):
                q = mag[i-1, j-1]
                r = mag[i+1, j+1]

            if (mag[i,j] >= q) and (mag[i,j] >= r):
                output[i,j] = mag[i,j]
            else:
                output[i,j] = 0
    return output

def threshold(img, low_ratio=0.05, high_ratio=0.15):
    high_threshold = img.max() * high_ratio
    low_threshold = high_threshold * low_ratio

    M, N = img.shape
    res = np.zeros((M,N), dtype=np.uint8)

    strong = 255
    weak = 75

    strong_i, strong_j = np.where(img >= high_threshold)
    zeros_i, zeros_j = np.where(img < low_threshold)

    weak_i, weak_j = np.where((img <= high_threshold) & (img >= low_threshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    return res, weak, strong

def hysteresis(img, weak=100, strong=200):
    M, N = img.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if img[i,j] == weak:
                if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                    or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                    or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                    img[i,j] = strong
                else:
                    img[i,j] = 0
    return img

def canny_edge_detector(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = gaussian_blur(gray)
    mag, angle = sobel_filters(blur)
    nms = non_max_suppression(mag, angle)
    thresh, weak, strong = threshold(nms)
    final = hysteresis(thresh, weak, strong)
    return strong

# Example usage
if __name__ == "__main__":
    image = cv2.imread("img/monkey.jpg")
    edges = canny_edge_detector(image)
    cv2.imshow("Canny Edge", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
