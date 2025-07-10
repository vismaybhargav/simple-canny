import sys
import cv2
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        prog="Vismay's Canny Implementation",
        description="A simple canny edge detector"
    )

    args = parser.parse_args()

    image = cv2.imread("img/monkey.jpg", cv2.IMREAD_GRAYSCALE)
    color_image = cv2.imread("img/monkey.jpg")
    cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    H, W = image.shape
    LOW_THRESH, HIGH_THRESH = 100, 200 # TODO: This needs to be tuned
    np.set_printoptions(threshold=sys.maxsize)

    sobel_gx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    sobel_gy = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    guass_image = cv2.filter2D(src=image, ddepth=-1, kernel=generate_guassian_kernel(size=5, sigma=2))

    gx = cv2.filter2D(src=guass_image, ddepth=-1, kernel=sobel_gx)
    gy = cv2.filter2D(src=guass_image, ddepth=-1, kernel=sobel_gy)

    gradient = np.hypot(gx, gy)

    direction = np.degrees(np.atan2(gy, gx))
    direction = (direction + 360) % 360 # Map the angles to [0, 180)
    np_print_name("Direction", direction)
    # direction = np.round(direction) # Now to do GMT

    for i in range(color_image.shape[0]):
        for j in range(color_image.shape[1]):
            color_image[i, j][0] = (direction[i, j] / 360.0) * 179.0
            color_image[i, j][1] = 255
            color_image[i, j][2] = 255

    print(color_image.shape)
    cv2.cvtColor(color_image, cv2.COLOR_HSV2BGR)
    cv2.imshow("Direction", color_image)

    angles = np.array([0, 45, 90, 135])

    gmt = weak = strong = np.zeros_like(image)

    # Ignore a 1 pixel border to make NOOB checks better
    # Double thresholding + GMT
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            direction[y, x] = angles[np.abs(angles - direction[y, x]).argmin()]
            ang = direction[y, x]
            mag = gradient[y, x]

            ov1 = ov2 = 0.0

            match ang:
                case 0:
                    ov1, ov2 = gradient[y, x - 1], gradient[y, x + 1]
                case 45:
                    ov1, ov2 = gradient[y - 1, x + 1], gradient[y + 1, x - 1]
                case 90:
                    ov1, ov2 = gradient[y - 1, x], gradient[y + 1, x]
                case 135:
                    ov1, ov2 = gradient[y - 1, x - 1], gradient[y + 1, x + 1]
                case _:
                    raise ValueError("Invalid Angle in Direction array")

            if mag >= HIGH_THRESH:
                strong[y, x] = mag
            elif mag < HIGH_THRESH and mag >= LOW_THRESH:
                weak[y, x] = mag

            if mag >= ov1 and mag >= ov2 and mag >= LOW_THRESH:
                gmt[y, x] = mag
    # Hysteresis/Blob Analysis
    # for y in range(1, H-1):
    #     for x in range(1, W-1):
    #         for i in range(y - 1, y + 1):
    #             for k in range(x - 1, x + 1):


    print("Min:", np.min(gmt), "Max:", np.max(gmt), "Non-zero count:", np.count_nonzero(gmt))
    gmt = cv2.normalize(gmt, np.empty_like(gmt), 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    cv2.imshow("Image", image)
    cv2.imshow("Guassian", guass_image)
    cv2.imshow("GMT", gmt)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def np_print_name(name, nparray):
    print(name)
    print(nparray)

def generate_guassian_kernel(size: int, sigma: int) -> np.ndarray:
    kernel = np.zeros((size, size), dtype=float)
    k = size // 2

    for i in range(size):
        for j in range(size):
            x = i - k
            y = j - k
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma))

    kernel /= (2 * np.pi * sigma ** 2)
    kernel /= kernel.sum()
    return kernel

if __name__ == "__main__":
    main()
