import cv2
import numpy as np

def main():
    image = cv2.imread("img/monkey.jpg", cv2.IMREAD_GRAYSCALE)
    H, W = image.shape

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

    guass_image = cv2.filter2D(src=image, ddepth=-1, kernel=generate_guassian_kernel(5, 2))

    gx = cv2.filter2D(src=guass_image, ddepth=-1, kernel=sobel_gx)
    gy = cv2.filter2D(src=guass_image, ddepth=-1, kernel=sobel_gy)

    gradient = np.hypot(gx, gy)

    direction = np.degrees(np.atan2(gx, gy))
    direction = (direction + 180) % 180 # Map the angles to [0, 180)
    direction = np.round(direction) # Now to do GMT

    print(direction)

    angles = np.array([0, 45, 90, 135])

    gmt = np.zeros_like(image)

    # Ignore a 1 pixel border to make NOOB checks better
    for y in range(1, H-1):
        for x in range(1, W-1):
            direction[y, x] = angles[np.abs(angles - direction[y, x]).argmin()]
            ang = direction[y, x]
            mag = gradient[y, x]

            ov1 = ov2 = 0.0

            match ang:
                case 0:
                    ov1, ov2 = gradient[y, x-1], gradient[y, x+1]
                case 45:
                    ov1, ov2 = gradient[y - 1, x + 1], gradient[y + 1, x - 1]
                case 90:
                    ov1, ov2 = gradient[y - 1, x], gradient[y + 1, x]
                case 135:
                    ov1, ov2 = gradient[y - 1, x - 1], gradient[y + 1, x + 1]
                case _:
                    raise ValueError("Invalid Angle in direction array")

            if mag >= ov1 and mag >= ov2:
                gmt[y, x] = mag

    gmt = cv2.normalize(gmt, np.empty_like(gmt), 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    print("Min:", np.min(gmt), "Max:", np.max(gmt), "Non-zero count:", np.count_nonzero(gmt))

    cv2.imshow("Image", image)
    cv2.imshow("Guassian", guass_image)
    cv2.imshow("GMT", gmt)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
