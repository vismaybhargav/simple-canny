import cv2
import numpy as np

LOW_THRESH, HIGH_THRESH = 100, 200  # TODO: This needs to be tuned

def main():
    image = cv2.imread("img/monkey.jpg", cv2.IMREAD_GRAYSCALE)
    hsv = cv2.cvtColor(cv2.imread("img/monkey.jpg"), cv2.COLOR_BGR2HSV)

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

    guass_image = cv2.filter2D(src=image, ddepth=-1, kernel=generate_guassian_kernel(size=5, sigma=2))

    gx = cv2.filter2D(src=guass_image, ddepth=cv2.CV_64F, kernel=sobel_gx)
    print_info("gx", gx)
    gy = cv2.filter2D(src=guass_image, ddepth=cv2.CV_64F, kernel=sobel_gy)
    print_info("gy", gy)

    gradient = np.hypot(gx, gy)
    print_info("gradient", gradient)
    gradient = cv2.normalize(gradient, np.empty_like(gradient), 0, 256, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imshow("sobel", gradient)

    direction = np.degrees(np.atan2(gy, gx))
    print_info("direction", direction)
    direction = np.mod(direction + 360, 360)
    print_info("360 direction", direction)

    for i in range(hsv.shape[0]):
        for j in range(hsv.shape[1]):
            hsv[i, j][0] = (direction[i, j] / 2.0)
            hsv[i, j][1] = 255
            hsv[i, j][2] = 255

    cv2.imshow("Direction", cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))

    direction = np.mod(direction + 180, 180)
    print_info("180 direction", direction)
    rounded = round_directions(direction)
    print_info("rounded", rounded)

    gmt = weak = strong = np.zeros_like(image)

    # Ignore a 1 pixel border to make NOOB checks better
    # Double thresholding + GMT
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            ang = rounded[y, x]
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

            # if mag >= HIGH_THRESH:
            #     strong[y, x] = mag
            # elif HIGH_THRESH > mag >= LOW_THRESH:
            #     weak[y, x] = mag

            if mag >= ov1 and mag >= ov2:
                gmt[y, x] = mag

    # Normalize again in case of overflow or values that don't reach the full range
    gmt = cv2.normalize(gmt, np.empty_like(gmt), 0, 256, cv2.NORM_MINMAX).astype(np.uint8)

    hysteresis = np.copy(strong)

    # for y in range(1, H - 1):
    #     for x in range(1, W - 1):
    #         if weak[y, x] > 0:
    #             for dy in [-1, 0, 1]:
    #                 for dx in [-1, 0, 1]:
    #                     if dy == 0 and dx == 0:
    #                         continue
    #                     nx, ny = x + dx, y + dy
    #                     if strong[ny, nx] > 0:
    #                         hysteresis[y, x] = strong[ny, nx]
    #                         break

    cv2.imshow("Image", image)
    cv2.imshow("Guassian", guass_image)
    cv2.imshow("GMT", gmt)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def round_directions(direction: np.ndarray) -> np.ndarray:
    H, W = direction.shape
    quantized = np.zeros_like(direction)

    for y in range(H):
        for x in range(W):
            angle = direction[y, x]
            if (angle < 22.5) or (angle >= 157.5):
                quantized[y, x] = 0
            elif angle < 67.5:
                quantized[y, x] = 45
            elif angle < 112.5:
                quantized[y, x] = 90
            elif angle < 157.5:
                quantized[y, x] = 135
            else:
                quantized[y, x] = 0  # fallback

    return quantized

def generate_guassian_kernel(size: int, sigma: int) -> np.ndarray:
    kernel = np.zeros((size, size), dtype=float)
    k = size // 2

    for i in range(size):
        for j in range(size):
            x = i - k
            y = j - k
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    kernel /= (2 * np.pi * sigma ** 2)
    kernel /= kernel.sum()
    return kernel

def print_info(title: str, arr: np.ndarray):
    print(title, arr.shape, np.max(arr), np.min(arr))

def validate_rounded_values(rounded: ) {

}
if __name__ == "__main__":
    main()
