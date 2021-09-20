import numpy as np


# свертка
def conv(A, V):
    H = (V.shape[0] - 1) // 2
    W = (V.shape[1] - 1) // 2
    Apad = np.pad(A, (H, W), 'edge')
    A_height = Apad.shape[0]
    A_width = Apad.shape[1]
    V_height = V.shape[0]
    V_width = V.shape[1]
    # output numpy matrix with height and width
    out = np.zeros((A_height - 2 * H, A_width - 2 * W))
    # итерация по изображению
    for i in range(H, A_height - H):
        for j in range(W, A_width - W):
            sum = 0
            # итерация по фильтру
            for k in range(-H, V_height - H):
                for m in range(-W, V_width - W):
                    a = Apad[i + k, j + m]
                    v = V[k + H, m + W]
                    sum += (a * v)
            out[i - H, j - W] = sum
    return out


# математическая морфология
def my_dilation(img, struct):
    out = np.zeros(img.shape, dtype='int')
    H = (struct.shape[0] - 1) // 2
    W = (struct.shape[1] - 1) // 2
    for i in range(H, img.shape[0] - H):
        for j in range(W, img.shape[1] - W):
            if img[i, j] == 255:
                for m in range(-H, H + 1):
                    for n in range(-W, W + 1):
                        out[i + m, j + n] = struct[H + m, W + n]
    return out.astype(img.dtype)


def gener(p, q):
    for i in range(-p, p + 1):
        for j in range(-q, q + 1):
            yield i, j


def my_erosion(img, struct):
    out = np.zeros(img.shape, dtype='int')
    H = (struct.shape[0] - 1) // 2
    W = (struct.shape[1] - 1) // 2
    for i in range(H, img.shape[0] - H):
        for j in range(W, img.shape[1] - W):
            for m, n in gener(H, W):
                if not struct[H + m, W + n]:
                    continue
                if struct[H + m, W + n] != img[i + m, j + n]:
                    break
            else:
                out[i, j] = 255
    return out.astype(img.dtype)


# Canny
def gaussian_kernel(size, sigma):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g


def gradients(img):
    sobelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobGx = conv(img, sobelX)
    sobGy = conv(img, sobelY)
    sob_out = np.sqrt(np.power(sobGx, 2) + np.power(sobGy, 2))

    G = sob_out / sob_out.max() * 255
    theta = np.arctan2(sobGy, sobGx)

    return G, theta


def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z


def threshold(img, lowThresholdRatio, highThresholdRatio):
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(100)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res, weak, strong


def hysteresis(img, weak, strong):
    M, N = img.shape

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if img[i, j] == weak:
                try:
                    if ((img[i + 1, j - 1] == strong) or
                            (img[i + 1, j] == strong) or
                            (img[i + 1, j + 1] == strong) or
                            (img[i, j - 1] == strong) or
                            (img[i, j + 1] == strong) or
                            (img[i - 1, j - 1] == strong) or
                            (img[i - 1, j] == strong) or
                            (img[i - 1, j + 1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass

    return img
