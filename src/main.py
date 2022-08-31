import cv2
import numpy as np
from numba import njit
import numba as nb


@njit(parallel=True, fastmath=True)
def render(img, vertices, colors, faces, N):
    height, width = img.shape[:2]
    buffer = np.zeros((img.shape[0], img.shape[1], 3, N), dtype=np.uint8)
    for i in nb.prange(height):
        for j in nb.prange(width):
            Py = 2.0 * ((height - i) / height) - 1.0
            Px = 2.0 * (j / width) - 1.0
            for n in nb.prange(N):
                for k in nb.prange(3):    
                    idx1 = faces[n, k]
                    idx2 = faces[n, (k + 1) % 3]
                    V0x = vertices[idx1, 0]
                    V0y = vertices[idx1, 1]
                    V1x = vertices[idx2, 0]
                    V1y = vertices[idx2, 1]
                    if (Px - V0x) * (V1y - V0y) - (Py - V0y) * (V1x - V0x) < 0:
                        buffer[i, j, k, n] = 255

    img_new = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    p = np.sqrt(2)
    for i in nb.prange(height):
        for j in nb.prange(width):
            for n in nb.prange(N):
                idx1 = faces[n, 0]
                idx2 = faces[n, 1]
                idx3 = faces[n, 2]
                x1 = vertices[idx1, 0]
                y1 = vertices[idx1, 1]
                x2 = vertices[idx2, 0]
                y2 = vertices[idx2, 1]
                x3 = vertices[idx3, 0]
                y3 = vertices[idx3, 1]
                c1 = colors[idx1, :]
                c2 = colors[idx2, :]
                c3 = colors[idx3, :]
                if buffer[i, j, 0, n] == 255 and buffer[i, j, 1, n] == 255 and buffer[i, j, 2, n] == 255:
                    y = 2.0 * ((height - i) / height) - 1.0
                    x = 2.0 * (j / width) - 1.0
                    mat = np.array([
                        [x2 * y3 - x3 * y2, y2 - y3, x3 - x2],
                        [x3 * y1 - x1 * y3, y3 - y1, x1 - x3],
                        [x1 * y2 - x2 * y1, y1 - y2, x2 - x1]
                    ])
                    l1, l2, l3 = mat @ np.array([1, x, y])
                    img_new[i, j, :] = (255 * np.float_power((l1 * np.power(c1, p) + l2 * np.power(c2, p) + l3 * np.power(c3, p)) / (l1 + l2 + l3), 1 / p)).astype(np.uint8)
    return img_new
    
                        

                
                
    return buffer

if __name__ == "__main__":
    theta_x = 0
    theta_z = 0
    size = 0.5
    while True:
        vertices = np.array([
            [-size, -size, 0.0],
            [size, -size, 0.0],
            [-size, size, 0.0],
            [size, size, 0.0],
        ], dtype=np.float32)

        colors = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 1],
        ], dtype=np.float32)
        faces = np.array([
            [0, 1, 2],
            [3, 2, 1],
            [2, 1, 0],
            [1, 2, 3]
        ], dtype=int)
        rotation_z = np.array([
            [np.cos(theta_z), -np.sin(theta_z), 0],
            [np.sin(theta_z), np.cos(theta_z), 0],
            [0, 0, 1]
        ])

        rotation_x = np.array([
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x), np.cos(theta_x)]
        ])
        vertices = np.einsum('ij,kj->ki', rotation_z, vertices)
        vertices = np.einsum('ij,kj->ki', rotation_x, vertices)
        img = np.zeros((400, 400))
        img_new = render(img, vertices, colors, faces, faces.shape[0])
        
        cv2.imshow("Image", img_new)
        if cv2.waitKey(1) == ord(' '):
            c = cv2.waitKey(1)
            while c != ord(' '):
                c = cv2.waitKey(1)
        theta_x += 2.0 * np.pi / 500
        theta_x = theta_x % (2.0 * np.pi)
        theta_z += 2.0 * np.pi / 2000
        theta_z = theta_z % (2.0 * np.pi)