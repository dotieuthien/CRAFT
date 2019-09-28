import cv2
import numpy as np
from math import exp
import re


def create_2d_gaussian():
    """
    Return 2D Gaussian heatmap
    """
    img_size = 512
    isotropicGrayscaleImage = np.zeros((img_size, img_size), np.uint8)
    scaledGaussian = lambda x : exp(-(1 / 2) * (x**2))

    for i in range(img_size):
        for j in range(img_size):
            # Find euclidian distance from center of image (imgSize / 2, imgSize / 2) 
            # and scale it to range of 0 to 2.5 as scaled Gaussian
            # Returns highest probability for x = 0 and approximately
            # Zero probability for x > 2.5

            distanceFromCenter = np.linalg.norm(np.array([i - img_size / 2, j - img_size / 2]))
            distanceFromCenter = 3 * distanceFromCenter / (img_size / 2)
            scaledGaussianProb = scaledGaussian(distanceFromCenter)
            isotropicGrayscaleImage[i, j] = np.clip(scaledGaussianProb * 255, 4, 255)
            if isotropicGrayscaleImage[i, j] == 4:
                isotropicGrayscaleImage[i, j] = 0
    return isotropicGrayscaleImage


def perspective_transform(img_size, pts, gauss_img):
    """
    pts: 4 points with coodinate of [[col, row]]
    gauss_img: 2D Gaussian heatmap
    """
    max_x, max_y = img_size[0], img_size[1]
    dst = np.array([[40, 40],
                    [(gauss_img.shape[1] - 1) - 40, 40],
                    [(gauss_img.shape[1] - 1) - 40, (gauss_img.shape[0] - 1) - 40],
                    [40, (gauss_img.shape[0] - 1) - 40]], dtype='float32')
    M = cv2.getPerspectiveTransform(dst, pts)
    warped_img = cv2.warpPerspective(gauss_img, M, (max_x, max_y))
    return warped_img


def translate_pts(pts):
    """
    pts: 4 points with coodinate of [[col, row]]
    """
    min_col = min(pts[:, 0])
    min_row = min(pts[:, 1])
    trans_pts = pts.copy()
    trans_pts[:, 0] = trans_pts[:, 0] - min_col
    trans_pts[:, 1] = trans_pts[:, 1] - min_row
    return trans_pts


def process_label(string):
    label = []
    for i in range(np.shape(string)[0]):
        words = re.split('[\n ]', string[i].strip())
        for word in words:
            if word == '':
                continue
            else:
                label.append(word)
    return label


def create_affinity_mask(img, gauss_img, char_bboxs, string):
    """
    char_box: [[col, row]]
    """
    background = np.zeros(np.shape(img), dtype=np.uint8)
    img_size = [np.shape(img)[1], np.shape(img)[0]]
    num_box = np.shape(char_bboxs)[2]
    start_box_id = 0
    # Pre-process string
    string = process_label(string)

    for i in range(np.shape(string)[0]):
        word = string[i].strip()
        len_word = len(word)
        end_box_id = start_box_id + len_word

        for j in range(start_box_id, end_box_id - 1, 1):
            # Point with [col, row]
            p1 = char_bboxs[:, 0, j]
            p2 = char_bboxs[:, 1, j]
            p3 = char_bboxs[:, 2, j]
            p4 = char_bboxs[:, 3, j]
            pts = np.array([p1, p2, p3, p4], dtype='float32')

            # Center of character box
            M = cv2.moments(pts)
            cX1 = int(M["m10"] / M["m00"])
            cY1 = int(M["m01"] / M["m00"])

            # The top-left of aff box
            tri = np.array([p1, p2, [cX1, cY1]], dtype='float32')
            M = cv2.moments(tri)
            aff_p1 = [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]

            # The bottom-left of aff box
            tri = np.array([p3, p4, [cX1, cY1]], dtype='float32')
            M = cv2.moments(tri)
            aff_p4 = [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]

            # Point with [col, row]
            p1 = char_bboxs[:, 0, j + 1]
            p2 = char_bboxs[:, 1, j + 1]
            p3 = char_bboxs[:, 2, j + 1]
            p4 = char_bboxs[:, 3, j + 1]
            pts = np.array([p1, p2, p3, p4], dtype='float32')

            # Center of character box
            M = cv2.moments(pts)
            cX2 = int(M["m10"] / M["m00"])
            cY2 = int(M["m01"] / M["m00"])

            # The top-right of aff box
            tri = np.array([p1, p2, [cX2, cY2]], dtype='float32')
            M = cv2.moments(tri)
            aff_p2 = [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]

            # The bottom-right of aff box
            tri = np.array([p3, p4, [cX2, cY2]], dtype='float32')
            M = cv2.moments(tri)
            aff_p3 = [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]

            # Aff box
            aff_box = np.array([aff_p1, aff_p2, aff_p3, aff_p4], dtype='float32')

            # Perspective projection
            warped_img = perspective_transform(img_size, aff_box, gauss_img)
            background[:] += warped_img[:]

            # Draw lines
            # cv2.line(img, (int(aff_p1[0]), int(aff_p1[1])), (int(aff_p2[0]), int(aff_p2[1])), (0, 255, 0), 1)
            # cv2.line(img, (int(aff_p2[0]), int(aff_p2[1])), (int(aff_p3[0]), int(aff_p3[1])), (0, 255, 0), 1)
            # cv2.line(img, (int(aff_p3[0]), int(aff_p3[1])), (int(aff_p4[0]), int(aff_p4[1])), (0, 255, 0), 1)
            # cv2.line(img, (int(aff_p1[0]), int(aff_p1[1])), (int(aff_p4[0]), int(aff_p4[1])), (0, 255, 0), 1)
            # cv2.imwrite('img.jpg', img)

        start_box_id = end_box_id

    # Convert to heatmap
    bacground = (np.clip(background, 0, 255)).astype(np.uint8)
    # background = cv2.applyColorMap(background, cv2.COLORMAP_JET)
    return background


def create_character_mask(img, gauss_img, char_bboxs):
    """
    img: image
    gauss_img: image of 2D Gaussian heatmap
    char_bboxs: list of bbox
    """
    background = np.zeros(np.shape(img), dtype=np.uint8)
    num_box = np.shape(char_bboxs)[2]
    img_size = [np.shape(img)[1], np.shape(img)[0]]

    for i in range(num_box):
        # Point with [col, row]
        p1 = char_bboxs[:, 0, i]
        p2 = char_bboxs[:, 1, i]
        p3 = char_bboxs[:, 2, i]
        p4 = char_bboxs[:, 3, i]
        pts = np.array([p1, p2, p3, p4], dtype='float32')

        # Draw lines
        # cv2.line(background, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 1)
        # cv2.line(background, (int(p2[0]), int(p2[1])), (int(p3[0]), int(p3[1])), (0, 255, 0), 1)
        # cv2.line(background, (int(p3[0]), int(p3[1])), (int(p4[0]), int(p4[1])), (0, 255, 0), 1)
        # cv2.line(background, (int(p1[0]), int(p1[1])), (int(p4[0]), int(p4[1])), (0, 255, 0), 1)

        # Perspective projection
        warped_img = perspective_transform(img_size, pts, gauss_img)
        background[:] += warped_img[:]

    # Convert to heatmap
    background = (np.clip(background, 0, 255)).astype(np.uint8)
    # background = cv2.applyColorMap(background, cv2.COLORMAP_JET)
    return background


if __name__ == '__main__':
    gauss_img = create_2d_gaussian()
    cv2.imwrite('gauss.jpg', gauss_img)
    pts = np.array([[300, 300], [430, 310], [450, 550], [300, 570]], dtype='float32')
    pts = translate_pts(pts)
    warped = perspective_transform([1000, 1500], pts, gauss_img)
    warped = cv2.applyColorMap(warped, cv2.COLORMAP_JET)
    for p in pts:
        cv2.circle(warped, (int(p[0]), int(p[1])) , 2, (0, 255, 0), -1)
    cv2.imwrite('warped.jpg', warped)
