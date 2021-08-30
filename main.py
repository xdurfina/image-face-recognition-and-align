import pickle
import cv2
import mtcnn
import os
import numpy as np
import dlib
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import euclidean


def detectFaceLandmarks():
    face_detector = mtcnn.MTCNN()
    cap = cv2.VideoCapture(0)

    while True:
        x, img = cap.read()
        results = face_detector.detect_faces(img)

        for res in results:
            x1, y1, width, height = res['box']
            x2, y2 = x1 + width, y1 + height

            key_points = res['keypoints'].values()

            for point in key_points:
                cv2.circle(img, point, 5, (0, 255, 0), thickness=-1)

        cv2.imshow('Image', img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break


def anotationsToSomethingNormal():
    annotations = {}
    with open("list_landmarks_celeba.txt", "r") as file:
        for line in file:
            line = line.split()
            key = line[0]
            line.pop(0)
            outputLine = [[line[0], line[1]], [line[2], line[3]], [line[4], line[5]], [line[6], line[7]],
                          [line[8], line[9]]]
            annotations.update({key: outputLine})

    with open('annotations.txt', 'wb') as fp:
        pickle.dump(annotations, fp)

    return annotations


def getDataFromPhotos():
    i = 0
    face_detector = mtcnn.MTCNN()
    dataFound = {}

    for filename in os.listdir('data'):
        img = cv2.imread('data/' + filename)
        results = face_detector.detect_faces(img)

        for res in results:
            key_points = res['keypoints']
            left_eye = list(key_points.get('left_eye'))
            right_eye = list(key_points.get('right_eye'))
            nose = list(key_points.get('nose'))
            mouth_left = list(key_points.get('mouth_left'))
            mouth_right = list(key_points.get('mouth_right'))

            outputList = [left_eye, right_eye, nose, mouth_left, mouth_right]

            print(i)
            print(filename)
            print(outputList)
            i += 1
            dataFound.update({filename: outputList})

    with open('foundCoordinates.txt', 'wb') as fo:
        pickle.dump(dataFound, fo)

    return dataFound


def calculateMSE(annotations, foundCoordinates):
    for x in annotations:
        annotations[x] = list(np.float_(annotations[x]))
    for i in foundCoordinates:
        foundCoordinates[i] = list(np.float_(foundCoordinates[i]))

    outputDictionary = {}
    totalMSE = 0.0
    for key in foundCoordinates:
        outputDictionary.update({key: [[foundCoordinates.get(key)], [annotations.get(key)],
                                       mean_squared_error(annotations.get(key), foundCoordinates.get(key))]})
        totalMSE = mean_squared_error(annotations.get(key), foundCoordinates.get(key)) + totalMSE

    totalMSE = totalMSE / len(foundCoordinates)
    print('Total MSE: ', totalMSE)
    return outputDictionary


def loadEverything():
    with open('annotations.txt', 'rb') as fp:
        annotations = pickle.load(fp)

    with open('foundCoordinates.txt', 'rb') as fo:
        foundCoordinates = pickle.load(fo)

    return annotations, foundCoordinates


# https://www.youtube.com/watch?v=81lCsiNBvrM&t=429s&ab_channel=PracticalAI
# https://github.com/Practical-AI/Face/blob/master/_04_face_alignment/01_face_alignment.py
def align(img, left_eye_pos, right_eye_pos, size=(150, 150), eye_pos=(0.35, 0.35)):
    width, height = size
    eye_pos_w, eye_pos_h = eye_pos

    l_e, r_e = left_eye_pos, right_eye_pos

    dy = r_e[1] - l_e[1]
    dx = r_e[0] - l_e[0]
    dist = euclidean(l_e, r_e)
    scale = (width * (1 - 2 * eye_pos_w)) / dist

    center = ((l_e[0] + r_e[0]) // 2, (l_e[1] + r_e[1]) // 2)
    angle = np.degrees(np.arctan2(dy, dx)) + 360

    m = cv2.getRotationMatrix2D(center, angle, scale)
    tx = width * 0.5
    ty = height * eye_pos_h
    m[0, 2] += (tx - center[0])
    m[1, 2] += (ty - center[1])

    aligned_face = cv2.warpAffine(img, m, (width, height))
    return aligned_face


def alignFaces(foundCoordinates):
    for key in foundCoordinates:
        img = cv2.imread('data/' + key)
        value = foundCoordinates.get(key)
        left_eye = tuple([value[0][0], value[0][1]])
        right_eye = tuple([value[1][0], value[1][1]])
        aligned = align(img, left_eye, right_eye)
        cv2.imwrite('faces/' + key, aligned)
        cv2.imshow('foto', aligned)
        cv2.waitKey(0)


# https://automaticaddison.com/how-to-blend-multiple-images-using-opencv/
def getAverageImage():
    image_data = []
    for filename in os.listdir('data'):
        img = cv2.imread('faces/' + filename)
        if img is not None:
            image_data.append(img)

    dst = image_data[0]

    for i in range(len(image_data)):
        if i == 0:
            pass
        else:
            alpha = 1.0 / (i + 1)
            beta = 1.0 - alpha
            dst = cv2.addWeighted(image_data[i], alpha, dst, beta, 0.0)

    cv2.imshow('Average face', dst)
    cv2.waitKey(0)
    return dst


def getMostSimilarImage(averageImage):
    err = 9999999
    for filename in os.listdir('faces'):
        img = cv2.imread('faces/' + filename)

        errNow = np.sum((averageImage.astype("float") - img.astype("float")) ** 2)
        errNow /= float(averageImage.shape[0] * averageImage.shape[1])

        print(errNow)

        if errNow < err:
            err = errNow
            mostSilimarImage = cv2.imread('faces/' + filename)

    cv2.imshow('Most Silimar Image', mostSilimarImage)
    cv2.waitKey(0)
    return mostSilimarImage


# https://pysource.com/2019/05/28/face-swapping-explained-in-8-steps-opencv-with-python/
def faceSwap(inputwebcam, secondface):
    def extract_index_nparray(nparray):
        index = None
        for num in nparray[0]:
            index = num
            break
        return index

    img = inputwebcam
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)
    img2 = secondface
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    height, width, channels = img2.shape
    img2_new_face = np.zeros((height, width, channels), np.uint8)

    # Face 1
    faces = detector(img_gray)
    for face in faces:
        landmarks = predictor(img_gray, face)
        landmarks_points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))

        points = np.array(landmarks_points, np.int32)
        convexhull = cv2.convexHull(points)
        # cv2.polylines(img, [convexhull], True, (255, 0, 0), 3)
        cv2.fillConvexPoly(mask, convexhull, 255)

        face_image_1 = cv2.bitwise_and(img, img, mask=mask)

        # Delaunay triangulation
        rect = cv2.boundingRect(convexhull)
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(landmarks_points)
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)

        indexes_triangles = []
        for t in triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            index_pt1 = np.where((points == pt1).all(axis=1))
            index_pt1 = extract_index_nparray(index_pt1)

            index_pt2 = np.where((points == pt2).all(axis=1))
            index_pt2 = extract_index_nparray(index_pt2)

            index_pt3 = np.where((points == pt3).all(axis=1))
            index_pt3 = extract_index_nparray(index_pt3)

            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                triangle = [index_pt1, index_pt2, index_pt3]
                indexes_triangles.append(triangle)

    # Face 2
    faces2 = detector(img2_gray)
    for face in faces2:
        landmarks = predictor(img2_gray, face)
        landmarks_points2 = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points2.append((x, y))

        points2 = np.array(landmarks_points2, np.int32)
        convexhull2 = cv2.convexHull(points2)

    lines_space_mask = np.zeros_like(img_gray)
    lines_space_new_face = np.zeros_like(img2)
    # Triangulation of both faces
    for triangle_index in indexes_triangles:
        # Triangulation of the first face
        tr1_pt1 = landmarks_points[triangle_index[0]]
        tr1_pt2 = landmarks_points[triangle_index[1]]
        tr1_pt3 = landmarks_points[triangle_index[2]]
        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

        rect1 = cv2.boundingRect(triangle1)
        (x, y, w, h) = rect1
        cropped_triangle = img[y: y + h, x: x + w]
        cropped_tr1_mask = np.zeros((h, w), np.uint8)

        points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                           [tr1_pt2[0] - x, tr1_pt2[1] - y],
                           [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

        # Lines space
        cv2.line(lines_space_mask, tr1_pt1, tr1_pt2, 255)
        cv2.line(lines_space_mask, tr1_pt2, tr1_pt3, 255)
        cv2.line(lines_space_mask, tr1_pt1, tr1_pt3, 255)
        lines_space = cv2.bitwise_and(img, img, mask=lines_space_mask)

        # Triangulation of second face
        tr2_pt1 = landmarks_points2[triangle_index[0]]
        tr2_pt2 = landmarks_points2[triangle_index[1]]
        tr2_pt3 = landmarks_points2[triangle_index[2]]
        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

        rect2 = cv2.boundingRect(triangle2)
        (x, y, w, h) = rect2

        cropped_tr2_mask = np.zeros((h, w), np.uint8)

        points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                            [tr2_pt2[0] - x, tr2_pt2[1] - y],
                            [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

        # Warp triangles
        points = np.float32(points)
        points2 = np.float32(points2)
        M = cv2.getAffineTransform(points, points2)
        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

        # Reconstructing destination face
        img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
        img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

        img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
        img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area

    # Face swapped (putting 1st face into 2nd face)
    img2_face_mask = np.zeros_like(img2_gray)
    img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
    img2_face_mask = cv2.bitwise_not(img2_head_mask)

    img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
    result = cv2.add(img2_head_noface, img2_new_face)

    (x, y, w, h) = cv2.boundingRect(convexhull2)
    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

    seamlessclone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)

    return seamlessclone
    # cv2.imshow("seamlessclone", seamlessclone)
    # cv2.waitKey(0)
    #
    # cv2.destroyAllWindows()


def webcamFaceswap():
    cap = cv2.VideoCapture(0)
    while True:
        xyz, image = cap.read()
        secondImage = cv2.imread('data/003147.jpg')
        outputImage = faceSwap(image, secondImage)
        cv2.imshow('Outputimage', outputImage)
        cv2.waitKey(0)


if __name__ == "__main__":
    detectFaceLandmarks()
    # annotations = anotationsToSomethingNormal()
    # foundCoordinates = getDataFromPhotos()

    annotations, foundCoordinates = loadEverything()

    dictionaryMSE = calculateMSE(annotations, foundCoordinates)

    #alignFaces(foundCoordinates)

    averageImage = getAverageImage()

    mostSimilarImage = getMostSimilarImage(averageImage)

    webcamFaceswap()

