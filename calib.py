import cv2
import numpy as np
import os

from calib_cls import CalibParam, CalibResult


def Intrinsic(img_path):
    chessboard_params = CalibParam()
    calib_result = CalibResult()

    chessboard_params.set_params(6, 9, 40, img_path)
    params = chessboard_params.get_params()
    # world coordinate for the chessboard
    obj_point = np.zeros((params[0] * params[1], 3), np.float32)
    obj_point[:, :2] = np.mgrid[0:params[0], 0:params[1]].T.reshape(-1, 2)
    cube_size = params[2]
    # top left corner of the intersection point on the calibration board as the center of the world coordinate
    obj_point = obj_point * cube_size

    # save all the object points in world coordinate and image coordinates
    obj_points = []
    img_points = []
    # read images only
    images = [os.path.join(img_path, x) for x in os.listdir(img_path)
              if any(x.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])
              ]
    # gray = None
    for img in images:
        gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        # find inner corners of the chessboard
        ret, corners = cv2.findChessboardCorners(gray, (params[0], params[1]), None,
                                                 flags=cv2.CALIB_CB_FAST_CHECK)
        # print("finding chess board corners")
        if ret:
            # print("**Found chess board corners")
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),  # get a more precise corner
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
            img_points.append(corners)
            obj_points.append(obj_point)

            # for visualisation
            cv2.drawChessboardCorners(gray, (params[0], params[1]), corners, ret)
            # cv2.imshow("img", gray)
            cv2.waitKey(500)
    cv2.destroyAllWindows()

    # mtx is the intrinsic parameter matrix and dist is the distortion vector
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    calib_result.set_int_calib_result(mtx, dist)

    for img in images:
        gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        h, w = gray.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv2.undistort(gray, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        # cv2.imshow("img", dst)
        cv2.waitKey(500)

    cv2.destroyAllWindows()
    return calib_result


def calib_videos(video_pth = "MAH01462.mp4", img_pth = 'calibration images', write_path = "calibrated.mp4"):
    orig_video = cv2.VideoCapture(video_pth)
    video_fps = round(orig_video.get(cv2.CAP_PROP_FPS))
    out_frame_width, out_frame_height = 871,880
    fourecc = cv2.VideoWriter_fourcc(*'mp4v')

    calib_result = Intrinsic(img_pth)

    mat, coef = calib_result.get_int_calib_result()
    return mat, coef

    processed_video = cv2.VideoWriter(write_path, fourecc, video_fps, (out_frame_width, out_frame_height))
    if orig_video.isOpened():
        while True:
            ret, frame = orig_video.read()
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if not ret:
                break
            else:
                h, w = frame.shape[:2]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mat, coef, (w, h), 1, (w, h))
                dst = cv2.undistort(frame, mat, coef, None, newcameramtx)

                # crop the image
                x, y, w, h = roi
                dst = dst[y:y + h, x:x + w]
                processed_video.write(dst)

    orig_video.release()
    processed_video.release()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    calib_videos()

