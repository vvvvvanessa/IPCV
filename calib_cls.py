# Parameters used for camera calibration (input from UI)
class CalibParam:

    _vertical_corner = 6
    _horizontal_corner = 9
    _cube_size = 40
    _intrinsic_img_file = ""


    def get_params(self):
        """
        return vertical_corner, horizontal_corner, cube_size
        """
        return self._vertical_corner, self._horizontal_corner, self._cube_size

    def get_intrisic_path(self):
        return self._intrinsic_img_file

    def get_extrisic_path(self):
        return self._extrinsic_img_file

    def set_params(self, vertical_corner, horizontal_corner, cube_size,
                   intrinsic_img_file):
        self._cube_size = cube_size
        self._vertical_corner = vertical_corner
        self._horizontal_corner = horizontal_corner
        self._intrinsic_img_file = intrinsic_img_file

# Calibration Results obtained
class CalibResult:
    def __init__(self):
        self._camera_mat = None
        self._dist_coeff = None
        self._rvec = None
        self._tvec = None

    def set_int_calib_result(self, camera_mat, dist_coeff):
        self._camera_mat = camera_mat
        self._dist_coeff = dist_coeff

    def set_ext_calib_result(self, rvec, tvec):
        self._tvec = tvec
        self._rvec = rvec

    def get_int_calib_result(self):
        """
        to get the intrinsic results
        :return: camera_mat, dist_coeff
        """
        return self._camera_mat, self._dist_coeff

    def get_ext_calib_result(self):
        """
        to get the intrinsic results
        :return: rvec, tvec
        """
        return self._rvec, self._tvec
