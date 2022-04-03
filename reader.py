import abc
from typing import List

import numpy as np


class Label3D:
    def __init__(self, classification: str, centroid: np.ndarray, dimension: np.ndarray, yaw: float):
        self.classification = classification
        self.centroid = centroid
        self.dimension = dimension
        self.yaw = yaw
        #print(centroid.shape)
    def __str__(self):
        return " Label 3D | Cls: %s, x: %f, y: %f, l: %f, w: %f, yaw: %f" % (
            self.classification, self.centroid[0], self.centroid[1], self.dimension[0], self.dimension[1], self.yaw)


class DataReader:

    @staticmethod
    @abc.abstractmethod
    def read_lidar(file_path: str) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def read_label(file_path: str) -> List[Label3D]:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def read_calibration(file_path: str) -> np.ndarray:
        raise NotImplementedError


class KittiDataReader(DataReader):

    def __init__(self):
        super(KittiDataReader, self).__init__()

    @staticmethod
    def read_lidar(file_path: str):
        return np.fromfile(file_path, dtype=np.float32).reshape((-1, 4))

    @staticmethod
    def read_label(file_path: str):
        with open(file_path, "r") as f:

            lines = f.readlines()

            elements = []
            for line in lines:

                values = line.split()

                element = Label3D(
                    str(values[0]), # 类别
                    np.array(values[11:14], dtype=np.float32), # 在相机坐标系的坐标
                    np.array(values[8:11], dtype=np.float32), # 3D bounding box的长宽高
                    float(values[14]) # yaw角
                )

                if element.classification == "DontCare":
                    continue
                else:
                    elements.append(element)

        return elements

    @staticmethod
    def read_calibration(file_path:str=None):
        """
        警告：暂时用现成的进行测试
        """ 
        Tr_velo_to_cam = np.array([[6.927964000000e-03,-9.999722000000e-01,-2.757829000000e-03,-2.457729000000e-02],
                                   [-1.162982000000e-03,2.749836000000e-03, -9.999955000000e-01,-6.127237000000e-02],
                                   [9.999753000000e-01,6.931141000000e-03,-1.143899000000e-03,-3.321029000000e-01]], dtype=np.float32)
        R, t = Tr_velo_to_cam[:, :3], Tr_velo_to_cam[:, 3]
        return R, t


if __name__=='__main__':

    # test_path = './000000.bin'
    # pc = KittiDataReader.read_lidar(test_path)
    # print(pc.shape) # 115384 * 4

    # test_path = './000000.txt'
    # label = KittiDataReader.read_label(test_path)
    # print(label[0])

    R, t = KittiDataReader.read_calibration()
    print(R, t)