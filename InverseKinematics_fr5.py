"""
逆向运动学算法,通过
"""

import logging
import math
import numpy as np

from typing import List
# from NFCWebots.kinematics.ForwardKinematics import ForwardKinematics
from NFCWebots.FR5RobotArmLimitDict import FR5RobotArmLimitDict
from NFCWebots.logging_main import logging_main


class InverseKinematics:
    # FR参数 FR5 global  d,a,alpha
    d_FR5 = [0.152, 0, 0, 0.102, 0.102, 0.100]  # 连杆距离d,单位m
    a_FR5 = [0, -0.425, -0.395, 0, 0, 0]  # 连杆长度ai,单位m
    alpha_FR5 = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]  # alpha 连杆扭角α,单位rad

    d = d_FR5
    a = a_FR5
    alpha = alpha_FR5

    def __init__(self, model_type, d6_offset=0.0):
        '''
        初始化逆解运算当中涉及到的DH参数
        :param model_type:  机械臂的型号类型,用于加载对应型号的DH参数信息
        :param d6_offset:  对于在机械臂末端执行器处添加执行器时的垂直高度确认,单位是m,默认是0m
        '''
        # 获取已经配置好的logger   获取logger实例
        self.logger = logging.getLogger("main_log.InverseKinematics.py")

        global d, a, alpha
        d = InverseKinematics.d_FR5
        a = InverseKinematics.a_FR5
        alpha = InverseKinematics.alpha_FR5
        d[5] = d[5] + d6_offset

        self.logger.info(f"InverseKinematics模型类型为: {model_type}")
        self.logger.info(f"InverseKinematics DH参数-d为: {d}")
        self.logger.info(f"InverseKinematics DH参数-a为: {a}")
        self.logger.info(f"InverseKinematics DH参数-alpha为: {alpha}")

    @staticmethod
    def calculate_transformation_matrix(px, py, pz, rx, ry, rz):
        """
        根据末端坐标和位姿信息进行构建变换矩阵
        :param px: x轴坐标
        :param py: y轴坐标
        :param pz: z轴坐标
        :param rx: x轴偏移量
        :param ry: y轴偏移量
        :param rz: z轴偏移量
        :return: 位姿构成的齐次变换矩阵,其中最后一行为固定的0,0,0,1
        """
        cosR = np.cos(rx)
        sinR = np.sin(rx)
        cosP = np.cos(ry)
        sinP = np.sin(ry)
        cosY = np.cos(rz)
        sinY = np.sin(rz)

        rotation_matrix = np.array([
            [cosY * cosP, cosY * sinP * sinR - sinY * cosR, cosY * sinP * cosR + sinY * sinR, px],
            [sinY * cosP, sinY * sinP * sinR + cosY * cosR, sinY * sinP * cosR - cosY * sinR, py],
            [-sinP, cosP * sinR, cosP * cosR, pz],
            [0, 0, 0, 1]
        ])

        return rotation_matrix

    @staticmethod
    def inv_kine(desired_pos):
        """
        逆解算法
        :param desired_pos: 位姿构成的齐次变换矩阵
        :return: 构成逆向运动学解的8组六轴机械臂的旋转角度信息
        """
        th = np.zeros((8, 6))

        nx, ox, ax = desired_pos[0, 0], desired_pos[0, 1], desired_pos[0, 2]
        ny, oy, ay = desired_pos[1, 0], desired_pos[1, 1], desired_pos[1, 2]
        nz, oz, az = desired_pos[2, 0], desired_pos[2, 1], desired_pos[2, 2]
        px, py, pz = desired_pos[0, 3], desired_pos[1, 3], desired_pos[2, 3]

        try:
            for i in range(8):
                m = (d[5] * ay - py)
                n = (ax * d[5] - px)
                psi = np.arctan2(m, n)
                phiz = np.arctan2(d[3], np.sqrt(m ** 2 + n ** 2 - d[3] ** 2))
                phif = np.arctan2(d[3], -np.sqrt(m ** 2 + n ** 2 - d[3] ** 2))
                phiz1 = psi - phiz
                phiz2 = psi - phif

                if i < 4:
                    th[i, 0] = InverseKinematics.complementAngle(phiz1)
                else:
                    th[i, 0] = InverseKinematics.complementAngle(phiz2)

                if i < 4:
                    c51 = ax * np.sin(th[i, 0]) - ay * np.cos(th[i, 0])
                    theta5z = np.arccos(c51)
                    if i < 2:
                        th[i, 4] = InverseKinematics.complementAngle(theta5z)
                    else:
                        th[i, 4] = InverseKinematics.complementAngle(-theta5z)
                else:
                    c52 = ax * np.sin(th[i, 0]) - ay * np.cos(th[i, 0])
                    theta52z = np.arccos(c52)
                    if i < 6:
                        th[i, 4] = InverseKinematics.complementAngle(theta52z)
                    else:
                        th[i, 4] = InverseKinematics.complementAngle(-theta52z)

                mm = nx * np.sin(th[i, 0]) - ny * np.cos(th[i, 0])
                nn = ox * np.sin(th[i, 0]) - oy * np.cos(th[i, 0])
                theta6 = np.arctan2(mm, nn) - np.arctan2(np.sin(th[i, 4]), 0)
                th[i, 5] = InverseKinematics.complementAngle(theta6)

                ncns = nx * np.cos(th[i, 0]) + ny * np.sin(th[i, 0])
                ocos = ox * np.cos(th[i, 0]) + oy * np.sin(th[i, 0])
                acas = ax * np.cos(th[i, 0]) + ay * np.sin(th[i, 0])
                ocns = oz * np.cos(th[i, 5]) + nz * np.sin(th[i, 5])

                mmm = d[4] * (np.sin(th[i, 5]) * ncns + np.cos(th[i, 5]) * ocos) - d[5] * acas + px * np.cos(
                    th[i, 0]) + py * np.sin(th[i, 0])
                nnn = d[4] * ocns + pz - d[0] - az * d[5]
                theta3 = np.arccos((mmm ** 2 + nnn ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
                if i % 2 == 0:
                    th[i, 2] = InverseKinematics.complementAngle(theta3)
                else:
                    th[i, 2] = InverseKinematics.complementAngle(-theta3)

                s21 = (a[2] * np.cos(th[i, 2]) + a[1]) * nnn - a[2] * np.sin(th[i, 2]) * mmm
                s22 = a[1] ** 2 + a[2] ** 2 + 2 * a[1] * a[2] * np.cos(th[i, 2])
                s2 = s21 / s22
                c21 = mmm + a[2] * np.sin(th[i, 2]) * s2
                c22 = a[2] * np.cos(th[i, 2]) + a[1]
                c2 = c21 / c22
                theta2 = np.arctan2(s2, c2)
                th[i, 1] = InverseKinematics.complementAngle(theta2)

                theta4a = -np.sin(th[i, 5]) * ncns - np.cos(th[i, 5]) * ocos
                theta4 = np.arctan2(theta4a, ocns) - th[i, 1] - th[i, 2]
                th[i, 3] = InverseKinematics.complementAngle(theta4)

        except Exception as e:
            print("InverseKinematics逆解算法计算时出现异常,原因 >>>> ", e)

        return th

    @staticmethod
    def complementAngle(theta):
        """
        角度取全补的最小值
        :param theta: 单位为rad,当前计算得出的旋转弧度
        :return: 根据2π进行比较取全补交的最小值
        """
        theta_mod = theta
        if abs(theta_mod) > 2 * np.pi:
            if theta_mod > 0:
                theta_mod -= 2 * np.pi
            else:
                theta_mod += 2 * np.pi

        if abs(theta_mod) > np.pi:
            if theta_mod < 0:
                theta_mod += 2 * np.pi
            else:
                theta_mod -= 2 * np.pi

        return theta_mod

    @staticmethod
    def complementAngle(angle):
        # 补全角度到范围[-pi, pi]
        return np.arctan2(np.sin(angle), np.cos(angle))

    @staticmethod
    def inverseCalculate(logger, unit_type, input_xyz_and_rxryrz, baseXOffset, baseYOffset, baseZOffset, endXOffset,
                         endYOffset,
                         endZOffset):
        """
        计算逆向运动学操作
        :param input_xyz_and_rxryrz: 坐标与位姿列表信息,单位是mm和rad
        :unit_type UR: mm_rad,FR: mm_du,webots: m_rad m_du
        :return:
        """
        global px, py, pz, rx, ry, rz
        if unit_type == "mm_rad":  # UR
            # 计算要求单位 m
            px = input_xyz_and_rxryrz[0] / 1000.0
            py = input_xyz_and_rxryrz[1] / 1000.0
            pz = input_xyz_and_rxryrz[2] / 1000.0
            # 计算要求单位 rad
            rx = input_xyz_and_rxryrz[3]
            ry = input_xyz_and_rxryrz[4]
            rz = input_xyz_and_rxryrz[5]
        elif unit_type == "mm_du":  # FR
            # 计算要求单位 m
            px = input_xyz_and_rxryrz[0] / 1000.0
            py = input_xyz_and_rxryrz[1] / 1000.0
            pz = input_xyz_and_rxryrz[2] / 1000.0
            # 计算要求单位 rad
            rx = math.radians(input_xyz_and_rxryrz[3])
            ry = math.radians(input_xyz_and_rxryrz[4])
            rz = math.radians(input_xyz_and_rxryrz[5])
        elif unit_type == "m_rad":  # webots
            # 计算要求单位 m
            px = input_xyz_and_rxryrz[0]
            py = input_xyz_and_rxryrz[1]
            pz = input_xyz_and_rxryrz[2]
            # 计算要求单位 rad
            rx = input_xyz_and_rxryrz[3]
            ry = input_xyz_and_rxryrz[4]
            rz = input_xyz_and_rxryrz[5]
        elif unit_type == "m_du":
            # 计算要求单位 m
            px = input_xyz_and_rxryrz[0]
            py = input_xyz_and_rxryrz[1]
            pz = input_xyz_and_rxryrz[2]
            # 计算要求单位 rad
            rx = math.radians(input_xyz_and_rxryrz[3])
            ry = math.radians(input_xyz_and_rxryrz[4])
            rz = math.radians(input_xyz_and_rxryrz[5])

        # 根据单位得到的位姿信息，变换为对应的4 * 4 的旋转矩阵
        transformation_matrix_result = InverseKinematics.calculate_transformation_matrix(px, py, pz, rx, ry, rz)
        # 根据4*4的旋转矩阵得到对应的八组逆运算的解，然后再根据限位或者最优解的情况，得到计算出合适的角度，进行控制旋转机械臂
        result = InverseKinematics.inv_kine(transformation_matrix_result)

        resultArrays = []
        # print()
        # print("=========================================")
        for i in range(8):
            logger.info(f"[第{i + 1}组逆向运动学解] --> 即六自由度机械臂每个关节转动的角度: [单位: °]")

            doubles = []
            for j in range(6):
                theta_degrees = np.degrees(result[i][j])
                theta_rad = result[i][j]

                # print(f"theta{j + 1}={round(theta_degrees,4)}°[{round(theta_rad,2)} rad]", end=', ')
                # print(f"{round(theta_degrees, 4)}°", end=', ')
                # print(f"theta{j + 1}={round(theta_rad,2)} rad", end=', ')
                doubles.append(theta_degrees)
            logger.info(doubles)
            # 使用 any 函数和 isnan 函数检查是否有 NaN 元素
            if any(np.isnan(d) for d in doubles):
                print()
                logger.info(">>>>>>>>>>>>    逆向解中存在NaN值,无法进行正向解验证    <<<<<<<<<<<<")
                logger.info("====================================================")
                print()
                continue
            else:
                resultArrays.append(doubles)

            print()
            # logger.info("对逆向解进行正向学验证:入参单位为°")
            # # 这里添加正向运动学的验证逻辑
            # logger.info("====================================================")
            # transformation_matrix = ForwardKinematics.robot_arm_forward_base_end_position_calculator(doubles, unit_type,
            #                                                                                          baseXOffset,
            #                                                                                          baseYOffset,
            #                                                                                          baseZOffset,
            #                                                                                          endXOffset,
            #                                                                                          endYOffset,
            #                                                                                          endZOffset)
            # ForwardKinematics.forward_kinematics_output(transformation_matrix)
            # ForwardKinematics.calculate_forward_kinematics_ref_rpy(transformation_matrix)
            print()

        return resultArrays

    @staticmethod
    def robot_arm_inverse_base_end_position_calculator(logger, unit_type, inputXYZAndRxRyRz,
                                                       baseXOffset=0.0, baseYOffset=0.0, baseZOffset=0.0,
                                                       endXOffset=0.0, endYOffset=0.0, endZOffset=0.0):
        '''
        基坐标相对于原点的偏移量，用于计算基坐标不在原始点的情况
        :param logger:
        :param inputXYZAndRxRyRz:
        :param baseXOffset:
        :param baseYOffset:
        :param baseZOffset:
        :param endXOffset:
        :param endYOffset:
        :param endZOffset:
        :param unit_type:
        :return:
        '''
        logger.info(f"robotArmInversePositionCalculator 当前参与逆运算的入参为:{inputXYZAndRxRyRz},单位为:{unit_type}")
        # 基坐标的偏移量调整
        inputXYZAndRxRyRz[0] -= baseXOffset
        inputXYZAndRxRyRz[1] -= baseYOffset
        inputXYZAndRxRyRz[2] -= baseZOffset

        # 末端执行器的偏移量调整
        inputXYZAndRxRyRz[0] -= endXOffset
        inputXYZAndRxRyRz[1] -= endYOffset
        inputXYZAndRxRyRz[2] -= endZOffset

        logger.info(
            f"进行基坐标base的变换之后与末端法兰执行偏移量变换之后，当前参与逆运算的入参为:{inputXYZAndRxRyRz},单位为:{unit_type}")

        return InverseKinematics.inverseCalculate(logger, unit_type, inputXYZAndRxRyRz, baseXOffset, baseYOffset,
                                                  baseZOffset, endXOffset, endYOffset, endZOffset)

    @staticmethod
    def limit_principle(logger, inverse_kinematics: List[List[float]], initPose) -> List[List[float]]:
        """
        限位原则函数
        :param inverse_kinematics:
        :return:
        """
        print()
        logger.info(
            "………………………………………………………………………限位原则: 排除超出关节角度限制范围的解。每个关节都有其角度范围,超出这个范围的解可能不符合机器人的运动能力。………………………………………………………………………………………………………")
        filtered_list = []
        count = 1
        logger.info("逆向运动学,工具位姿求解关节位置,根据FR的每个关节运动范围,进行计算符合的输出结果:")
        for doubles_value in inverse_kinematics:
            is_available = FR5RobotArmLimitDict.checkAngleIsAvailableCommon(logger, doubles_value, initPose)
            logger.info(f"第{count}组解:{'符合   ' if is_available else '不符合   '} :{doubles_value}")
            # end=""参数在logger.info中是不必要的,因为logger.info不会像在print函数中那样在字符串末尾添加换行符。logger.info会自动为每次记录添加换行符（除非在日志配置中特别指定了不同的格式）。
            # print(f"第{count}组解:{'符合   ' if is_available else '不符合   '}", end="")
            if is_available:
                filtered_list.append(doubles_value)
            count += 1
            print()

        # print()
        # count = 1
        # for doubles_value in filtered_list:
        #     print(f"第{count}次  -->  ", end="")
        #     for a_double in doubles_value:
        #         print(f"{round(a_double, 4)}°   ", end="")
        #     count += 1
        #     print()

        return filtered_list

    @staticmethod
    def minimumRotationAngle(logger, limitPrinciple, lastJointsDegrees):
        """
        根据逆解得到的8组解与当前机械臂六轴停留的角度进行计算比较,获取最小转动角度
        :param limitPrinciple: 当前转动的角度
        :param lastJointsDegrees: 机械臂停留的六轴转动角度信息,也就是相对运动
        :return:
        """
        doubleMap = InverseKinematics.sumOfRotationAngles(limitPrinciple, lastJointsDegrees)
        angleMin = min(doubleMap.items(), key=lambda x: x[0])[1]
        # logger.info("逆向运动学,工具位姿求解关节位置,参考指定关节位置之:[当前位置到达目标位置转动角度之和最小]的解为:")
        # for angle in angleMin:
        #     logger.info(f"{round(angle, 4)} '°'")
        # logger.info()
        return angleMin

    @staticmethod
    def sumOfRotationAngles(limitPrinciple, lastJointsDegrees):
        """
        计算需要制定转动角度与当前停留的角度之间的差值,然后进行求和操作,比较之后,取当中的最小值
        :param limitPrinciple: 当前转动的角度
        :param lastJointsDegrees: 机械臂停留的六轴转动角度信息,也就是相对运动
        :return:
        """
        resuleMap = {}
        a = 1
        for solution in limitPrinciple:
            angleSum = sum(abs(sol - last) for sol, last in zip(solution, lastJointsDegrees))
            # print("第", a, "次  -->  计算出的转动角度之和为:", angleSum)
            resuleMap[angleSum] = solution
            a += 1
        return resuleMap

    @staticmethod
    def shortestPathPrinciple(limitPrinciple, lastJointsDegrees):
        # print(
        #     "\n………………………………………………………………………最短路径原则: 选择距离上一次关节角度最小的解………………………………………………………………………………………………………")
        optimalSolution = InverseKinematics.selectOptimalSolution(limitPrinciple, lastJointsDegrees)
        # print("逆向运动学,工具位姿求解关节位置,参考指定关节位置之:[最短路径原则]的解为:")
        # for angle in optimalSolution:
        #     print(round(angle, 4), "°", end="   ")
        # print()
        return optimalSolution

    @staticmethod
    def selectOptimalSolution(solutions, lastJoints):
        minDistance = float("inf")
        optimalSolution = None
        a = 1
        for solution in solutions:
            distance = InverseKinematics.calculateDistance(solution, lastJoints)
            # print("第", a, "次  -->  计算出的距离为:", distance)
            if distance < minDistance:
                minDistance = distance
                optimalSolution = solution
            a += 1
        return optimalSolution

    @staticmethod
    def calculateDistance(solution1, lastJoints):
        distance = sum((sol - last) ** 2 for sol, last in zip(solution1, lastJoints))
        return math.sqrt(distance)


if __name__ == "__main__":
    # 假设你的JSON配置文件路径为'logging_config.json'
    logging_main.setup_logging(default_path="config/inner_logging.json")

    """
        根据如入参类型,确定启用哪个类型的机械臂对应的DH参数信息
        UR10e 或 Fr5Robot
    """

    # ====================================FR5===========================================
    # 案例一：
    # mm du
    modol_name = "Fr5Robot"
    unit_type = "mm_du"
    # mm du
    inputFR5LastBase = [-102.879, -289.639, 299.966, -90, -0.672, 179.689]
    # [-90.1015286822664, -98.59478463733666, -152.75649079275146, 71.35127543008811, 89.7905286822664, -0.6719999999999913]
    # modol_name = const.WEBOTS_ROBOT_ARM_FR5
    # unit_type = "mm_du"
    # 使用的D-H参数
    # d_FR5 = [0.152, 0, 0, 0.102, 0.102, 0.100]  # 连杆距离d,单位m
    # a_FR5 = [0, -0.425, -0.395, 0, 0, 0]  # 连杆长度ai,单位m
    # alpha_FR5 = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]  # alpha 连杆扭角α,单位rad

    inv = InverseKinematics(modol_name)
    # fk = ForwardKinematics(modol_name)

    inputXYZAndRxRyRz = inputFR5LastBase
    inv.logger.info(
        "逆向运动学: 已知末端执行器相对于基坐标系的位置和姿态,求对应的六个关节角度. [入参的坐标单位: mm ; 姿态单位: °]")

    # inv.logger.info(", ".join(map(str, inputXYZAndRxRyRz)))

    # 相当于基坐标偏移原始点的位置信息
    baseXOffset = 0.0  # 单位mm
    baseYOffset = 0.0  # 单位mm
    baseZOffset = 0.0  # 单位mm

    # 末端法兰处添加的执行器未在执行器的正中间，假设的激光笔相对于末端执行器的偏移量,dx, dy, dz 是x, y, z轴上的偏移量
    endXOffset = 0.0  # 单位mm
    endYOffset = 0.0  # 单位mm
    endZOffset = 0.0  # 单位mm

    inverseKinematics = InverseKinematics.robot_arm_inverse_base_end_position_calculator(inv.logger, unit_type,
                                                                                         inputXYZAndRxRyRz,
                                                                                         baseXOffset, baseYOffset,
                                                                                         baseZOffset,
                                                                                         endXOffset, endYOffset,
                                                                                         endZOffset)
    optimalSolution = []
    if not inverseKinematics:
        inv.logger.info("逆解运算之后无解")
    else:
        # if const.WEBOTS_ROBOT_ARM_FR5 == modol_name:
        # inverseKinematics = inv.limit_principle(inv.logger, inverseKinematics, 'l')
        # if not inverseKinematics:
        #     print("限位无解")
        lastJointsDegrees = [-90.102, -98.594, -152.756, 71.35, 89.792, -0.672]
        # optimalSolution = InverseKinematics.shortestPathPrinciple(inverseKinematics, lastJointsDegrees)
        optimalSolution = InverseKinematics.minimumRotationAngle(inv.logger, inverseKinematics, lastJointsDegrees)
        print(optimalSolution)
