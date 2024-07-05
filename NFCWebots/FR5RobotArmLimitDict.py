"""
对于FR5机械臂每个关节可以运动的正向与反向可到达的范围编写，通过
"""
import sys

sys.path.append('/home/wedo/opt/fr5eg')

import logging

from NFCWebots.kinematics.logging_main import logging_main


class FR5RobotArmLimitDict:
    JOINT_1_RANGE_MOTION_ANGLE = (175, -175)  # 右是负值，左是正值
    JOINT_21_RANGE_MOTION_ANGLE = (85, -265)
    JOINT_22_RANGE_MOTION_ANGLE = (0, -180)  # 平面以上
    JOINT_3_RANGE_MOTION_ANGLE = (160, -160)  # 与第二个关节取相反值
    JOINT_4_RANGE_MOTION_ANGLE = (85, -265)
    JOINT_5_RANGE_MOTION_ANGLE = (175, -175)
    JOINT_6_RANGE_MOTION_ANGLE = (175, -175)

    '''
    姿态范围：单位:°
    rx:[24,163]
    ry:[1.6,90]
    rz:[90,-110]
    '''

    def __init__(self, positiveNumber, negativeNumber):
        # 获取已经配置好的logger   获取logger实例
        self.logger = logging.getLogger("main_log.fr5robotarmlimit.py")

        self.positiveNumber = positiveNumber
        self.negativeNumber = negativeNumber

    @staticmethod
    def checkAngleIsAvailableCommon(logger, jointDoubles, initPose):
        if initPose:
            # logger.info("使用左右臂方式计算限位")
            return FR5RobotArmLimitDict.checkAngleIsAvailableRightOrLeft(logger, jointDoubles, initPose)
        else:
            # logger.info("使用非左右臂方式计算限位")
            return FR5RobotArmLimitDict.checkAngleIsAvailable(jointDoubles)

    @staticmethod
    def checkAngleIsAvailableRightOrLeft(logger, jointDoubles, initPose):
        """
        :param jointDoubles: 逆解对应的八组解
        :param initPose: 表示左右姿态，第一个关节左代表正数范围，右代表负数范围
        :return:
        """
        result = False
        joint1, joint2, joint3, joint4, joint5, joint6 = jointDoubles
        # 第一关节:需要根据使用左臂还是右臂进行确认初始移动的范围
        if initPose == 'r':
            if joint1 > 0:
                joint1 = -360 + joint1
        elif initPose == 'l':
            if joint1 < 0:
                joint1 = 360 + joint1
        joint1CheckResult = FR5RobotArmLimitDict.checkJoint1Range(joint1)
        # logger.info(
        #     f"根据指定的{'右[负数范围]' if initPose == 'r' else '左[正数范围]'}臂，确定对应基坐标第一个关节的转动角度:{joint1},验证限位结果为:{joint1CheckResult}")

        # 第二关节:需要注意取值应该在地平面之上，范围为
        joint2CheckResult = FR5RobotArmLimitDict.checkJoint22Range(joint2)
        if not joint2CheckResult:
            joint2 = -360.0 + joint2 if joint2 > 0.0 else 360 + joint2
            joint2CheckResult = FR5RobotArmLimitDict.checkJoint22Range(joint2)
            if joint2CheckResult:
                jointDoubles[1] = joint2
        # print(f"第二个关节需要在地平面之上进行运动:{joint2},验证限位结果为:{joint2CheckResult}")

        # 第三个关节需要与第二个关机
        joint3CheckResult = FR5RobotArmLimitDict.checkJoint3Range(joint3)
        if not joint3CheckResult:
            joint3 = -360.0 + joint3 if joint3 > 0.0 else 360 + joint3
            joint3CheckResult = FR5RobotArmLimitDict.checkJoint3Range(joint3)
            if joint3CheckResult:
                jointDoubles[2] = joint3
        joint4CheckResult = FR5RobotArmLimitDict.checkJoint4Range(joint4)
        if not joint4CheckResult:
            joint4 = -360.0 + joint4 if joint4 > 0.0 else 360 + joint4
            joint4CheckResult = FR5RobotArmLimitDict.checkJoint4Range(joint4)
            if joint4CheckResult:
                jointDoubles[3] = joint4
        joint5CheckResult = FR5RobotArmLimitDict.checkJoint5Range(joint5)
        if not joint5CheckResult:
            joint5 = -360.0 + joint5 if joint5 > 0.0 else 360 + joint5
            joint5CheckResult = FR5RobotArmLimitDict.checkJoint5Range(joint5)
            if joint5CheckResult:
                jointDoubles[4] = joint5
        joint6CheckResult = FR5RobotArmLimitDict.checkJoint6Range(joint6)
        if not joint6CheckResult:
            joint6 = -360.0 + joint6 if joint6 > 0.0 else 360 + joint6
            joint6CheckResult = FR5RobotArmLimitDict.checkJoint6Range(joint6)
            if joint6CheckResult:
                jointDoubles[5] = joint6
        rs = joint1CheckResult and joint2CheckResult and joint3CheckResult and joint4CheckResult and joint5CheckResult and joint6CheckResult
        return rs

    @staticmethod
    def checkAngleIsAvailable(jointDoubles):
        rs = False
        joint1, joint2, joint3, joint4, joint5, joint6 = jointDoubles
        joint1CheckResult = FR5RobotArmLimitDict.checkJoint1Range(joint1)
        if not joint1CheckResult:
            joint1 = -360.0 + joint1 if joint1 > 0.0 else 360 + joint1
            joint1CheckResult = FR5RobotArmLimitDict.checkJoint1Range(joint1)
            if joint1CheckResult:
                jointDoubles[0] = joint1
        joint2CheckResult = FR5RobotArmLimitDict.checkJoint22Range(joint2)
        if not joint2CheckResult:
            joint2 = -360.0 + joint2 if joint2 > 0.0 else 360 + joint2
            joint2CheckResult = FR5RobotArmLimitDict.checkJoint22Range(joint2)
            if joint2CheckResult:
                jointDoubles[1] = joint2
        joint3CheckResult = FR5RobotArmLimitDict.checkJoint3Range(joint3)
        if not joint3CheckResult:
            joint3 = -360.0 + joint3 if joint3 > 0.0 else 360 + joint3
            joint3CheckResult = FR5RobotArmLimitDict.checkJoint3Range(joint3)
            if joint3CheckResult:
                jointDoubles[2] = joint3
        joint4CheckResult = FR5RobotArmLimitDict.checkJoint4Range(joint4)
        if not joint4CheckResult:
            joint4 = -360.0 + joint4 if joint4 > 0.0 else 360 + joint4
            joint4CheckResult = FR5RobotArmLimitDict.checkJoint4Range(joint4)
            if joint4CheckResult:
                jointDoubles[3] = joint4
        joint5CheckResult = FR5RobotArmLimitDict.checkJoint5Range(joint5)
        if not joint5CheckResult:
            joint5 = -360.0 + joint5 if joint5 > 0.0 else 360 + joint5
            joint5CheckResult = FR5RobotArmLimitDict.checkJoint5Range(joint5)
            if joint5CheckResult:
                jointDoubles[4] = joint5
        joint6CheckResult = FR5RobotArmLimitDict.checkJoint6Range(joint6)
        if not joint6CheckResult:
            joint6 = -360.0 + joint6 if joint6 > 0.0 else 360 + joint6
            joint6CheckResult = FR5RobotArmLimitDict.checkJoint6Range(joint6)
            if joint6CheckResult:
                jointDoubles[5] = joint6
        rs = joint1CheckResult and joint2CheckResult and joint3CheckResult and joint4CheckResult and joint5CheckResult and joint6CheckResult
        return rs

    @staticmethod
    def checkJoint1Range(joint1):
        return FR5RobotArmLimitDict.JOINT_1_RANGE_MOTION_ANGLE[0] >= joint1 >= \
            FR5RobotArmLimitDict.JOINT_1_RANGE_MOTION_ANGLE[1]

    @staticmethod
    def checkJoint21Range(joint2):
        return FR5RobotArmLimitDict.JOINT_21_RANGE_MOTION_ANGLE[0] >= joint2 >= \
            FR5RobotArmLimitDict.JOINT_21_RANGE_MOTION_ANGLE[1]

    @staticmethod
    def checkJoint22Range(joint2):
        return FR5RobotArmLimitDict.JOINT_22_RANGE_MOTION_ANGLE[0] >= joint2 >= \
            FR5RobotArmLimitDict.JOINT_22_RANGE_MOTION_ANGLE[1]

    @staticmethod
    def checkJoint3Range(joint3):
        return FR5RobotArmLimitDict.JOINT_3_RANGE_MOTION_ANGLE[0] >= joint3 >= \
            FR5RobotArmLimitDict.JOINT_3_RANGE_MOTION_ANGLE[1]

    @staticmethod
    def checkJoint4Range(joint4):
        return FR5RobotArmLimitDict.JOINT_4_RANGE_MOTION_ANGLE[0] >= joint4 >= \
            FR5RobotArmLimitDict.JOINT_4_RANGE_MOTION_ANGLE[1]

    @staticmethod
    def checkJoint5Range(joint5):
        return FR5RobotArmLimitDict.JOINT_5_RANGE_MOTION_ANGLE[0] >= joint5 >= \
            FR5RobotArmLimitDict.JOINT_5_RANGE_MOTION_ANGLE[1]

    @staticmethod
    def checkJoint6Range(joint6):
        return FR5RobotArmLimitDict.JOINT_6_RANGE_MOTION_ANGLE[0] >= joint6 >= \
            FR5RobotArmLimitDict.JOINT_6_RANGE_MOTION_ANGLE[1]


if __name__ == '__main__':
    # 假设你的JSON配置文件路径为'logging_config.json'
    logging_main.setup_logging(default_path="../kinematics/config/inner_logging.json")

    fr5 = FR5RobotArmLimitDict(30, 40)
    print(f"输出对应的信息为:{FR5RobotArmLimitDict.checkJoint6Range(30)}")
    fr5.logger.info(f"输出对应的信息为:{FR5RobotArmLimitDict.checkJoint6Range(30)}")

    a = None
    print(a is None)
    print(a is not None)
    print(a  != None)
    print(a  == None)
