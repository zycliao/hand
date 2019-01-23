# coding=utf-8
from robotcontrol import *

xxx, yyy, zzz = 0.1882, 0.1407, 0.264

class WuziRobot(Auboi5Robot):
    def __init__(self, logger):
        Auboi5Robot.__init__(self)
        self.logger = logger
        self.zero_radian = (0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000)
        self.init_pos = (0.045555930042795366, -0.6322379766245128, 0.4419410554923796)
        self.init_ori = (0., 1., 0., 0.)
        self.zero_z = 0.178

    def prepare(self, ip='192.168.1.1', port=8899):
        # 创建并打印上下文
        handle = self.create_context()
        self.logger.info("robot.rshd={0}".format(handle))
        result = self.connect(ip, port)

        if result != RobotErrorType.RobotError_SUCC:
            self.logger.info("connect server{0}:{1} failed.".format(ip, port))
            raise RobotErrorType.RobotError_NotLogin
        else:
            # # 重新上电
            # self.robot_shutdown()
            #
            # # 上电
            # self.robot_startup()
            #
            # # 设置碰撞等级
            # self.set_collision_class(7)

            # 设置工具端电源为１２ｖ
            # self.set_tool_power_type(RobotToolPowerType.OUT_12V)

            # 设置工具端ＩＯ_0为输出
            self.set_tool_io_type(RobotToolIoAddr.TOOL_DIGITAL_IO_0, RobotToolDigitalIoDir.IO_OUT)

            # 获取工具端ＩＯ_0当前状态
            tool_io_status = self.get_tool_io_status(RobotToolIoName.tool_io_0)
            self.logger.info("tool_io_0={0}".format(tool_io_status))

            # 设置工具端ＩＯ_0状态
            self.set_tool_io_status(RobotToolIoName.tool_io_0, 1)

            # 获取控制柜用户DI
            io_config = self.get_board_io_config(RobotIOType.User_DI)

            # 输出DI配置
            self.logger.info(io_config)

            # 获取控制柜用户DO
            io_config = self.get_board_io_config(RobotIOType.User_DO)

            # 输出DO配置
            self.logger.info(io_config)

            # 当前机械臂是否运行在联机模式
            self.logger.info("self online mode is {0}".format(self.is_online_mode()))
            joint_status = self.get_joint_status()
            self.logger.info("joint_status={0}".format(joint_status))

            # 初始化全局配置文件
            self.init_profile()

            # 设置关节最大加速度
            self.set_joint_maxacc((1.5, 1.5, 1.5, 1.5, 1.5, 1.5))

            # 设置关节最大加速度
            self.set_joint_maxvelc((1.5, 1.5, 1.5, 1.5, 1.5, 1.5))
            # 获取关节最大加速度
            self.logger.info(self.get_joint_maxacc())

            self.logger.info(self.get_current_waypoint())

    def move_to_init(self):
        logger.info("move to initial position")
        ik_result = self.inverse_kin(self.current_waypoint['joint'], self.init_pos, self.init_ori)
        self.move_joint(ik_result['joint'])

    def move_to_zero(self):
        logger.info("move to zero position")
        self.move_joint(self.zero_radian)

    def move_to_zero_z(self):
        dst_pos = list(self.current_waypoint['pos'])
        dst_pos[2] = self.zero_z
        ik_result = self.inverse_kin(self.current_waypoint['joint'], dst_pos, self.current_waypoint['ori'])
        self.move_joint(ik_result['joint'])

    @property
    def current_waypoint(self):
        return self.get_current_waypoint()


if __name__ == '__main__':
    # 初始化logger
    logger_init()

    # 系统初始化
    WuziRobot.initialize()

    # 创建机械臂控制类
    robot = WuziRobot(logger)
    try:
        robot.prepare()
        robot.set_tool_power_type(power_type=RobotToolPowerType.OUT_0V)
        import time
        time.sleep(5)
        robot.move_to_init()
        robot.move_to_zero_z()
        # robot.set_tool_power_type(power_type=RobotToolPowerType.OUT_24V)
        robot.move_to_init()
        robot.move_to_zero_z()
        robot.set_tool_power_type(power_type=RobotToolPowerType.OUT_24V)
        robot.move_to_init()

        # 断开服务器链接
        robot.disconnect()
    except RobotError, e:
        logger.error("{0} robot Event:{1}".format(robot.get_local_time(), e))

    finally:
        # 断开服务器链接
        if robot.connected:
            # 关闭机械臂
            robot.robot_shutdown()
            # 断开机械臂链接
            robot.disconnect()
        # 释放库资源
        Auboi5Robot.uninitialize()
        logger.info("{0} test completed.".format(Auboi5Robot.get_local_time()))