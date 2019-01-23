import pyrealsense2 as rs
import numpy as np
import cv2

num = 0

def print_coord(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print x
        print y
    if event == cv2.EVENT_RBUTTONDOWN:
        global num
        # cv2.imwrite('./images/'+str(num)+'.jpg', params[0])
        num += 1


if __name__ == '__main__':
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    while 1:

        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        cv2.imwrite('ref2.jpg', color_image)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('RealSense', print_coord, color_image)
        cv2.imshow('RealSense', images)
        k = cv2.waitKey(1)
        if k == ord('q'):
            cv2.destroyAllWindows()
            break