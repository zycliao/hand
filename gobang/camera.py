import pyrealsense2 as rs
import numpy as np
import cv2






def detection(image,x,y):
	res=[]
	img1=image
	x_num=300
	y_num=300
	#cv.imwrite("001.jpg", img1)

	#裁剪原图
	#image_crop=src[:,1000:2500]
	image_out=img1[1000:2500,:]
	print(len(img1),len(img1[0]))
	#转灰度处理
	gray = cv.cvtColor(image_out,cv.COLOR_BGR2GRAY)
	circles1 = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1,100,param1=100,param2=32,minRadius=30,maxRadius=150)  

	circles = circles1[0,:,:]

	circles = np.uint16(np.around(circles))
	for i in circles[:]:   
	    cv.circle(image_out,(i[0],i[1]),i[2],(255,0,0),5)  
	    cv.circle(image_out,(i[0],i[1]),2,(255,0,255),10)  
	    #cv.rectangle(image_out,(i[0]-i[2],i[1]+i[2]),(i[0]+i[2],i[1]-i[2]),(255,255,0),5)

	    res.append([int((i[0]-x)/x_num),int((i[1]-y)/y_num)])
	    #print("圆心坐标",i[0],i[1])
	    cv.circle(image_out, (i[0], i[1]), 7, (255, 255, 255), -1)
	return  res
	


def camera(center_array):
	# 配置
	pipeline = rs.pipeline()
	config = rs.config()
	config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
	config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

	# 打开摄像头
	pipeline.start(config)


	center_array_last=center_array
	# Wait for a coherent pair of frames: depth and color
	frames = pipeline.wait_for_frames()
	depth_frame = frames.get_depth_frame()
	color_frame = frames.get_color_frame()


	# Convert images to numpy arrays
	depth_image = np.asanyarray(depth_frame.get_data())
	color_image = np.asanyarray(color_frame.get_data())

	#判定条件
	depth_image[depth_image==0]=100000
	if np.min(depth_image)>30000:
	    center_x_array,center_y_array,center_array=detection(color_image)
	
	return answer=[i for i in center_array if i not in center_array1],center_array
		
		

		
             

     	
'''      
	# Show images
	cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
	cv2.imshow('RealSense', depth_image)
	cv2.imshow('1111', color_image)
	cv2.waitKey(3000)

'''

