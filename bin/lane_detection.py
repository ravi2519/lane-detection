import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def resize_img ( img, ht ):

    width = img.shape[0]
    height = img.shape[1]
    
    show_ht = ht
    show_wt = int((show_ht/height)*width)
    
    img = cv.resize( img, ( show_ht, show_wt ) )

    return img


def filter_white_and_yellow ( img ):
    
    # RGB to HLS
    img = img.astype('uint8')
    hsv = cv.cvtColor( img, cv.COLOR_BGR2HSV )

    # RGB thresholding for white
    # White lower and upper threshold using color_picker.py -> [ 28   1 209] [ 48  21 289]
    lower = np.array( [ 0, 0, 150 ] )
    upper = np.array( [ 40, 50, 240 ] )
    hsv_w = cv.inRange( hsv, lower, upper )
    #hsv_w = cv.bitwise_and( img, img, mask = mask )
    
    # HLS thresholding for yellow
    lower = np.array([20,100,100])
    upper = np.array([30,255,255])
    hsv_y = cv.inRange( hsv, lower, upper )
    #hls_y = cv.bitwise_and( img, img, mask = mask )

    y_nd_w = cv.bitwise_or( hsv_w, hsv_y )
    
    return ( hsv_w, hsv_y, y_nd_w )


def region_of_interest( img, vertices ):
    mask = np.zeros_like( img )

    if len( img.shape ) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv.fillPoly( mask, np.array([vertices], dtype=np.int32), ignore_mask_color )

    masked_image = cv.bitwise_and( img, mask )
    return masked_image


def draw_lines(img, lines, color=[0, 0, 255], thickness=8):
    x_size = img.shape[1]
    y_size = img.shape[0]
    lines_slope_intercept = np.zeros(shape=(len(lines),2))
    for index,line in enumerate(lines):
        for x1,y1,x2,y2 in line:
            slope = (y2-y1)/(x2-x1)
            intercept = y1 - x1 * slope
            lines_slope_intercept[index]=[slope,intercept]
    max_slope_line = lines_slope_intercept[lines_slope_intercept.argmax(axis=0)[0]]
    min_slope_line = lines_slope_intercept[lines_slope_intercept.argmin(axis=0)[0]]
    left_slopes = []
    left_intercepts = []
    right_slopes = []
    right_intercepts = []
    # this gets slopes and intercepts of lines similar to the lines with the max (immediate left) and min
    # (immediate right) slopes (i.e. slope and intercept within x%)
    for line in lines_slope_intercept:
        if abs(line[0] - max_slope_line[0]) < 0.15 and abs(line[1] - max_slope_line[1]) < (0.15 * x_size):
            left_slopes.append(line[0])
            left_intercepts.append(line[1])
        elif abs(line[0] - min_slope_line[0]) < 0.15 and abs(line[1] - min_slope_line[1]) < (0.15 * x_size):
            right_slopes.append(line[0])
            right_intercepts.append(line[1])
    # left and right lines are averages of these slopes and intercepts, extrapolate lines to edges and center*
    # *roughly
    new_lines = np.zeros(shape=(1,2,4), dtype=np.int32)
    if len(left_slopes) > 0:
        left_line = [sum(left_slopes)/len(left_slopes),sum(left_intercepts)/len(left_intercepts)]
        left_bottom_x = (y_size - left_line[1])/left_line[0]
        left_top_x = (y_size*.700 - left_line[1])/left_line[0]
        if (left_bottom_x >= 0):
            new_lines[0][0] =[left_bottom_x,y_size,left_top_x,y_size*.700]
    if len(right_slopes) > 0:
        right_line = [sum(right_slopes)/len(right_slopes),sum(right_intercepts)/len(right_intercepts)]
        right_bottom_x = (y_size - right_line[1])/right_line[0]
        right_top_x = (y_size*.700 - right_line[1])/right_line[0]
        if (right_bottom_x <= x_size):
            new_lines[0][1]=[right_bottom_x,y_size,right_top_x,y_size*.700]


    for line in new_lines:
        for x1,y1,x2,y2 in line:
            cv.line(img, (x1, y1), (x2, y2), color, thickness)

def process_image( img ):

    img1 = img
    
    # filter white and yellow
    ( img_w, img_y, w_nd_y_img ) = filter_white_and_yellow ( img1 )
    
    # Convert to gray scale
    gray_img = cv.cvtColor( img, cv.COLOR_BGR2GRAY )
    subdued_gray = (gray_img / 2).astype('uint8')

    boosted_lanes = cv.bitwise_or(subdued_gray, w_nd_y_img )
    
    # Apply gaussian blur
    
    gaussianblur = cv.GaussianBlur( boosted_lanes, (5,5), 0 )
    
    # Canny edge detection
    
    cannyedge = cv.Canny ( gaussianblur, 100, 200 )
    
    # get the region of interest
    height = cannyedge.shape[1]
    width = cannyedge.shape[0]
    roi = [ (0, width), ( height*.475, width*.575 ), ( height*.525, width*.575 ), ( height, width) ]
    roi_img = region_of_interest( cannyedge, roi )
    
    #cv.imshow('ROI Image',roi_img)
    
    lines = cv.HoughLinesP( roi_img, 3, np.pi/180, 70, minLineLength=70, maxLineGap=250 )

    if lines is None:
        return img
    
    #lines_img = resize_img( img, 600 ) 
    #for line in lines:
    #    for x1,y1,x2,y2 in line:
    #        cv.line(lines_img, (x1, y1), (x2, y2), [0, 255, 0], 1)
    #
    #cv.imshow('Lines Image',lines_img)
    
    line_img = np.zeros( ( roi_img.shape[0], roi_img.shape[1], 3 ), dtype=np.uint8 )
    draw_lines( line_img, lines )
    
    #final_img = resize_img( img, 600 ) 
    final_img = cv.addWeighted( img, 0.8, line_img, 1., 0. )

    return ( final_img, img_w, img_y, boosted_lanes, gaussianblur, cannyedge, line_img ) 

def create_video( inpfile, opfile ):
    cap = cv.VideoCapture( inpfile )
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    video=cv.VideoWriter(opfile,cv.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    
    while(1):
    
        # take each frame
        ret, frame = cap.read()
        if ret == True:
            ( res, img_w, img_y, boosted_lanes, gaussianblur, cannyedge, line_img ) = process_image( frame )
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText( res, '@ravi2519', (10,50), font, 1, (255,255,255),2,cv.LINE_AA )
            video.write( res )
    
            k = cv.waitKey(5) & 0xFF
            if k == 27:
                break
        else:
            break
    
    cap.release()
    video.release()
    
    cv.destroyAllWindows()

def create_image ( inpfile ):
    img = cv.imread( inpfile )
    ( final_img, img_w, img_y, boosted_lanes, gaussianblur, cannyedge, line_img ) = process_image( img )

    plt.subplot(331),plt.imshow(img),plt.title("Original")
    plt.subplot(332),plt.imshow(img_w),plt.title("HSV White")
    plt.subplot(333),plt.imshow(img_y),plt.title("HSL Yellow")
    plt.subplot(334),plt.imshow(boosted_lanes),plt.title("White and Yellow")
    plt.subplot(335),plt.imshow(gaussianblur),plt.title("Gaussian Image")
    plt.subplot(336),plt.imshow(cannyedge),plt.title("Canny Image")
    plt.subplot(337),plt.imshow(line_img),plt.title("Image With Lines")
    plt.subplot(338),plt.imshow(final_img),plt.title("Final Image")
    
    plt.show()


