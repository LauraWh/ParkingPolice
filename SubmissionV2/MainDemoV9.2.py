#Parking Police
#written by Leon Harmon, Moya Clarke & Laura Whelan


#[1]A. Chaudhary and S. Mayne, "How to convert a nested list into a one-dimensional list in Python?", Stack Overflow, 2013. 
#[Online]. Available: https://stackoverflow.com/questions/17485747/how-to-convert-a-nested-list-into-a-one-dimensional-list-in-python. [Accessed: 20- Nov- 2020].

#===================================================
#Import libraries
#===================================================
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui  as  egui
import os
from collections import Iterable
#===================================================
#0.5 Declare functions
#===================================================
print('')
print('0.5. Declaring Functions...')

#[1]
def flatten(lis):
     for item in lis:
         if isinstance(item, Iterable) and not isinstance(item, str):
             for x in flatten(item):
                 yield x
         else:        
             yield item

#===================================================
#1. Read in Input Images
#===================================================
print('1. Read in Images...')

#Empty=egui.fileopenbox()#choose image
Empty = cv2.imread('EmptyA.jpg')
#Full=egui.fileopenbox()#choose image
Full = cv2.imread('img6.jpg') #use this to test wasted output
#Full = cv2.imread('LeonTest3.jpg') #use this to test alternative output
Full_original=Full.copy()

#===================================================
#2. Grayscale Image and Thresholding
#===================================================
print('2. Grayscale Images...')#Bug Check Print

EmptyG=cv2.cvtColor(Empty,cv2.COLOR_BGR2GRAY) # Convert Empty car park to grayscale
FullG=cv2.cvtColor(Full,cv2.COLOR_BGR2GRAY) # Convert Full car park to grayscale

print('2.5. Thresholding...')#Bug Check Print

TEO, EThreshO = cv2.threshold(EmptyG,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU) # Thresholding Empty car park
TE, EThresh = cv2.threshold(EmptyG,0,255,cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU) # Inverted Thresholding Empty car park
TF, FThresh = cv2.threshold(FullG,0,255,cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU) # Thresholding full car park
#===================================================
#3. Line Detection Empty Carpark 
#===================================================
print('3. Line Detection...')#Bug Check Print

h = Full.shape[0] #image height(Y axis) in pixels
w = Full.shape[1] #image width(X axis) in pixels

if h>w: #Choose longer side as a base length
    l=w
else:
    l=h

edges = cv2.Canny(EThreshO,50,255,apertureSize = 7) # Edge Detection

lines = cv2.HoughLinesP(edges,
                        1,np.pi/180,80,
                        minLineLength=l*0.3,
                        maxLineGap=l*0.05) # Line detection from edges

if lines is not None: # only do this if there is lines
    for line in lines: # Draw lines
        x1,y1,x2,y2 = line[0]
        cv2.line(Full,
                (x1,y1),
                (x2,y2),
                (255,0,0),2)
    
    #===================================================
    #4. Detect Car Park Area From Lines
    #===================================================
    # Find top left and bottom right car park corners
    # (by finding the smallest and largest x and y values)
    print('4. Car Park Area Detection...')#Bug Check Print
    
    n=len(lines)
    for i in range(n):
        if lines[i][0][0]<=x1:
            x1=lines[i][0][0]
    #print('l ',x1)
    for i in range(n):
        if lines[i][0][2]>=x2:
            x2=lines[i][0][2]
    #print('r ',x2)
    for i in range(n):
        if lines[i][0][1]>=y1:
            y1=lines[i][0][1]
    #print('d ',y1)
    for i in range(n):
        if lines[i][0][3]<=y2:
            y2=lines[i][0][3]
    #================================================
    

#===================================================
#5. Draw Carpark Rectangle (Visual only)
#===================================================   
print('5. Draw Car Parking Area...')#Bug Check Print 

# cv2.rectangle(Full, #Draw Rectangle from corners
            # pt1=(x1,y1), 
            # pt2=(x2,y2),
            # color=(0,255,0),
            # thickness=3) 
            
#===================================================
#6. Crop All Images To Carpark Area
#===================================================  
print('6. Crop to Car Parking Area...')#Bug Check Print

FullC=Full[y2:y1,x1:x2]
EmptyC=Empty[y2:y1,x1:x2]
EThreshC=EThresh[y2:y1,x1:x2]
EThreshOC=EThreshO[y2:y1,x1:x2] 
FThreshC=FThresh[y2:y1,x1:x2]    

#===================================================
#7. Subtract Full Image threshold From Empty Carpark Threshold
#=================================================== 
print('7. Subtracting Full Image Threshold From Empty Carpark Threshold...')#Bug Check Print

Subtract=FThreshC-EThreshC
#Add=FThreshC+EThreshC
#Multiply=FThreshC*EThreshC
#Divide=FThreshC/EThreshC

#===================================================
#8. Plot Section 7
#===================================================
print('8. Plotting Previous Section...')#Bug Check Print

# plt.figure(1)
# plt.subplot(2,3,1)
# plt.imshow(FThreshC,cmap='gray')
# plt.title('FThreshC ')
# plt.subplot(2,3,2)
# plt.imshow(EThreshC,cmap='gray')
# plt.title('EThreshC ')
# plt.subplot(2,3,3)
# plt.imshow(Subtract,cmap='gray')
# plt.title('Subtract ')
# plt.subplot(2,3,4)
# plt.imshow(Add,cmap='gray')
# plt.title('Add ')
# plt.subplot(2,3,5)
# plt.imshow(Multiply,cmap='gray')
# plt.title('Multiply ')
# plt.subplot(2,3,6)
# plt.imshow(Divide,cmap='gray')
# plt.title('Divide')
         
#===================================================
#9. Prepare for Contours
#===================================================
print('9. Preparing for Contours...')#Bug Check Print

h = EThreshOC.shape[0] #cropped image height(Y axis) in pixels
w = EThreshOC.shape[1] #cropped image width(X axis) in pixels

Blank = np.zeros([h,w], np.uint8)
Blank1 = cv2.cvtColor( Blank , cv2.COLOR_GRAY2BGR )
Blank2 = Blank1.copy()
Blank3 =Blank1.copy()
Blank4 =Blank1.copy()

shape =cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))

#===================================================
#10. Apply Contours to Cars
#===================================================
print('10. Applying Car Contours...')#Bug Check Print

Cars_isolated =cv2.morphologyEx(Subtract,cv2.MORPH_OPEN,shape)
Car_contours,_ = cv2.findContours(Cars_isolated, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
Car_count=Car_contours[0]
car1=[]
car2=[]
car3=[]
car4=[]
car5=[]
car6=[]

print('10.5. Locating Each Car Individually...')#Bug Check Print

i=1
for Car_count in Car_contours: 
    Car_approx = cv2.approxPolyDP(Car_count, 0.003 * cv2.arcLength(Car_count, True), True)
    if i==1:
        car1.append(Car_approx)
    elif i==2:
        car2.append(Car_approx)
    elif i==3:
        car3.append(Car_approx)
    elif i==4:
        car4.append(Car_approx)
    elif i==5:
        car5.append(Car_approx)  
    elif i==6:
        car6.append(Car_approx)         
    i=i+1

#===================================================
#11. Apply Contours to Parking Area
#===================================================
print('11. Applying Contours to Parking Area...')#Bug Check Print

Lines_isolated =cv2.morphologyEx(EThreshC,cv2.MORPH_OPEN,shape)
Line_contours,_ = cv2.findContours(Lines_isolated, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
Line_contours,_ = cv2.findContours(Lines_isolated,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)

#===================================================
#12. Draw Contours and Combine
#===================================================
print('12. Drawing Contours and Combining Them...')#Bug Check Print

Cars_contoured = cv2.drawContours(Blank1, Car_contours, contourIdx=-1,color=(0,255,0), thickness=5) 
Lines_contoured = cv2.drawContours(Blank2, Line_contours, contourIdx=-1,color=(255,0,255), thickness=5) 
Combined_contoured=Cars_contoured + Lines_contoured

#print('Plotting Contours...')#Bug Check Print
# plt.figure(2)
# plt.subplot(1,3,1)
# plt.imshow(Cars_contoured)
# plt.title('cars_contoured')
# plt.subplot(1,3,2)
# plt.imshow(Lines_contoured)
# plt.title('Lines_contoured')
# plt.subplot(1,3,3)
# plt.imshow(Combined_contoured)
# plt.title('Combined_contoured')

#===================================================
#13. Greyscale Combined_contoured Image, threshold to find intersect coords
#===================================================
print('13. Searching Car-Line Intersection Co-ordinates...')#Bug Check Print

final_g=cv2.cvtColor(Combined_contoured,cv2.COLOR_BGR2GRAY)
_, Final_Threshold = cv2.threshold(final_g,254,255,cv2.THRESH_BINARY)

intersect_coords=[]  
for i2 in range(w):
   for i3 in range(h):
     if Final_Threshold[i3,i2]==255:
        location=[i3,i2]
        intersect_coords.append(location)
        
#===================================================
#14. Prepare intersect coord lists
#===================================================
print('14. Preparing Intersect Co-ords Lists...')#Bug Check Print

intersect_coords_list=list(flatten(intersect_coords))
intersect_list_len=len(intersect_coords_list)

intersect_coords_list_x=[]
intersect_coords_list_y=[]
for i4 in intersect_coords_list:
    if(i4 % 2): 
        intersect_coords_list_x.append(intersect_coords_list[i4])
    else:
        intersect_coords_list_y.append(intersect_coords_list[i4])
        
# print('===================================================')
# print('intersect_coords_list_x\n', intersect_coords_list_x)
# print('===================================================')
# print('intersect_coords_list_y\n', intersect_coords_list_y)      
# print('===================================================')

#===================================================
#15. Arrange car contours into lists for x and y co-ords
#===================================================
print('15. Arranging Car Contours Into Lists for x and y Co-ords...')#Bug Check Print

car1_list=list(flatten(car1))
car2_list=list(flatten(car2))
car3_list=list(flatten(car3))
car4_list=list(flatten(car4))
car5_list=list(flatten(car5))
car6_list=list(flatten(car6))

# car 1--------------------------------------------------------------
car1_list_len=len(car1_list)
car1_list_x=[]
car1_list_y=[]
for i5 in range (car1_list_len):
    if(i5 % 2): 
        car1_list_x.append(car1_list[i5])
    else:
        car1_list_y.append(car1_list[i5])

# car 2--------------------------------------------------------------
car2_list_len=len(car2_list)
car2_list_x=[]
car2_list_y=[]
for i5 in range (car2_list_len):
    if(i5 % 2): 
        car2_list_y.append(car2_list[i5])
    else:
        car2_list_x.append(car2_list[i5])

# car 3--------------------------------------------------------------
car3_list_len=len(car3_list)
car3_list_x=[]
car3_list_y=[]
for i5 in range (car3_list_len):
    if(i5 % 2): 
        car3_list_y.append(car3_list[i5])
    else:
        car3_list_x.append(car3_list[i5])

# car 4--------------------------------------------------------------
car4_list_len=len(car4_list)
car4_list_x=[]
car4_list_y=[]
for i5 in range (car4_list_len):
    if(i5 % 2): 
        car4_list_y.append(car4_list[i5])
    else:
        car4_list_x.append(car4_list[i5])

# car 5--------------------------------------------------------------
car5_list_len=len(car5_list)
car5_list_x=[]
car5_list_y=[]
for i5 in range (car5_list_len):
    if(i5 % 2): 
        car5_list_y.append(car5_list[i5])
    else:
        car5_list_x.append(car5_list[i5])

# car 6--------------------------------------------------------------
car6_list_len=len(car6_list)
car6_list_x=[]
car6_list_y=[]
for i5 in range (car6_list_len):
    if(i5 % 2): 
        car6_list_y.append(car6_list[i5])
    else:
        car6_list_x.append(car6_list[i5])
#------------------------------------------------------------------

#===================================================
#16. Print out each car list with list of x and y coords
#===================================================
print('16. Printing Out Each Car List with a List of x and y Co-ords...')#Bug Check Print

# print('==================================')
# print('car 1_list_x',car1_list_x)
# car1_len_x=len(car1_list_x)
# print('car 1_len_x ',car1_len_x)
# print('==================================')
# print('car 1_list_y',car1_list_y)
# car1_len_y=len(car1_list_y) 
# print('car 1_len_y ',car1_len_y)
# print('==================================')


# print('==================================')
# print('car 2_list_x',car2_list_x)
# car2_len_x=len(car2_list_x)
# print('car 2_len_x ',car2_len_x)
# print('==================================')
# print('car 2_list_y',car2_list_y)
# car2_len_y=len(car2_list_y) 
# print('car 2_len_y ',car2_len_y)
# print('==================================')


# print('==================================')
# print('car 3_list_x',car3_list_x)
# car3_len_x=len(car3_list_x)
# print('car 3_len_x ',car3_len_x)
# print('==================================')
# print('car 3_list_y',car3_list_y)
# car3_len_y=len(car3_list_y) 
# print('car 3_len_y ',car3_len_y)
# print('==================================')


# print('==================================')
# print('car 4_list_x',car4_list_x)
# car4_len_x=len(car4_list_x)
# print('car 4_len_x ',car4_len_x)
# print('==================================')
# print('car 4_list_y',car4_list_y)
# car4_len_y=len(car4_list_y) 
# print('car 4_len_y ',car4_len_y)
# print('==================================')

#------------------------------------------------------------------
#need to check if any x coords are in intersect (0,2,4...) if so,
# then check the y's (1,3,5...) matching the matched x 
#e.g. if carlist_x[0] matched a coord for an x coord in intersect (intersect_coords_list_x[34]), 
# then check if carlist_y[0] matches intersect_coords_list_y[34]
#----------------------------------------------------------------------------

#===================================================
#17. Check which cars broke the line
#===================================================
print('17. Checking Which Cars Broke the Line...')
intersect_list_x_len=len(intersect_coords_list_x)
intersect_list_y_len=len(intersect_coords_list_y)

contour_allowance=1
output_option=1

# car 1______________________________________________________ 
print('car 1 __________________________________________ \n')
car1_list_x_len=len(car1_list_x)
car1_list_y_len=len(car1_list_y)

for i7 in range(intersect_list_x_len):
    for i6 in range(car1_list_x_len):
        if car1_list_x[i6]==intersect_coords_list_x[i7]:
            if car1_list_y[i6]== intersect_coords_list_y[i7]:
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car1, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 1 Point Co-ordinates: ', car1_list_x[i6], ',',car1_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])

            elif car1_list_y[i6]== (intersect_coords_list_y[i7]+contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car1, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 1 Point Co-ordinates: ', car1_list_x[i6], ',',car1_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])
                
            elif car1_list_y[i6]== (intersect_coords_list_y[i7]-contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car1, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 1 Point Co-ordinates: ', car1_list_x[i6], ',',car1_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])

        if car1_list_x[i6]==(intersect_coords_list_x[i7]+contour_allowance):
            if car1_list_y[i6]== intersect_coords_list_y[i7]:
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car1, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 1 Point Co-ordinates: ', car1_list_x[i6], ',',car1_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])

            elif car1_list_y[i6]== (intersect_coords_list_y[i7]+contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car1, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 1 Point Co-ordinates: ', car1_list_x[i6], ',',car1_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])

            elif car1_list_y[i6]== (intersect_coords_list_y[i7]-contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car1, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 1 Point Co-ordinates: ', car1_list_x[i6], ',',car1_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])

        if car1_list_x[i6]==(intersect_coords_list_x[i7]-contour_allowance):
            if car1_list_y[i6]== intersect_coords_list_y[i7]:
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car1, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 1 Point Co-ordinates: ', car1_list_x[i6], ',',car1_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])

            elif car1_list_y[i6]== (intersect_coords_list_y[i7]+contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car1, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 1 Point Co-ordinates: ', car1_list_x[i6], ',',car1_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])

            elif car1_list_y[i6]== (intersect_coords_list_y[i7]-contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car1, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 1 Point Co-ordinates: ', car1_list_x[i6], ',',car1_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])



# car 2__________________________________________  
print('car 2 __________________________________________ \n')
car2_list_x_len=len(car2_list_x)
car2_list_y_len=len(car2_list_y)

for i7 in range(intersect_list_x_len):
    for i6 in range(car2_list_x_len):
        if car2_list_x[i6]==intersect_coords_list_x[i7]:
            if car2_list_y[i6]== intersect_coords_list_y[i7]:
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car2, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 2 Point Co-ordinates: ', car2_list_x[i6], ',',car2_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])
            elif car2_list_y[i6]== (intersect_coords_list_y[i7]+contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car2, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 2 Point Co-ordinates: ', car2_list_x[i6], ',',car2_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])
            elif car2_list_y[i6]== (intersect_coords_list_y[i7]-contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car2, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 2 Point Co-ordinates: ', car2_list_x[i6], ',',car2_list_y[i6])
                print('     Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])
  
        if car2_list_x[i6]==(intersect_coords_list_x[i7]+contour_allowance):
            if car2_list_y[i6]== intersect_coords_list_y[i7]:
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car2, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 2 Point Co-ordinates: ', car2_list_x[i6], ',',car2_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])
            elif car3_list_y[i6]== (intersect_coords_list_y[i7]+contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car2, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 2 Point Co-ordinates: ', car2_list_x[i6], ',',car2_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])
            elif car3_list_y[i6]== (intersect_coords_list_y[i7]-contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car2, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 2 Point Co-ordinates: ', car2_list_x[i6], ',',car2_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])

        if car2_list_x[i6]==(intersect_coords_list_x[i7]-contour_allowance):
            if car2_list_y[i6]== intersect_coords_list_y[i7]:
                #draw car contour here on black 
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car2, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 2 Point Co-ordinates: ', car2_list_x[i6], ',',car2_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])
            elif car2_list_y[i6]== (intersect_coords_list_y[i7]+contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car2, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 2 Point Co-ordinates: ', car2_list_x[i6], ',',car2_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])
            elif car2_list_y[i6]== (intersect_coords_list_y[i7]-contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car2, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 2 Point Co-ordinates: ', car2_list_x[i6], ',',car2_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])


# car 3__________________________________________  
print('car 3 __________________________________________ \n')
car3_list_x_len=len(car3_list_x)
car3_list_y_len=len(car3_list_y)

for i7 in range(intersect_list_x_len):
    for i6 in range(car3_list_x_len):
        if car3_list_x[i6]==intersect_coords_list_x[i7]:
            if car3_list_y[i6]== intersect_coords_list_y[i7]:
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car3, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 3 Point Co-ordinates: ', car3_list_x[i6], ',',car3_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])
            elif car3_list_y[i6]== (intersect_coords_list_y[i7]+contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car3, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 3 Point Co-ordinates: ', car3_list_x[i6], ',',car3_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])
            elif car3_list_y[i6]== (intersect_coords_list_y[i7]-contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car3, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 3 Point Co-ordinates: ', car3_list_x[i6], ',',car3_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])

        if car3_list_x[i6]==(intersect_coords_list_x[i7]+contour_allowance):
            if car3_list_y[i6]== intersect_coords_list_y[i7]:
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car3, contourIdx=-1,color=(0,255,0), thickness=5) 
                print('      Car 3 Point Co-ordinates: ', car3_list_x[i6], ',',car3_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])
            elif car3_list_y[i6]== (intersect_coords_list_y[i7]+contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car3, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 3 Point Co-ordinates: ', car3_list_x[i6], ',',car3_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])
            elif car3_list_y[i6]== (intersect_coords_list_y[i7]-contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car3, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 3 Point Co-ordinates: ', car3_list_x[i6], ',',car3_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])

        if car3_list_x[i6]==(intersect_coords_list_x[i7]-contour_allowance):
            if car3_list_y[i6]== intersect_coords_list_y[i7]:
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car3, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 3 Point Co-ordinates: ', car3_list_x[i6], ',',car3_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])
            elif car3_list_y[i6]== (intersect_coords_list_y[i7]+contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car3, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 3 Point Co-ordinates: ', car3_list_x[i6], ',',car3_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])
            elif car3_list_y[i6]== (intersect_coords_list_y[i7]-contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car3, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 3 Point Co-ordinates: ', car3_list_x[i6], ',',car3_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])

# car 4__________________________________________  
print('car 4 __________________________________________ \n')
car4_list_x_len=len(car4_list_x)
car4_list_y_len=len(car4_list_y)

for i7 in range(intersect_list_x_len):
    for i6 in range(car4_list_x_len):
        if car4_list_x[i6]==intersect_coords_list_x[i7]:
            if car4_list_y[i6]== intersect_coords_list_y[i7]:
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car4, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 4 Point Co-ordinates: ', car4_list_x[i6], ',',car4_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])
            elif car4_list_y[i6]== (intersect_coords_list_y[i7]+contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car4, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 4 Point Co-ordinates: ', car4_list_x[i6], ',',car4_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])
            elif car4_list_y[i6]== (intersect_coords_list_y[i7]-contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car4, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 4 Point Co-ordinates: ', car4_list_x[i6], ',',car4_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])

        if car4_list_x[i6]==(intersect_coords_list_x[i7]+contour_allowance):
            if car4_list_y[i6]== intersect_coords_list_y[i7]:
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car4, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 4 Point Co-ordinates: ', car4_list_x[i6], ',',car4_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])
            elif car4_list_y[i6]== (intersect_coords_list_y[i7]+contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car4, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 4 Point Co-ordinates: ', car4_list_x[i6], ',',car4_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])
            elif car4_list_y[i6]== (intersect_coords_list_y[i7]-contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car4, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 4 Point Co-ordinates: ', car4_list_x[i6], ',',car4_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])

        if car4_list_x[i6]==(intersect_coords_list_x[i7]-contour_allowance):
            if car4_list_y[i6]== intersect_coords_list_y[i7]:
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car4, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 4 Point Co-ordinates: ', car4_list_x[i6], ',',car4_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])
            elif car4_list_y[i6]== (intersect_coords_list_y[i7]+contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car4, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 4 Point Co-ordinates: ', car4_list_x[i6], ',',car4_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])
            elif car4_list_y[i6]== (intersect_coords_list_y[i7]-contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car4, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 4 Point Co-ordinates: ', car4_list_x[i6], ',',car4_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])

# car 5__________________________________________  
print('car 5 __________________________________________ \n')
car5_list_x_len=len(car5_list_x)
car5_list_y_len=len(car5_list_y)

for i7 in range(intersect_list_x_len):
    for i6 in range(car5_list_x_len):
        if car5_list_x[i6]==intersect_coords_list_x[i7]:
            if car5_list_y[i6]== intersect_coords_list_y[i7]:
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car5, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 5 Point Co-ordinates: ', car5_list_x[i6], ',',car5_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])
            elif car5_list_y[i6]== (intersect_coords_list_y[i7]+contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car5, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 5 Point Co-ordinates: ', car5_list_x[i6], ',',car5_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])
            elif car5_list_y[i6]== (intersect_coords_list_y[i7]-contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car5, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 5 Point Co-ordinates: ', car5_list_x[i6], ',',car5_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])
 
        if car5_list_x[i6]==(intersect_coords_list_x[i7]+contour_allowance):
            if car5_list_y[i6]== intersect_coords_list_y[i7]:
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car5, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 5 Point Co-ordinates: ', car5_list_x[i6], ',',car5_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])
            elif car5_list_y[i6]== (intersect_coords_list_y[i7]+contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car5, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 5 Point Co-ordinates: ', car5_list_x[i6], ',',car5_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])
            elif car5_list_y[i6]== (intersect_coords_list_y[i7]-contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car5, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 5 Point Co-ordinates: ', car5_list_x[i6], ',',car5_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])

        if car5_list_x[i6]==(intersect_coords_list_x[i7]-contour_allowance):
            if car5_list_y[i6]== intersect_coords_list_y[i7]:
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car5, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 5 Point Co-ordinates: ', car5_list_x[i6], ',',car5_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])
            elif car5_list_y[i6]== (intersect_coords_list_y[i7]+contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car5, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 5 Point Co-ordinates: ', car5_list_x[i6], ',',car5_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])
            elif car5_list_y[i6]== (intersect_coords_list_y[i7]-contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car5, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 5 Point Co-ordinates: ', car5_list_x[i6], ',',car5_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])

# car 6_________________________________________  
print('car 6 __________________________________________ \n')
car6_list_x_len=len(car6_list_x)
car6_list_y_len=len(car6_list_y)

for i7 in range(intersect_list_x_len):
    for i6 in range(car6_list_x_len):
        if car6_list_x[i6]==intersect_coords_list_x[i7]:
            if car6_list_y[i6]== intersect_coords_list_y[i7]:
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car6, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 6 Point Co-ordinates: ', car6_list_x[i6], ',',car6_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])
            elif car6_list_y[i6]== (intersect_coords_list_y[i7]+contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car6, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 6 Point Co-ordinates: ', car6_list_x[i6], ',',car6_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])
            elif car6_list_y[i6]== (intersect_coords_list_y[i7]-contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car6, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 6 Point Co-ordinates: ', car6_list_x[i6], ',',car6_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])

        if car6_list_x[i6]==(intersect_coords_list_x[i7]+contour_allowance):
            if car6_list_y[i6]== intersect_coords_list_y[i7]:
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car6, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 6 Point Co-ordinates: ', car6_list_x[i6], ',',car6_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])
            elif car6_list_y[i6]== (intersect_coords_list_y[i7]+contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car6, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 6 Point Co-ordinates: ', car6_list_x[i6], ',',car6_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])
            elif car6_list_y[i6]== (intersect_coords_list_y[i7]-contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car6, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 6 Point Co-ordinates: ', car6_list_x[i6], ',',car6_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])

        if car6_list_x[i6]==(intersect_coords_list_x[i7]-contour_allowance):
            if car6_list_y[i6]== intersect_coords_list_y[i7]:
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car6, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 6 Point Co-ordinates: ', car6_list_x[i6], ',',car6_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])
            elif car6_list_y[i6]== (intersect_coords_list_y[i7]+contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car6, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 6 Point Co-ordinates: ', car6_list_x[i6], ',',car6_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])
            elif car6_list_y[i6]== (intersect_coords_list_y[i7]-contour_allowance):
                #draw car contour here on black background
                output_option=2
                final_output_contoured = cv2.drawContours(Blank4, car6, contourIdx=-1,color=(0,255,0), thickness=5)
                print('      Car 6 Point Co-ordinates: ', car6_list_x[i6], ',',car6_list_y[i6])
                print('      Intersect Co-ordinate Crossed: ', intersect_coords_list_x[i7], ',',intersect_coords_list_y[i7])
  
#===================================================
#18. Produce final images
#===================================================
print('18. Producing Final Images...')

# if blank4 is entirely black theres no bad parking so show good parking output
if output_option==1:
    #print out alternate final output
    print('18.5. Print Out Alternate Final Output')
    Good_Park = cv2.imread("good_park2.jpg")
    Good_Park_RGB = cv2.cvtColor(Good_Park, cv2.COLOR_BGR2RGB)
    Full_RGB = cv2.cvtColor(Full_original, cv2.COLOR_BGR2RGB)
    # cv2.imshow('final good parking',Good_Park)
    # cv2.waitKey(0) 
    
    plt.figure(3)
    plt.subplot(1,2,1)
    plt.imshow(Full_RGB)
    plt.title('original')
    plt.subplot(1,2,2)
    plt.imshow(Good_Park_RGB)
    plt.title('final good parking')

# if blank4 is not entirely black,i.e. has contours on it, show output for boundaries broken     
elif output_option==2:     
#draw intersect on image that was black background, now with car contours
#addweight to wasted
    print('18.5. Print Out Wasted Final Output')
    Full_RGB = cv2.cvtColor(Full_original, cv2.COLOR_BGR2RGB)
    
    W = cv2.imread("wasted.jpg")
    hw, ww, dw = W.shape
    # print('wasted h :', hw) 
    # print('wasted w :', ww) 
    Wasted=W.copy()

    C=Blank4
    hc, wc, dc = C.shape
    # print('Crime h :', hc) 
    # print('Crime w :', wc) 
    Criminal=C.copy()

    scale_percent_w = wc/ww
    scale_percent_h = hc/hw

    width = int(Wasted.shape[1] * scale_percent_w)
    height = int(Wasted.shape[0] * scale_percent_h)
    dim = (width, height)
    # resize image
    Wasted = cv2.resize(Wasted, dim, interpolation = cv2.INTER_AREA)

    lower = np.array([5, 5,5]) 
    upper = np.array([255,255,255]) 
      
    # preparing the mask to overlay 
    mask = cv2.inRange(Wasted, lower, upper) 
          
    result = cv2.bitwise_and(Wasted,Wasted,mask=mask)

    mask_reverse = cv2.bitwise_not(mask)
    background_mask = cv2.bitwise_and(Criminal,Criminal,mask=mask_reverse)

    Wasted_weight=.5
    Criminal_weight = (1.0 - Wasted_weight)
    gamma=10
    Added_scaled_weighted = cv2.addWeighted(result, Wasted_weight, background_mask, Criminal_weight, gamma)
    
    Added_scaled_weighted_RGB = cv2.cvtColor(Added_scaled_weighted, cv2.COLOR_BGR2RGB)
    
    
    plt.figure(4)
    plt.subplot(1,2,1)
    plt.imshow(Full_RGB)
    plt.title('original')
    plt.subplot(1,2,2)
    plt.imshow(Added_scaled_weighted_RGB)
    plt.title('final bad parking')
#==============================================================================================================

plt.show() 