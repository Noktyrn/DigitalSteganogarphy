import cv2, os
base_path = "images/grey/"
new_path = "images/grey_jpeg/"
for infile in os.listdir(base_path):
    print ("file : " + infile)
    if infile[-5:] == ".tiff":
        read = cv2.imread(base_path + infile)
        outfile = infile[:-5] + '.jpg'
        cv2.imwrite(new_path+outfile,read,[int(cv2.IMWRITE_JPEG_QUALITY), 200])