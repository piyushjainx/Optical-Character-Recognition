"""
Character Detection
The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks
Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.
Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy

BLACK = 1
WHITE = 0

def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.
    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.
    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # TODO Add your code here. Do not modify the return and input arguments

    # print(characters)
    # characters = enrollment(characters)
    # plt.imshow(test_img)
    # plt.show()
    detection(test_img)
    
    # recognition()

    #raise NotImplementedError

def show_piyush_char(image,delay = 1000):
    image[image == 0 ] = 255
    image[image != 255 ] = 0
    sift = cv2.SIFT_create()
    kp,des =sift.detectAndCompute(image,None)
    img = cv2.drawKeypoints(image,kp,None)
    # show_image(img,2000)
    # show_image(image,delay)
    
            

def convert_to_binary_char(image):
    COLOR_THRESHOLD = 127
    rows = image.shape[0]
    columns = image.shape[1]
    newshape = (rows, columns, 1)
    print(rows,columns)
    flat_img = np.reshape(image,newshape)

    for i in range(rows):
        for j in range(columns):
            if flat_img[i][j]>COLOR_THRESHOLD:
                flat_img[i][j] = 255
            else:
                flat_img[i][j] = 0

    # show_image(image)
    # plt.imshow(flat_img)
    # plt.show()
    # input()

    return image

def convert_to_binary(image):
    COLOR_THRESHOLD = 127
    rows = image.shape[0]
    columns = image.shape[1]
    newshape = (rows, columns, 1)
    #print(rows,columns)
    flat_img = np.reshape(image,newshape)

    for i in range(rows):
        for j in range(columns):
            if flat_img[i][j]>COLOR_THRESHOLD:
                flat_img[i][j] = WHITE
            else:
                flat_img[i][j] = BLACK

    # show_image(image)
    # plt.imshow(flat_img)
    # plt.show()
    # input()

    return image

def enrollment(characters):
    """ Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 1 : Your Enrollment code should go here.
    #feature extraction of characters
    # sift = cv2.xfeatures2d.SIFT_create()
    sift = cv2.SIFT_create()
    for i in range(len(characters)):
        characters[i][1] = convert_to_binary_char(characters[i][1])
        # show_image(bin_img)
        # show_image(characters[i][1])
        # kp = sift.detect(characters[i][1],None)
        # A = sift.detectAndCompute(characters[i][1], None)
        # sift_image = cv2.drawKeypoints(characters[i][1], keypoints, img)
        # keypoints, descriptors = A
    # for i,j in character_list:
        # print(len(j))
        
        kp,des =sift.detectAndCompute(characters[i][1],None)
        img = cv2.drawKeypoints(characters[i][1],kp,None)
        show_image(img,2000)
        # cv2.imshow("Image",img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        print(kp)
        print("a")

    #run ORB on the image



    return characters

def set_add_to_eqlist_2(eq_list, key, val):
    #if key<200:
        #print(key,val)
        #raise Exception()
    if int(key) not in eq_list.keys():
        eq_list[int(key)] = set()
        eq_list[int(key)].update(val)
    else:
        eq_list[int(key)].update(val)
    #if(min(eq_list[int(key)]) in eq_list.keys()):
        #eq_list = set_add_to_eqlist(eq_list, int(key), eq_list[min(eq_list[int(key)])])
    return eq_list

def set_add_to_eqlist(eq_list, key, val):
    #if key<200:
        #print(key,val)
        #raise Exception()
    if int(key) not in eq_list.keys():
        eq_list[int(key)] = set()
        eq_list[int(key)].update(val)
    else:
        eq_list[int(key)].update(val)
    if(min(eq_list[int(key)]) in eq_list.keys()):
        eq_list = set_add_to_eqlist_2(eq_list, int(key), eq_list[min(eq_list[int(key)])])
    return eq_list

def add_to_eqlist(eq_list, key, val):
    #if key<200:
        #print(key,val)
        #raise Exception()
    if int(key) not in eq_list.keys():
        eq_list[int(key)] = set()
        eq_list[int(key)].update({int(val)})
    else:
        eq_list[int(key)].update({int(val)})
    if(min(eq_list[int(key)]) in eq_list.keys()):
        eq_list = set_add_to_eqlist(eq_list, int(key), eq_list[min(eq_list[int(key)])])
    return eq_list

def detection(test_image):
    """ 
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 2 : Your Detection code should go here.
    #get number of char and bounding boxes
    #grayscale
    #conversion to binary
    #test_img = test_image
    test_img = copy.deepcopy(test_image)
    test_img = convert_to_binary(test_img)
    #connected component labeling
    rows = test_img.shape[0]
    columns = test_img.shape[1]

    labelled_img = np.zeros(test_img.shape)
    cur_label = 200
    eq_list = {}
    # show_image(test_img,5000)
    count_black = 0
    for i in range(rows):
        for j in range(columns):
            if test_img[i][j] == BLACK:
                if i == 0 and j == 0:
                    labelled_img[i][j] = cur_label
                    eq_list = add_to_eqlist(eq_list,cur_label,cur_label)
                    cur_label+=1
                elif i == 0: #first row
                    if test_img[i][j-1] == BLACK:
                        labelled_img[i][j] = labelled_img[i][j-1]
                    elif test_img[i][j-1] == WHITE:
                        labelled_img[i][j] = cur_label
                        eq_list = add_to_eqlist(eq_list,cur_label,cur_label)
                        cur_label += 1 #increment cur_label
                    else:
                        print("Unexpected : 1")
                elif j == 0: #first column
                    if test_img[i-1][j] == BLACK:
                        labelled_img[i][j] = labelled_img[i-1][j]
                    elif test_img[i-1][j] == WHITE:
                        labelled_img[i][j] = cur_label
                        eq_list = add_to_eqlist(eq_list,cur_label,cur_label)
                        cur_label += 1
                    else:
                        print("Unexpected : 2")
                else:
                    if test_img[i][j-1] == WHITE and test_img[i-1][j] == WHITE:
                        labelled_img[i][j] = cur_label
                        eq_list = add_to_eqlist(eq_list,cur_label,cur_label)
                        cur_label += 1
                    elif test_img[i][j-1] == WHITE: #top
                        labelled_img[i][j] = labelled_img[i-1][j]
                    elif test_img[i-1][j] == WHITE: #left
                        labelled_img[i][j] = labelled_img[i][j-1]
                    else:
                        labelled_img[i][j] = min(
                            labelled_img[i][j-1],
                            labelled_img[i-1][j],
                        )
                        eq_list = add_to_eqlist(
                            eq_list,
                            max(
                                labelled_img[i][j-1],
                                labelled_img[i-1][j],
                            ),
                            min(
                                labelled_img[i][j-1],
                                labelled_img[i-1][j],
                            )
                        )
    show_image(labelled_img,1000)
    
    for i in range(rows):
        for j in range(columns):
            if labelled_img[i][j] in eq_list.keys():
                labelled_img[i][j] = min(eq_list[labelled_img[i][j]])
    # print("black pixels = ",count_black)
    show_image(labelled_img,1000)
    unique_chars = np.unique(labelled_img)
    for count,i in enumerate(unique_chars):
        if i==0:
            continue
        labelled_img[labelled_img == i] = count
    unique_chars_clean = np.unique(labelled_img)
    print(unique_chars_clean)

    #get bounding boxes
    bounds = {}
    for i in unique_chars_clean:
        if i!=0:
            bounds[i] = {
                'xmin' : 999999,
                'xmax' : 0,
                'ymin' : 999999,
                'ymax' : 0,
            }
    for i in range(rows):
        for j in range(columns):
                if labelled_img[i][j] in bounds.keys():
                    bounds[labelled_img[i][j]]['xmin'] = min(bounds[labelled_img[i][j]]['xmin'],j)
                    bounds[labelled_img[i][j]]['xmax'] = max(bounds[labelled_img[i][j]]['xmax'],j)
                    bounds[labelled_img[i][j]]['ymin'] = min(bounds[labelled_img[i][j]]['ymin'],i)
                    bounds[labelled_img[i][j]]['ymax'] = max(bounds[labelled_img[i][j]]['ymax'],i)
    # print(bounds)

    for key,val in bounds.items():
        print(key,val)
        if val['ymin'] == val['ymax'] or val['xmin'] == val['xmax']:
            continue
        # show_image(test_image[val['xmin']:val['xmax'],val['ymin']:val['ymax']],3000)
        character = test_img[val['ymin']:val['ymax'],val['xmin']:val['xmax']]
        show_piyush_char(character,3000)
        #show_image(character,3000)
        # show_image(test_image[0:100,0:100],3000)

    # orb = cv2.ORB_create()
    print("")

    # queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw,None)

    # raise NotImplementedError

def recognition():
    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 3 : Your Recognition code should go here.
    #match features from enrollment and detection

    raise NotImplementedError


def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = []
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()
    
    characters = []

    all_character_imgs = glob.glob(args.character_folder_path+ "/*")
    
    for each_character in all_character_imgs :
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])

    test_img = read_image(args.test_img)
    # show_image(test_img,1000)

    results = ocr(test_img, characters)

    save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()