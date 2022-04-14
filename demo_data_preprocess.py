import os

import pydicom
import numpy as np
# from PIL import Image
import skimage.transform as trans
from torch import conv2d
from tqdm import tqdm
import cv2

def apply_window_center(x, window, center, y_min=0., y_max=1.):
    """ Apply the RGB Look-Up Table for the given data and window/center value.
            if (x <= c - 0.5 - (w-1)/2), then y = y_min
            else if (x > c - 0.5 + (w-1)/2), then y = y_max ,
            else y = ((x - (c - 0.5)) / (w-1) + 0.5) * (y_max - y_min) + y_min
        See https://www.dabsoft.ch/dicom/3/C.11.2.1.2/ for details
    """
    try:
        window = float(window[0])
    except TypeError:
        pass
    try:
        center = float(center[0])
    except TypeError:
        pass
    
    c_min = x <= (center - 0.5 - (window - 1) / 2)
    c_max = x > (center - 0.5 + (window - 1) / 2)
    fn = lambda c: ((c - (center - 0.5)) / (window - 1) + 0.5) * (y_max - y_min) + y_min
    
    return np.piecewise(x.astype('float'), [c_min, c_max], [y_min, y_max, fn])

def decode_dicom(dataset, view_idx=0):
    """ Decode dicom file into images. """
    if ('PixelData' not in dataset):
        raise TypeError("DICOM dataset does not have pixel data")
    # can only apply LUT if these window info exists
    if ('WindowWidth' not in dataset) or ('WindowCenter' not in dataset):
        import PIL.Image
        
        bits = dataset.BitsAllocated
        samples = dataset.SamplesPerPixel
        if bits == 8 and samples == 1:
            mode = "L"
        elif bits == 8 and samples == 3:
            mode = "RGB"
        elif bits == 16:
            # not sure about this -- PIL source says is 'experimental'
            # and no documentation. Also, should bytes swap depending
            # on endian of file and system??
            mode = "I;16"
        else:
            raise TypeError("Don't know PIL mode for %d BitsAllocated "
                            "and %d SamplesPerPixel" % (bits, samples))

        # PIL size = (width, height)
        size = (dataset.Columns, dataset.Rows)

        # Recommended to specify all details
        # by http://www.pythonware.com/library/pil/handbook/image.htm
        im = PIL.Image.frombuffer(mode, size, dataset.PixelData,
                                  "raw", mode, 0, 1)
        
        return np.array(im)
    
    else:
        arr = dataset.pixel_array.astype('float')
        if ('RescaleIntercept' in dataset) and ('RescaleSlope' in dataset):
            intercept = float(dataset.RescaleIntercept)  # single value
            slope = float(dataset.RescaleSlope)
            arr = slope * arr + intercept
        
        if isinstance(dataset.WindowCenter, pydicom.multival.MultiValue):
            window_center = float(dataset.WindowCenter[view_idx])
            window_width = float(dataset.WindowWidth[view_idx])
        else:
            window_center = float(dataset.WindowCenter)
            window_width = float(dataset.WindowWidth)
            
        arr = apply_window_center(arr, window_width, window_center)
        if dataset.PhotometricInterpretation == 'MONOCHROME1':
            # return 2 ** dataset.BitsStored - 1 - x
            return 1 - arr
        elif dataset.PhotometricInterpretation == 'MONOCHROME2':
            return arr


def get_imageData(sample_folder, index=-1):
    sample_folder += '/'
    files = os.listdir(sample_folder)
    slices = []
    for temp_file in files:
        try:
            slices.append(pydicom.dcmread(sample_folder+temp_file))
        except Exception as e:
            pass

    ps = slices[0].PixelSpacing
    ss = slices[0].SliceThickness
    slices = sorted(slices, key=lambda s: s.SliceLocation , reverse=True)
    img_shape = (slices[0].Rows, slices[0].Columns, len(slices))

    image_3d = []
    for s in slices[:index]:
        image_3d.append(decode_dicom(s, view_idx=0))
    image_3d = np.array(image_3d)

    # real_shape = (int(img_shape[0] * ps[0]), int(img_shape[1] * ps[1]), int(img_shape[2] * ss))
    # img3d = trans.resize(image_3d, (128,128,64), preserve_range=True, anti_aliasing=True)
    # print(img3d.shape, img3d.dtype, img3d.min(), img3d.max())
    # print('file count:{}'.format(len(files)))
    return image_3d



def get_imageData_with_real(sample_folder, index=-1):
    sample_folder += '/'
    files = os.listdir(sample_folder)
    slices = []
    for temp_file in files:
        try:
            slices.append(pydicom.dcmread(sample_folder+temp_file))
        except Exception as e:
            pass

    ps = slices[0].PixelSpacing
    ss = slices[0].SliceThickness
    slices = sorted(slices, key=lambda s: s.SliceLocation , reverse=True)
    img_shape = (slices[0].Rows, slices[0].Columns, len(slices))

    image_3d = []
    for s in slices[:index]:
        image_3d.append(decode_dicom(s, view_idx=0))
    image_3d = np.array(image_3d)

    real_shape = (int(img_shape[0] * ps[0]), int(img_shape[1] * ps[1]), int(img_shape[2] * ss))
    return image_3d, real_shape



import p_tqdm

def convert_data():
    list_test_name = [
    '1.000000-CTs from rtog conversion-577.2.xlsx.csv', 
    '1.000000-CTs from rtog conversion-600.2.xlsx.csv', 
    '4.000000-CTnormalFOV500-94964.xlsx.csv', 
    '5.000000-CTnormalFOV500-79953.xlsx.csv', 
    '5577.000000-NA-40311.xlsx.csv', 
    '712152553.000000-kVCT Image Set-57586.xlsx.csv', 
    'CT WB  3.0  B30fCHEST-05638.xlsx.csv', 
    'CT WB 5.0 B40sCHEST-52338.xlsx.csv'
    ]

    data_source_dir = './data/'
    data_save_dir = './data_np_train/'
    if os.path.exists(data_save_dir) is False:
        os.makedirs(data_save_dir)
    list_data_names = os.listdir(data_source_dir)

    
    def convert_data(temp_name):
        if temp_name+'.xlsx.csv' not in list_test_name:
            temp_sample_dir = data_source_dir + temp_name
            try:
                temp_data = get_imageData(temp_sample_dir, 300)
                np.save(data_save_dir + temp_name, temp_data)
            except Exception as e:
                print(e)
            return 1
        else:
            return 0


    iterator_ml = p_tqdm.p_imap(convert_data, list_data_names, num_cpus=8, ncols=200)

    temp_temp = 0
    for result in iterator_ml:
        temp_temp += result

    # for temp_name in tqdm(list_data_names[:], ncols=200):
    #     if temp_name+'.xlsx.csv' not in list_test_name:
    #         temp_sample_dir = data_source_dir + temp_name
    #         try:
    #             temp_data = get_imageData(temp_sample_dir, 300)
    #             np.save(data_save_dir + temp_name, temp_data)
    #         except Exception as e:
    #             print(e)



def temp_test():
    # sample_folder = 'E:/task_data/data/1.000000-CTs from rtog conversion-555.2/'
    # get_imageData(sample_folder)
    list_test_name = []
    info_root_dir = './all_kps_3d/label_csv/'
    data_root_dir = 'E:/task_data/data/'
    list_info_names = os.listdir(info_root_dir)
    for info_name in tqdm(list_info_names):
        temp_info_dir = info_root_dir + info_name
        f = open(temp_info_dir, mode='r')
        temp_lines = f.readlines()
        f.close()

        temp_line = temp_lines[10].strip('\n').split(',')
        # print(temp_line)
        temp_id = int(float(temp_line[-1]))
        if temp_id > 180:
            temp_line = temp_lines[3].strip('\n').split(',')
            temp_id = int(float(temp_line[-1]))
            temp_x = int(float(temp_line[-3]))
            temp_y = int(float(temp_line[-2]))

            try:
                temp_image_data = get_imageData(data_root_dir + info_name[:-9]+'/')
                print(info_name)
                print(temp_image_data.shape)
                list_test_name.append(info_name)
                num_id = temp_image_data.shape[0]
                temp_show = np.uint8(temp_image_data[temp_id]*255)
                temp_show = cv2.cvtColor(temp_show, cv2.COLOR_GRAY2BGR)
                temp_show = cv2.circle(temp_show, (temp_x, temp_y), 5, (0,255,0), -1)
                cv2.imwrite('./temp/'+info_name[:-9]+'.jpg', temp_show)
                cv2.imshow('temp', temp_show)
                cv2.waitKey()
                

        #         print(temp_id, num_id)
        #         # print(temp_id)
            except Exception as e:
                print(e)

    print(list_test_name)


if __name__ == "__main__":
    convert_data()
    # temp_test()




