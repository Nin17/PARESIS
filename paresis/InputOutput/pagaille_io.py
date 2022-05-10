"""_summary_
    """
import fabio
import fabio.edfimage as edf
import fabio.tifimage as tif
import numpy as np


def open_image(filename):
    """_summary_

    Parameters
    ----------
    filename : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    filename = str(filename)
    image = fabio.open(filename)
    imarray = image.data
    return imarray


def get_header(filename):
    """_summary_

    Parameters
    ----------
    filename : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    image = fabio.open(filename)
    header = image.header
    return header


def save_tiff_16bit(data, filename, min_im=0, max_im=0, header=None):
    """_summary_

    Parameters
    ----------
    data : _type_
        _description_
    filename : _type_
        _description_
    min_im : int, optional
        _description_, by default 0
    max_im : int, optional
        _description_, by default 0
    header : _type_, optional
        _description_, by default None
    """
    if min_im == max_im:
        min_im = np.amin(data)
        max_im = np.amax(data)
    data_to_store = 65536*(data-min_im)/(max_im-min_im)
    data_to_store[data_to_store > 65635] = 65535
    data_to_store[data_to_store < 0] = 0
    data_to_store = np.asarray(data_to_store, np.uint16)

    if header:
        tif.TifImage(data=data_to_store, header=header).write(filename)
    else:
        tif.TifImage(data=data_to_store).write(filename)


def open_seq(filenames):
    """_summary_

    Parameters
    ----------
    filenames : _type_
        _description_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    Exception
        _description_
    """
    if filenames:
        data = open_image(str(filenames[0]))
        height, width = data.shape
        to_return = np.zeros((len(filenames), height, width), dtype=np.float32)
        for i, file in enumerate(filenames):
            data = open_image(str(file))
            to_return[i, :, :] = data
        return to_return
    raise Exception('spytlabIOError')


def make_dark_mean(darkfields):
    """_summary_

    Parameters
    ----------
    darkfields : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # nbslices, height, width = darkfields.shape # ??? Why is this here
    mean_slice = np.mean(darkfields, axis=0)
    print('---------------  mean Dark calculation done ---------------- ')
    output_filename = '/Users/helene/PycharmProjects/spytlab/meanDarkTest.edf'
    output_edf = edf.EdfFile(output_filename, access='wb+')  # ??? What is this
    output_edf.WriteImage({}, mean_slice)
    return mean_slice


def save_edf(data, filename):
    """_summary_

    Parameters
    ----------
    data : _type_
        _description_
    filename : _type_
        _description_
    """
    # print(filename)
    data_to_store = data.astype(np.float32)
    edf.EdfImage(data=data_to_store).write(filename)


def save_edf_3d(data, filename):
    """_summary_

    Parameters
    ----------
    data : _type_
        _description_
    filename : _type_
        _description_
    """
    nb_slices, height, width = data.shape
    for i in range(nb_slices):
        text_slice = f'{i:04d}'
        data_to_save = data[i, :, :]
        filename_slice = filename+text_slice+'.edf'
        save_edf(data_to_save, filename_slice)


# def savePNG(data,filename,min=0,max=0):
    # if min == max:
    #    min=np.amin(data)
    #    max= np.amax(data)
    # data16bit=data-min/(max-min)
    # data16bit=np.asarray(data16bit,dtype=np.uint16)

    # scipy.misc.imsave(filename,data16bit)


if __name__ == "__main__":

    # filename='ref1-1.edf'
    # filenames=glob.glob('*.edf')
    # data=openImage(filename)
    # savePNG(data,'ref.png',100,450)
    # print( data.shape)
    #
    # rootfolder =
    # '/Volumes/VISITOR/md1097/id17/Phantoms/TwoDimensionalPhantom/GrilleFils/Absorption52keV/'
    # referencesFilenames = glob.glob(rootfolder + 'Projref/*.edf')
    # sampleFilenames = glob.glob(rootfolder + 'Proj/*.edf')
    # referencesFilenames.sort()
    # sampleFilenames.sort()
    # print(' lalalal ')
    # print (referencesFilenames)
    # print (sampleFilenames)

    INPUT_IMAGE_FILENAME = '''/Volumes/ID17/speckle/md1097/id17/Phantoms/\
ThreeDimensionalPhantom/OpticalFlow/dx32/\
dx_Speckle_Foam1_52keV_6um_xss_bis_012_0000.edf'''
    DATA = open_image(INPUT_IMAGE_FILENAME)
    print(DATA.dtype)
    print(DATA)
    OUTPUT_IMAGE_FILENAME = '''/Volumes/ID17/speckle/md1097/id17/Phantoms/\
ThreeDimensionalPhantom/OpticalFlowTest26Apr/dx0001_32bit.edf'''
    save_edf(DATA, OUTPUT_IMAGE_FILENAME)
    print(DATA)
    print('At the end '+str(DATA.dtype))
