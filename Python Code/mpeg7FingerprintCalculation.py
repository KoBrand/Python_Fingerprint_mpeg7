__author__ = 'KoBra'

import numpy as np
from scipy.io import wavfile
import os


def nextpow2(x):
    ''' tells the minimum binary exponent to discribe this number
    Argrs:
        x: (int) number of interest
    Returns:
         n: (int) minimum binary exponent
    '''
    n = np.log2(abs(x))

    if n-np.floor(n) != 0:
        n =np.floor(n)+1

    return n


def h_specgram2(x, nfft, fs, window, shift):
    """ Calculates a spectrogram
    Args:
        x: (ndarray) mono audiofile
        nfft: (int) windowsize of FFT (2048)
        fs: (int) samplingrate (4100)
        window: (ndarray) np.hamming(windosize) windowsize = 1323
        shift: (int) hopsize (441)
    Return:
         y4:
    """

    nx = x.shape[0]
    nwind = window.shape[0]
    if nx < nwind:   # zero - pad x if it has length less than the window length
        x[nwind] = 0
        nx = nwind

    # no overlap
    noverlap = nwind - np.mean(shift)
    ncol = int(np.fix((nx - noverlap) / (nwind - noverlap)))

    totalshift = np.sum(shift)
    try:
        numshift = shift.shape[0]
    except:
        numshift = 1
    progshift = 0  # (cumsum(shift) - shift(1))
    overshift = np.arange(0, nx, totalshift)

    shifts = overshift
    colindex = shifts
    colindex = colindex[0:ncol]
    rowindex = np.arange(0, nwind)
    if x.shape[0] < (nwind + colindex[ncol-1]):
        x[nwind + colindex[ncol] - 1] = 0  # % zero - pad x

    rowmat = rowindex
    rowmat = np.append([rowmat], [rowindex], axis=0)

    for i in np.arange(ncol-2):
        rowmat = np.append(rowmat, [rowindex], axis=0)

    rowmat = np.rot90(rowmat, 3)

    colmat = colindex
    colmat = np.append([colmat], [colindex], axis=0)
    for i in np.arange(nwind-2):
        colmat= np.append(colmat, [colindex], axis=0)

    # put x into columns of y with the proper offset
    y2 = x[rowmat+colmat]

    windowmat = window
    windowmat = np.append([windowmat], [window], axis=0)
    for i in np.arange(ncol - 2):
        windowmat = np.append(windowmat, [window], axis=0)
    windowmat = np.rot90(windowmat, 3)

    # Apply the window to the array of offset signal segments.
    y = windowmat*y2
    # now fft y which does the columns
    y3 = np.fft.fft(y, nfft, axis=0)
    if np.any(np.imag(x)) == False:
        if np.fmod(nfft, 2) != 0:
            select = np.arange(nfft+1 / 2)
        else:
            select = np.floor(np.arange(nfft / 2 + 1))
            select = select.astype(int)

        y4 = y3[select, :]
    else:
        select = np.arange(nfft)

    f = (select)*fs/nfft
    t = colindex/fs

    return y4, f, t


def mpeg7getspec(data, hopsize, window, windowsize, fftSize, fs):
    """ basic steps from the MPEG 7 standard
    Args:
        data:
        hopsize: (ndarray) [10 1000]
        window : (ndarray) np.hamming(windosize)
    Return:
         fftout:
         phase:
    """
    ####################################################
    #calculate hopsize
    ######################################################

    hs = np.mean(hopsize)
    hops = 1
    num_f = np.ceil(data.shape[0]/hs)
    rem_f = np.mod(num_f, hops)
    pad = 0 - (data.shape[0]-(num_f-rem_f)*hs)
    data = np.append(data, np.zeros(pad))   # adds pad zeros to data
    NormWindow = sum(window*window)
    spec = h_specgram2(data, fftSize, fs, window, hopsize)/np.sqrt(fftSize*NormWindow)
    fftout = np.abs(spec)
    phase = np.angle(spec)

    if pad:
        fftout[:, -1] = fftout[:, -1]*np.sqrt(windowsize/(windowsize-pad))
        # change last row because of added zeros! If there are no added zeros this step is not needed

    return fftout, phase


def AudioSpectrumFlatness(wavArray, samplingRate, hopSize = 'PT10N1000F', loEdge = 250, hiEdge =16000):
    """ Calculation of an Audio Spectrum Flatess of an audio file
    Args:
        audioFile: (str) Path of audio file wave only
        samplingRate: (int) 44100
        hopSize: (int) Hopsize (30ms is recomanded)
        loEdge: (int) Lower edge frequency (a default value of 250 is assumed)
        hiEdge: (int) Upper edge frequency (a default value of 16000 is assumed)
    Returns:
        audioSpectrumFlatness:  (2d ndarray) Matrix (rowwsxcols)
                                Description of the audio spectral
                                flatness of the audio signal
    """

    # Start calculating descriptors
    # hopsize is allways PT10N1000F
    hop = np.array([10, 1000])

    # Check loEdge and hiEdge
    if loEdge != 250:
        return

    if hiEdge != 16000:
        return

    ######################################################
    # STFT with no overlap; window size equal to  hopsize.
    # Zero padding of the last few frames will occur, to ensure there is one spectral frame
    # for each corresponding power estimate in the power descriptor.
    #######################################################

    # AudioSpectrumFlatness calculation:
    windowsize = 1323  # always
    window = np.hamming(windowsize)
    FFTsize= int(2**nextpow2(windowsize))
    struc_FFtsize = FFTsize

    struck_hopsize = 1323  # 441 or 1323
    fs = samplingRate
    N = struc_FFtsize

    fftout, phase = mpeg7getspec(wavArray, struck_hopsize, window, windowsize, struc_FFtsize, samplingRate)

    numbands = int(np.floor(4*np.log2(hiEdge/loEdge)))
    firstband = int(np.round(np.log2(loEdge/1000)*4))
    overlap = 0.05

    check = 1

    for k in range(1, numbands+1):
        f_lo = loEdge*(2**((k-1)/4))*(1-overlap)
        f_hi = loEdge*(2**(k/4))*(1+overlap)
        i_lo = round(f_lo/(fs/N))+1  # Calculate wht the frequencies are in the array
        i_hi = round(f_hi/(fs/N))+1

        # Rounding of upper index according due to coefficient grouping
        if (k+firstband-1 >= 0):  #Start grouping at 1kHz
            grpsize = 2**np.ceil((k+firstband)/4)
            i_hi = round((i_hi-i_lo+1)/grpsize)*grpsize + i_lo-1
        else:
            grpsize = 1

        tmp = fftout[i_lo-1:i_hi, :]**2  # PSD coefficients
        ncoeffs = i_hi-i_lo+1

        if k+firstband-1 >= 0:
            tmp2 = tmp[:ncoeffs:grpsize, :]
            for g in np.arange(1, grpsize):
                tmp2 = tmp2 + tmp[g:ncoeffs:grpsize, :]
            tmp = tmp2

        # Actual calculation
        ncoeffs = ncoeffs/grpsize
        tmp = tmp + 1e-50  # to avoid devition by zero

        a = 0
        if check == 1:
            gmmat = np.exp(np.sum(np.log(tmp), axis=0)/ncoeffs)   # geometrical mean of PSD
            ammat  = np.sum(tmp, axis=0)/ncoeffs    # Arethmemtic mean of PSD
            check = 2
        elif check == 2 :
            gm = np.exp(np.sum(np.log(tmp), axis=0)/ncoeffs)
            am  = np.sum(tmp, axis=0)/ncoeffs

            gmmat = np.append([gmmat], [gm], axis=0)
            ammat = np.append([ammat], [am], axis=0)
            check = 3
        elif check == 3:
            gm = np.exp(np.sum(np.log(tmp), axis=0)/ncoeffs)
            am = np.sum(tmp, axis=0)/ncoeffs

            gmmat = np.append(gmmat, [gm], axis=0)
            ammat = np.append(ammat, [am], axis=0)

    audiospectralflatness = np.transpose(gmmat/ammat)
    return audiospectralflatness


def AudioSignature(audioSpectrumFlatness, loEdge, highEdge, decim):
    """ This function extracts the values of the MPEG-7 Audio AudioSignature DS
        where audiosignal contains the raw data to be analysed
    Args:
        audioSpectrumFlatness: (nd2array) Audiospectral faltness type described by Mpeg7 standard
        loEdge: (int) lo edge = 250 default
        highEdge: (int) high edge = 16000 default
        decim: (int) optionally specifies the decimation factor = 32 default
    Returns:
        asMean:
        asVar:
    """
    as_mean = []
    as_var = []
    num_blocks = int(np.floor(audioSpectrumFlatness.shape[0] / decim))
    check = 1
    for k in range(num_blocks):
        block_data = audioSpectrumFlatness[k*decim:(k+1)*decim, :]
        if check == 1:
            as_mean = np.mean(block_data, axis=0)
            check = 2
        elif check == 2:
            block_mean = np.mean(block_data, axis=0)

            as_mean = np.append([as_mean], [block_mean], axis=0)
            check = 3
        elif check == 3:
            block_mean = np.mean(block_data, axis=0)
            as_mean = np.append(as_mean, [block_mean], axis=0)
    return as_mean, as_var


def readAudiofile(audioName):
    """ Reads a wave file and converts it to mono
    Args:
        audiname: (sting) path to the audioFile
    Return:
        wavArray: (array) mono wave array
        samplingRate: (int) used samplingrate
    """
    if (audioName.split(".")[-1] == "wav") | (audioName.split(".")[-1] == "WAV"):
        samplingRate, wavArray = wavfile.read(audioName)
    else:
        print('wrong file format! only .WAV is supported')
    try:
        if wavArray.shape[1] == 2:
            left = wavArray[:, 0]
            right = wavArray[:, 1]
            wavArray = left+right
    except:
        print('Wavefile is already mono')

    wavArray = wavArray/np.max(wavArray)
    return wavArray, samplingRate


def euclideanDistance(smallMatrix, bigMatrix):
    """ Calculate the euclidean distance of to matrixes using a slidingwindow comparison
    Args:
        smallMatrix: (ndarray) Matrix of interest
        bigMatrix: (ndarrray) Matrix you what to check
    Returs:
        euclideanDistances: (array) euclidean distances of all comparisons
    """
    samleMatrixLength = smallMatrix.shape[0]
    bigMatrixLength = bigMatrix.shape[0]
    searchinterval = abs(bigMatrixLength - samleMatrixLength) + 1
    euclideanDistances = []
    for k in range(0, searchinterval):
        C = abs(smallMatrix[:, :] - bigMatrix[k:samleMatrixLength + k, :])
        C1 = np.sum(C, axis=1)
        C2 = np.sum(C1)
        euclideanDistances.append(C2)
    return euclideanDistances


################################################################
# Fingerprint calculation of one song
################################################################

file = 'path/to/file.wav'
audioFile, samplingrate = readAudiofile(file)
spectralFlatness = AudioSpectrumFlatness(audioFile, samplingrate)
signature = AudioSignature(spectralFlatness, 250, 1600, 32)

# Save fingerprint as .csv file
np.savetxt(file[-4]+'.csv', signature[0], delimiter=",")


################################################################
# Automatic fingerprint calculation of a folder
################################################################
#
# location = os.path.abspath(os.path.dirname(__file__))  # same folder
# namesAll = fileimporter(location, 'wav')
#
# for file in namesAll:
#     fileNew = file.replace(' ', '')[:-4]
#     audioFile, samplingrate = readAudiofile(file)
#     spectralFlatness = AudioSpectrumFlatness(audioFile, samplingrate)
#     signature = AudioSignature(spectralFlatness, 250, 1600, 32)
#     # remove existing file
#     if not os.path.exists(location +'/csvFiles/'):
#         os.makedirs(location +'/csvFiles/')
#     try:
#         os.remove(location +'/csvFiles/'+ fileNew + '.csv')
#         print('Remove old .csv file')
#     except OSError:
#         print('no old file '+fileNew+'.csv has not been found')
#     # save as CSV
#     np.savetxt(location +'/csvFiles/'+ fileNew +'.csv', signature[0], delimiter=",")

