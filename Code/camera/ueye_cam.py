import numpy as np
from pyueye import ueye

"""
Class to manage initialization of ueye camera module and reading from its live video capture.
"""
class ueye_cam:
    #sInfo = ueye.SENSORINFO()
    #cInfo = ueye.CAMINFO()
    pcImageMemory = ueye.c_mem_p()
    MemID = ueye.int()
    rectAOI = ueye.IS_RECT()
    pitch = ueye.INT()
    nBitsPerPixel = ueye.INT(8)  # monochrome
    m_nColorMode = ueye.INT()  # Y8
    bytes_per_pixel = int(nBitsPerPixel / 8)

    def __init__(self, camId=0):

        print("Initializing camera module")
        self.hCam = ueye.HIDS(camId)

        # Starts the driver and establishes the connection to the camera
        nRet = ueye.is_InitCamera(self.hCam, None)
        if nRet != ueye.IS_SUCCESS:
            print("is_InitCamera ERROR")

        nRet = ueye.is_ResetToDefault(self.hCam)
        if nRet != ueye.IS_SUCCESS:
            print("is_ResetToDefault ERROR")

        nRet = ueye.is_SetDisplayMode(self.hCam, ueye.IS_SET_DM_DIB)

        # for monochrome camera models use Y8 mode
        self.m_nColorMode = ueye.IS_CM_MONO8
        self.nBitsPerPixel = ueye.INT(8)
        self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
        print("Monochrome Mode")

        # Area of interest can be set here
        nRet = ueye.is_AOI(self.hCam, ueye.IS_AOI_IMAGE_GET_AOI, self.rectAOI, ueye.sizeof(self.rectAOI))
        if nRet != ueye.IS_SUCCESS:
            print("is_AOI ERROR")

        self.width = self.rectAOI.s32Width
        self.height = self.rectAOI.s32Height

        # Prints out some information about the camera and the sensor
        #print("Camera model:\t\t", self.sInfo.strSensorName.decode('utf-8'))
        #print("Camera serial no.:\t", self.cInfo.SerNo.decode('utf-8'))
        print("Maximum image width:\t", self.width)
        print("Maximum image height:\t", self.height)

        # Allocates an image memory for an image having its dimensions defined by width and height and its color depth defined by nBitsPerPixel
        nRet = ueye.is_AllocImageMem(self.hCam, self.width, self.height, self.nBitsPerPixel, self.pcImageMemory, self.MemID)
        if nRet != ueye.IS_SUCCESS:
            print("is_AllocImageMem ERROR")
        else:
            # Makes the specified image memory the active memory
            nRet = ueye.is_SetImageMem(self.hCam, self.pcImageMemory, self.MemID)
            if nRet != ueye.IS_SUCCESS:
                print("is_SetImageMem ERROR")
            else:
                # Set the desired color mode
                nRet = ueye.is_SetColorMode(self.hCam, self.m_nColorMode)

        # Activates the camera's live video mode (free run mode)
        nRet = ueye.is_CaptureVideo(self.hCam, ueye.IS_DONT_WAIT)
        if nRet != ueye.IS_SUCCESS:
            print("is_CaptureVideo ERROR")

        ueye.is_SetRopEffect(self.hCam, ueye.IS_SET_ROP_MIRROR_UPDOWN, 1, 0)
        ueye.is_SetRopEffect(self.hCam, ueye.IS_SET_ROP_MIRROR_LEFTRIGHT, 1, 0)

        # Enables the queue mode for existing image memory sequences
        nRet = ueye.is_InquireImageMem(self.hCam, self.pcImageMemory, self.MemID, self.width, self.height, self.nBitsPerPixel, self.pitch)
        if nRet != ueye.IS_SUCCESS:
            print("is_InquireImageMem ERROR")

        # enable trigger for new frame
        ueye.is_EnableEvent(self.hCam, ueye.IS_SET_EVENT_FRAME)

        print("Initialized!")

    """
    def set_pixel_clock(self, clock_mhz):
        clk_float = ueye.INT(clock_mhz)
        ueye.is_PixelClock(self.hCam, ueye.IS_PIXELCLOCK_CMD_SET, clk_float, ueye.sizeof(clk_float))

    def set_frame_rate(self, fps):
        dfps = ueye.DOUBLE(0)
        ueyeFps = ueye.DOUBLE(fps)
        ueye.is_SetFrameRate(self.hCam, ueyeFps, dfps)
        print("new Framerate: ", dfps)

    def set_exposure(self, exp):
        exp_double = ueye.DOUBLE(exp)
        ueye.is_Exposure(self.hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, exp_double, ueye.sizeof(exp_double));
        print("new exposure: ", exp_double)

    def set_hw_gain_factor(self, gain):
        gain_int = ueye.INT(gain)
        ueye.is_SetHWGainFactor(self.hCam, ueye.IS_SET_MASTER_GAIN_FACTOR, gain_int);
        
    """

    def read(self):
        nRet = ueye.is_WaitEvent(self.hCam, ueye.IS_SET_EVENT_FRAME, 1000)
        if (nRet != ueye.IS_SUCCESS):
            print("pic capture failed")

        #extract image data from memory
        array = ueye.get_data(self.pcImageMemory, self.width, self.height, self.nBitsPerPixel, self.pitch, copy=False)

        # reshape into numpy array
        gray = np.reshape(array, (self.height.value, self.width.value, self.bytes_per_pixel))

        return gray

    def __del__(self):
        ueye.is_DisableEvent(self.hCam, ueye.IS_SET_EVENT_FRAME)
        # Releases an image memory that was allocated using is_AllocImageMem() and removes it from the driver management
        ueye.is_FreeImageMem(self.hCam, self.pcImageMemory, self.MemID)
        # Disables the self.self.hCam camera handle and releases the data structures and memory areas taken up by the uEye camera
        ueye.is_ExitCamera(self.hCam)
