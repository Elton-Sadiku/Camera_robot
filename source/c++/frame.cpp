// frame.cpp - FrameCamera class implementation (the code behind frame.hpp)

#include "frame.hpp"

#include <fcntl.h>       // open()
#include <unistd.h>      // close()
#include <sys/ioctl.h>   // ioctl() - used to talk to the camera
#include <sys/mman.h>    // mmap()  - used to see the camera frames in memory
#include <cstdio>        // printf()

// "FrameCamera::" means "this function belongs to the FrameCamera class"

FrameCamera::FrameCamera() {
    width = 0;
    height = 0;
    fileDescriptor = -1;
    for (int i = 0; i < BUFFER_COUNT; i++) {
        buffers[i] = NULL;
        bufferSizes[i] = 0;
    }
    currentBuffer = {};
}

// Stops the camera and frees everything we used
FrameCamera::~FrameCamera() {
    if (fileDescriptor < 0) {
        return;  // camera was never opened
    }

    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(fileDescriptor, VIDIOC_STREAMOFF, &type);

    for (int i = 0; i < BUFFER_COUNT; i++) {
        if (buffers[i] != NULL) {
            munmap(buffers[i], bufferSizes[i]);
        }
    }
    close(fileDescriptor);
}

bool FrameCamera::open(const char* devicePath, int width, int height, int fps) {
    return openStream(devicePath, V4L2_PIX_FMT_YUYV, width, height, fps);
}

bool FrameCamera::openStream(const char* devicePath, unsigned int pixelFormat,
                             int width, int height, int fps) {
    // Remember the size ("this->" = the class member, not the parameter)
    this->width = width;
    this->height = height;

    // 1. Open the device file (:: means "the Linux open(), not our method")
    fileDescriptor = ::open(devicePath, O_RDWR);
    if (fileDescriptor < 0) {
        printf("Error: cannot open %s\n", devicePath);
        return false;
    }

    // 2. Tell the camera the frame size and format
    v4l2_format format = {};
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    format.fmt.pix.width = width;
    format.fmt.pix.height = height;
    format.fmt.pix.pixelformat = pixelFormat;

    if (ioctl(fileDescriptor, VIDIOC_S_FMT, &format) < 0) {
        printf("Error: %s does not support this format\n", devicePath);
        return false;
    }

    // 3. Tell the camera how many frames per second we want
    v4l2_streamparm streamSettings = {};
    streamSettings.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    streamSettings.parm.capture.timeperframe.numerator = 1;
    streamSettings.parm.capture.timeperframe.denominator = fps;
    ioctl(fileDescriptor, VIDIOC_S_PARM, &streamSettings);

    // 4. Ask the camera to prepare some buffers (memory for frames)
    v4l2_requestbuffers request = {};
    request.count = BUFFER_COUNT;
    request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    request.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fileDescriptor, VIDIOC_REQBUFS, &request) < 0) {
        printf("Error: cannot get buffers for %s\n", devicePath);
        return false;
    }

    // 5. Connect each buffer to our program, then give it to the camera to fill
    for (int i = 0; i < BUFFER_COUNT; i++) {
        v4l2_buffer buffer = {};
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.index = i;

        // Ask where this buffer is and how big it is
        ioctl(fileDescriptor, VIDIOC_QUERYBUF, &buffer);

        // Make the buffer visible in our program (no copying needed)
        bufferSizes[i] = buffer.length;
        buffers[i] = (unsigned char*)mmap(NULL, buffer.length, PROT_READ,
                                          MAP_SHARED, fileDescriptor, buffer.m.offset);

        // Give the empty buffer to the camera
        ioctl(fileDescriptor, VIDIOC_QBUF, &buffer);
    }

    // 6. Start the camera
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fileDescriptor, VIDIOC_STREAMON, &type) < 0) {
        printf("Error: cannot start %s\n", devicePath);
        return false;
    }

    // Prepare the structure we use later to receive frames
    currentBuffer = {};
    currentBuffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    currentBuffer.memory = V4L2_MEMORY_MMAP;

    printf("Opened %s\n", devicePath);
    return true;
}

unsigned char* FrameCamera::getImage() {
    if (ioctl(fileDescriptor, VIDIOC_DQBUF, &currentBuffer) < 0) {
        return NULL;
    }
    return buffers[currentBuffer.index];
}

void FrameCamera::returnImage() {
    ioctl(fileDescriptor, VIDIOC_QBUF, &currentBuffer);
}

int FrameCamera::getFileDescriptor() {
    return fileDescriptor;
}

// In YUYV, every 4 bytes describe 2 pixels:
//   Y0  U  Y1  V
//   Y0 = brightness of pixel 1
//   Y1 = brightness of pixel 2
//   U, V = color, shared by both pixels
void FrameCamera::convertToRGB(unsigned char* data, unsigned int* output, int outputRowLength) {
    for (int y = 0; y < height; y++) {
        unsigned int* row = output + y * outputRowLength;  // start of this row

        for (int x = 0; x < width; x += 2) {
            int y0 = data[0];
            int u  = data[1];
            int y1 = data[2];
            int v  = data[3];
            data += 4;  // move to the next 2 pixels

            // Standard YUV -> RGB formula (using whole numbers, it is faster)
            int c0 = 298 * (y0 - 16);
            int c1 = 298 * (y1 - 16);
            int d = u - 128;
            int e = v - 128;

            int redPart   = 409 * e + 128;
            int greenPart = -100 * d - 208 * e + 128;
            int bluePart  = 516 * d + 128;

            // Pixel 1
            int red   = limitTo255((c0 + redPart) >> 8);
            int green = limitTo255((c0 + greenPart) >> 8);
            int blue  = limitTo255((c0 + bluePart) >> 8);
            row[x] = makePixel(red, green, blue);

            // Pixel 2
            red   = limitTo255((c1 + redPart) >> 8);
            green = limitTo255((c1 + greenPart) >> 8);
            blue  = limitTo255((c1 + bluePart) >> 8);
            row[x + 1] = makePixel(red, green, blue);
        }
    }
}
