// frame.hpp - FrameCamera class declaration
//
// FrameCamera reads frames from a V4L2 camera device.
// Used directly, it is the COLOR camera (YUYV format).
// DepthCamera (depth.hpp) inherits from it and reuses all the camera code.
//
// How to use it:
//   FrameCamera cam;
//   cam.open("/dev/video8", 640, 360, 90);
//   unsigned char* data = cam.getImage();    // wait for and take a frame
//   cam.convertToRGB(data, output, 640);     // turn it into RGB pixels
//   cam.returnImage();                        // give it back to the camera

#ifndef FRAME_HPP   // these 2 lines and the #endif at the bottom make sure
#define FRAME_HPP   // this file is included only once

#include <linux/videodev2.h>   // V4L2 camera definitions (v4l2_buffer, formats)

class FrameCamera {
public:
    FrameCamera();            // constructor: runs when the object is created
    virtual ~FrameCamera();   // destructor: stops the camera and frees memory

    // Open the color stream (YUYV format) and start it. Returns true if it worked.
    bool open(const char* devicePath, int width, int height, int fps);

    // Take a filled frame from the camera. Returns a pointer to its data,
    // or NULL if something went wrong.
    unsigned char* getImage();

    // Give the frame back to the camera so it can fill it again
    void returnImage();

    // Needed by poll() to wait for new frames
    int getFileDescriptor();

    // Convert a YUYV frame to RGB pixels (0x00RRGGBB).
    // output:          where to write the pixels
    // outputRowLength: how many pixels one row of the output has
    //                  (can be bigger than the frame width, e.g. a wider window)
    void convertToRGB(unsigned char* data, unsigned int* output, int outputRowLength);

protected:
    // "protected" = can be used here AND in child classes (like DepthCamera)

    // Open the device with any pixel format. open() above and
    // DepthCamera::open() both use this.
    bool openStream(const char* devicePath, unsigned int pixelFormat, int width, int height, int fps);

    int width;    // frame width in pixels
    int height;   // frame height in pixels

    // Small helpers for making pixels.
    // They are written here in the header (not in frame.cpp) so the compiler
    // can put them directly inside the pixel loops = no function call per pixel.

    // Keep a number between 0 and 255 (the range of one color channel)
    static int limitTo255(int value) {
        if (value < 0) {
            return 0;
        }
        if (value > 255) {
            return 255;
        }
        return value;
    }

    // Make one pixel number from red, green and blue -> 0x00RRGGBB
    static unsigned int makePixel(int red, int green, int blue) {
        return (red << 16) | (green << 8) | blue;
    }

private:
    // "private" = only this class can use these
    static const int BUFFER_COUNT = 2;      // 2 buffers: the minimum to avoid dropping frames

    int fileDescriptor;                     // the opened /dev/videoX
    unsigned char* buffers[BUFFER_COUNT];   // where the camera writes the frames
    unsigned int bufferSizes[BUFFER_COUNT]; // size of each buffer in bytes
    v4l2_buffer currentBuffer;              // information about the latest frame
};

#endif
