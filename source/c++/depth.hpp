// depth.hpp - DepthCamera class declaration
//
// DepthCamera is a FrameCamera that gives DEPTH frames (Z16 format).
// It inherits all the camera code from FrameCamera (getImage, returnImage, ...)
// and only adds what is special for depth.
//
// In Z16, every pixel is a 16-bit number = distance in millimeters (0 = no data).
//
// How to use it:
//   DepthCamera cam;
//   cam.open("/dev/video4", 640, 360, 90);
//   unsigned char* data = cam.getImage();
//   int mm = cam.getDistance(data, 320, 180);   // distance at the center
//   cam.returnImage();

#ifndef DEPTH_HPP
#define DEPTH_HPP

#include "frame.hpp"

class DepthCamera : public FrameCamera {   // ": public FrameCamera" = DepthCamera is a FrameCamera
public:
    // Open the depth stream (Z16 format) and start it. Returns true if it worked.
    bool open(const char* devicePath, int width, int height, int fps);

    // Distance in millimeters at pixel (x, y). 0 = no data.
    int getDistance(unsigned char* data, int x, int y);

    // Convert a depth frame to gray pixels for showing on screen:
    //   near -> bright, far -> dark, no data -> black (shows 0 to ~5 m)
    void convertToGray(unsigned char* data, unsigned int* output, int outputRowLength);
};

#endif
