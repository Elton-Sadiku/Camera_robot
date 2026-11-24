// depth.cpp - DepthCamera class implementation (the code behind depth.hpp)

#include "depth.hpp"

bool DepthCamera::open(const char* devicePath, int width, int height, int fps) {
    // Same camera code as FrameCamera, only with the Z16 (depth) format
    return openStream(devicePath, V4L2_PIX_FMT_Z16, width, height, fps);
}

int DepthCamera::getDistance(unsigned char* data, int x, int y) {
    unsigned short* depth = (unsigned short*)data;  // 2 bytes per pixel
    return depth[y * width + x];
}

void DepthCamera::convertToGray(unsigned char* data, unsigned int* output, int outputRowLength) {
    unsigned short* depth = (unsigned short*)data;  // 2 bytes per pixel

    for (int y = 0; y < height; y++) {
        unsigned int* row = output + y * outputRowLength;  // start of this row

        for (int x = 0; x < width; x++) {
            int millimeters = depth[y * width + x];

            int gray = 0;  // 0 = no data -> black
            if (millimeters > 0) {
                gray = 255 - limitTo255(millimeters / 20);  // 20 mm per gray step
            }

            row[x] = makePixel(gray, gray, gray);
        }
    }
}
