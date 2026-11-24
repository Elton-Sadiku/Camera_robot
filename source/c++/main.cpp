// main.cpp - reads COLOR and DEPTH frames from an Intel RealSense D455
// and shows them side by side in a window.
//
// Files:
//   frame.hpp / frame.cpp  - FrameCamera: reads frames, color (YUYV -> RGB)
//   depth.hpp / depth.cpp  - DepthCamera: depth frames (Z16), distance, gray image
//   main.cpp               - this file: opens the cameras and shows the frames
//
// No copies:
//   - camera frames are read directly from the camera memory (mmap)
//   - each frame is converted ONCE, straight into memory shared with the
//     X server (XShm), so the picture is not copied again to show it
//
// Build:  g++ -O2 main.cpp frame.cpp depth.cpp -o d455 -lX11 -lXext
// Run:    ./d455
// Stop:   press Ctrl+C in the terminal

#include "frame.hpp"                // FrameCamera (color)
#include "depth.hpp"                // DepthCamera

#include <poll.h>                   // poll() - used to wait for new frames
#include <sys/shm.h>                // shared memory
#include <X11/Xlib.h>               // window functions
#include <X11/extensions/XShm.h>    // X shared memory images (no copy to the X server)
#include <cstdio>                   // printf()
#include <ctime>                    // time()

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

const char* COLOR_DEVICE = "/dev/video8";  // color camera of the D455
const char* DEPTH_DEVICE = "/dev/video4";  // depth camera of the D455

const int WIDTH  = 640;   // frame width in pixels
const int HEIGHT = 360;   // frame height in pixels
const int FPS    = 90;    // frames per second (90 is the maximum at 640x360)

int main() {
    // ---- Open both cameras ----
    FrameCamera colorCamera;
    DepthCamera depthCamera;

    if (!colorCamera.open(COLOR_DEVICE, WIDTH, HEIGHT, FPS)) {
        return 1;
    }
    if (!depthCamera.open(DEPTH_DEVICE, WIDTH, HEIGHT, FPS)) {
        return 1;
    }

    // ---- Open a window ----
    Display* display = XOpenDisplay(NULL);
    if (display == NULL) {
        printf("Error: cannot open display\n");
        return 1;
    }
    if (!XShmQueryExtension(display)) {
        printf("Error: X shared memory (XShm) is not available\n");
        return 1;
    }

    int screen = DefaultScreen(display);
    Window window = XCreateSimpleWindow(display, RootWindow(display, screen),
                                        0, 0, WIDTH * 2, HEIGHT, 0, 0, 0);
    XStoreName(display, window, "D455 - color | depth");
    XMapWindow(display, window);
    GC gc = DefaultGC(display, screen);

    // ---- Create the window picture in shared memory ----
    // The picture is twice as wide as one frame: color left, depth right.
    // Both our program and the X server use the SAME memory, so nothing is copied.
    XShmSegmentInfo sharedInfo;
    XImage* image = XShmCreateImage(display, DefaultVisual(display, screen), 24, ZPixmap,
                                    NULL, &sharedInfo, WIDTH * 2, HEIGHT);

    int imageBytes = image->bytes_per_line * image->height;
    sharedInfo.shmid = shmget(IPC_PRIVATE, imageBytes, IPC_CREAT | 0600);
    sharedInfo.shmaddr = (char*)shmat(sharedInfo.shmid, NULL, 0);
    sharedInfo.readOnly = False;
    image->data = sharedInfo.shmaddr;

    XShmAttach(display, &sharedInfo);
    XSync(display, False);
    shmctl(sharedInfo.shmid, IPC_RMID, NULL);  // memory is freed automatically when we exit

    unsigned int* pixels = (unsigned int*)image->data;   // 0x00RRGGBB per pixel
    int rowLength = image->bytes_per_line / 4;            // pixels per row of the picture

    // ---- Prepare to wait for both cameras at the same time ----
    pollfd waitList[2];
    waitList[0].fd = colorCamera.getFileDescriptor();
    waitList[0].events = POLLIN;   // POLLIN = "a new frame is ready"
    waitList[1].fd = depthCamera.getFileDescriptor();
    waitList[1].events = POLLIN;

    // For measuring frames per second
    int colorFrames = 0;
    int depthFrames = 0;
    int centerDistance = 0;   // millimeters at the center of the depth frame
    time_t lastSecond = time(NULL);

    // ---- Main loop: runs forever until Ctrl+C ----
    while (true) {
        // Wait until at least one camera has a new frame
        poll(waitList, 2, -1);

        // New color frame? -> convert into the left half and show only that half
        if (waitList[0].revents & POLLIN) {
            unsigned char* colorData = colorCamera.getImage();
            if (colorData == NULL) {
                printf("Error: cannot read color frame\n");
                return 1;
            }
            colorCamera.convertToRGB(colorData, pixels, rowLength);
            colorCamera.returnImage();

            XShmPutImage(display, window, gc, image, 0, 0, 0, 0, WIDTH, HEIGHT, False);
            colorFrames++;
        }

        // New depth frame? -> convert into the right half and show only that half
        if (waitList[1].revents & POLLIN) {
            unsigned char* depthData = depthCamera.getImage();
            if (depthData == NULL) {
                printf("Error: cannot read depth frame\n");
                return 1;
            }
            centerDistance = depthCamera.getDistance(depthData, WIDTH / 2, HEIGHT / 2);
            depthCamera.convertToGray(depthData, pixels + WIDTH, rowLength);
            depthCamera.returnImage();

            XShmPutImage(display, window, gc, image, WIDTH, 0, WIDTH, 0, WIDTH, HEIGHT, False);
            depthFrames++;
        }

        XFlush(display);  // send the draw commands to the X server

        // Once per second, print the frames per second and the center distance
        if (time(NULL) != lastSecond) {
            printf("FPS  color: %d   depth: %d   center: %d mm\n",
                   colorFrames, depthFrames, centerDistance);
            fflush(stdout);
            colorFrames = 0;
            depthFrames = 0;
            lastSecond = time(NULL);
        }
    }

    return 0;
}
