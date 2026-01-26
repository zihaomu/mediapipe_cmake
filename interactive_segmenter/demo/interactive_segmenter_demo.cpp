#include <opencv2/opencv.hpp>
#include "interactive_segmenter.h"

using namespace mpp;
using namespace cv;
using namespace std;

cv::Mat image;        // Original image
cv::Mat display;      // Display image
bool drawing = false;
cv::Point prevPoint;
bool runSeg = false;

const string root_path = "/home/moo/work/my_lab/mpp_project/mediapiep_cmake_private/";
InterativeSegmenter segmenter(
    root_path + "interactive_segmenter/models/magic_touch.mnn"
);

vector<cv::Point> mouseClick;

void drawHint(cv::Mat& img)
{
    const std::string hint = "c: clear   esc: quit";
    int font = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.6;
    int thickness = 1;

    // Green outline
    cv::putText(img, hint, cv::Point(11, 31),
                font, scale, cv::Scalar(0, 255, 0), thickness, cv::LINE_AA);
    // Bright green text
    cv::putText(img, hint, cv::Point(10, 30),
                font, scale, cv::Scalar(0, 255, 0), thickness, cv::LINE_AA);
}

void onMouse(int event, int x, int y, int, void*)
{
    if (runSeg) return;

    if (event == EVENT_LBUTTONDOWN) {
        drawing = true;
        prevPoint = Point(x, y);
        mouseClick.push_back(prevPoint);
        circle(display, prevPoint, 2, Scalar(0, 0, 255), -1);
        imshow("Image", display);
    }
    else if (event == EVENT_MOUSEMOVE && drawing) {
        Point current(x, y);
        line(display, prevPoint, current, Scalar(0, 0, 255), 2);
        prevPoint = current;
        mouseClick.push_back(prevPoint);
        imshow("Image", display);
    }
    else if (event == EVENT_LBUTTONUP) {
        drawing = false;
        runSeg = true;
    }
}

int main()
{
    image = imread(root_path + "data/body_image/test3.jpeg");

    
    if (image.empty()) {
        cerr << "Error: Image not found." << endl;
        return -1;
    }

    display = image.clone();
    drawHint(display);
    
    namedWindow("Image", WINDOW_AUTOSIZE);
    imshow("Image", display);
    setMouseCallback("Image", onMouse);

    while (true)
    {
        if (runSeg)
        {
            Mat out;
            segmenter.run(image, mouseClick, out);
            imshow("maskOut", out);
            runSeg = false;
        }

        int key = waitKey(1);
        if (key == 27) break;  // ESC

        if (key == 'c' || key == 'C') {
            display = image.clone();
            mouseClick.clear();
            drawing = false;
            runSeg = false;
            drawHint(display);
            imshow("Image", display);
        }
    }

    destroyAllWindows();
    return 0;
}
