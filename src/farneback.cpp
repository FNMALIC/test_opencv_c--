#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
using namespace cv;
using namespace std;
int main()
{
 VideoCapture capture(samples::findFile("D:/Y4P/code/test_opencv/1.mp4"));
 
 int deviceID = 0; // 0 = open default camera
 int apiID = cv::CAP_ANY; // 0 = autodetect default API
 // open selected camera using selected API
 capture.open(deviceID, apiID);

 if (!capture.isOpened()){
 //error in opening the video input
 cerr << "Unable to open file!" << endl;
 return 0;
 }

 Mat frame, frame1, prvs;
 capture >> frame1;
 
 cvtColor(frame1, prvs, COLOR_BGR2GRAY);

 while(true){
    Mat frame2, next;
    capture >> frame2;
 if (frame2.empty())
    break;
 
 cvtColor(frame2, next, COLOR_BGR2GRAY);
 Mat flow(prvs.size(), CV_32FC2);


 calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
 // visualization
 Mat flow_parts[2];
 split(flow, flow_parts);
 Mat magnitude, angle, magn_norm;
 cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
 normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
 angle *= ((1.f / 360.f) * (180.f / 255.f));
 //build hsv image
 Mat _hsv[3], hsv, hsv8, bgr;
 _hsv[0] = angle;
 _hsv[1] = Mat::ones(angle.size(), CV_32F);
 _hsv[2] = magn_norm;
 merge(_hsv, 3, hsv);
 hsv.convertTo(hsv8, CV_8U, 255.0);
 cvtColor(hsv8, bgr, COLOR_HSV2BGR);

 // wait for a new frame from camera and store it into 'frame'
 capture.read(frame);
 // check if we succeeded
 if (frame.empty()) {
 cerr << "ERROR! blank frame grabbed\n";
 break;
 }
 imshow("frame2", bgr);
 imshow("Live", frame);
 int keyboard = waitKey(30);
 if (keyboard == 'q' || keyboard == 27)
    break;
 prvs = next;
 }
}



