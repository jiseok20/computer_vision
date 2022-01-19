#include<opencv2/opencv.hpp>
#include<iostream>
#include<math.h>
#include<opencv2/imgproc.hpp>
using namespace std;
using namespace cv;

Mat LINE_ROI(Mat edge) {
	int width = edge.cols;
	int heigth = edge.rows;
	Point points[4];
	points[0] = Point(170, 270);
	points[1] = Point(520, 270);
	points[2] = Point(680, 400);
	points[3] = Point(90, 400);

	Mat img_mask = Mat::zeros(heigth, width, CV_8UC1);
	const Point* ppt[1] = { points };
	int npt[] = { 4 };
	fillPoly(img_mask, ppt, npt, 1, Scalar(255, 255, 255), LINE_8);
	Mat img_result;
	bitwise_and(edge, img_mask, img_result);
	return img_result;
}
Mat CAR_ROI(Mat hsv) {

	int width = hsv.cols;
	int heigth = hsv.rows;
	erode(hsv, hsv, Mat::ones(Size(5,5), CV_8UC1), Point(-1, -1), 1);
	Point points[4];
	points[0] = Point(0, 150);
	points[1] = Point(350, 150);
	points[2] = Point(450, 300);
	points[3] = Point(0, 300);

	Mat img_mask = Mat::zeros(heigth, width,CV_8UC3);
	const Point* ppt[1] = { points };
	int npt[] = { 4 };
	fillPoly(img_mask, ppt, npt, 1, Scalar(255, 255, 255), LINE_8);
	Mat img_result;
	bitwise_and(hsv, img_mask, img_result);
	return img_result;
}
void DRAW_CAR(Mat car_roi, Mat frame) {

	Mat canny_output;
	vector<vector<Point>> contours;
	vector<Vec4i>hierarchy;
	Canny(car_roi, canny_output, 100, 200);
	findContours(canny_output, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	vector<vector<Point>> conPoly(contours.size());
	vector<Rect>boundRect(contours.size());

	int maxArea = 0;

	for (int i = 0; i < contours.size(); i++) {
		int area = contourArea(contours[i]);
		if (area > 20) {
			float peri = arcLength(contours[i], true);
			approxPolyDP(contours[i], conPoly[i], 0.01 * peri, true);
			drawContours(frame, conPoly, i, Scalar(0, 0,255), 2);
		}
	}	
}
void DRAW_LINE(Mat ROI_edge,Mat frame) {
	
	vector<Vec4i> lines;
	int delta;
	HoughLinesP(ROI_edge, lines, 1, CV_PI / 180, 30, 30, 10);
	for (size_t i = 0; i < lines.size(); i++) {
		Vec4i l = lines[i];
		delta = abs(cvFastArctan(int(l[0] - l[2]), int(l[1] - l[3])) / 3.14);
		if (delta > 95) {
			//cout << "왼쪽 차선" << delta << endl;
			line(frame, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 0), 4, CV_AA);
		}

		else if (delta < 76 && delta > 70) {
			line(frame, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 255,0), 4, CV_AA);
			//cout << "오른쪽 차선" << delta << endl;
		}
	}
}
int main() {
	VideoCapture cap("data/blackbox.mp4");	
	if (!cap.isOpened())
		return -1;
	while (1) {
		Mat frame;
		Mat gray, blur, edge,ROI_edge, car_roi,car_edge;
		Mat white, white_img, img_hsv,car,car_blur;
		cap >> frame;
		if (frame.empty()) {
			return 0;
		}
		resize(frame, frame, Size(640,480));
		cvtColor(frame,gray, COLOR_BGR2GRAY);
		cvtColor(frame, img_hsv, COLOR_BGR2HSV);

		GaussianBlur(gray, blur, Size(5,5), 0);
		Canny(blur, edge, 90, 150);
		ROI_edge = LINE_ROI(edge);
		DRAW_LINE(ROI_edge, frame);

		inRange(img_hsv, Scalar(0, 0, 200), Scalar(180, 10, 255), white);
		bitwise_and(frame,img_hsv , white_img, white);

		car_roi = CAR_ROI(white_img);
		DRAW_CAR(car_roi, frame);
		imshow("roi_line", ROI_edge);
		imshow("car_HSV_ROI", car_roi);
		imshow("result", frame);
		if (waitKey(30) >= 0)
			break;
	}
	
	return 0;
}