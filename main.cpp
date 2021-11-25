////---------------------------------��ͷ�ļ��������ռ�������֡�----------------------------
////        ����������������ʹ�õ�ͷ�ļ��������ռ�
////------------------------------------------------------------------------------------------------
////#include "stdafx.h"
//#include "include/opencv2/highgui/highgui.hpp"
//#include "include/opencv2/imgproc/imgproc.hpp"
//#include <iostream>
//using namespace cv;
//using namespace std;


////-----------------------------------���궨�岿�֡�--------------------------------------------
////        ����������һЩ������
////------------------------------------------------------------------------------------------------
//#define WINDOW_NAME1 "��ԭʼͼ���ڡ�"            //Ϊ���ڱ��ⶨ��ĺ�
//#define WINDOW_NAME2 "������ͼ��"                    //Ϊ���ڱ��ⶨ��ĺ�


////-----------------------------------��ȫ�ֱ����������֡�--------------------------------------
////        ������ȫ�ֱ���������
////-----------------------------------------------------------------------------------------------
//Mat g_srcImage;
//Mat g_grayImage;
//int g_nThresh = 80;
//int g_nThresh_max = 255;
//RNG g_rng(12345);
//Mat g_cannyMat_output;
//vector<vector<Point>> g_vContours;
//vector<Vec4i> g_vHierarchy;


////-----------------------------------��ȫ�ֺ����������֡�--------------------------------------
////        ������ȫ�ֺ���������
////-----------------------------------------------------------------------------------------------
//static void ShowHelpText( );
//void on_ThreshChange(int, void* );


////-----------------------------------��main( )������--------------------------------------------
////        ����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼִ��
////-----------------------------------------------------------------------------------------------
//int main( int argc, char** argv )
//{
//    //��0���ı�console������ɫ
//    system("color 1F");

//    // ����Դͼ��
//    g_srcImage = imread( "F:/qt/canny/canny/images/539.jpg", 1 );
//    if(!g_srcImage.data ) { printf("��ȡͼƬ������ȷ��Ŀ¼���Ƿ���imread����ָ����ͼƬ����~�� \n"); return false; }

//    // ת�ɻҶȲ�ģ��������
//    cvtColor( g_srcImage, g_grayImage, COLOR_BGR2GRAY );
//    blur( g_grayImage, g_grayImage, Size(3,3) );

//    // ��������
//    namedWindow( WINDOW_NAME1, WINDOW_AUTOSIZE );
//    imshow( WINDOW_NAME1, g_srcImage );

//    //��������������ʼ��
//    createTrackbar( "canny��ֵ", WINDOW_NAME1, &g_nThresh, g_nThresh_max, on_ThreshChange );
//    on_ThreshChange( 0, 0 );

//    waitKey(0);
//    return(0);
//}

////-----------------------------------��on_ThreshChange( )������------------------------------
////      �������ص�����
////----------------------------------------------------------------------------------------------
//void on_ThreshChange(int, void* )
//{

//    // ��Canny���Ӽ���Ե
//    Canny( g_grayImage, g_cannyMat_output, g_nThresh, g_nThresh*2, 3 );

//    // Ѱ������
//    findContours( g_cannyMat_output, g_vContours, g_vHierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

//    // �������
//    Mat drawing = Mat::zeros( g_cannyMat_output.size(), CV_8UC3 );
//    for( int i = 0; i< g_vContours.size(); i++ )
//    {
//        //Scalar color = Scalar( g_rng.uniform(0, 255), g_rng.uniform(0,255), g_rng.uniform(0,255) );//����ֵ
//        Scalar color = Scalar(255, 182, 193);
//        drawContours( drawing, g_vContours, i, color, 2, 8, g_vHierarchy, 0, Point() );
//    }

//    vector<Point> tempPoint;     // �㼯
//    // �����е㼯�洢��tempPoint
//    for (int k = 0; k < g_vContours.size(); k++)
//    {
//        for (int m = 0; m < g_vContours[k].size(); m++)
//        {
//            tempPoint.push_back(g_vContours[k][m]);
//        }
//    }
//    //�Ը����� 2D �㼯��Ѱ����С����İ�Χ����
//    RotatedRect box = minAreaRect(Mat(tempPoint));
//    Point2f vertex[4];
//    box.points(vertex);

//    //���Ƴ���С����İ�Χ����
//    for (int i = 0; i < 4; i++)
//    {
//        line(drawing, vertex[i], vertex[(i + 1) % 4], Scalar(100, 200, 211), 2, LINE_AA);
//    }

//    imshow(WINDOW_NAME2, drawing);
//}



////��ɫʶ��2
//#include <iostream>
//#include <include/opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <include/opencv2/opencv.hpp>

//using namespace cv;
//using namespace std;

//int main(int argc, char** argv)
//{
//    VideoCapture cap(0); //capture the video from web cam

//    if (!cap.isOpened())  // if not success, exit program
//    {
//        cout << "Cannot open the web cam" << endl;
//        return -1;
//    }

//    namedWindow("control", 1);
//    int ctrl = 0;
//    createTrackbar("ctrl", "control", &ctrl, 7);

//    while (true)
//    {
//        Mat imgOriginal;

//        bool bSuccess = cap.read(imgOriginal); // read a new frame from video
//        if (!bSuccess) //if not success, break loop
//        {
//            cout << "Cannot read a frame from video stream" << endl;
//            break;
//        }

//        // imgOriginal = imread("4.jpg");

//        Mat imgHSV, imgBGR;
//        Mat imgThresholded;

//        if(0)
//        {
//            vector<Mat> hsvSplit;   //�����������������HSV����ͨ������
//            cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
//            split(imgHSV, hsvSplit);			//����ԭͼ���HSV��ͨ��
//            equalizeHist(hsvSplit[2], hsvSplit[2]);    //��HSV������ͨ������ֱ��ͼ����
//            merge(hsvSplit, imgHSV);				   //�ϲ�����ͨ��
//            cvtColor(imgHSV, imgBGR, COLOR_HSV2BGR);    //��HSV�ռ�ת����RGB�ռ䣬Ϊ����������ɫʶ����׼��
//        }
//        else
//        {
//            imgBGR = imgOriginal.clone();
//        }



//        switch(ctrl)
//        {
//        case 0:
//            {
//                inRange(imgBGR, Scalar(128, 0, 0), Scalar(255, 127, 127), imgThresholded); //��ɫ
//                break;
//            }
//        case 1:
//            {
//                inRange(imgBGR, Scalar(128, 128, 128), Scalar(255, 255, 255), imgThresholded); //��ɫ
//                break;
//            }
//        case 2:
//            {
//                inRange(imgBGR, Scalar(128, 128, 0), Scalar(255, 255, 127), imgThresholded); //��ɫ
//                break;
//            }
//        case 3:
//            {
//                inRange(imgBGR, Scalar(128, 0, 128), Scalar(255, 127, 255), imgThresholded); //��ɫ
//                break;
//            }
//        case 4:
//            {
//                inRange(imgBGR, Scalar(0, 128, 128), Scalar(127, 255, 255), imgThresholded); //��ɫ
//                break;
//            }
//        case 5:
//            {
//                inRange(imgBGR, Scalar(0, 128, 0), Scalar(127, 255, 127), imgThresholded); //��ɫ
//                break;
//            }
//        case 6:
//            {
//                inRange(imgBGR, Scalar(0, 0, 128), Scalar(127, 127, 255), imgThresholded); //��ɫ
//                break;
//            }
//        case 7:
//            {
//                inRange(imgBGR, Scalar(0, 0, 0), Scalar(127, 127, 127), imgThresholded); //��ɫ
//                break;
//            }
//        }

//        imshow("��̬ѧȥ����ǰ", imgThresholded);

//        Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
//        morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);
//        morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, element);

//        imshow("Thresholded Image", imgThresholded); //show the thresholded image
//        imshow("ֱ��ͼ�����Ժ�", imgBGR);
//        imshow("Original", imgOriginal); //show the original image

//        char key = (char)waitKey(300);
//        if (key == 27)
//            break;
//    }

//    return 0;

//}








////pnp
//#include <ctime>
//#include <string>
//#include <cstdio>
//#include <iostream>
//#include <math.h>
//#include <stdlib.h>
//#include <include/opencv2/opencv.hpp>
//#include <include/opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <include/opencv2/aruco.hpp>
//using namespace std;
//using namespace cv;



//const Mat  intrinsic_matrix = (Mat_<float>(3, 3)
//                               << 868.2187,  0,         345.93152,
//                                  0,         870.4765,  252.03741,
//                                  0,         0,         1);

////k1,k2,p1,p2,k3
//const Mat  distCoeffs = (Mat_<float>(5, 1) << 0.17153, 0.06634, 0.00465, 0.00868, 0);

////����������
//const Mat  arucodistCoeffs = (Mat_<float>(1, 5) << 0, 0, 0, 0, 0);//

////�ж�solvePnPRansac�Ƿ����
//bool PnPRansac;

//int main(int args, char *argv[])
//{
//    //ͼ����������
//    //vector<vector<Point3f> > projectedPoints;
//    vector<Point3f> imgpoint;

//    Point3f pointimg0, pointimg1, pointimg2, pointimg3;

//    pointimg0 = cv::Point3f(-0.046f, -0.046f, 0);
//    pointimg1 = cv::Point3f(+0.046f, -0.046f, 0);
//    pointimg2 = cv::Point3f(+0.046f, +0.046f, 0);
//    pointimg3 = cv::Point3f(-0.046f, +0.046f, 0);


//    imgpoint.push_back(cv::Point3f(-0.046f, -0.046f, 0));
//    imgpoint.push_back(cv::Point3f(+0.046f, -0.046f, 0));
//    imgpoint.push_back(cv::Point3f(+0.046f, +0.046f, 0));
//    imgpoint.push_back(cv::Point3f(-0.046f, +0.046f, 0));


//    //��ȡ���
//    VideoCapture cap(1);

//    Mat frame,framecopy;
//    //�����ֵ�
//    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

//    while (true)
//    {
//        cap>>frame;
//        frame.copyTo(framecopy);

//        vector<int> ids;
//        vector<vector<Point2f> > corners;
//        vector<vector<Point2f> > outArry;
//        //���markers�������marker��ͼ���ֵ���󣬽ǵ��б�marker��id�б�
//        cv::aruco::detectMarkers(framecopy, dictionary, corners, ids);
//        if (ids.size() > 0)//��⵽marker
//        {
//            cout<<"numberr of ids:  "<<ids.size()<<endl;
//            for(int i=0;i<ids.size();i++)
//            {
//                //���ÿһ��id
//                cout<<"ids:  "<<ids[i]<<endl;
//            }
//            double x= (corners[0][0].x+corners[0][1].x+corners[0][2].x+corners[0][3].x)/4;
//            cout<<"x = "<<x<<endl;
//            double y= (corners[0][0].y+corners[0][1].y+corners[0][2].y+corners[0][3].y)/4;
//            cout<<"y = "<<y<<endl;
//            cout<<corners[0][0]<<" "<<corners[0][1]<<" "<<corners[0][2]<<" "<<corners[0][3]<<" "<<endl;
//             //trun the car and if in center line ,stop the car
//            if (abs(x-320)<=5)
//            {
//                cout<<"aruco in center line"<<endl;
//                // whether left or right
//            }
//            double side=corners[0][3].y-corners[0][0].y-(corners[0][2].y-corners[0][1].y);
//            if(side>0)
//                cout<<"in the left side: "<<side<<endl;
//            else if (side<0)
//                cout<<"in the right side: "<<side<<endl;
//            else
//                cout<<"airedy in the center line"<<endl;
//            //���Ƽ�������markers
//            cv::aruco::drawDetectedMarkers(frame, corners, ids);
//            vector< Vec3d > rvecs, tvecs;

//            //����SplvePnP������������̬
//            cv::aruco::estimatePoseSingleMarkers(corners,0.092, intrinsic_matrix, arucodistCoeffs, rvecs, tvecs); // draw axis for eac marker
//            //����solvePnPRansac������������̬
//            PnPRansac = solvePnPRansac(projectedPoints, corners, intrinsic_matrix, arucodistCoeffs, rvecs, tvecs, false, 100, 8, 0.99, outArry, SOLVEPNP_ITERATIVE);


//            //X:red Y: green Z:blue ԭ�������ĵ��ϣ����б�������ʱ�򣬣��ᳯ��ˮƽ�������ϣ�����ֱֽ�����⣬���������ϵ���ǣ���������󷽿��������ǳ��ұߵģ������ǳ��µģ������ǳ�ǰ��
//            for(int i=0; i<ids.size(); i++)
//            {
//                cout<<"R :"<<rvecs[i]<<endl;
//                //T:��ά���������������������ϵ�µ�����
//                cout<<"T :"<<tvecs[i]<<endl;
//                cv::aruco::drawAxis(frame, intrinsic_matrix, arucodistCoeffs, rvecs[i], tvecs[i], 0.1);
//                if (tvecs[i][2]<=0.5)
//                {
//                    int left= corners[0][3].y-corners[0][0].y;
//                    int right=corners[0][2].y-corners[0][1].y;
//                    if (left>right)
//                        cout<<"left"<<endl;
//                    if(right>left)
//                        cout<<"right"<<endl;
//                    if(right=left)
//                        cout<<"bingo,no need to trun"<<endl;
//                  }

//            }
//        }

//        imshow("frame", frame);
//        if( char(waitKey(30))==' ')
//        {
//            imwrite("detect.jpg",frame);
//            break;
//        }
//    }
//    return 0;
//}





















//�ǵ�ʶ��

//#include "stdafx.h"
#include "include/opencv2/opencv.hpp"
#include "include/opencv2/highgui/highgui.hpp"
#include "include/opencv2/imgproc/imgproc.hpp"
#include<cmath>
// ��λ���ؿ�/��(cm/pixel)
#define UNIT_PIXEL_W 0.00762
#define UNIT_PIXEL_H 0.00762

using namespace std;
using namespace cv;

int findMax(vector<float> vec) {
    int max = -999;
    for (auto v : vec) {
        if (max < v) max = v;
    }
    return max;
}

int findMin(vector<float> vec) {
    int min = 1300;
    for (auto v : vec) {
        if (min > v) min = v;
    }
    return min;
}

int main(int argv, char** argc)
{

    //const double f = 5.17;//4.8/1920*1071.83;  // ����
    const double f = 8.17;//4.8/1920*1071.83;  // ����
    const double w = 23;   // ����������
    const double h = 19.5;   // ��������߶�

    int maxcorners = 200;
    double qualityLevel = 0.2;  //�ǵ���ɽ��ܵ���С����ֵ
    double minDistance = 100;	//�ǵ�֮����С����
    int blockSize = 3;//���㵼������ؾ���ʱָ��������Χ
    double  k = 0.04; //Ȩ��ϵ��


    //������ͷ������Ƶ
    VideoCapture capture(0);//������ͷ
    if (!capture.isOpened())//û�д�����ͷ�Ļ����ͷ��ء�
        return -1;
    //Mat edges; //����һ��Mat���������ڴ洢ÿһ֡��ͼ��
    //ѭ����ʾÿһ֡
    while (1)
    {
        Mat srcImage; //����һ��Mat���������ڴ洢ÿһ֡��ͼ��
        capture >> srcImage;  //��ȡ��ǰ֡
        imshow("Video0", srcImage);
        if (srcImage.empty())
        {
            break;
        }
        else
        {

            int frame_num = capture.get(cv::CAP_PROP_FRAME_COUNT);
            std::cout << "total frame number is: " << frame_num << std::endl;

            int width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
            int height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);

            cv::VideoWriter out;
            //���ڱ�������
            out.open("F:/qt/canny/canny/test_result0.mp4", CV_FOURCC('m', 'p', '4', 'v'), 25.0, cv::Size(1280, 720), true);

            Mat srcgray, dstImage, normImage,scaledImage;
            cvtColor(srcImage, srcgray, CV_BGR2GRAY);

            Mat srcbinary;
            threshold(srcgray, srcbinary,0,255, THRESH_OTSU | THRESH_BINARY);

            Mat kernel = getStructuringElement(MORPH_RECT, Size(15, 15), Point(-1, -1));
            morphologyEx(srcbinary, srcbinary, MORPH_OPEN, kernel, Point(-1, -1));

            //2��Shi-Tomasi�㷨��ȷ��ͼ��ǿ�ǵ�
            vector<Point2f> corners;//�ṩ��ʼ�ǵ������λ�ú;�ȷ�������λ��
            //������� x/y
            vector<float>c_x;
            vector<float>c_y;


            //�ǵ���ȡ
            goodFeaturesToTrack(srcgray, corners, maxcorners, qualityLevel, minDistance, Mat(), blockSize, false, k);
            //Mat():��ʾ����Ȥ����false:��ʾ����Harris�ǵ���
            //����ǵ���Ϣ
            cout << "�ǵ���ϢΪ��" << corners.size() << endl;
            //���ƽǵ�
            RNG rng(12345);
            for (unsigned j = 0; j < corners.size(); j++)
            {
                circle(srcImage, corners[j], 5, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);
                cout << "�ǵ����꣺" << corners[j] << endl;
                //cout << "�ǵ�����x��" << corners[i].x << endl;
                //cout << "�ǵ�����y��" << corners[i].y << endl;
                //cout<<corners.begin()<<endl;;

                c_x.push_back(corners[j].x);
                c_y.push_back(corners[j].y);
                //imshow("111",srcImage);
            }
            //��ȡx�������ֵ����Сֵ
            float c_x_max = findMax(c_x);
            float c_x_min = findMin(c_x);
            cout<<"c_x_max = "<<c_x_max<<endl;
            cout<<"c_x_min = "<<c_x_min<<endl;
            //��ȡy�������ֵ����Сֵ
            float c_y_max = findMax(c_y);
            float c_y_min = findMin(c_y);
            cout<<"c_y_max = "<<c_y_max<<endl;
            cout<<"c_y_min = "<<c_y_min<<endl;

            //��ȡ����Ȥ����
            cv::Rect m_select = cv::Rect(c_x_min,c_y_min,(c_x_max-c_x_min),(c_y_max-c_y_min));
            //Mat ROI = srcImage(m_select);
            //imshow("111",ROI);

            //���ƾ���
            cv::rectangle(srcImage, m_select, cv::Scalar(0, 0, 255), 2);
            //cv::imshow("111",srcImage);

            // ��������/��/ƫ�þ���
            double widths = m_select.width * UNIT_PIXEL_W;
            double heights = m_select.height * UNIT_PIXEL_H;
            double px = (m_select.x + (m_select.width / 2) - 640) * UNIT_PIXEL_H;
            cout<<width<<endl;
            cout<<height<<endl;
            cout<<px<<endl;
            // �ֱ��Կ�/��Ϊ��׼�������
            double distanceW = w * f / widths;
            double distanceH = h * f / heights;
            double distancepx = px * w / widths;
            double angel = atan(distancepx/distanceW) / (3.1415926/180);
            cout<<"angel"<<angel<<endl;
            //�����ӡ����ֵ��ƫ��ֵ
            char disW[50], disH[50], dispx[50], disangel[50];
            sprintf_s(disW, "Distance_W : %.2fcm", distanceW);
            sprintf_s(disH, "Distance_H : %.2fcm", distanceH);
            sprintf_s(dispx,"Distance_px : %.2fcm", distancepx);
            sprintf_s(disangel,"Distance_angel : %.2f", angel);
            cv::putText(srcImage, disW, cv::Point(5, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 255, 0), 1, 8);
            cv::putText(srcImage, disH, cv::Point(5, 40), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 255, 0), 1, 8);
            cv::putText(srcImage, dispx, cv::Point(5, 60), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 255, 0), 1, 8);
            cv::putText(srcImage, disangel, cv::Point(5, 80), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 255, 0), 1, 8);

            cv::imshow("Frame", srcImage);
            if ((cv::waitKey(10) & 0XFF) == 27) break;

        }
        waitKey(10); //��ʱ10ms
    }

//    //��������
//    out << srcImage;

    if (cv::waitKey(30) == 'q')
    {
        //break;
    }

    capture.release();
    //out.release();
    destroyAllWindows();//�ر����д���
    //waitKey(0);
    return(0);
}





////�ǵ�ʶ��--����

////#include "stdafx.h"
//#include "include/opencv2/opencv.hpp"
//#include "include/opencv2/highgui/highgui.hpp"
//#include "include/opencv2/imgproc/imgproc.hpp"
//#include<cmath>
//// ��λ���ؿ�/��(cm/pixel)
//#define UNIT_PIXEL_W 0.00762
//#define UNIT_PIXEL_H 0.00762

//using namespace std;
//using namespace cv;

//int findMax(vector<float> vec) {
//    int max = -999;
//    for (auto v : vec) {
//        if (max < v) max = v;
//    }
//    return max;
//}

//int findMin(vector<float> vec) {
//    int min = 1300;
//    for (auto v : vec) {
//        if (min > v) min = v;
//    }
//    return min;
//}

//int main(int argv, char** argc)
//{

//    //const double f = 5.17;//4.8/1920*1071.83;  // ����
//    const double f = 8.17;//4.8/1920*1071.83;  // ����
//    const double w = 23;   // ����������
//    const double h = 19.5;   // ��������߶�

//    int maxcorners = 200;
//    double qualityLevel = 0.2;  //�ǵ���ɽ��ܵ���С����ֵ
//    double minDistance = 100;	//�ǵ�֮����С����
//    int blockSize = 3;//���㵼������ؾ���ʱָ��������Χ
//    double  k = 0.04; //Ȩ��ϵ��

//    /**************������Ƶ����HOG���******************/

//    cv::VideoCapture capture("F:/qt/canny/canny/test4.mp4");

//    if (!capture.isOpened())
//    {
//        std::cout << "Read video Failed !" << std::endl;
//        //return;
//    }

//    cv::Mat srcImage;

//    int frame_num = capture.get(cv::CAP_PROP_FRAME_COUNT);
//    std::cout << "total frame number is: " << frame_num << std::endl;

//    int width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
//    int height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);

//    cv::VideoWriter out;

//    //���ڱ�������
//    out.open("F:/qt/canny/canny/test_result4.mp4", CV_FOURCC('m', 'p', '4', 'v'), 25.0, cv::Size(1280, 720), true);

//    for (int i = 0; i < 10000; ++i)
//    {
//        capture >> srcImage;

//        //    Mat srcImage = imread("F:/qt/canny/canny/images/103.jpg");
//        //    if (srcImage.empty())
//        //    {
//        //        printf("could not load image..\n");
//        //        return false;
//        //    }
//        Mat srcgray;
//        cvtColor(srcImage, srcgray, CV_BGR2GRAY);

//        Mat srcbinary;
//        threshold(srcgray, srcbinary,0,255, THRESH_OTSU | THRESH_BINARY);

//        Mat kernel = getStructuringElement(MORPH_RECT, Size(15, 15), Point(-1, -1));
//        morphologyEx(srcbinary, srcbinary, MORPH_OPEN, kernel, Point(-1, -1));

//        //2��Shi-Tomasi�㷨��ȷ��ͼ��ǿ�ǵ�
//        vector<Point2f> corners;//�ṩ��ʼ�ǵ������λ�ú;�ȷ�������λ��
//        //������� x/y
//        vector<float>c_x;
//        vector<float>c_y;


//        //�ǵ���ȡ
//        goodFeaturesToTrack(srcgray, corners, maxcorners, qualityLevel, minDistance, Mat(), blockSize, false, k);
//        //Mat():��ʾ����Ȥ����false:��ʾ����Harris�ǵ���
//        //����ǵ���Ϣ
//        cout << "�ǵ���ϢΪ��" << corners.size() << endl;
//        //���ƽǵ�
//        RNG rng(12345);
//        for (unsigned j = 0; j < corners.size(); j++)
//        {
//            circle(srcImage, corners[j], 5, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);
//            cout << "�ǵ����꣺" << corners[j] << endl;
//            //cout << "�ǵ�����x��" << corners[i].x << endl;
//            //cout << "�ǵ�����y��" << corners[i].y << endl;
//            //cout<<corners.begin()<<endl;;

//            c_x.push_back(corners[j].x);
//            c_y.push_back(corners[j].y);
//            //imshow("111",srcImage);
//        }
//        //��ȡx�������ֵ����Сֵ
//        float c_x_max = findMax(c_x);
//        float c_x_min = findMin(c_x);
//        cout<<"c_x_max = "<<c_x_max<<endl;
//        cout<<"c_x_min = "<<c_x_min<<endl;
//        //��ȡy�������ֵ����Сֵ
//        float c_y_max = findMax(c_y);
//        float c_y_min = findMin(c_y);
//        cout<<"c_y_max = "<<c_y_max<<endl;
//        cout<<"c_y_min = "<<c_y_min<<endl;

//        //��ȡ����Ȥ����
//        cv::Rect m_select = cv::Rect(c_x_min,c_y_min,(c_x_max-c_x_min),(c_y_max-c_y_min));
//        //Mat ROI = srcImage(m_select);
//        //imshow("111",ROI);

//        //���ƾ���
//        cv::rectangle(srcImage, m_select, cv::Scalar(0, 0, 255), 2);
//        //cv::imshow("111",srcImage);

//        // ��������/��/ƫ�þ���
//        double width = m_select.width * UNIT_PIXEL_W;
//        double height = m_select.height * UNIT_PIXEL_H;
//        double px = (m_select.x + (m_select.width / 2) - 640) * UNIT_PIXEL_H;
//        cout<<width<<endl;
//        cout<<height<<endl;
//        cout<<px<<endl;
//        // �ֱ��Կ�/��Ϊ��׼�������
//        double distanceW = w * f / width;
//        double distanceH = h * f / height;
//        double distancepx = px * w / width;
//        double angel = atan(distancepx/distanceW) / (3.1415926/180);
//        cout<<"angel"<<angel<<endl;
//        //�����ӡ����ֵ��ƫ��ֵ
//        char disW[50], disH[50], dispx[50], disangel[50];
//        sprintf_s(disW, "Distance_W : %.2fcm", distanceW);
//        sprintf_s(disH, "Distance_H : %.2fcm", distanceH);
//        sprintf_s(dispx,"Distance_px : %.2fcm", distancepx);
//        sprintf_s(disangel,"Distance_angel : %.2f", angel);
//        cv::putText(srcImage, disW, cv::Point(5, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 255, 0), 1, 8);
//        cv::putText(srcImage, disH, cv::Point(5, 40), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 255, 0), 1, 8);
//        cv::putText(srcImage, dispx, cv::Point(5, 60), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 255, 0), 1, 8);
//        cv::putText(srcImage, disangel, cv::Point(5, 80), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 255, 0), 1, 8);

//        cv::imshow("Frame", srcImage);
//        //if ((cv::waitKey(10) & 0XFF) == 27) break;

//        //��������
//        out << srcImage;
//        if (cv::waitKey(30) == 'q')
//        {
//            break;
//        }

//    }
////    cv::imshow("detect result", srcImage);
////    //��������
////    out << srcImage;
////    if (cv::waitKey(30) == 'q')
////    {
////        //break;
////    }
//    capture.release();
//    out.release();

//    waitKey(0);
//    return(0);

//}






////�ǵ�ʶ��
////�������������ֳ���

////#include "stdafx.h"
//#include "include/opencv2/opencv.hpp"
//#include "include/opencv2/highgui/highgui.hpp"
//#include "include/opencv2/imgproc/imgproc.hpp"

//// ��λ���ؿ�/��(cm/pixel)
//#define UNIT_PIXEL_W 0.00762
//#define UNIT_PIXEL_H 0.00762


//using namespace std;
//using namespace cv;


//int main(int argv, char** argc)
//{

//    const double f = 5.17;//4.8/1920*1071.83;  // ����
//    const double w = 28.7;   // ����������
//    const double h = 14.5;   // ��������߶�

//    Mat srcImage = imread("F:/qt/canny/canny/images/539.jpg");

//    if (srcImage.empty())
//    {
//        printf("could not load image..\n");
//        return false;
//    }
//    Mat srcgray, dstImage, normImage,scaledImage;

//    cvtColor(srcImage, srcgray, CV_BGR2GRAY);

//    Mat srcbinary;
//    threshold(srcgray, srcbinary,0,255, THRESH_OTSU | THRESH_BINARY);

//    Mat kernel = getStructuringElement(MORPH_RECT, Size(15, 15), Point(-1, -1));
//    morphologyEx(srcbinary, srcbinary, MORPH_OPEN, kernel, Point(-1, -1));

///*
//    //1��Harris�ǵ���
//    cornerHarris(srcgray, dstImage, 3, 3, 0.01, BORDER_DEFAULT);
//    //��һ����ת��
//    normalize(dstImage, normImage, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
//    convertScaleAbs(normImage, scaledImage);
//    Mat binaryImage;
//    threshold(scaledImage, binaryImage, 0, 255, THRESH_OTSU | THRESH_BINARY);
//*/


//    //2��Shi-Tomasi�㷨��ȷ��ͼ��ǿ�ǵ�
//    vector<Point2f> corners;//�ṩ��ʼ�ǵ������λ�ú;�ȷ�������λ��
//    int maxcorners = 200;
//    double qualityLevel = 0.2;  //�ǵ���ɽ��ܵ���С����ֵ
//    double minDistance = 30;	//�ǵ�֮����С����
//    int blockSize = 3;//���㵼������ؾ���ʱָ��������Χ
//    double  k = 0.04; //Ȩ��ϵ��

//    goodFeaturesToTrack(srcgray, corners, maxcorners, qualityLevel, minDistance, Mat(), blockSize, false, k);
//    //Mat():��ʾ����Ȥ����false:��ʾ����Harris�ǵ���

//    //����ǵ���Ϣ
//    cout << "�ǵ���ϢΪ��" << corners.size() << endl;

//    //���ƽǵ�
//    RNG rng(12345);
//    for (unsigned i = 0; i < corners.size(); i++)
//    {
//        circle(srcImage, corners[i], 5, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);
//        cout << "�ǵ����꣺" << corners[i] << endl;
//        imshow("111",srcImage);
//    }


////    //3��Ѱ�������ؽǵ�
////    Size winSize = Size(5, 5);  //���ش��ڵ�һ��ߴ�
////    Size zeroZone = Size(-1, -1);//��ʾ������һ��ߴ�
////    //��ǵ�ĵ������̵���ֹ���������ǵ�λ�õ�ȷ��
////    TermCriteria criteria = TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40,0.001);
////    //TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 40, 0.001);

////    cornerSubPix(srcgray, corners, winSize, zeroZone, criteria);


////    //����ǵ���Ϣ
////    cout << "�ǵ���ϢΪ��" << corners.size() << endl;

////    //���ƽǵ�
////    for (unsigned i = 0; i < corners.size(); i++)
////    {
////        circle(srcImage, corners[i], 2, Scalar(255,0,0), -1, 8, 0);
////        cout << "�ǵ����꣺" << corners[i] << endl;
////        cout << "�ǵ�����x��" << corners[i].x << endl;
////        cout << "�ǵ�����y��" << corners[i].y << endl;
////        imshow("111",srcImage);

////    }


//    waitKey(0);
//    return(0);

//}










////��Ŀ��ࣨ������ɼ�����Ƶ�� c++��
//#include<opencv2/opencv.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <iostream>
//#include <opencv2/imgproc.hpp>
//#include <opencv2\imgproc\types_c.h>
//#include<stdio.h>
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include<opencv2/opencv_modules.hpp>
//#include "opencv2/imgproc/imgproc_c.h"
//using namespace cv;
//using namespace std;



////�������������С��������
//bool ascendSort(vector<Point> a, vector<Point> b)
//{
//    return a.size() < b.size();
//}
////�������������С��������
//bool descendSort(vector<Point> a, vector<Point> b) {
//    return a.size() > b.size();
//}
//static inline bool ContoursSortFun(vector<cv::Point> contour1, vector<cv::Point> contour2)
//{
//    return (cv::contourArea(contour1) > cv::contourArea(contour2));
//}
//int main()
//{
//    //������ͷ������Ƶ
//    VideoCapture capture(0);//������ͷ
//    if (!capture.isOpened())//û�д�����ͷ�Ļ����ͷ��ء�
//        return -1;
//    Mat edges; //����һ��Mat���������ڴ洢ÿһ֡��ͼ��
//               //ѭ����ʾÿһ֡
//    while (1)
//    {
//        Mat frame; //����һ��Mat���������ڴ洢ÿһ֡��ͼ��
//        capture >> frame;  //��ȡ��ǰ֡
//        imshow("Video0", frame);
//        if (frame.empty())
//        {
//            break;
//        }
//        else
//        {
//            //waitKey(2000);����ѡ����д���֡����ʱ��
//            cvtColor(frame, edges, CV_BGR2GRAY);//��ɫת���ɻҶ�
//            GaussianBlur(edges, edges, Size(3, 3), 0, 0);//ģ����
//                                                         //Canny(edges, edges, 35, 125, 3);//��Ե��
//            threshold(edges, edges, 220, 255, CV_THRESH_BINARY);
//            imshow("Video1", edges);
//            Mat mask = Mat::zeros(edges.size(), CV_8UC1);
//            vector<vector<Point>>contours;
//            vector<Vec4i>hierarchy;
//            findContours(edges, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);//��������
//            vector<RotatedRect> rectangle(contours.size()); //��С��Ӿ���    ***��С��Ӿ��κ���С����Ӿ��λ��ǲ�һ����***
//            Point2f rect[4];
//            float width = 0;//��Ӿ��εĿ�͸�
//            float height = 0;

//            for (int i = 0; i < contours.size(); i++)
//            {
//                rectangle[i] = minAreaRect(Mat(contours[i]));
//                rectangle[i].points(rect);          //��С��Ӿ��ε�4���˵�
//                width = rectangle[i].size.width;
//                height = rectangle[i].size.height;
//                if (height >= width)
//                {
//                    float x = 0;
//                    x = height;
//                    height = width;
//                    width = x;
//                }
//                cout << "��" << width << " " << "��" << height<< endl;
//                for (int j = 0; j < 4; j++)
//                {
//                    cout << "0" << rect[j] << " " << "1" << rect[(j + 1) % 4 ]<< endl;
//                    line(frame, rect[j], rect[(j + 1) % 4], Scalar(0, 0, 255), 1, 8);//������С��Ӿ��ε�ÿ����
//                }
//            }
//            float D = (210 * 509.57) / width;
//            char tam[100];
//            sprintf(tam, "D=:%lf", D);
//            putText(frame, tam, Point(100, 100), FONT_HERSHEY_SIMPLEX, 1, cvScalar(255, 0, 255), 1, 8);
//            imshow("Video2", mask); //��ʾ��ǰ֡
//            imshow("Video3", frame);
//        }
//        waitKey(10); //��ʱ30ms
//    }
//    capture.release();//�ͷ���Դ
//    destroyAllWindows();//�ر����д���
//    return 0;
//}








////��Ŀ���
//#include <iostream>
//#include <include/opencv2/opencv.hpp>
//#include <include/opencv2/highgui.hpp>
//#include <stdio.h>
//#include <vector>

//// ��λ���ؿ�/��(cm/pixel)
//#define UNIT_PIXEL_W 0.00762
//#define UNIT_PIXEL_H 0.00762

//using namespace std;

//int main(void)
//{
////    const double f = 5.17;//4.8/1920*1071.83;  // ����
////    const double w = 31;   // ����������
////    const double h = 17.5;   // ��������߶�

//    const double f = 5.17;//4.8/1920*1071.83;  // ����
//    const double w = 20.5;   // ����������
//    const double h = 17.5;   // ��������߶�


//    cv::Mat frame;
//    cv::VideoCapture capture(0);

//    if (!capture.isOpened()) {
//        printf("The camera is not opened.\n");
//        return EXIT_FAILURE;
//    }

//    for (;;) {
//        capture >> frame;
//        if (frame.empty()) {
//            printf("The frame is empty.\n");
//            break;
//        }
//        cv::medianBlur(frame, frame, 3);

//        cv::Mat grayImage;
//        cv::cvtColor(frame, grayImage, cv::COLOR_BGR2GRAY);
//        // otsu ���Ի��ö�̬��ֵ
//        cv::threshold(grayImage, grayImage, NULL, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

//        vector<vector<cv::Point>> contours;
//        vector<cv::Point> maxAreaContour;

//        cv::findContours(grayImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//        //cv::drawContours(frame, contours, -1, cv::Scalar(0, 0, 255), 2, 8);

//        // ��ȡ����������
//        double maxArea = 0;
//        for (size_t i = 0; i < contours.size(); i++) {
//            double area = fabs(cv::contourArea(contours[i]));
//            if (area > maxArea) {
//                maxArea = area;
//                maxAreaContour = contours[i];
//            }
//        }
//        // �������������
//        cv::Rect rect = cv::boundingRect(maxAreaContour);
//        cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), cv::Scalar(255, 0, 0), 2, 8);

//        // ��������/��
//        double width = rect.width * UNIT_PIXEL_W;
//        double height = rect.height * UNIT_PIXEL_H;
//        cout<<width<<endl;
//        cout<<height<<endl;    // �ֱ��Կ�/��Ϊ��׼�������
//        double distanceW = w * f / width;
//        double distanceH = h * f / height;

//        char disW[50], disH[50];
//        sprintf_s(disW, "DistanceW : %.2fcm", distanceW);
//        sprintf_s(disH, "DistanceH : %.2fcm", distanceH);
//        cv::putText(frame, disW, cv::Point(5, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 255, 0), 1, 8);
//        cv::putText(frame, disH, cv::Point(5, 40), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 255, 0), 1, 8);

//        cv::imshow("Frame", frame);
//        cv::imshow("Gray", grayImage);
//        if ((cv::waitKey(10) & 0XFF) == 27) break;
//    }
//    cv::destroyAllWindows();
//    capture.release();

//    return EXIT_SUCCESS;
//}


////��Ŀ���
//#include <iostream>
//#include <include/opencv2/opencv.hpp>
//#include <include/opencv2/highgui.hpp>
//#include <stdio.h>
//#include <vector>

//// ��λ���ؿ�/��(cm/pixel)
//#define UNIT_PIXEL_W 0.00762
//#define UNIT_PIXEL_H 0.00762

//using namespace std;

//int main(void)
//{
////    const double f = 5.17;//4.8/1920*1071.83;  // ����
////    const double w = 31;   // ����������
////    const double h = 17.5;   // ��������߶�

//    const double f = 5.17;//4.8/1920*1071.83;  // ����
//    const double w = 20.5;   // ����������
//    const double h = 17.5;   // ��������߶�


//    cv::Mat frame;
//    cv::VideoCapture capture(0);

//    if (!capture.isOpened()) {
//        printf("The camera is not opened.\n");
//        return EXIT_FAILURE;
//    }

//    for (;;) {
//        capture >> frame;
//        if (frame.empty()) {
//            printf("The frame is empty.\n");
//            break;
//        }
//        cv::medianBlur(frame, frame, 3);

//        cv::Mat grayImage;
//        cv::cvtColor(frame, grayImage, cv::COLOR_BGR2GRAY);
//        // otsu ���Ի��ö�̬��ֵ
//        cv::threshold(grayImage, grayImage, NULL, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

//        vector<vector<cv::Point>> contours;
//        vector<cv::Point> maxAreaContour;

//        cv::findContours(grayImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//        //cv::drawContours(frame, contours, -1, cv::Scalar(0, 0, 255), 2, 8);

//        // ��ȡ����������
//        double maxArea = 0;
//        for (size_t i = 0; i < contours.size(); i++) {
//            double area = fabs(cv::contourArea(contours[i]));
//            if (area > maxArea) {
//                maxArea = area;
//                maxAreaContour = contours[i];
//            }
//        }
//        // �������������
//        cv::Rect rect = cv::boundingRect(maxAreaContour);
//        cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), cv::Scalar(255, 0, 0), 2, 8);

//        // ��������/��
//        double width = rect.width * UNIT_PIXEL_W;
//        double height = rect.height * UNIT_PIXEL_H;
//        cout<<width<<endl;
//        cout<<height<<endl;    // �ֱ��Կ�/��Ϊ��׼�������
//        double distanceW = w * f / width;
//        double distanceH = h * f / height;

//        char disW[50], disH[50];
//        sprintf_s(disW, "DistanceW : %.2fcm", distanceW);
//        sprintf_s(disH, "DistanceH : %.2fcm", distanceH);
//        cv::putText(frame, disW, cv::Point(5, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 255, 0), 1, 8);
//        cv::putText(frame, disH, cv::Point(5, 40), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 255, 0), 1, 8);

//        cv::imshow("Frame", frame);
//        cv::imshow("Gray", grayImage);
//        if ((cv::waitKey(10) & 0XFF) == 27) break;
//    }
//    cv::destroyAllWindows();
//    capture.release();

//    return EXIT_SUCCESS;
//}




////��Ŀ����궨
//#include <iostream>
//#include <sstream>
//#include <time.h>
//#include <stdio.h>
//#include <fstream>

//#include <include/opencv2/core/core.hpp>
//#include <include/opencv2/imgproc/imgproc.hpp>
//#include <include/opencv2/calib3d/calib3d.hpp>
//#include <include/opencv2/highgui/highgui.hpp>

//using namespace cv;
//using namespace std;
//#define calibration

//int main()
//{
//#ifdef calibration

//    ifstream fin("F:/qt/canny/canny/chassread.txt");             /* �궨����ͼ���ļ���·�� */
//    ofstream fout("F:/qt/canny/canny/caliberation_result_right.txt");  /* ����궨������ļ� */

//    // ��ȡÿһ��ͼ�񣬴�����ȡ���ǵ㣬Ȼ��Խǵ���������ؾ�ȷ��
//    int image_count = 0;  /* ͼ������ */
//    Size image_size;      /* ͼ��ĳߴ� */
//    Size board_size = Size(11,8);             /* �궨����ÿ�С��еĽǵ��� */
//    vector<Point2f> image_points_buf;         /* ����ÿ��ͼ���ϼ�⵽�Ľǵ� */
//    vector<vector<Point2f>> image_points_seq; /* �����⵽�����нǵ� */
//    string filename;      // ͼƬ��
//    vector<string> filenames;

//    while (getline(fin, filename))
//    {
//        ++image_count;
//        Mat imageInput = imread(filename);
//        filenames.push_back(filename);

//        // �����һ��ͼƬʱ��ȡͼƬ��С
//        if (image_count == 1)
//        {
//            image_size.width = imageInput.cols;
//            image_size.height = imageInput.rows;
//        }

//        /* ��ȡ�ǵ� */
//        if (0 == findChessboardCorners(imageInput, board_size, image_points_buf))
//        {
//            //cout << "can not find chessboard corners!\n";  // �Ҳ����ǵ�
//            cout << "**" << filename << "** can not find chessboard corners!\n";
//            exit(1);
//        }
//        else
//        {
//            Mat view_gray;
//            cvtColor(imageInput, view_gray, CV_RGB2GRAY);  // ת�Ҷ�ͼ

//            /* �����ؾ�ȷ�� */
//            // image_points_buf ��ʼ�Ľǵ�����������ͬʱ��Ϊ����������λ�õ����
//            // Size(5,5) �������ڴ�С
//            // ��-1��-1����ʾû������
//            // TermCriteria �ǵ�ĵ������̵���ֹ����, ����Ϊ���������ͽǵ㾫�����ߵ����
//            cornerSubPix(view_gray, image_points_buf, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

//            image_points_seq.push_back(image_points_buf);  // ���������ؽǵ�

//            /* ��ͼ������ʾ�ǵ�λ�� */
//            drawChessboardCorners(view_gray, board_size, image_points_buf, false); // ������ͼƬ�б�ǽǵ�

//            imshow("Camera Calibration", view_gray);       // ��ʾͼƬ

//            waitKey(500); //��ͣ0.5S
//        }
//    }
//    int CornerNum = board_size.width * board_size.height;  // ÿ��ͼƬ���ܵĽǵ���

//    //-------------������������궨------------------

//    /*������ά��Ϣ*/
//    Size square_size = Size(16, 16);         /* ʵ�ʲ����õ��ı궨����ÿ�����̸�Ĵ�С */
//    vector<vector<Point3f>> object_points;   /* ����궨���Ͻǵ����ά���� */

//    /*�������*/
//    Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));  /* ������ڲ������� */
//    vector<int> point_counts;   // ÿ��ͼ���нǵ������
//    Mat distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0));       /* �������5������ϵ����k1,k2,p1,p2,k3 */
//    vector<Mat> tvecsMat;      /* ÿ��ͼ�����ת���� */
//    vector<Mat> rvecsMat;      /* ÿ��ͼ���ƽ������ */

//    /* ��ʼ���궨���Ͻǵ����ά���� */
//    int i, j, t;
//    for (t = 0; t<image_count; t++)
//    {
//        vector<Point3f> tempPointSet;
//        for (i = 0; i<board_size.height; i++)
//        {
//            for (j = 0; j<board_size.width; j++)
//            {
//                Point3f realPoint;

//                /* ����궨�������������ϵ��z=0��ƽ���� */
//                realPoint.x = i * square_size.width;
//                realPoint.y = j * square_size.height;
//                realPoint.z = 0;
//                tempPointSet.push_back(realPoint);
//            }
//        }
//        object_points.push_back(tempPointSet);
//    }

//    /* ��ʼ��ÿ��ͼ���еĽǵ��������ٶ�ÿ��ͼ���ж����Կ��������ı궨�� */
//    for (i = 0; i<image_count; i++)
//    {
//        point_counts.push_back(board_size.width * board_size.height);
//    }

//    /* ��ʼ�궨 */
//    // object_points ��������ϵ�еĽǵ����ά����
//    // image_points_seq ÿһ���ڽǵ��Ӧ��ͼ�������
//    // image_size ͼ������سߴ��С
//    // cameraMatrix ������ڲξ���
//    // distCoeffs ���������ϵ��
//    // rvecsMat �������ת����
//    // tvecsMat �����λ������
//    // 0 �궨ʱ�����õ��㷨
//    calibrateCamera(object_points, image_points_seq, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, 0);

//    //------------------------�궨���------------------------------------

//    // -------------------�Ա궨�����������------------------------------

//    double total_err = 0.0;         /* ����ͼ���ƽ�������ܺ� */
//    double err = 0.0;               /* ÿ��ͼ���ƽ����� */
//    vector<Point2f> image_points2;  /* �������¼���õ���ͶӰ�� */
//    fout << "ÿ��ͼ��ı궨��\n";

//    for (i = 0; i<image_count; i++)
//    {
//        vector<Point3f> tempPointSet = object_points[i];

//        /* ͨ���õ������������������Կռ����ά���������ͶӰ���㣬�õ��µ�ͶӰ�� */
//        projectPoints(tempPointSet, rvecsMat[i], tvecsMat[i], cameraMatrix, distCoeffs, image_points2);

//        /* �����µ�ͶӰ��;ɵ�ͶӰ��֮������*/
//        vector<Point2f> tempImagePoint = image_points_seq[i];
//        Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
//        Mat image_points2Mat = Mat(1, image_points2.size(), CV_32FC2);

//        for (int j = 0; j < tempImagePoint.size(); j++)
//        {
//            image_points2Mat.at<Vec2f>(0, j) = Vec2f(image_points2[j].x, image_points2[j].y);
//            tempImagePointMat.at<Vec2f>(0, j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
//        }
//        err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
//        total_err += err /= point_counts[i];
//        fout << "��" << i + 1 << "��ͼ���ƽ����" << err << "����" << endl;
//    }
//    fout << "����ƽ����" << total_err / image_count << "����" << endl << endl;

//    //-------------------------�������---------------------------------------------

//    //-----------------------���涨����-------------------------------------------
//    Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0));  /* ����ÿ��ͼ�����ת���� */
//    fout << "����ڲ�������" << endl;
//    fout << cameraMatrix << endl << endl;
//    fout << "����ϵ����\n";
//    fout << distCoeffs << endl << endl << endl;
//    for (int i = 0; i<image_count; i++)
//    {
//        fout << "��" << i + 1 << "��ͼ�����ת������" << endl;
//        fout << tvecsMat[i] << endl;

//        /* ����ת����ת��Ϊ���Ӧ����ת���� */
//        Rodrigues(tvecsMat[i], rotation_matrix);
//        fout << "��" << i + 1 << "��ͼ�����ת����" << endl;
//        fout << rotation_matrix << endl;
//        fout << "��" << i + 1 << "��ͼ���ƽ��������" << endl;
//        fout << rvecsMat[i] << endl << endl;
//    }
//    fout << endl;

//    //--------------------�궨����������-------------------------------

//    //----------------------��ʾ������--------------------------------

//    Mat mapx = Mat(image_size, CV_32FC1);
//    Mat mapy = Mat(image_size, CV_32FC1);
//    Mat R = Mat::eye(3, 3, CV_32F);
//    string imageFileName;
//    std::stringstream StrStm;
//    for (int i = 0; i != image_count; i++)
//    {
//        initUndistortRectifyMap(cameraMatrix, distCoeffs, R, cameraMatrix, image_size, CV_32FC1, mapx, mapy);
//        Mat imageSource = imread(filenames[i]);
//        Mat newimage = imageSource.clone();
//        remap(imageSource, newimage, mapx, mapy, INTER_LINEAR);
//        StrStm.clear();
//        imageFileName.clear();
//        StrStm << i + 1;
//        StrStm >> imageFileName;
//        imageFileName += "_d.jpg";
//        imwrite(imageFileName, newimage);
//    }

//    fin.close();
//    fout.close();

//#else
//        /// ��ȡһ��ͼƬ�����ı�ͼƬ�������ɫ���ͣ��ö�ȡ��ʽΪDOS����ģʽ��
//        Mat src = imread("F:\\lane_line_detection\\left_img\\1.jpg");
//        Mat distortion = src.clone();
//        Mat camera_matrix = Mat(3, 3, CV_32FC1);
//        Mat distortion_coefficients;


//        //��������ڲκͻ���ϵ������
//        FileStorage file_storage("F:\\lane_line_detection\\left_img\\Intrinsic.xml", FileStorage::READ);
//        file_storage["CameraMatrix"] >> camera_matrix;
//        file_storage["Dist"] >> distortion_coefficients;
//        file_storage.release();

//        //����
//        cv::undistort(src, distortion, camera_matrix, distortion_coefficients);

//        cv::imshow("img", src);
//        cv::imshow("undistort", distortion);
//        cv::imwrite("undistort.jpg", distortion);

//        cv::waitKey(0);
//#endif // DEBUG
//        cout<<"q"<<endl;
//    return 0;
//}














//#include <iostream>
//#include <include/opencv2/opencv.hpp>
//#include <include/opencv2/highgui.hpp>

//using namespace std;
//using namespace cv;

///*
// * Canny��Ե�����
// */

//int main() {
//    //������ͷ
////    char c = 0;
////    VideoCapture capture(0);

////    Mat src;
////    capture >> src;
////    imshow("camera",src);

//    Mat src = imread("F:/qt/canny/canny/images/44.jpg");
//    if (src.empty()) {
//        cout << "could not load image.." << endl;
//    }
//    imshow("input", src);
//    //���лҶȱ任
//    Mat dst;
//    cvtColor(src,dst,COLOR_BGR2GRAY);

//    //���и�˹�˲�
//    GaussianBlur(dst, dst, Size(7,7), 2, 2);
//    vector<Vec3f>circles;

//    //ִ��canny��Ե���
//    Mat edges;
//    Canny(dst, edges, 100, 300);

//    //ִ�����ͺ͸�ʴ����
//    Mat dresult, eresult;
//    //����ṹԪ��3*3��С�ľ���
//    Mat se = getStructuringElement(MORPH_RECT, Size(3, 3));
//    //����
//    dilate(edges, dresult, se);
//    //��ʴ
//    erode(dresult, eresult, se);

//    imshow("dresult",dresult);
//    imshow("eresult",eresult);

////    waitKey(3000);


//    //���Բ�����Բ����뾶
//    vector<vector<Point>>contours;
//    vector<Vec4i>hierarchy;
//    findContours(eresult, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//���ҳ����е�Բ�߽�
//    int index = 0;
//    for (; index >= 0; index = hierarchy[index][0])
//    {
//        Scalar color(rand() & 255, rand() & 255, rand() & 255);
//        drawContours(eresult, contours, index, color, CV_FILLED, 8, hierarchy);

//    }

//    namedWindow("detected circles", CV_NORMAL);
//    //imshow("edges",edges);
//    //imshow("detected circles", eresult);
//    //��׼Բ��ͼƬ��һ������Բ�����Բ���OpenCV�������Բ�ķ����������
//    Mat pointsf;
//    Mat(contours[0]).convertTo(pointsf, CV_32F);
//    RotatedRect box = fitEllipse(pointsf);
//    cout << box.center<<endl;
//    circle( eresult, box.center, 3, Scalar(0,255,0), -1, 8, 0 );
//    imshow("detected circles", eresult);
//    cout<<"box.size.height"<<box.size.height<<endl;
//    cout<<"box.size.width"<<box.size.width<<endl;

//    waitKey(10000);
//    return 0;
//}

////����С���˷����ֱ��
//#include <iostream>

//using namespace std;

//void LinearFit(double abr[],double x[],double y[],int n) {//�������ax+b
//   double xsum, ysum,x2sum,xysum;
//   xsum = 0; ysum = 0; x2sum = 0; xysum = 0;
//   for (int  i = 0; i < n; i++)
//   {
//       xsum += x[i];
//       ysum += y[i];
//       x2sum += x[i] * x[i];
//       xysum += x[i] * y[i];
//   }
//   abr[0] = (n*xysum - xsum * ysum) / (n*x2sum - xsum * xsum);//a
//   abr[1] = (ysum - abr[0] * xsum) / n;//b
//   double yavg = ysum / n;
//   double dy2sum1 = 0, dy2sum2 = 0;
//   for (int i = 0; i < n; i++)
//   {
//       dy2sum1 += ((abr[0] * x[i] + abr[1]) - yavg)*((abr[0] * x[i] + abr[1]) - yavg);//r^2�ķ���
//       dy2sum2 += (y[i] - yavg)*(y[i] - yavg);//r^2�ķ�ĸ
//   }
//   abr[2] = dy2sum1 / dy2sum2;//r^2
//}
//void HalfLogLine(double y[], int n) {//��������
//   for (int i = 0; i < n; i++)
//   {
//       y[i] = log10(y[i]);

//   }
//}
//void LogtoLine(double x[], double y[], int n) {//�������

//   for (int i = 0; i < n; i++)
//   {
//       y[i] = log(y[i]);
//       x[i] = log(x[i]);

//   }
//}
//int main()
//{
//   int const N = 12;//12;
//   //double x[N] = {0.96,0.94,0.92,0.90,0.88,0.86,0.84,0.82,0.80,0.78,0.76,0.74 };//�����
//   //double y[N] = {558.0,313.0,174.0,97.0,55.8,31.3,17.4,9.70,5.58,3.13,1.74,1.00 };
//   double x[N] = { 215.0, 230.6, 248.5, 255.6, 277.5, 284.0, 294.8, 296.8, 314.5, 317.5, 332.8, 362.5};//����
//   double y[N] = { 93.0, 81.0, 71.6, 62.0, 56.3, 51.5, 46.0, 44.0, 37.5, 35.0, 30.5, 19.0};
//   double abr[3];
//   //HalfLogLine(y, N);
//   LogtoLine(x, y, N);
//   LinearFit(abr, x, y, N);
//   abr[1] = exp(abr[1]);
//   cout << showpos;//��ʾ������
//   cout <<"���ϵ�����ֱ��:y=" << abr[0] << "x" << abr[1] << endl;
//   cout <<"���ϵ��:r^2"<< abr[2] << endl;
//   system("pause");
//   return 0;
//}


////��Բ���
//#include <iostream>
//#include <include/opencv2/opencv.hpp>
//#include <include/opencv2/highgui.hpp>
//#include <include/opencv2/imgproc.hpp>

//using namespace cv;
//using namespace std;

//int main( )
//{

////    char c = 0;
////    VideoCapture capture(0);

////    Mat src;
////    capture >> src;
////    imshow("camera",src);




//    Mat src = imread("F:/qt/canny/canny/images/10.jpg");
//    if (src.empty()) {
//        cout << "could not load image.." << endl;
//    }
//    imshow("input", src);


////    cv::Mat src;
////    cv::VideoCapture capture(0);



////    if (!capture.isOpened()) {
////        printf("The camera is not opened.\n");
////        return EXIT_FAILURE;
////    }

////    for (;;) {
////        capture >> src;
////        if (src.empty()) {
////            printf("The frame is empty.\n");
////            break;
////        }
////        cv::medianBlur(src, src, 3);
////    }


//    //���лҶȱ任
//    Mat dst;
//    cvtColor(src,dst,COLOR_BGR2GRAY);

//    //���и�˹�˲�
//    GaussianBlur(dst, dst, Size(7,7), 2, 2);
//    vector<Vec3f>circles;

//    //ִ��canny��Ե���
//    Mat edges;
//    Canny(dst, edges, 100, 300);

//    //ִ�����ͺ͸�ʴ����
//    Mat dresult, eresult;
//    //����ṹԪ��3*3��С�ľ���
//    Mat se = getStructuringElement(MORPH_RECT, Size(3, 3));
////    //����
////    dilate(edges, dresult, se);
//    //��ʴ
//    erode(edges, eresult, se);
//    //����
//    dilate(edges, dresult, se);

//    imshow("dresult",dresult);
//    imshow("eresult",eresult);
//    imshow("edges",edges);

//    //����
//    vector<vector<Point>> contours;


//    //��Ե׷�٣�û�д洢��Ե����֯�ṹ
//    findContours(dresult, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//    Mat cimage = Mat::zeros(dresult.size(), CV_8UC3);

//    for(size_t i = 0; i < contours.size(); i++)
//    {
//        //��ϵĵ�����Ϊ200
//        size_t count = contours[i].size();
//        if( count < 200 )
//            continue;

//        //��Բ���
//        RotatedRect box = fitEllipse(contours[i]);

//        //�������ȴ���3�����ų����������
//        if( MAX(box.size.width, box.size.height) > MIN(box.size.width, box.size.height)*3)
//            continue;
//        cout<<"box.size.height = "<<box.size.height<<endl;
//        cout<<"box.size.width = " <<box.size.width<<endl;

//        cout<<"box.center"<<box.center<<endl;
//        //����׷�ٳ�������
//        drawContours(cimage, contours, (int)i, Scalar::all(255), 1, 8);

//        //������ϵ���Բ
//        ellipse(cimage, box, Scalar(0,0,255), 1, CV_AA);


//    }
//    imshow("��Ͻ��", cimage);

//    waitKey();
//    return 0;
//}

////OpenCV��Ŀ�Ӿ���λ��������
//#include  <iostream>
//#include <include/opencv2/opencv.hpp>
////#include <include/opencv2/highgui.hpp>
////#include <include/opencv2/imgproc.hpp>


//using namespace std;
//using namespace cv;

////ȫ�ֱ���
//Mat src, gray, gray_blur, contours_image,dstThreshold;
////������ֱ���Ϊ640*480������ͼ������Ϊͼ��ԭ�㣬Ҳ����Ӧ���������λ�ã�����λ��
////�Զ���ͼ��ԭ������
//float oriX = 640.0f;
//float oriY = 360.0f;

//float targetImage_X, targetImage_Y;  //Ŀ�����ͼ��ԭ���X,Y��������ؾ���
//float mm_per_pixel;                  //���سߴ�
//float targetLength=100.0f;           //Ŀ����ʵ�ʳ���
//float targetActualX, targetActualY;  //��ά�ռ�ʵ������

////�Ӿ���λ����
//void Location();

//int main()
//{
//    cout <<" ������������������������������������'q'�˳����򡣡���������������������������" << endl;
//    VideoCapture cap(0);
//    if (!cap.isOpened()){
//        cout << "Failed To Open capture";
//        return -1;
//    }
//    while (1){
//        cap >> src;
//        cvtColor(src, gray, CV_BGR2GRAY);
//        imshow("gray", gray);

//        contours_image = Mat::zeros(src.rows, src.cols, CV_8UC3);
//        Mat gray_Contrast;
//        gray_Contrast = Mat::zeros(gray.size(), gray.type());

//        Location();
//        char key = waitKey(10);
//        if (key == 'q'){
//            break;
//        }
//    }
//}


////��λ����
//void Location(){

//    //��ֵ�˲�
//    blur(gray, gray_blur, Size(3, 3));

//    //��Ե�����ȡ��Ե��Ϣ
//    Canny(gray_blur, dstThreshold, 150, 450);
//    imshow("canny��Ե���", dstThreshold);

//    //�Ա�Եͼ����ȡ������Ϣ
//    vector<vector<Point> >contours;
//    findContours(dstThreshold, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

//    //��������
//    drawContours(contours_image, contours, -1, Scalar(0, 0, 255));
//    imshow("contours", contours_image);

//    //���������ԭ��
//    circle(src, Point2f(oriX, oriY), 2, Scalar(0, 0, 255), 3);

//    //����ֱ�ƽ������κ�����ε�����
//    vector< vector<Point> > Contour1_Ok, Contour2_Ok;

//    //��������
//    vector<Point> approx;
//    for (int i = 0; i < contours.size(); i++){
//        approxPolyDP(Mat(contours[i]), approx, cv::arcLength(cv::Mat(contours[i]), true)*0.04, true);

//        //ȥ�� С������ֻ��ȡ͹����
//        if (std::fabs(cv::contourArea(contours[i])) < 600 || !cv::isContourConvex(approx))
//            continue;

//        //����ƽ������ε����� �� Contour1_Ok
//         if (approx.size() == 4){
//            Contour1_Ok.push_back(contours[i]);
//        }
//        //����ƽ�����ε����� �� Contour2_Ok
//         else if (approx.size() == 4){
//             Contour2_Ok.push_back(contours[i]);
//        }

//    }

//    //�����з���Ҫ��������Σ�������������з���
//    //ʶ����Զ��������Ĺؼ��ǣ�
//    //1.�����κ��������������С��Ӿ��ε����Ļ�����ͬһ��
//    //2.��������������С��Ӿ��ε���һ�߳������������������С��Ӿ��ε���һ�߳�
//    for (int i = 0; i < Contour1_Ok.size(); i++){
//        for (int j = 0; j < Contour2_Ok.size(); j++){
//            RotatedRect minRect1 = minAreaRect(Mat(Contour1_Ok[i]));  //��������������С��Ӿ���
//            RotatedRect minRect2 = minAreaRect(Mat(Contour2_Ok[j]));  //�������������С������
//            //�ҳ�����Ҫ�����������С��Ӿ���
//            if ( fabs(minRect1.center.x - minRect2.center.x) < 30 && fabs(minRect1.center.y - minRect2.center.y)<30 && minRect1.size.width > minRect2.size.width){
//                Point2f vtx[4];
//                minRect1.points(vtx);

//                //�����ҵ����������С��Ӿ���
//                for (int j = 0; j < 4; j++)
//                    line(src, vtx[j], vtx[(j + 1) % 4], Scalar(0, 0, 255), 2, LINE_AA);

//                //����Ŀ�������ĵ�ͼ��ԭ���ֱ��
//                line(src, minRect1.center, Point2f(oriX, oriY), Scalar(0, 255, 0), 1, LINE_AA);

//                //Ŀ�����ͼ��ԭ���X,Y��������ؾ���
//                targetImage_X = minRect1.center.x - oriX;
//                targetImage_Y = oriY - minRect1.center.y;

//                line(src, minRect1.center, Point2f(minRect1.center.x, oriY), Scalar(255, 0, 0), 1, LINE_AA);
//                line(src, Point2f(oriX, oriY), Point2f(minRect1.center.x, oriY), Scalar(255, 0, 0), 1, LINE_AA);

//                Point2f pointX((oriX + minRect1.center.x) / 2, oriY);
//                Point2f pointY(minRect1.center.x, (oriY + minRect1.center.y) / 2);

//                //�ҳ�����
//                float a = minRect1.size.height, b = minRect1.size.width;
//                if (a < b) a = b;

//                mm_per_pixel = targetLength / a;               //�������سߴ� = Ŀ�����ʵ�ʳ��ȣ�cm��/ Ŀ������ͼ���ϵ����س��ȣ�pixels��
//                targetActualX = mm_per_pixel *targetImage_X;   //����ʵ�ʾ���X��cm��
//                targetActualY = mm_per_pixel *targetImage_Y;   //����ʵ�ʾ���Y��cm��

//                //��ӡ��Ϣ��ͼƬ��
//                String text1 = "X:"+format("%f", targetImage_X);
//                String text2 = "Y:"+format("%f", targetImage_Y);
//                putText(src, text1, pointX, FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 255), 1, 8);
//                putText(src, text2, pointY, FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 255), 1, 8);

//                String text3 = "Target_X:"+format("%f", targetActualX);
//                String text4 = "Target_Y:"+format("%f", targetActualY);
//                putText(src, text3, Point(10,30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 1, 8);
//                putText(src, text4, Point(10,60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 1, 8);


//            }
//            break;
//        }
//        break;
//    }

//    imshow("SRC", src);

//}


////��Ŀ��ࣨ������ɼ�����Ƶ�� c++��
//#include<opencv2/opencv.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <iostream>
//#include <opencv2/imgproc.hpp>
//#include <opencv2\imgproc\types_c.h>
//#include<stdio.h>
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include<opencv2/opencv_modules.hpp>
//#include "opencv2/imgproc/imgproc_c.h"
//using namespace cv;
//using namespace std;



////�������������С��������
//bool ascendSort(vector<Point> a, vector<Point> b)
//{
//    return a.size() < b.size();
//}
////�������������С��������
//bool descendSort(vector<Point> a, vector<Point> b) {
//    return a.size() > b.size();
//}
//static inline bool ContoursSortFun(vector<cv::Point> contour1, vector<cv::Point> contour2)
//{
//    return (cv::contourArea(contour1) > cv::contourArea(contour2));
//}
//int main()
//{
//    //������ͷ������Ƶ
//    VideoCapture capture(0);//������ͷ
//    if (!capture.isOpened())//û�д�����ͷ�Ļ����ͷ��ء�
//        return -1;
//    Mat edges; //����һ��Mat���������ڴ洢ÿһ֡��ͼ��
//               //ѭ����ʾÿһ֡
//    while (1)
//    {
//        Mat frame; //����һ��Mat���������ڴ洢ÿһ֡��ͼ��
//        capture >> frame;  //��ȡ��ǰ֡
//        imshow("Video0", frame);
//        if (frame.empty())
//        {
//            break;
//        }
//        else
//        {
//            //waitKey(2000);����ѡ����д���֡����ʱ��
//            cvtColor(frame, edges, CV_BGR2GRAY);//��ɫת���ɻҶ�
//            GaussianBlur(edges, edges, Size(3, 3), 0, 0);//ģ����
//                                                         //Canny(edges, edges, 35, 125, 3);//��Ե��
//            threshold(edges, edges, 220, 255, CV_THRESH_BINARY);
//            imshow("Video1", edges);
//            Mat mask = Mat::zeros(edges.size(), CV_8UC1);
//            vector<vector<Point>>contours;
//            vector<Vec4i>hierarchy;
//            findContours(edges, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);//��������
//            vector<RotatedRect> rectangle(contours.size()); //��С��Ӿ���    ***��С��Ӿ��κ���С����Ӿ��λ��ǲ�һ����***
//            Point2f rect[4];
//            float width = 0;//��Ӿ��εĿ�͸�
//            float height = 0;

//            for (int i = 0; i < contours.size(); i++)
//            {
//                rectangle[i] = minAreaRect(Mat(contours[i]));
//                rectangle[i].points(rect);          //��С��Ӿ��ε�4���˵�
//                width = rectangle[i].size.width;
//                height = rectangle[i].size.height;
//                if (height >= width)
//                {
//                    float x = 0;
//                    x = height;
//                    height = width;
//                    width = x;
//                }
//                cout << "��" << width << " " << "��" << height<< endl;
//                for (int j = 0; j < 4; j++)
//                {
//                    cout << "0" << rect[j] << " " << "1" << rect[(j + 1) % 4 ]<< endl;
//                    line(frame, rect[j], rect[(j + 1) % 4], Scalar(0, 0, 255), 1, 8);//������С��Ӿ��ε�ÿ����
//                }
//            }
//            float D = (210 * 509.57) / width;
//            char tam[100];
//            sprintf(tam, "D=:%lf", D);
//            putText(frame, tam, Point(100, 100), FONT_HERSHEY_SIMPLEX, 1, cvScalar(255, 0, 255), 1, 8);
//            imshow("Video2", mask); //��ʾ��ǰ֡
//            imshow("Video3", frame);
//        }
//        waitKey(10); //��ʱ30ms
//    }
//    capture.release();//�ͷ���Դ
//    destroyAllWindows();//�ر����д���
//    return 0;
//}

















////��Ŀ���+��Բ���
//#include <iostream>
//#include <include/opencv2/opencv.hpp>
//#include <include/opencv2/highgui.hpp>
//#include <stdio.h>
//#include <vector>


//using namespace std;
//using namespace cv;

//int main(void)
//{
//    cv::Mat frame;
//    cv::VideoCapture capture(0);

////    capture.set(CV_CAP_PROP_FRAME_WIDTH, 1080);//���
////    capture.set(CV_CAP_PROP_FRAME_HEIGHT, 960);//�߶�
//    capture.set(CV_CAP_PROP_FPS, 0.5);//֡�� ֡/��
////    capture.set(CV_CAP_PROP_BRIGHTNESS, 1);//����
////    capture.set(CV_CAP_PROP_CONTRAST,40);//�Աȶ� 40
////    capture.set(CV_CAP_PROP_SATURATION, 50);//���Ͷ� 50
////    capture.set(CV_CAP_PROP_HUE, 50);//ɫ�� 50
////    capture.set(CV_CAP_PROP_EXPOSURE, 50);//�ع� 50 ��ȡ����ͷ����

//    if (!capture.isOpened()) {
//        printf("The camera is not opened.\n");
//        return EXIT_FAILURE;
//    }

//    for (;;) {
//        capture >> frame;
//        if (frame.empty()) {
//            printf("The frame is empty.\n");
//            break;
//        }
//        cv::medianBlur(frame, frame, 3);



//        //���лҶȱ任
//        cv::Mat dst;
//        cvtColor(frame,dst,cv::COLOR_BGR2GRAY);
//        //���и�˹�˲�
//        cv::GaussianBlur(dst, dst, cv::Size(7,7), 2, 2);
//        vector<cv::Vec3f>circles;

//        //ִ��canny��Ե���
//        cv::Mat edges;
//        Canny(dst, edges, 100, 300);

////        //ִ�����ͺ͸�ʴ����
////        cv::Mat dresult, eresult;
////        //����ṹԪ��3*3��С�ľ���
////        cv::Mat se = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

////        //��ʴ
////        erode(dst, eresult, se);
////        //����
////        dilate(dst, dresult, se);

//        //imshow("dresult",dresult);
//        //imshow("eresult",eresult);
//        //imshow("edges",dst);

//        //����
//        vector<vector<cv::Point>> contours;


//        //��Ե׷�٣�û�д洢��Ե����֯�ṹ
//        findContours(edges, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//        cv::Mat cimage = cv::Mat::zeros(edges.size(), CV_8UC3);

//        for(size_t i = 0; i < contours.size(); i++)
//        {
//            //��ϵĵ�����Ϊ200
//            size_t count = contours[i].size();
//            if( count < 200 )
//                continue;

//            //��Բ���
//            cv::RotatedRect box = fitEllipse(contours[i]);

//            //�������ȴ���3�����ų����������
//            if( MAX(box.size.width, box.size.height) > MIN(box.size.width, box.size.height)*3 )
//                continue;
//            //if( 400 > box.size.height > 250 )
//                //continue;
//            //if( 250 > box.size.width > 150 )
//                //continue;
//            cout<<"box.size.height = "<<box.size.height<<endl;
//            cout<<"box.size.width = " <<box.size.width<<endl;
//            cout<<"box.center"<<box.center<<endl;
//            //����׷�ٳ�������
//            drawContours(cimage, contours, (int)i, cv::Scalar::all(255), 1, 8);

//            //������ϵ���Բ
//            ellipse(cimage, box, cv::Scalar(0,0,255), 1, CV_AA);
//            imshow("��Ͻ��0", cimage);

//        }
//        imshow("��Ͻ��1", cimage);

//        cv::imshow("Frame", frame);
//        cv::imshow("Gray", dst);
//        if ((cv::waitKey(10) & 0XFF) == 27) break;
//    }
//    cv::destroyAllWindows();
//    capture.release();

//    return EXIT_SUCCESS;
//}




////��Ŀ���+��Բ���
//#include <iostream>
//#include <include/opencv2/opencv.hpp>
//#include <include/opencv2/highgui.hpp>
//#include <stdio.h>
//#include <vector>

//// ��λ���ؿ�/��(cm/pixel)
//#define UNIT_PIXEL_W 0.03
//#define UNIT_PIXEL_H 0.03

//using namespace std;

//int main(void)
//{
//    cv::Mat frame;
//    cv::VideoCapture capture(0);

////    const double f = 2.8;  // ����
////    const double w = 100;   // ����������
////    const double h = 60;   // ��������߶�

//    if (!capture.isOpened()) {
//        printf("The camera is not opened.\n");
//        return EXIT_FAILURE;
//    }

//    for (;;) {
//        capture >> frame;
//        if (frame.empty()) {
//            printf("The frame is empty.\n");
//            break;
//        }
//        cv::medianBlur(frame, frame, 3);

//        cv::Mat dst;
//        cv::cvtColor(frame, dst, cv::COLOR_BGR2GRAY);
//        // otsu ���Ի��ö�̬��ֵ
//        cv::threshold(dst, dst, NULL, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

//        vector<vector<cv::Point>> contours;
//        //vector<cv::Point> maxAreaContour;

//        cv::findContours(dst, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//        //cv::drawContours(frame, contours, -1, cv::Scalar(0, 0, 255), 2, 8);

//        //���лҶȱ任
//        //cv::Mat dst;
//        //cvtColor(dst,dst,cv::COLOR_BGR2GRAY);
//        //���и�˹�˲�
//        cv::GaussianBlur(dst, dst, cv::Size(7,7), 2, 2);
//        vector<cv::Vec3f>circles;

//        //ִ��canny��Ե���
//        cv::Mat edges;
//        Canny(dst, edges, 100, 300);

//        //ִ�����ͺ͸�ʴ����
//        cv::Mat dresult, eresult;
//        //����ṹԪ��3*3��С�ľ���
//        cv::Mat se = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
//        //    //����
//        //    dilate(edges, dresult, se);
//        //��ʴ
//        erode(dst, eresult, se);
//        //����
//        dilate(dst, dresult, se);

//        //imshow("dresult",dresult);
//        //imshow("eresult",eresult);
//        //imshow("edges",dst);

////        //����
////        vector<vector<cv::Point>> contours;


//        //��Ե׷�٣�û�д洢��Ե����֯�ṹ
//        findContours(dresult, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//        cv::Mat cimage = cv::Mat::zeros(dresult.size(), CV_8UC3);

//        for(size_t i = 0; i < contours.size(); i++)
//        {
//            //��ϵĵ�����Ϊ6
//            size_t count = contours[i].size();
//            if( count < 6 )
//                continue;

//            //��Բ���
//            cv::RotatedRect box = fitEllipse(contours[i]);

//            //�������ȴ���30�����ų����������
//            if( MAX(box.size.width, box.size.height) > MIN(box.size.width, box.size.height)*30 )
//                continue;
//            cout<<"box.size.height = "<<box.size.height<<endl;
//            cout<<"box.size.width = " <<box.size.width<<endl;
//            //����׷�ٳ�������
//            drawContours(cimage, contours, (int)i, cv::Scalar::all(255), 1, 8);

//            //������ϵ���Բ
//            ellipse(cimage, box, cv::Scalar(0,0,255), 1, CV_AA);


//        }
//        imshow("��Ͻ��", cimage);




//        // ��ȡ����������
//        double maxArea = 0;
//        for (size_t i = 0; i < contours.size(); i++) {
//            double area = fabs(cv::contourArea(contours[i]));
//            if (area > maxArea) {
//                maxArea = area;
//                maxAreaContour = contours[i];
//            }
//        }
//        // �������������
//        cv::Rect rect = cv::boundingRect(maxAreaContour);


//        cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), cv::Scalar(255, 0, 0), 2, 8);
//        cv::imshow("frame",frame);
//        // ��������/��
//        double width = rect.width * UNIT_PIXEL_W;
//        double height = rect.height * UNIT_PIXEL_H;
//        // �ֱ��Կ�/��Ϊ��׼�������
//        double distanceW = w * f / width;
//        double distanceH = h * f / height;

//        char disW[50], disH[50];
//        sprintf_s(disW, "DistanceW : %.2fcm", distanceW);
//        sprintf_s(disH, "DistanceH : %.2fcm", distanceH);
//        cv::putText(frame, disW, cv::Point(5, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 255, 0), 1, 8);
//        cv::putText(frame, disH, cv::Point(5, 40), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 255, 0), 1, 8);

//        cv::imshow("Frame", frame);
//        cv::imshow("Gray", dst);
//        if ((cv::waitKey(10) & 0XFF) == 27) break;
//    }
//    cv::destroyAllWindows();
//    capture.release();

//    return EXIT_SUCCESS;
//}





















////��Ŀ���
//#include <iostream>
//#include <include/opencv2/opencv.hpp>
//#include <include/opencv2/highgui.hpp>
//#include <stdio.h>
//#include <vector>

//// ��λ���ؿ�/��(cm/pixel)
//#define UNIT_PIXEL_W 0.0003
//#define UNIT_PIXEL_H 0.0003

//using namespace std;

//int main(void)
//{
//    cv::Mat frame;
//    cv::VideoCapture capture(0);

//    const double f = 0.268;//4.8/1920*1071.83;  // ����
//    const double w = 60;   // ����������
//    const double h = 100;   // ��������߶�

//    if (!capture.isOpened()) {
//        printf("The camera is not opened.\n");
//        return EXIT_FAILURE;
//    }

//    for (;;) {
//        capture >> frame;
//        if (frame.empty()) {
//            printf("The frame is empty.\n");
//            break;
//        }
//        cv::medianBlur(frame, frame, 3);

//        cv::Mat grayImage;
//        cv::cvtColor(frame, grayImage, cv::COLOR_BGR2GRAY);
//        // otsu ���Ի��ö�̬��ֵ
//        cv::threshold(grayImage, grayImage, NULL, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

//        vector<vector<cv::Point>> contours;
//        vector<cv::Point> maxAreaContour;

//        cv::findContours(grayImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//        //cv::drawContours(frame, contours, -1, cv::Scalar(0, 0, 255), 2, 8);

//        // ��ȡ����������
//        double maxArea = 0;
//        for (size_t i = 0; i < contours.size(); i++) {
//            double area = fabs(cv::contourArea(contours[i]));
//            if (area > maxArea) {
//                maxArea = area;
//                maxAreaContour = contours[i];
//            }
//        }
//        // �������������
//        cv::Rect rect = cv::boundingRect(maxAreaContour);
//        cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), cv::Scalar(255, 0, 0), 2, 8);

//        // ��������/��
//        double width = rect.width * UNIT_PIXEL_W;
//        double height = rect.height * UNIT_PIXEL_H;
//        cout<<width<<endl;
//        cout<<height<<endl;    // �ֱ��Կ�/��Ϊ��׼�������
//        double distanceW = w * f / width;
//        double distanceH = h * f / height;

//        char disW[50], disH[50];
//        sprintf_s(disW, "DistanceW : %.2fcm", distanceW);
//        sprintf_s(disH, "DistanceH : %.2fcm", distanceH);
//        cv::putText(frame, disW, cv::Point(5, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 255, 0), 1, 8);
//        cv::putText(frame, disH, cv::Point(5, 40), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 255, 0), 1, 8);

//        cv::imshow("Frame", frame);
//        cv::imshow("Gray", grayImage);
//        if ((cv::waitKey(10) & 0XFF) == 27) break;
//    }
//    cv::destroyAllWindows();
//    capture.release();

//    return EXIT_SUCCESS;
//}




////��Ե+�������

////opencv�汾:OpenCV3.0
////VS�汾:VS2013
////Author:opencv66.net

//#include <include/opencv2/core/core.hpp>
//#include <include/opencv2/imgproc/imgproc.hpp>
//#include <include/opencv2/imgproc/types_c.h>
//#include <include/opencv2/highgui/highgui.hpp>
//#include <include/opencv2/highgui/highgui_c.h>

//#include <iostream>

//using namespace cv;
//using namespace std;

//int main()
//{

//    Mat image = imread("F:/qt/canny/canny/images/10.jpg");

//    namedWindow("ԭͼ");
//    imshow("ԭͼ", image);

//    cvtColor(image, image, CV_BGR2GRAY);//תΪ�Ҷ�ͼ��

//    vector<vector<Point>> contours;
//    vector<Vec4i> hierarchy;
//    // �������
//    findContours(image, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

//    // ��������
//    Mat result(image.size(), CV_8UC3, Scalar(0));
//    drawContours(result, contours, -1, Scalar(255, 255, 255), 1);

//    Mat result_PolyDP = result.clone();
//    Mat result_boundingRect = result.clone();
//    Mat result_Circle = result.clone();


//    //conPoint�洢����õ�����Ӷ����
//    vector<vector<Point> > conPoint(contours.size());

//    //boundRect�洢����õ�����С��ʽ����
//    vector<Rect> boundRect(contours.size());

//    //center��radius�洢����õ�����С���Բ
//    vector<Point2f>center(contours.size());
//    vector<float>radius(contours.size());

//    for (int i = 0; i < contours.size(); i++)
//    {
//        // ������Ӷ����
//        approxPolyDP(Mat(contours[i]), conPoint[i], 3, true);
//        // ������С�����ʽ����
//        boundRect[i] = boundingRect(Mat(conPoint[i]));
//        //������С���Բ
//        minEnclosingCircle(conPoint[i], center[i], radius[i]);
//    }

//    for (int i = 0; i< contours.size(); i++)
//    {
//        Scalar color = Scalar(0, 0, 255);
//        //������Ӷ����
//        drawContours(result_PolyDP, conPoint, i, color, 2, 8, vector<Vec4i>(), 0, Point());
//        // ������С�����ʽ����
//        rectangle(result_boundingRect, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
//        // ������С���Բ
//        circle(result_Circle, center[i], (int)radius[i], color, 2, 8, 0);
//    }

//    namedWindow("����ͼ");
//    imshow("����ͼ", result);
//    namedWindow("PolyDP");
//    imshow("PolyDP", result_PolyDP);

//    namedWindow("boundingRect");
//    imshow("boundingRect", result_boundingRect);

//    namedWindow("Circle");
//    imshow("Circle", result_Circle);

//    waitKey(5000);
//    return 0;
//}



















////����ڽӾ���
//#include <iostream>
//#include <include/opencv2/opencv.hpp>
//#include <include/opencv2/highgui.hpp>
//#include<vector>

//using namespace cv;
//using namespace std;

///**

//* @brief expandEdge ��չ�߽纯��

//* @param img:����ͼ�񣬵�ͨ����ֵͼ�����Ϊ8

//* @param edge  �߽����飬���4���߽�ֵ

//* @param edgeID ��ǰ�߽��

//* @return ����ֵ ȷ����ǰ�߽��Ƿ������չ

//*/

//bool expandEdge(const Mat & img, int edge[], const int edgeID)
//{
//    //[1] --��ʼ������
//    int nc = img.cols;
//    int nr = img.rows;
//    switch (edgeID) {
//    case 0:
//        if (edge[0]>nr)
//            return false;
//        for (int i = edge[3]; i <= edge[1]; ++i)
//        {
//            if (img.at<uchar>(edge[0], i) == 255)//����255���ر���������Ե��
//                return false;
//        }
//        edge[0]++;
//        return true;
//        break;
//    case 1:
//        if (edge[1]>nc)
//            return false;
//        for (int i = edge[2]; i <= edge[0]; ++i)
//        {
//            if (img.at<uchar>(i, edge[1]) == 255)//����255���ر���������Ե��
//                return false;
//        }
//        edge[1]++;
//        return true;
//        break;
//    case 2:
//        if (edge[2]<0)
//            return false;
//        for (int i = edge[3]; i <= edge[1]; ++i)
//        {
//            if (img.at<uchar>(edge[2], i) == 255)//����255���ر���������Ե��
//                return false;
//        }
//        edge[2]--;
//        return true;
//        break;
//    case 3:
//        if (edge[3]<0)
//            return false;
//        for (int i = edge[2]; i <= edge[0]; ++i)
//        {
//            if (img.at<uchar>(i, edge[3]) == 255)//����255���ر���������Ե��
//                return false;
//        }
//        edge[3]--;
//        return true;
//        break;
//    default:
//        return false;
//        break;
//    }

//}

///**

//* @brief ��ȡ��ͨ�����ڽӾ�

//* @param img:����ͼ�񣬵�ͨ����ֵͼ�����Ϊ8

//* @param center:��С��Ӿص�����

//* @return  ����ڽӾ���

//* ����������չ�㷨

//*/

//cv::Rect InSquare(Mat &img, const Point center)
//{
//    // --[1]�������
//    if (img.empty() ||img.channels()>1|| img.depth()>8)
//        return Rect();
//    // --[2] ��ʼ������
//    int edge[4];
//    edge[0] = center.y + 1;//top
//    edge[1] = center.x + 1;//right
//    edge[2] = center.y - 1;//bottom
//    edge[3] = center.x - 1;//left
//                           //[2]
//                           // --[3]�߽���չ(������ɢ��)

//    bool EXPAND[4] = { 1,1,1,1 };//��չ���λ
//    int n = 0;
//    while (EXPAND[0] || EXPAND[1] || EXPAND[2] || EXPAND[3])
//    {
//        int edgeID = n % 4;
//        EXPAND[edgeID] = expandEdge(img, edge, edgeID);
//        n++;
//    }
//    //[3]
//    //qDebug() << edge[0] << edge[1] << edge[2] << edge[3];
//    Point tl = Point(edge[3], edge[0]);
//    Point br = Point(edge[1], edge[2]);
//    return Rect(tl, br);
//}




//int main()
//{

//    bool isExistence = false;
//    float first_area = 0;
//    /// ����Դͼ��
//    Mat src;
//    src = imread("F:/qt/canny/canny/images/1.jpg", 1);
//    //src = imread("C:\\Users\\Administrator\\Desktop\\����ͼƬ\\xxx\\20190308152516.jpg",1);
//    //src = imread("C:\\Users\\Administrator\\Desktop\\����ͼƬ\\xx\\20190308151912.jpg",1);
//    //src = imread("C:\\Users\\Administrator\\Desktop\\����ͼ��\\2\\BfImg17(x-247 y--91 z--666)-(492,280).jpg",1);
//    cvtColor(src, src, CV_RGB2GRAY);
//    threshold(src, src, 100, 255, THRESH_BINARY);
//    Rect ccomp;
//    Point center(src.cols / 2, src.rows / 2);
//    //floodFill(src, center, Scalar(255, 255, 55), &ccomp, Scalar(20, 20, 20), Scalar(20, 20, 20));
//    if (src.empty())
//    {
//        cout << "fali" << endl;
//    }
//    //resize(src, src, cv::Size(496, 460), cv::INTER_LINEAR);
//    imshow("src", src);
//    Rect rr = InSquare(src, center);
//    rectangle(src, rr, Scalar(255), 1, 8);
//    imshow("src2", src);

//    waitKey(0);
//    getchar();
//    return 0;
//}










////��ɫʶ��2
//#include <iostream>
//#include <include/opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <include/opencv2/opencv.hpp>

//using namespace cv;
//using namespace std;

//int main(int argc, char** argv)
//{
//    VideoCapture cap(0); //capture the video from web cam

//    if (!cap.isOpened())  // if not success, exit program
//    {
//        cout << "Cannot open the web cam" << endl;
//        return -1;
//    }

//    namedWindow("control", 1);
//    int ctrl = 0;
//    createTrackbar("ctrl", "control", &ctrl, 7);

//    while (true)
//    {
//        Mat imgOriginal;

//        bool bSuccess = cap.read(imgOriginal); // read a new frame from video
//        if (!bSuccess) //if not success, break loop
//        {
//            cout << "Cannot read a frame from video stream" << endl;
//            break;
//        }

//        // imgOriginal = imread("4.jpg");

//        Mat imgHSV, imgBGR;
//        Mat imgThresholded;

//        if(0)
//        {
//            vector<Mat> hsvSplit;   //�����������������HSV����ͨ������
//            cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
//            split(imgHSV, hsvSplit);			//����ԭͼ���HSV��ͨ��
//            equalizeHist(hsvSplit[2], hsvSplit[2]);    //��HSV������ͨ������ֱ��ͼ����
//            merge(hsvSplit, imgHSV);				   //�ϲ�����ͨ��
//            cvtColor(imgHSV, imgBGR, COLOR_HSV2BGR);    //��HSV�ռ�ת����RGB�ռ䣬Ϊ����������ɫʶ����׼��
//        }
//        else
//        {
//            imgBGR = imgOriginal.clone();
//        }



//        switch(ctrl)
//        {
//        case 0:
//            {
//                inRange(imgBGR, Scalar(128, 0, 0), Scalar(255, 127, 127), imgThresholded); //��ɫ
//                break;
//            }
//        case 1:
//            {
//                inRange(imgBGR, Scalar(128, 128, 128), Scalar(255, 255, 255), imgThresholded); //��ɫ
//                break;
//            }
//        case 2:
//            {
//                inRange(imgBGR, Scalar(128, 128, 0), Scalar(255, 255, 127), imgThresholded); //��ɫ
//                break;
//            }
//        case 3:
//            {
//                inRange(imgBGR, Scalar(128, 0, 128), Scalar(255, 127, 255), imgThresholded); //��ɫ
//                break;
//            }
//        case 4:
//            {
//                inRange(imgBGR, Scalar(0, 128, 128), Scalar(127, 255, 255), imgThresholded); //��ɫ
//                break;
//            }
//        case 5:
//            {
//                inRange(imgBGR, Scalar(0, 128, 0), Scalar(127, 255, 127), imgThresholded); //��ɫ
//                break;
//            }
//        case 6:
//            {
//                inRange(imgBGR, Scalar(0, 0, 128), Scalar(127, 127, 255), imgThresholded); //��ɫ
//                break;
//            }
//        case 7:
//            {
//                inRange(imgBGR, Scalar(0, 0, 0), Scalar(127, 127, 127), imgThresholded); //��ɫ
//                break;
//            }
//        }

//        imshow("��̬ѧȥ����ǰ", imgThresholded);

//        Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
//        morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);
//        morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, element);

//        imshow("Thresholded Image", imgThresholded); //show the thresholded image
//        imshow("ֱ��ͼ�����Ժ�", imgBGR);
//        imshow("Original", imgOriginal); //show the original image

//        char key = (char)waitKey(300);
//        if (key == 27)
//            break;
//    }

//    return 0;

//}










//��ɫʶ��1
//#include <iostream>
//#include <include/opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <include/opencv2/opencv.hpp>


//using namespace cv;
//using namespace std;

// int main( int argc, char** argv )
// {
//    VideoCapture cap(0); //capture the video from web cam

//    if ( !cap.isOpened() )  // if not success, exit program
//    {
//         cout << "Cannot open the web cam" << endl;
//         return -1;
//    }

//  namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"

//  int iLowH = 100;
//  int iHighH = 140;

//  int iLowS = 90;
//  int iHighS = 255;

//  int iLowV = 90;
//  int iHighV = 255;

//  //Create trackbars in "Control" window
//  cvCreateTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
//  cvCreateTrackbar("HighH", "Control", &iHighH, 179);

//  cvCreateTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
//  cvCreateTrackbar("HighS", "Control", &iHighS, 255);

//  cvCreateTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)
//  cvCreateTrackbar("HighV", "Control", &iHighV, 255);

//    while (true)
//    {
//        Mat imgOriginal;

//        bool bSuccess = cap.read(imgOriginal); // read a new frame from video

//         if (!bSuccess) //if not success, break loop
//        {
//             cout << "Cannot read a frame from video stream" << endl;
//             break;
//        }

//   Mat imgHSV;
//   vector<Mat> hsvSplit;
//   cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

//   //��Ϊ���Ƕ�ȡ���ǲ�ɫͼ��ֱ��ͼ���⻯��Ҫ��HSV�ռ���
//   split(imgHSV, hsvSplit);
//   equalizeHist(hsvSplit[2],hsvSplit[2]);
//   merge(hsvSplit,imgHSV);
//   Mat imgThresholded;

//   inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image

//   //������ (ȥ��һЩ���)
//   Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
//   morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);

//   //�ղ��� (����һЩ��ͨ��)
//   morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, element);

//   imshow("Thresholded Image", imgThresholded); //show the thresholded image
//   imshow("Original", imgOriginal); //show the original image

//   char key = (char) waitKey(300);
//   if(key == 27)
//         break;
//    }

//   return 0;

//}










//#include <opencv2/opencv.hpp>
//#include <iostream>
//using namespace cv;
//using namespace std;



//static double angle(Point pt1, Point pt2, Point pt0)
//{
//    double dx1 = pt1.x - pt0.x;
//    double dy1 = pt1.y - pt0.y;
//    double dx2 = pt2.x - pt0.x;
//    double dy2 = pt2.y - pt0.y;
//    return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
//}

////������
////��һ�������Ǵ����ԭʼͼ�񣬵ڶ��������ͼ��
//void findSquares(const Mat& image, Mat &out)
//{
//    int thresh = 50, N = 5;
//    vector<vector<Point> > squares;
//    squares.clear();

//    Mat src,dst, gray_one, gray;

//    src = image.clone();
//    out = image.clone();
//    gray_one = Mat(src.size(), CV_8U);
//    //�˲���ǿ��Ե���
//    medianBlur(src, dst, 9);
//    //bilateralFilter(src, dst, 25, 25 * 2, 35);

//    vector<vector<Point> > contours;
//    vector<Vec4i> hierarchy;

//    //��ͼ���ÿ����ɫͨ���в��Ҿ���
//    for (int c = 0; c < image.channels(); c++)
//    {
//        int ch[] = { c, 0 };

//        //ͨ������
//        mixChannels(&dst, 1, &gray_one, 1, ch, 1);

//        // ���Լ�����ֵ
//        for (int l = 0; l < N; l++)
//        {
//            // ��canny()��ȡ��Ե
//            if (l == 0)
//            {
//                //����Ե
//                Canny(gray_one, gray, 5, thresh, 5);
//                //��Û
//                dilate(gray, gray, Mat(), Point(-1, -1));
//                imshow("dilate", gray);
//            }
//            else
//            {
//                gray = gray_one >= (l + 1) * 255 / N;
//            }

//            // ��������
//            //findContours(gray, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
//            findContours(gray, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

//            vector<Point> approx;

//            // ������ҵ�������
//            for (size_t i = 0; i < contours.size(); i++)
//            {
//                //ʹ��ͼ����������ж�������
//                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

//                //������������󣬵õ�����4������
//                if (approx.size() == 4 &&fabs(contourArea(Mat(approx))) > 1000 &&isContourConvex(Mat(approx)))
//                {
//                    double maxCosine = 0;

//                    for (int j = 2; j < 5; j++)
//                    {
//                        // ��������Ե֮��Ƕȵ��������
//                        double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
//                        maxCosine = MAX(maxCosine, cosine);
//                    }

//                    if (maxCosine < 0.3)
//                    {
//                        squares.push_back(approx);
//                    }
//                }
//            }
//        }
//    }


//    for (size_t i = 0; i < squares.size(); i++)
//    {
//        const Point* p = &squares[i][0];

//        int n = (int)squares[i].size();
//        if (p->x > 3 && p->y > 3)
//        {
//            polylines(out, &p, &n, 1, true, Scalar(0, 255, 0), 3, LINE_AA);
//        }
//    }
//    imshow("dst",out);
//}




//int main()
//{
//    Mat src = imread("F:/qt/canny/canny/images/3.jpg");
//    Mat out;
//    findSquares(src,out);

//    waitKey(10000);
//    return 0;


//}











//#include <opencv2/opencv.hpp>
//#include <iostream>
//using namespace cv;
//using namespace std;

//int main()
//{
//    Mat src = imread("F:/qt/canny/canny/images/3.jpg");
//    if (src.empty())
//    {
//        printf("read image error\n");
//        system("pause");
//        return -1;
//    }

//    imshow("src", src);

//    Scalar colorTab[] = {
//        Scalar(0, 0, 255),
//        Scalar(0, 255, 0),
//        Scalar(255, 0, 0),
//        Scalar(0, 255, 255),
//        Scalar(255, 0, 255)
//    };

//    int width = src.cols;
//    int height = src.rows;
//    int dims = src.channels();

//    // ��ʼ������
//    int sampleCount = width * height;
//    int clusterCount = 4;
//    Mat points(sampleCount, dims, CV_32F, Scalar(10));
//    Mat labels;
//    Mat centers(clusterCount, 1, points.type());

//    // RGB ��������ת������������
//    int index = 0;
//    for (int row = 0; row < height; row++)
//    {
//        for (int col = 0; col < width; col++)
//        {
//            // ��άתһά
//            index = row * width + col;
//            Vec3b bgr = src.at<Vec3b>(row, col);
//            points.at<float>(index, 0) = static_cast<int>(bgr[0]);
//            points.at<float>(index, 1) = static_cast<int>(bgr[1]);
//            points.at<float>(index, 2) = static_cast<int>(bgr[2]);
//        }
//    }

//    // KMeans
//    TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1);
//    kmeans(points, clusterCount, labels, criteria, 3, KMEANS_PP_CENTERS, centers);

//    // ��ʾͼ��ָ��Ľ����һάת��ά
//    Mat result = Mat::zeros(src.size(), src.type());
//    for (int row = 0; row < height; row++)
//    {
//        for (int col = 0; col < width; col++)
//        {
//            index = row * width + col;
//            int label = labels.at<int>(index, 0);
//            result.at<Vec3b>(row, col)[0] = colorTab[label][0];
//            result.at<Vec3b>(row, col)[1] = colorTab[label][1];
//            result.at<Vec3b>(row, col)[2] = colorTab[label][2];
//        }
//    }

//    // ���ĵ���ʾ
//    for (int i = 0; i < centers.rows; i++)
//    {
//        int x = centers.at<float>(i, 0);
//        int y = centers.at<float>(i, 1);
//        circle(result, Point(x, y), 10, Scalar(255,255,255), 1, LINE_AA);
//    }

//    imshow("result", result);

//    waitKey(0);
//    return(0);
//}








//#include <iostream>
//#include <include/opencv2/opencv.hpp>
//#include <include/opencv2/highgui.hpp>
//#include <vector>

//using namespace std;
//using namespace cv;

//const int bins = 256;
//Mat src;
//const char *winTitle = "input image";

//void showHistogram();

///*
// * ͼ��ֱ��ͼ
// */
//int main() {
//    src = imread("F:/qt/canny/canny/images/3.jpg");
//    if (src.empty()) {
//        cout << "could not load image.." << endl;
//    }
//    imshow(winTitle, src);
//    showHistogram();

//    waitKey(0);
//    return 0;
//}

//void showHistogram() {
//    // ��ͨ������
//    vector<Mat> bgr_plane;
//    split(src, bgr_plane);
//    // �����������
//    const int channels[1] = {0};
//    const int bins[1] = {256};
//    float hranges[2] = {0, 255};
//    const float *ranges[1] = {hranges};
//    Mat b_hist, g_hist, r_hist;
//    // ������ͨ��ֱ��ͼ
//    calcHist(&bgr_plane[0], 1, 0, Mat(), b_hist, 1, bins, ranges);
//    calcHist(&bgr_plane[1], 1, 0, Mat(), g_hist, 1, bins, ranges);
//    calcHist(&bgr_plane[2], 1, 0, Mat(), r_hist, 1, bins, ranges);
//    /*
//     * ��ʾֱ��ͼ
//     */
//    int hist_w = 512;
//    int hist_h = 400;
//    int bin_w = cvRound((double) hist_w / bins[0]);
//    Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);
//    // ��һ��ֱ��ͼ����
//    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1);
//    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1);
//    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1);
//    // ����ֱ��ͼ����
//    for (int i = 1; i < bins[0]; ++i) {
//        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
//             Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))), Scalar(255, 0, 0),
//             2, 8, 0);
//        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
//             Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))), Scalar(0, 255, 0),
//             2, 8, 0);
//        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
//             Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))), Scalar(0, 0, 255),
//             2, 8, 0);

//    }
//    imshow("Histogram", histImage);
//}







//#include <iostream>
//#include <include/opencv2/opencv.hpp>
//#include <include/opencv2/highgui.hpp>

//using namespace std;
//using namespace cv;


//int main()
//{
//    cv::Mat image = cv::imread("F:/qt/canny/canny/images/3.jpg", 1);
//    cv::Ptr<cv::MSER> ptrMSER =
//        cv::MSER::create(5, // �ֲ����ʱʹ�õ�����ֵ
//            200, // �������С���
//            20000); // �����������
//    std::vector<std::vector<cv::Point> > points;
//    std::vector<cv::Rect> rects;
//    ptrMSER->detectRegions(image, points, rects);

//    cv::Mat output(image.size(), CV_8UC3);
//    output = cv::Scalar(255, 255, 255);
//    cv::RNG rng;
//    // ���ÿ����⵽�����������ڲ�ɫ������ʾ MSER
//    // ������������ʾ�ϴ�� MSER
//    for (std::vector<std::vector<cv::Point> >::reverse_iterator
//        it = points.rbegin();
//        it != points.rend(); ++it) {
//        // ���������ɫ
//        cv::Vec3b c(rng.uniform(0, 254),
//            rng.uniform(0, 254), rng.uniform(0, 254));
//        // ��� MSER �����е�ÿ����
//        for (std::vector<cv::Point>::iterator itPts = it->begin();
//            itPts != it->end(); ++itPts) {
//            // ����д MSER ������
//            if (output.at<cv::Vec3b>(*itPts)[0] == 255) {
//                output.at<cv::Vec3b>(*itPts) = c;
//            }
//        }
//    }

//    cv::imshow("image", image);
//    cv::imshow("output", output);
//    waitKey(0);
//}









//#include <iostream>
//#include <stdio.h>
//#include <include/opencv2/opencv.hpp>
//#include <include/opencv2/highgui.hpp>

//using namespace std;
//using namespace cv;

//int hmin = 170, hmax = 180, smin = 43, smax = 255, vmin = 46, vmax = 255;
//int g_nStructElementSize = 3;
//int g_nGaussianBlurValue = 6;

//int main()
//{
//    Mat img = imread("F:/qt/canny/canny/images/3.jpg");
//    Mat imghsv;
//    cvtColor(img, imghsv, COLOR_BGR2HSV);//RGB to HSV
//    imshow("hs", imghsv);

//    Mat mask;
//    inRange(imghsv, Scalar(hmin, smin, vmin), Scalar(hmax, smax, vmax), mask);//filter red color
//    imshow("mask", mask);

//    Mat out2;
//    Mat element = getStructuringElement(MORPH_RECT, Size(2 * g_nStructElementSize + 1, 2 * g_nStructElementSize + 1), Point(g_nStructElementSize, g_nStructElementSize));
//    erode(mask, out2, element); //erode
//    imshow("��ʴ", out2);

//    Mat gaussian;
//    GaussianBlur(out2, gaussian, Size(g_nGaussianBlurValue * 2 + 1, g_nGaussianBlurValue * 2 + 1), 0, 0);//ģ����
//    imshow("��˹�˲�", gaussian);

//    vector<vector<Point>>contours;
//    vector<Vec4i>hierarchy;
//    Mat imgcontours;
//    Point2f center;
//    float radius;

//    cv::findContours(gaussian, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
//    double maxarea = 0;
//    int maxareaidx = 0;
//    for (int index = contours.size() - 1; index >= 0; index --)// find the maxarea return contour index
//    {
//        double tmparea = fabs(contourArea(contours[index]));
//        if (tmparea > maxarea)
//        {
//            maxarea = tmparea;
//            maxareaidx = index;
//        }
//    }
//    minEnclosingCircle(contours[maxareaidx], center, radius);//using index ssearching the min circle
//    circle(img, center, (int)radius, Scalar(255,0,0), 3);//using contour index to drawing circle
//    imshow("����", img);

//    waitKey();
//}




//#include <iostream>
//#include <include/opencv2/opencv.hpp>
//#include <include/opencv2/highgui.hpp>

//using namespace std;
//using namespace cv;



//int main( )
//{
//    //��1������ԭʼͼ��Mat��������
//    Mat srcImage = imread("F:/qt/canny/canny/images/08.jpg");  //����Ŀ¼��Ӧ����һ����Ϊ1.jpg���ز�ͼ
//    Mat midImage,dstImage;//��ʱ������Ŀ��ͼ�Ķ���

//    //��2����ʾԭʼͼ
//    imshow("��ԭʼͼ��", srcImage);

//    //��3��תΪ�Ҷ�ͼ������ͼ��ƽ��
//    cvtColor(srcImage,midImage, CV_BGR2GRAY);//ת����Ե�����ͼΪ�Ҷ�ͼ
//    GaussianBlur( midImage, midImage, Size(9, 9), 2, 2 );

//    //��4�����л���Բ�任
//    vector<Vec3f> circles;
//    HoughCircles( midImage, circles, CV_HOUGH_GRADIENT,1.5, 10, 200, 100, 0, 0 );

//    //��5��������ͼ�л��Ƴ�Բ
//    for( size_t i = 0; i < circles.size(); i++ )
//    {
//        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
//        int radius = cvRound(circles[i][2]);
//        //����Բ��
//        circle( srcImage, center, 3, Scalar(0,255,0), -1, 8, 0 );
//        //����Բ����
//        circle( srcImage, center, radius, Scalar(155,50,255), 3, 8, 0 );
//    }

//    //��6����ʾЧ��ͼ
//    imshow("��Ч��ͼ��", srcImage);

//    waitKey(0);

//    return 0;
//}











//#include <iostream>
//#include <include/opencv2/opencv.hpp>
//#include <include/opencv2/imgproc/imgproc.hpp>

//using namespace std;
//using namespace cv;

//int main()
//{

//    Mat edges; //����ת���ĻҶ�ͼ
//    namedWindow("Ч��ͼ",CV_WINDOW_NORMAL);


//    Mat frame;
//    Mat img = imread("F:/qt/canny/canny/images/3.jpg");

//    if(!img.data)
//    {
//        return -1;
//    }
//    cvtColor(img, edges, CV_BGR2GRAY);
//    //��˹�˲�
//    GaussianBlur(edges, edges, Size(7,7), 2, 2);
//    vector<Vec3f>circles;

//    Mat edges2, edges_src;
//    Canny(img, edges2, 100, 300);
//    // ��ȡ��ɫ��Ե
//    bitwise_and(img, img, edges_src, edges2);
//    imshow("edges2", edges2);
//    imshow("edges_src", edges_src);

//    waitKey(0);
//    return 0;



////    //����Բ
////    HoughCircles(edges2, circles, CV_HOUGH_GRADIENT, 1.5, 10, 200, 100, 0, 0);
////    for(size_t i = 0; i < circles.size(); i++)
////    {
////        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
////        int radius = cvRound(circles[i][2]);
////        //����Բ��
////        circle(img, center, 3, Scalar(0, 255, 0), -1, 8, 0);
////        //����Բ����
////        circle(img, center, radius, Scalar(155, 50, 255), 2, 8, 0);

////    }

////    imshow("Ч��ͼ2",img);
////    waitKey(30000);


//    return 0;
//}



//#include <iostream>
//#include <include/opencv2/opencv.hpp>
//#include <include/opencv2/imgproc/imgproc.hpp>

//using namespace std;
//using namespace cv;



//int main() {
//    Mat src = imread("F:/qt/canny/canny/images/3.jpg");
//    if (src.empty()) {
//        cout << "could not load image.." << endl;
//    }
//    imshow("input", src);

//    // ȥ�������ֵ��
//    Mat binary;
//    Canny(src, binary, 80, 160);

//    // ��׼����ֱ�߼��
//    vector<Vec2f> lines;
//    HoughLines(binary, lines, 1, CV_PI / 180, 150);

//    // ����ֱ��
//    Point pt1, pt2;
//    for (size_t i = 0; i < lines.size(); ++i) {
//        float rho = lines[i][0];
//        float theta = lines[i][1];
//        double a = cos(theta), b = sin(theta);
//        double x0 = a * rho, y0 = b * rho;
//        pt1.x = cvRound(x0 + 1000 * (-b));
//        pt1.y = cvRound(y0 + 1000 * (a));
//        pt2.x = cvRound(x0 - 1000 * (-b));
//        pt2.y = cvRound(y0 - 1000 * (a));
//        line(src, pt1, pt2, Scalar(0, 0, 255), 2, LINE_AA);
//    }

//    imshow("contours", src);

//    waitKey(0);
//    return 0;
//}


//#include <iostream>
//#include <opencv2/opencv.hpp>

//using namespace std;
//using namespace cv;

///*
// * ��ֵͼ�����(������Բ����Բ���)
// */
//int main() {
//    Mat src = imread("F:/qt/canny/canny/images/06.jpg");
//    if (src.empty()) {
//        cout << "could not load image.." << endl;
//    }
//    imshow("input", src);

//    // ȥ�������ֵ��
//    Mat dst, gray, binary;
//    Canny(src, binary, 80, 160);

//    Mat k = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
//    dilate(binary, binary, k);

//    // �������������
//    vector<vector<Point>> contours;
//    vector<Vec4i> hierarchy;
//    findContours(binary, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());
//    for (size_t t = 0; t < contours.size(); t++) {
//        // Ѱ������Բ
//        RotatedRect rrt = fitEllipse(contours[t]);
//        ellipse(src, rrt, Scalar(0,0,255), 2);
//    }

//    imshow("contours", src);

//    waitKey(0);
//    return 0;
//}



//Բ���
//#include <iostream>
//#include <include/opencv2/opencv.hpp>
//#include <include/opencv2/highgui.hpp>
//#include <include/opencv2/imgproc/imgproc_c.h>
//#include <include/opencv2/imgproc.hpp>

//using namespace std;
//using namespace cv;

//bool circleLeastFit(CvSeq* points, double &center_x, double &center_y, double &radius);//��С���˷���Ϻ���


//int main()
//{
//    const char* winname  ="winname";
//    //const char* winname1  ="winname1";
//    //const char* winname2  ="winname2";
//    //const char* winname3  ="winname3";
//    char * picname = "P11.jpg";
//    //����ԭͼ
//    IplImage * pImage = cvLoadImage(picname);

//    //����ͼ��
//    IplImage *pR = cvCreateImage(cvGetSize(pImage),IPL_DEPTH_8U,1);
//    IplImage *pG = cvCreateImage(cvGetSize(pImage),IPL_DEPTH_8U,1);
//    IplImage *pB = cvCreateImage(cvGetSize(pImage),IPL_DEPTH_8U,1);

//    IplImage *temp = cvCreateImage(cvGetSize(pImage),IPL_DEPTH_8U,1);
//    IplImage *binary = cvCreateImage(cvGetSize(pImage),IPL_DEPTH_8U,1);
//    //trackbar�ı���ֵ    //��Ӧ����ͨ��
//    int b_low =20;
//    int b_high = 100;
//    int g_low = 20;
//    int g_high = 100;
//    int r_low = 0;
//    int r_high = 100;

//    //�������
//    CvMemStorage *storage = cvCreateMemStorage(0);
//    CvSeq * seq = cvCreateSeq(CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint), storage);

//    //����
//    cvNamedWindow(winname);
//    cvShowImage(winname, pImage);  //��ʾԭͼ
//    cvNamedWindow("r",2);
//    cvNamedWindow("g",2);
//    cvNamedWindow("b",2); //����ͨ��
//    cvNamedWindow("binary",2);//��ֵ��ͼ

//    //����Ӧ�Ĵ��ڽ���������
//    cvCreateTrackbar(  "b1","b", &b_low,  254,   NULL); //Hͨ��������Χ0-180
//    cvSetTrackbarPos("b1","b",0 );                        //����Ĭ��λ��
//    cvCreateTrackbar(  "b2","b", &b_high,  254,   NULL);//Hͨ��������Χ0-180
//    cvSetTrackbarPos("b2","b",110 );

//    cvCreateTrackbar(  "g1","g", &g_low,  254,   NULL);
//    cvSetTrackbarPos("g1","g",0 );
//    cvCreateTrackbar(  "g2","g", &g_high,  254,   NULL);
//    cvSetTrackbarPos("g2","g",158 );

//    cvCreateTrackbar(  "r1","r", &r_low,  254,   NULL);
//    cvSetTrackbarPos("r1","r",68 );
//    cvCreateTrackbar(  "r2","r", &r_high,  254,   NULL);
//    cvSetTrackbarPos("r2","r",137);

//    while(1)
//    {
//        //����ͨ������
//        cvSplit(pImage,pB,pG,pR,NULL);

//        //��ֵ��
//        cvThreshold(pB, temp,b_low , 255, CV_THRESH_BINARY);
//        cvThreshold(pB, pB,b_high , 255, CV_THRESH_BINARY_INV);
//        cvAnd(temp,pB,pB,NULL);//��������ϳ�һ��ͼ

//        cvThreshold(pG, temp,g_low , 255, CV_THRESH_BINARY);
//        cvThreshold(pG, pG,g_high , 255, CV_THRESH_BINARY_INV);
//        cvAnd(temp,pG,pG,NULL);//��������ϳ�һ��ͼ

//        cvThreshold(pR, temp,r_low , 255, CV_THRESH_BINARY);
//        cvThreshold(pR, pR,r_high , 255, CV_THRESH_BINARY_INV);
//        cvAnd(temp,pR,pR,NULL);//��������ϳ�һ��ͼ

//        //��ʾ����ͨ����ͼ��
//        cvShowImage("b",pB);
//        cvShowImage("g",pG);
//        cvShowImage("r",pR);

//        //�ϳɵ�һ��ͼ��
//        cvAnd(pB, pG, binary, NULL);
//        cvAnd(pR, binary, binary, NULL);

//        //���͸�ʴ����ȥ���ڵ�
//        //cvDilate(binary,binary);
//        //cvErode(binary,binary);

//        //��ʾ�ϳɵĶ�ֵ��ͼ
//        cvShowImage("binary",binary);
//        //cvSaveImage("erzhitu.jpg",binary);

//        // ��������
//        int cnt = cvFindContours(binary,storage,&seq,sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);//������������Ŀ
//        CvSeq* _contour =seq;
//        cout<<"number of contours "<<cnt<<endl;
//////////////////////_
//        //�ҵ��������������
//        double maxarea=0;
//        int ind_max = -1;
//        int m=0;
//         for( ; seq != 0; seq = seq->h_next )
//         {
//             m++;
//            double tmparea = abs(cvArcLength(seq,CV_WHOLE_SEQ,-1));
//              //double contArea = fabs(cvContourArea(pcontour,CV_WHOLE_SEQ));
//            if(tmparea > maxarea)
//            {
//                maxarea = tmparea;
//                ind_max=m;
//            }
//            // cout<<"seqfor:  "<<seq->total<<endl;
//         }
//         m=0;
//         seq = _contour;
//         for( ; seq != 0; seq = seq->h_next )
//         {
//            m++;
//            if(m == ind_max)
//            {
//                break;
//            }
//         }
//         CvSeq*  cur_cont = seq;
//         cout<<"seq:  "<<seq->total<<endl;
//         cout<<"cur_cont:  "<<cur_cont->total<<endl;
//         //for (int i=0;i<cur_cont->total;++i)
//         //{
//            // CvPoint* p = CV_GET_SEQ_ELEM(CvPoint,cur_cont,i);//��������ϵ������
//            // printf("(%d,%d)\n",p->x,p->y);
//         //}
//         //cvWaitKey(0);

//         //������ɫ���ͼ��
//         IplImage *pOutlineImage = cvCreateImage(cvGetSize(pImage), IPL_DEPTH_8U, 3);
//         cvCopy(pImage,pOutlineImage);

//         //int nLevels = 5;
//         //��ȡ���������͹���㼯
//         CvSeq* hull=NULL;
//         hull = cvConvexHull2(cur_cont,0,CV_CLOCKWISE,0);
//         cout<<"hull total points number:"<<hull->total<<endl;
//         CvPoint pt0 = **(CvPoint**)cvGetSeqElem(hull,hull->total - 1);
//         for(int i = 0;i<hull->total;++i){
//             CvPoint pt1 = **(CvPoint**)cvGetSeqElem(hull,i);
//             //cvLine(pOutlineImage,pt0,pt1,CV_RGB(0,0,255));
//             cvLine(pOutlineImage, pt0, pt1, CV_RGB(0,0,255),1,LINE_AA);
//             pt0 = pt1;
//         }

//         //��С���˷����Բ
//         double center_x=0;
//         double center_y=0;
//         double radius=0;
//         cout<<"nihe :"<<circleLeastFit(hull, center_x, center_y, radius);
//         cout<<"canshu: "<<center_x<<endl<<center_y<<endl<<radius<<endl;

//         //����Բ
//         cvCircle(pOutlineImage,Point2f(center_x,center_y),radius,CV_RGB(0,100,100));
//         //cvCircle(pOutlineImage,Point2f(center_x,center_y),radius,CV_RGB(0,100,100));

////////////////////////////////////////////////////////////////////////////

//        //��������
//        //cvDrawContours(pOutlineImage, cur_cont, CV_RGB(255,0,0), CV_RGB(0,255,0),0);
//        //cvDrawContours(dst,contour,CV_RGB(255,0,0),CV_RGB(0,255,0),0);
//        cvShowImage(winname, pOutlineImage);  //��ʾԭͼjiangshang luokuo

//        if (cvWaitKey(1000) == 27)
//        {
//            cvSaveImage("tutu.jpg",pOutlineImage);

//            break;
//        }
//        cvClearMemStorage( storage );  //���������ռ�õ��ڴ�
//        cvReleaseImage(&pOutlineImage);//�����ɫ���ͼ��
//    }

//    cvDestroyAllWindows();
//    cvReleaseImage(&pImage);
//    cvReleaseImage(&pR);
//    cvReleaseImage(&pG);
//    cvReleaseImage(&pB);
//    cvReleaseImage(&temp);
//    cvReleaseImage(&binary);
//    return 0;
//}

////��С���˷���ϣ����Բ�ĵ�xy����ֵ�Ͱ뾶��С��
//bool circleLeastFit(CvSeq* points, double &center_x, double &center_y, double &radius)
//{
//    center_x = 0.0f;
//    center_y = 0.0f;
//    radius = 0.0f;
//    if (points->total < 3)
//    {
//        return false;
//    }

//    double sum_x = 0.0f, sum_y = 0.0f;
//    double sum_x2 = 0.0f, sum_y2 = 0.0f;
//    double sum_x3 = 0.0f, sum_y3 = 0.0f;
//    double sum_xy = 0.0f, sum_x1y2 = 0.0f, sum_x2y1 = 0.0f;

//    int N = points->total ;
//    for (int i = 0; i < N; i++)
//    {
//         CvPoint pt1 = **(CvPoint**)cvGetSeqElem(points,i);
//        double x =pt1.x;
//        double y = pt1.y ;
//        double x2 = x * x;
//        double y2 = y * y;
//        sum_x += x;
//        sum_y += y;
//        sum_x2 += x2;
//        sum_y2 += y2;
//        sum_x3 += x2 * x;
//        sum_y3 += y2 * y;
//        sum_xy += x * y;
//        sum_x1y2 += x * y2;
//        sum_x2y1 += x2 * y;
//    }

//    double C, D, E, G, H;
//    double a, b, c;

//    C = N * sum_x2 - sum_x * sum_x;
//    D = N * sum_xy - sum_x * sum_y;
//    E = N * sum_x3 + N * sum_x1y2 - (sum_x2 + sum_y2) * sum_x;
//    G = N * sum_y2 - sum_y * sum_y;
//    H = N * sum_x2y1 + N * sum_y3 - (sum_x2 + sum_y2) * sum_y;
//    a = (H * D - E * G) / (C * G - D * D);
//    b = (H * C - E * D) / (D * D - G * C);
//    c = -(a * sum_x + b * sum_y + sum_x2 + sum_y2) / N;

//    center_x = a / (-2);
//    center_y = b / (-2);
//    radius = sqrt(a * a + b * b - 4 * c) / 2;
//    return true;
//}

////2d--2d
//#include <iostream>
//#include <include/opencv2/core/core.hpp>
//#include <include/opencv2/features2d/features2d.hpp>
//#include <include/opencv2/highgui/highgui.hpp>
//#include <include/opencv2/calib3d/calib3d.hpp>
//// #include "extra.h" // use this if in OpenCV2
//using namespace std;
//using namespace cv;

///****************************************************
// * ��������ʾ�����ʹ��2D-2D������ƥ���������˶�
// * **************************************************/

//void find_feature_matches (
//    const Mat& img_1, const Mat& img_2,
//    std::vector<KeyPoint>& keypoints_1,
//    std::vector<KeyPoint>& keypoints_2,
//    std::vector< DMatch >& matches );

//void pose_estimation_2d2d (
//    std::vector<KeyPoint> keypoints_1,
//    std::vector<KeyPoint> keypoints_2,
//    std::vector< DMatch > matches,
//    Mat& R, Mat& t );

//// ��������ת�����һ������
//Point2d pixel2cam ( const Point2d& p, const Mat& K );

//int main ( int argc, char** argv )
//{
////    if ( argc != 3 )
////    {
////        cout<<"usage: pose_estimation_2d2d img1 img2"<<endl;
////        return 1;
////    }
////    //-- ��ȡͼ��
////    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
////    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );


//    Mat img_1 = imread ("F:/qt/canny/canny/images/12.jpg");
//    Mat img_2 = imread ("F:/qt/canny/canny/images/13.jpg");

//    vector<KeyPoint> keypoints_1, keypoints_2;
//    vector<DMatch> matches;
//    find_feature_matches ( img_1, img_2, keypoints_1, keypoints_2, matches );
//    cout<<"һ���ҵ���"<<matches.size() <<"��ƥ���"<<endl;

//    //-- ��������ͼ����˶�
//    Mat R,t;
//    pose_estimation_2d2d ( keypoints_1, keypoints_2, matches, R, t );

//    //-- ��֤E=t^R*scale
//    Mat t_x = ( Mat_<double> ( 3,3 ) <<
//                0,                      -t.at<double> ( 2,0 ),     t.at<double> ( 1,0 ),
//                t.at<double> ( 2,0 ),      0,                      -t.at<double> ( 0,0 ),
//                -t.at<double> ( 1.0 ),     t.at<double> ( 0,0 ),      0 );

//    cout<<"t^R="<<endl<<t_x*R<<endl;

//    //-- ��֤�Լ�Լ��
//    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
//    for ( DMatch m: matches )
//    {
//        Point2d pt1 = pixel2cam ( keypoints_1[ m.queryIdx ].pt, K );
//        Mat y1 = ( Mat_<double> ( 3,1 ) << pt1.x, pt1.y, 1 );
//        Point2d pt2 = pixel2cam ( keypoints_2[ m.trainIdx ].pt, K );
//        Mat y2 = ( Mat_<double> ( 3,1 ) << pt2.x, pt2.y, 1 );
//        Mat d = y2.t() * t_x * R * y1;
//        cout << "epipolar constraint = " << d << endl;
//    }
//    return 0;
//}

//void find_feature_matches ( const Mat& img_1, const Mat& img_2,
//                            std::vector<KeyPoint>& keypoints_1,
//                            std::vector<KeyPoint>& keypoints_2,
//                            std::vector< DMatch >& matches )
//{
//    //-- ��ʼ��
//    Mat descriptors_1, descriptors_2;
//    // used in OpenCV3
//    Ptr<FeatureDetector> detector = ORB::create();
//    Ptr<DescriptorExtractor> descriptor = ORB::create();
//    // use this if you are in OpenCV2
//    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
//    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
//    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
//    //-- ��һ��:��� Oriented FAST �ǵ�λ��
//    detector->detect ( img_1,keypoints_1 );
//    detector->detect ( img_2,keypoints_2 );

//    //-- �ڶ���:���ݽǵ�λ�ü��� BRIEF ������
//    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
//    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

//    //-- ������:������ͼ���е�BRIEF�����ӽ���ƥ�䣬ʹ�� Hamming ����
//    vector<DMatch> match;
//    //BFMatcher matcher ( NORM_HAMMING );
//    matcher->match ( descriptors_1, descriptors_2, match );

//    //-- ���Ĳ�:ƥ����ɸѡ
//    double min_dist=10000, max_dist=0;

//    //�ҳ�����ƥ��֮�����С�����������, ���������Ƶĺ�����Ƶ������֮��ľ���
//    for ( int i = 0; i < descriptors_1.rows; i++ )
//    {
//        double dist = match[i].distance;
//        if ( dist < min_dist ) min_dist = dist;
//        if ( dist > max_dist ) max_dist = dist;
//    }

//    printf ( "-- Max dist : %f \n", max_dist );
//    printf ( "-- Min dist : %f \n", min_dist );

//    //��������֮��ľ��������������С����ʱ,����Ϊƥ������.����ʱ����С�����ǳ�С,����һ������ֵ30��Ϊ����.
//    for ( int i = 0; i < descriptors_1.rows; i++ )
//    {
//        if ( match[i].distance <= max ( 2*min_dist, 30.0 ) )
//        {
//            matches.push_back ( match[i] );
//        }
//    }
//}


//Point2d pixel2cam ( const Point2d& p, const Mat& K )
//{
//    return Point2d
//           (
//               ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
//               ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
//           );
//}


//void pose_estimation_2d2d ( std::vector<KeyPoint> keypoints_1,
//                            std::vector<KeyPoint> keypoints_2,
//                            std::vector< DMatch > matches,
//                            Mat& R, Mat& t )
//{
//    // ����ڲ�,TUM Freiburg2
//    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );

//    //-- ��ƥ���ת��Ϊvector<Point2f>����ʽ
//    vector<Point2f> points1;
//    vector<Point2f> points2;

//    for ( int i = 0; i < ( int ) matches.size(); i++ )
//    {
//        points1.push_back ( keypoints_1[matches[i].queryIdx].pt );
//        points2.push_back ( keypoints_2[matches[i].trainIdx].pt );
//    }

//    //-- �����������
//    Mat fundamental_matrix;
//    fundamental_matrix = findFundamentalMat ( points1, points2, CV_FM_8POINT );
//    cout<<"fundamental_matrix is "<<endl<< fundamental_matrix<<endl;

//    //-- ���㱾�ʾ���
//    Point2d principal_point ( 325.1, 249.7 );	//�������, TUM dataset�궨ֵ
//    double focal_length = 521;			//�������, TUM dataset�궨ֵ
//    Mat essential_matrix;
//    essential_matrix = findEssentialMat ( points1, points2, focal_length, principal_point );
//    cout<<"essential_matrix is "<<endl<< essential_matrix<<endl;

//    //-- ���㵥Ӧ����
//    Mat homography_matrix;
//    homography_matrix = findHomography ( points1, points2, RANSAC, 3 );
//    cout<<"homography_matrix is "<<endl<<homography_matrix<<endl;

//    //-- �ӱ��ʾ����лָ���ת��ƽ����Ϣ.
//    recoverPose ( essential_matrix, points1, points2, R, t, focal_length, principal_point );
//    cout<<"R is "<<endl<<R<<endl;
//    cout<<"t is "<<endl<<t<<endl;

//}
