////---------------------------------【头文件、命名空间包含部分】----------------------------
////        描述：包含程序所使用的头文件和命名空间
////------------------------------------------------------------------------------------------------
////#include "stdafx.h"
//#include "include/opencv2/highgui/highgui.hpp"
//#include "include/opencv2/imgproc/imgproc.hpp"
//#include <iostream>
//using namespace cv;
//using namespace std;


////-----------------------------------【宏定义部分】--------------------------------------------
////        描述：定义一些辅助宏
////------------------------------------------------------------------------------------------------
//#define WINDOW_NAME1 "【原始图窗口】"            //为窗口标题定义的宏
//#define WINDOW_NAME2 "【轮廓图】"                    //为窗口标题定义的宏


////-----------------------------------【全局变量声明部分】--------------------------------------
////        描述：全局变量的声明
////-----------------------------------------------------------------------------------------------
//Mat g_srcImage;
//Mat g_grayImage;
//int g_nThresh = 80;
//int g_nThresh_max = 255;
//RNG g_rng(12345);
//Mat g_cannyMat_output;
//vector<vector<Point>> g_vContours;
//vector<Vec4i> g_vHierarchy;


////-----------------------------------【全局函数声明部分】--------------------------------------
////        描述：全局函数的声明
////-----------------------------------------------------------------------------------------------
//static void ShowHelpText( );
//void on_ThreshChange(int, void* );


////-----------------------------------【main( )函数】--------------------------------------------
////        描述：控制台应用程序的入口函数，我们的程序从这里开始执行
////-----------------------------------------------------------------------------------------------
//int main( int argc, char** argv )
//{
//    //【0】改变console字体颜色
//    system("color 1F");

//    // 加载源图像
//    g_srcImage = imread( "F:/qt/canny/canny/images/539.jpg", 1 );
//    if(!g_srcImage.data ) { printf("读取图片错误，请确定目录下是否有imread函数指定的图片存在~！ \n"); return false; }

//    // 转成灰度并模糊化降噪
//    cvtColor( g_srcImage, g_grayImage, COLOR_BGR2GRAY );
//    blur( g_grayImage, g_grayImage, Size(3,3) );

//    // 创建窗口
//    namedWindow( WINDOW_NAME1, WINDOW_AUTOSIZE );
//    imshow( WINDOW_NAME1, g_srcImage );

//    //创建滚动条并初始化
//    createTrackbar( "canny阈值", WINDOW_NAME1, &g_nThresh, g_nThresh_max, on_ThreshChange );
//    on_ThreshChange( 0, 0 );

//    waitKey(0);
//    return(0);
//}

////-----------------------------------【on_ThreshChange( )函数】------------------------------
////      描述：回调函数
////----------------------------------------------------------------------------------------------
//void on_ThreshChange(int, void* )
//{

//    // 用Canny算子检测边缘
//    Canny( g_grayImage, g_cannyMat_output, g_nThresh, g_nThresh*2, 3 );

//    // 寻找轮廓
//    findContours( g_cannyMat_output, g_vContours, g_vHierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

//    // 绘出轮廓
//    Mat drawing = Mat::zeros( g_cannyMat_output.size(), CV_8UC3 );
//    for( int i = 0; i< g_vContours.size(); i++ )
//    {
//        //Scalar color = Scalar( g_rng.uniform(0, 255), g_rng.uniform(0,255), g_rng.uniform(0,255) );//任意值
//        Scalar color = Scalar(255, 182, 193);
//        drawContours( drawing, g_vContours, i, color, 2, 8, g_vHierarchy, 0, Point() );
//    }

//    vector<Point> tempPoint;     // 点集
//    // 将所有点集存储到tempPoint
//    for (int k = 0; k < g_vContours.size(); k++)
//    {
//        for (int m = 0; m < g_vContours[k].size(); m++)
//        {
//            tempPoint.push_back(g_vContours[k][m]);
//        }
//    }
//    //对给定的 2D 点集，寻找最小面积的包围矩形
//    RotatedRect box = minAreaRect(Mat(tempPoint));
//    Point2f vertex[4];
//    box.points(vertex);

//    //绘制出最小面积的包围矩形
//    for (int i = 0; i < 4; i++)
//    {
//        line(drawing, vertex[i], vertex[(i + 1) % 4], Scalar(100, 200, 211), 2, LINE_AA);
//    }

//    imshow(WINDOW_NAME2, drawing);
//}



////颜色识别2
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
//            vector<Mat> hsvSplit;   //创建向量容器，存放HSV的三通道数据
//            cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
//            split(imgHSV, hsvSplit);			//分类原图像的HSV三通道
//            equalizeHist(hsvSplit[2], hsvSplit[2]);    //对HSV的亮度通道进行直方图均衡
//            merge(hsvSplit, imgHSV);				   //合并三种通道
//            cvtColor(imgHSV, imgBGR, COLOR_HSV2BGR);    //将HSV空间转回至RGB空间，为接下来的颜色识别做准备
//        }
//        else
//        {
//            imgBGR = imgOriginal.clone();
//        }



//        switch(ctrl)
//        {
//        case 0:
//            {
//                inRange(imgBGR, Scalar(128, 0, 0), Scalar(255, 127, 127), imgThresholded); //蓝色
//                break;
//            }
//        case 1:
//            {
//                inRange(imgBGR, Scalar(128, 128, 128), Scalar(255, 255, 255), imgThresholded); //白色
//                break;
//            }
//        case 2:
//            {
//                inRange(imgBGR, Scalar(128, 128, 0), Scalar(255, 255, 127), imgThresholded); //靛色
//                break;
//            }
//        case 3:
//            {
//                inRange(imgBGR, Scalar(128, 0, 128), Scalar(255, 127, 255), imgThresholded); //紫色
//                break;
//            }
//        case 4:
//            {
//                inRange(imgBGR, Scalar(0, 128, 128), Scalar(127, 255, 255), imgThresholded); //黄色
//                break;
//            }
//        case 5:
//            {
//                inRange(imgBGR, Scalar(0, 128, 0), Scalar(127, 255, 127), imgThresholded); //绿色
//                break;
//            }
//        case 6:
//            {
//                inRange(imgBGR, Scalar(0, 0, 128), Scalar(127, 127, 255), imgThresholded); //红色
//                break;
//            }
//        case 7:
//            {
//                inRange(imgBGR, Scalar(0, 0, 0), Scalar(127, 127, 127), imgThresholded); //黑色
//                break;
//            }
//        }

//        imshow("形态学去噪声前", imgThresholded);

//        Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
//        morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);
//        morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, element);

//        imshow("Thresholded Image", imgThresholded); //show the thresholded image
//        imshow("直方图均衡以后", imgBGR);
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

////相机畸变参数
//const Mat  arucodistCoeffs = (Mat_<float>(1, 5) << 0, 0, 0, 0, 0);//

////判断solvePnPRansac是否完成
//bool PnPRansac;

//int main(int args, char *argv[])
//{
//    //图像物理坐标
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


//    //获取相机
//    VideoCapture cap(1);

//    Mat frame,framecopy;
//    //设置字典
//    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

//    while (true)
//    {
//        cap>>frame;
//        frame.copyTo(framecopy);

//        vector<int> ids;
//        vector<vector<Point2f> > corners;
//        vector<vector<Point2f> > outArry;
//        //检测markers（待检测marker的图像，字典对象，角点列表，marker的id列表）
//        cv::aruco::detectMarkers(framecopy, dictionary, corners, ids);
//        if (ids.size() > 0)//检测到marker
//        {
//            cout<<"numberr of ids:  "<<ids.size()<<endl;
//            for(int i=0;i<ids.size();i++)
//            {
//                //输出每一个id
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
//            //绘制检测出来的markers
//            cv::aruco::drawDetectedMarkers(frame, corners, ids);
//            vector< Vec3d > rvecs, tvecs;

//            //利用SplvePnP函数求解相机姿态
//            cv::aruco::estimatePoseSingleMarkers(corners,0.092, intrinsic_matrix, arucodistCoeffs, rvecs, tvecs); // draw axis for eac marker
//            //利用solvePnPRansac函数求解相机姿态
//            PnPRansac = solvePnPRansac(projectedPoints, corners, intrinsic_matrix, arucodistCoeffs, rvecs, tvecs, false, 100, 8, 0.99, outArry, SOLVEPNP_ITERATIVE);


//            //X:red Y: green Z:blue 原点在中心点上，当靶标是正的时候，ｘ轴朝右水平，ｙ朝上，ｚ垂直纸面向外，摄像机坐标系的是（在摄像机后方看，ｘ轴是朝右边的，ｙ轴是朝下的，ｚ轴是朝前的
//            for(int i=0; i<ids.size(); i++)
//            {
//                cout<<"R :"<<rvecs[i]<<endl;
//                //T:二维码中心坐标在摄像机坐标系下的坐标
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





















//角点识别

//#include "stdafx.h"
#include "include/opencv2/opencv.hpp"
#include "include/opencv2/highgui/highgui.hpp"
#include "include/opencv2/imgproc/imgproc.hpp"
#include<cmath>
// 单位像素宽/高(cm/pixel)
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

    //const double f = 5.17;//4.8/1920*1071.83;  // 焦距
    const double f = 8.17;//4.8/1920*1071.83;  // 焦距
    const double w = 23;   // 被测物体宽度
    const double h = 19.5;   // 被测物体高度

    int maxcorners = 200;
    double qualityLevel = 0.2;  //角点检测可接受的最小特征值
    double minDistance = 100;	//角点之间最小距离
    int blockSize = 3;//计算导数自相关矩阵时指定的领域范围
    double  k = 0.04; //权重系数


    //从摄像头读入视频
    VideoCapture capture(0);//打开摄像头
    if (!capture.isOpened())//没有打开摄像头的话，就返回。
        return -1;
    //Mat edges; //定义一个Mat变量，用于存储每一帧的图像
    //循环显示每一帧
    while (1)
    {
        Mat srcImage; //定义一个Mat变量，用于存储每一帧的图像
        capture >> srcImage;  //读取当前帧
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
            //用于保存检测结果
            out.open("F:/qt/canny/canny/test_result0.mp4", CV_FOURCC('m', 'p', '4', 'v'), 25.0, cv::Size(1280, 720), true);

            Mat srcgray, dstImage, normImage,scaledImage;
            cvtColor(srcImage, srcgray, CV_BGR2GRAY);

            Mat srcbinary;
            threshold(srcgray, srcbinary,0,255, THRESH_OTSU | THRESH_BINARY);

            Mat kernel = getStructuringElement(MORPH_RECT, Size(15, 15), Point(-1, -1));
            morphologyEx(srcbinary, srcbinary, MORPH_OPEN, kernel, Point(-1, -1));

            //2、Shi-Tomasi算法：确定图像强角点
            vector<Point2f> corners;//提供初始角点的坐标位置和精确的坐标的位置
            //存放坐标 x/y
            vector<float>c_x;
            vector<float>c_y;


            //角点提取
            goodFeaturesToTrack(srcgray, corners, maxcorners, qualityLevel, minDistance, Mat(), blockSize, false, k);
            //Mat():表示感兴趣区域；false:表示不用Harris角点检测
            //输出角点信息
            cout << "角点信息为：" << corners.size() << endl;
            //绘制角点
            RNG rng(12345);
            for (unsigned j = 0; j < corners.size(); j++)
            {
                circle(srcImage, corners[j], 5, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);
                cout << "角点坐标：" << corners[j] << endl;
                //cout << "角点坐标x：" << corners[i].x << endl;
                //cout << "角点坐标y：" << corners[i].y << endl;
                //cout<<corners.begin()<<endl;;

                c_x.push_back(corners[j].x);
                c_y.push_back(corners[j].y);
                //imshow("111",srcImage);
            }
            //获取x坐标最大值与最小值
            float c_x_max = findMax(c_x);
            float c_x_min = findMin(c_x);
            cout<<"c_x_max = "<<c_x_max<<endl;
            cout<<"c_x_min = "<<c_x_min<<endl;
            //获取y坐标最大值与最小值
            float c_y_max = findMax(c_y);
            float c_y_min = findMin(c_y);
            cout<<"c_y_max = "<<c_y_max<<endl;
            cout<<"c_y_min = "<<c_y_min<<endl;

            //截取感兴趣区域
            cv::Rect m_select = cv::Rect(c_x_min,c_y_min,(c_x_max-c_x_min),(c_y_max-c_y_min));
            //Mat ROI = srcImage(m_select);
            //imshow("111",ROI);

            //绘制矩形
            cv::rectangle(srcImage, m_select, cv::Scalar(0, 0, 255), 2);
            //cv::imshow("111",srcImage);

            // 计算成像宽/高/偏置距离
            double widths = m_select.width * UNIT_PIXEL_W;
            double heights = m_select.height * UNIT_PIXEL_H;
            double px = (m_select.x + (m_select.width / 2) - 640) * UNIT_PIXEL_H;
            cout<<width<<endl;
            cout<<height<<endl;
            cout<<px<<endl;
            // 分别以宽/高为标准计算距离
            double distanceW = w * f / widths;
            double distanceH = h * f / heights;
            double distancepx = px * w / widths;
            double angel = atan(distancepx/distanceW) / (3.1415926/180);
            cout<<"angel"<<angel<<endl;
            //输出打印距离值与偏置值
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
        waitKey(10); //延时10ms
    }

//    //保存检测结果
//    out << srcImage;

    if (cv::waitKey(30) == 'q')
    {
        //break;
    }

    capture.release();
    //out.release();
    destroyAllWindows();//关闭所有窗口
    //waitKey(0);
    return(0);
}





////角点识别--备份

////#include "stdafx.h"
//#include "include/opencv2/opencv.hpp"
//#include "include/opencv2/highgui/highgui.hpp"
//#include "include/opencv2/imgproc/imgproc.hpp"
//#include<cmath>
//// 单位像素宽/高(cm/pixel)
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

//    //const double f = 5.17;//4.8/1920*1071.83;  // 焦距
//    const double f = 8.17;//4.8/1920*1071.83;  // 焦距
//    const double w = 23;   // 被测物体宽度
//    const double h = 19.5;   // 被测物体高度

//    int maxcorners = 200;
//    double qualityLevel = 0.2;  //角点检测可接受的最小特征值
//    double minDistance = 100;	//角点之间最小距离
//    int blockSize = 3;//计算导数自相关矩阵时指定的领域范围
//    double  k = 0.04; //权重系数

//    /**************读入视频进行HOG检测******************/

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

//    //用于保存检测结果
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

//        //2、Shi-Tomasi算法：确定图像强角点
//        vector<Point2f> corners;//提供初始角点的坐标位置和精确的坐标的位置
//        //存放坐标 x/y
//        vector<float>c_x;
//        vector<float>c_y;


//        //角点提取
//        goodFeaturesToTrack(srcgray, corners, maxcorners, qualityLevel, minDistance, Mat(), blockSize, false, k);
//        //Mat():表示感兴趣区域；false:表示不用Harris角点检测
//        //输出角点信息
//        cout << "角点信息为：" << corners.size() << endl;
//        //绘制角点
//        RNG rng(12345);
//        for (unsigned j = 0; j < corners.size(); j++)
//        {
//            circle(srcImage, corners[j], 5, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);
//            cout << "角点坐标：" << corners[j] << endl;
//            //cout << "角点坐标x：" << corners[i].x << endl;
//            //cout << "角点坐标y：" << corners[i].y << endl;
//            //cout<<corners.begin()<<endl;;

//            c_x.push_back(corners[j].x);
//            c_y.push_back(corners[j].y);
//            //imshow("111",srcImage);
//        }
//        //获取x坐标最大值与最小值
//        float c_x_max = findMax(c_x);
//        float c_x_min = findMin(c_x);
//        cout<<"c_x_max = "<<c_x_max<<endl;
//        cout<<"c_x_min = "<<c_x_min<<endl;
//        //获取y坐标最大值与最小值
//        float c_y_max = findMax(c_y);
//        float c_y_min = findMin(c_y);
//        cout<<"c_y_max = "<<c_y_max<<endl;
//        cout<<"c_y_min = "<<c_y_min<<endl;

//        //截取感兴趣区域
//        cv::Rect m_select = cv::Rect(c_x_min,c_y_min,(c_x_max-c_x_min),(c_y_max-c_y_min));
//        //Mat ROI = srcImage(m_select);
//        //imshow("111",ROI);

//        //绘制矩形
//        cv::rectangle(srcImage, m_select, cv::Scalar(0, 0, 255), 2);
//        //cv::imshow("111",srcImage);

//        // 计算成像宽/高/偏置距离
//        double width = m_select.width * UNIT_PIXEL_W;
//        double height = m_select.height * UNIT_PIXEL_H;
//        double px = (m_select.x + (m_select.width / 2) - 640) * UNIT_PIXEL_H;
//        cout<<width<<endl;
//        cout<<height<<endl;
//        cout<<px<<endl;
//        // 分别以宽/高为标准计算距离
//        double distanceW = w * f / width;
//        double distanceH = h * f / height;
//        double distancepx = px * w / width;
//        double angel = atan(distancepx/distanceW) / (3.1415926/180);
//        cout<<"angel"<<angel<<endl;
//        //输出打印距离值与偏置值
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

//        //保存检测结果
//        out << srcImage;
//        if (cv::waitKey(30) == 'q')
//        {
//            break;
//        }

//    }
////    cv::imshow("detect result", srcImage);
////    //保存检测结果
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






////角点识别
////代码里面有三种程序

////#include "stdafx.h"
//#include "include/opencv2/opencv.hpp"
//#include "include/opencv2/highgui/highgui.hpp"
//#include "include/opencv2/imgproc/imgproc.hpp"

//// 单位像素宽/高(cm/pixel)
//#define UNIT_PIXEL_W 0.00762
//#define UNIT_PIXEL_H 0.00762


//using namespace std;
//using namespace cv;


//int main(int argv, char** argc)
//{

//    const double f = 5.17;//4.8/1920*1071.83;  // 焦距
//    const double w = 28.7;   // 被测物体宽度
//    const double h = 14.5;   // 被测物体高度

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
//    //1、Harris角点检测
//    cornerHarris(srcgray, dstImage, 3, 3, 0.01, BORDER_DEFAULT);
//    //归一化与转换
//    normalize(dstImage, normImage, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
//    convertScaleAbs(normImage, scaledImage);
//    Mat binaryImage;
//    threshold(scaledImage, binaryImage, 0, 255, THRESH_OTSU | THRESH_BINARY);
//*/


//    //2、Shi-Tomasi算法：确定图像强角点
//    vector<Point2f> corners;//提供初始角点的坐标位置和精确的坐标的位置
//    int maxcorners = 200;
//    double qualityLevel = 0.2;  //角点检测可接受的最小特征值
//    double minDistance = 30;	//角点之间最小距离
//    int blockSize = 3;//计算导数自相关矩阵时指定的领域范围
//    double  k = 0.04; //权重系数

//    goodFeaturesToTrack(srcgray, corners, maxcorners, qualityLevel, minDistance, Mat(), blockSize, false, k);
//    //Mat():表示感兴趣区域；false:表示不用Harris角点检测

//    //输出角点信息
//    cout << "角点信息为：" << corners.size() << endl;

//    //绘制角点
//    RNG rng(12345);
//    for (unsigned i = 0; i < corners.size(); i++)
//    {
//        circle(srcImage, corners[i], 5, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);
//        cout << "角点坐标：" << corners[i] << endl;
//        imshow("111",srcImage);
//    }


////    //3、寻找亚像素角点
////    Size winSize = Size(5, 5);  //搜素窗口的一半尺寸
////    Size zeroZone = Size(-1, -1);//表示死区的一半尺寸
////    //求角点的迭代过程的终止条件，即角点位置的确定
////    TermCriteria criteria = TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40,0.001);
////    //TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 40, 0.001);

////    cornerSubPix(srcgray, corners, winSize, zeroZone, criteria);


////    //输出角点信息
////    cout << "角点信息为：" << corners.size() << endl;

////    //绘制角点
////    for (unsigned i = 0; i < corners.size(); i++)
////    {
////        circle(srcImage, corners[i], 2, Scalar(255,0,0), -1, 8, 0);
////        cout << "角点坐标：" << corners[i] << endl;
////        cout << "角点坐标x：" << corners[i].x << endl;
////        cout << "角点坐标y：" << corners[i].y << endl;
////        imshow("111",srcImage);

////    }


//    waitKey(0);
//    return(0);

//}










////单目测距（摄像机采集的视频流 c++）
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



////轮廓按照面积大小升序排序
//bool ascendSort(vector<Point> a, vector<Point> b)
//{
//    return a.size() < b.size();
//}
////轮廓按照面积大小降序排序
//bool descendSort(vector<Point> a, vector<Point> b) {
//    return a.size() > b.size();
//}
//static inline bool ContoursSortFun(vector<cv::Point> contour1, vector<cv::Point> contour2)
//{
//    return (cv::contourArea(contour1) > cv::contourArea(contour2));
//}
//int main()
//{
//    //从摄像头读入视频
//    VideoCapture capture(0);//打开摄像头
//    if (!capture.isOpened())//没有打开摄像头的话，就返回。
//        return -1;
//    Mat edges; //定义一个Mat变量，用于存储每一帧的图像
//               //循环显示每一帧
//    while (1)
//    {
//        Mat frame; //定义一个Mat变量，用于存储每一帧的图像
//        capture >> frame;  //读取当前帧
//        imshow("Video0", frame);
//        if (frame.empty())
//        {
//            break;
//        }
//        else
//        {
//            //waitKey(2000);可以选择进行处理帧数的时间
//            cvtColor(frame, edges, CV_BGR2GRAY);//彩色转换成灰度
//            GaussianBlur(edges, edges, Size(3, 3), 0, 0);//模糊化
//                                                         //Canny(edges, edges, 35, 125, 3);//边缘化
//            threshold(edges, edges, 220, 255, CV_THRESH_BINARY);
//            imshow("Video1", edges);
//            Mat mask = Mat::zeros(edges.size(), CV_8UC1);
//            vector<vector<Point>>contours;
//            vector<Vec4i>hierarchy;
//            findContours(edges, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);//画出轮廓
//            vector<RotatedRect> rectangle(contours.size()); //最小外接矩形    ***最小外接矩形和最小正外接矩形还是不一样的***
//            Point2f rect[4];
//            float width = 0;//外接矩形的宽和高
//            float height = 0;

//            for (int i = 0; i < contours.size(); i++)
//            {
//                rectangle[i] = minAreaRect(Mat(contours[i]));
//                rectangle[i].points(rect);          //最小外接矩形的4个端点
//                width = rectangle[i].size.width;
//                height = rectangle[i].size.height;
//                if (height >= width)
//                {
//                    float x = 0;
//                    x = height;
//                    height = width;
//                    width = x;
//                }
//                cout << "宽" << width << " " << "高" << height<< endl;
//                for (int j = 0; j < 4; j++)
//                {
//                    cout << "0" << rect[j] << " " << "1" << rect[(j + 1) % 4 ]<< endl;
//                    line(frame, rect[j], rect[(j + 1) % 4], Scalar(0, 0, 255), 1, 8);//绘制最小外接矩形的每条边
//                }
//            }
//            float D = (210 * 509.57) / width;
//            char tam[100];
//            sprintf(tam, "D=:%lf", D);
//            putText(frame, tam, Point(100, 100), FONT_HERSHEY_SIMPLEX, 1, cvScalar(255, 0, 255), 1, 8);
//            imshow("Video2", mask); //显示当前帧
//            imshow("Video3", frame);
//        }
//        waitKey(10); //延时30ms
//    }
//    capture.release();//释放资源
//    destroyAllWindows();//关闭所有窗口
//    return 0;
//}








////单目测距
//#include <iostream>
//#include <include/opencv2/opencv.hpp>
//#include <include/opencv2/highgui.hpp>
//#include <stdio.h>
//#include <vector>

//// 单位像素宽/高(cm/pixel)
//#define UNIT_PIXEL_W 0.00762
//#define UNIT_PIXEL_H 0.00762

//using namespace std;

//int main(void)
//{
////    const double f = 5.17;//4.8/1920*1071.83;  // 焦距
////    const double w = 31;   // 被测物体宽度
////    const double h = 17.5;   // 被测物体高度

//    const double f = 5.17;//4.8/1920*1071.83;  // 焦距
//    const double w = 20.5;   // 被测物体宽度
//    const double h = 17.5;   // 被测物体高度


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
//        // otsu 可以换用动态阈值
//        cv::threshold(grayImage, grayImage, NULL, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

//        vector<vector<cv::Point>> contours;
//        vector<cv::Point> maxAreaContour;

//        cv::findContours(grayImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//        //cv::drawContours(frame, contours, -1, cv::Scalar(0, 0, 255), 2, 8);

//        // 提取面积最大轮廓
//        double maxArea = 0;
//        for (size_t i = 0; i < contours.size(); i++) {
//            double area = fabs(cv::contourArea(contours[i]));
//            if (area > maxArea) {
//                maxArea = area;
//                maxAreaContour = contours[i];
//            }
//        }
//        // 轮廓外包正矩形
//        cv::Rect rect = cv::boundingRect(maxAreaContour);
//        cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), cv::Scalar(255, 0, 0), 2, 8);

//        // 计算成像宽/高
//        double width = rect.width * UNIT_PIXEL_W;
//        double height = rect.height * UNIT_PIXEL_H;
//        cout<<width<<endl;
//        cout<<height<<endl;    // 分别以宽/高为标准计算距离
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


////单目测距
//#include <iostream>
//#include <include/opencv2/opencv.hpp>
//#include <include/opencv2/highgui.hpp>
//#include <stdio.h>
//#include <vector>

//// 单位像素宽/高(cm/pixel)
//#define UNIT_PIXEL_W 0.00762
//#define UNIT_PIXEL_H 0.00762

//using namespace std;

//int main(void)
//{
////    const double f = 5.17;//4.8/1920*1071.83;  // 焦距
////    const double w = 31;   // 被测物体宽度
////    const double h = 17.5;   // 被测物体高度

//    const double f = 5.17;//4.8/1920*1071.83;  // 焦距
//    const double w = 20.5;   // 被测物体宽度
//    const double h = 17.5;   // 被测物体高度


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
//        // otsu 可以换用动态阈值
//        cv::threshold(grayImage, grayImage, NULL, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

//        vector<vector<cv::Point>> contours;
//        vector<cv::Point> maxAreaContour;

//        cv::findContours(grayImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//        //cv::drawContours(frame, contours, -1, cv::Scalar(0, 0, 255), 2, 8);

//        // 提取面积最大轮廓
//        double maxArea = 0;
//        for (size_t i = 0; i < contours.size(); i++) {
//            double area = fabs(cv::contourArea(contours[i]));
//            if (area > maxArea) {
//                maxArea = area;
//                maxAreaContour = contours[i];
//            }
//        }
//        // 轮廓外包正矩形
//        cv::Rect rect = cv::boundingRect(maxAreaContour);
//        cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), cv::Scalar(255, 0, 0), 2, 8);

//        // 计算成像宽/高
//        double width = rect.width * UNIT_PIXEL_W;
//        double height = rect.height * UNIT_PIXEL_H;
//        cout<<width<<endl;
//        cout<<height<<endl;    // 分别以宽/高为标准计算距离
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




////单目相机标定
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

//    ifstream fin("F:/qt/canny/canny/chassread.txt");             /* 标定所用图像文件的路径 */
//    ofstream fout("F:/qt/canny/canny/caliberation_result_right.txt");  /* 保存标定结果的文件 */

//    // 读取每一幅图像，从中提取出角点，然后对角点进行亚像素精确化
//    int image_count = 0;  /* 图像数量 */
//    Size image_size;      /* 图像的尺寸 */
//    Size board_size = Size(11,8);             /* 标定板上每行、列的角点数 */
//    vector<Point2f> image_points_buf;         /* 缓存每幅图像上检测到的角点 */
//    vector<vector<Point2f>> image_points_seq; /* 保存检测到的所有角点 */
//    string filename;      // 图片名
//    vector<string> filenames;

//    while (getline(fin, filename))
//    {
//        ++image_count;
//        Mat imageInput = imread(filename);
//        filenames.push_back(filename);

//        // 读入第一张图片时获取图片大小
//        if (image_count == 1)
//        {
//            image_size.width = imageInput.cols;
//            image_size.height = imageInput.rows;
//        }

//        /* 提取角点 */
//        if (0 == findChessboardCorners(imageInput, board_size, image_points_buf))
//        {
//            //cout << "can not find chessboard corners!\n";  // 找不到角点
//            cout << "**" << filename << "** can not find chessboard corners!\n";
//            exit(1);
//        }
//        else
//        {
//            Mat view_gray;
//            cvtColor(imageInput, view_gray, CV_RGB2GRAY);  // 转灰度图

//            /* 亚像素精确化 */
//            // image_points_buf 初始的角点坐标向量，同时作为亚像素坐标位置的输出
//            // Size(5,5) 搜索窗口大小
//            // （-1，-1）表示没有死区
//            // TermCriteria 角点的迭代过程的终止条件, 可以为迭代次数和角点精度两者的组合
//            cornerSubPix(view_gray, image_points_buf, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

//            image_points_seq.push_back(image_points_buf);  // 保存亚像素角点

//            /* 在图像上显示角点位置 */
//            drawChessboardCorners(view_gray, board_size, image_points_buf, false); // 用于在图片中标记角点

//            imshow("Camera Calibration", view_gray);       // 显示图片

//            waitKey(500); //暂停0.5S
//        }
//    }
//    int CornerNum = board_size.width * board_size.height;  // 每张图片上总的角点数

//    //-------------以下是摄像机标定------------------

//    /*棋盘三维信息*/
//    Size square_size = Size(16, 16);         /* 实际测量得到的标定板上每个棋盘格的大小 */
//    vector<vector<Point3f>> object_points;   /* 保存标定板上角点的三维坐标 */

//    /*内外参数*/
//    Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));  /* 摄像机内参数矩阵 */
//    vector<int> point_counts;   // 每幅图像中角点的数量
//    Mat distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0));       /* 摄像机的5个畸变系数：k1,k2,p1,p2,k3 */
//    vector<Mat> tvecsMat;      /* 每幅图像的旋转向量 */
//    vector<Mat> rvecsMat;      /* 每幅图像的平移向量 */

//    /* 初始化标定板上角点的三维坐标 */
//    int i, j, t;
//    for (t = 0; t<image_count; t++)
//    {
//        vector<Point3f> tempPointSet;
//        for (i = 0; i<board_size.height; i++)
//        {
//            for (j = 0; j<board_size.width; j++)
//            {
//                Point3f realPoint;

//                /* 假设标定板放在世界坐标系中z=0的平面上 */
//                realPoint.x = i * square_size.width;
//                realPoint.y = j * square_size.height;
//                realPoint.z = 0;
//                tempPointSet.push_back(realPoint);
//            }
//        }
//        object_points.push_back(tempPointSet);
//    }

//    /* 初始化每幅图像中的角点数量，假定每幅图像中都可以看到完整的标定板 */
//    for (i = 0; i<image_count; i++)
//    {
//        point_counts.push_back(board_size.width * board_size.height);
//    }

//    /* 开始标定 */
//    // object_points 世界坐标系中的角点的三维坐标
//    // image_points_seq 每一个内角点对应的图像坐标点
//    // image_size 图像的像素尺寸大小
//    // cameraMatrix 输出，内参矩阵
//    // distCoeffs 输出，畸变系数
//    // rvecsMat 输出，旋转向量
//    // tvecsMat 输出，位移向量
//    // 0 标定时所采用的算法
//    calibrateCamera(object_points, image_points_seq, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, 0);

//    //------------------------标定完成------------------------------------

//    // -------------------对标定结果进行评价------------------------------

//    double total_err = 0.0;         /* 所有图像的平均误差的总和 */
//    double err = 0.0;               /* 每幅图像的平均误差 */
//    vector<Point2f> image_points2;  /* 保存重新计算得到的投影点 */
//    fout << "每幅图像的标定误差：\n";

//    for (i = 0; i<image_count; i++)
//    {
//        vector<Point3f> tempPointSet = object_points[i];

//        /* 通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点 */
//        projectPoints(tempPointSet, rvecsMat[i], tvecsMat[i], cameraMatrix, distCoeffs, image_points2);

//        /* 计算新的投影点和旧的投影点之间的误差*/
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
//        fout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
//    }
//    fout << "总体平均误差：" << total_err / image_count << "像素" << endl << endl;

//    //-------------------------评价完成---------------------------------------------

//    //-----------------------保存定标结果-------------------------------------------
//    Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0));  /* 保存每幅图像的旋转矩阵 */
//    fout << "相机内参数矩阵：" << endl;
//    fout << cameraMatrix << endl << endl;
//    fout << "畸变系数：\n";
//    fout << distCoeffs << endl << endl << endl;
//    for (int i = 0; i<image_count; i++)
//    {
//        fout << "第" << i + 1 << "幅图像的旋转向量：" << endl;
//        fout << tvecsMat[i] << endl;

//        /* 将旋转向量转换为相对应的旋转矩阵 */
//        Rodrigues(tvecsMat[i], rotation_matrix);
//        fout << "第" << i + 1 << "幅图像的旋转矩阵：" << endl;
//        fout << rotation_matrix << endl;
//        fout << "第" << i + 1 << "幅图像的平移向量：" << endl;
//        fout << rvecsMat[i] << endl << endl;
//    }
//    fout << endl;

//    //--------------------标定结果保存结束-------------------------------

//    //----------------------显示定标结果--------------------------------

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
//        /// 读取一副图片，不改变图片本身的颜色类型（该读取方式为DOS运行模式）
//        Mat src = imread("F:\\lane_line_detection\\left_img\\1.jpg");
//        Mat distortion = src.clone();
//        Mat camera_matrix = Mat(3, 3, CV_32FC1);
//        Mat distortion_coefficients;


//        //导入相机内参和畸变系数矩阵
//        FileStorage file_storage("F:\\lane_line_detection\\left_img\\Intrinsic.xml", FileStorage::READ);
//        file_storage["CameraMatrix"] >> camera_matrix;
//        file_storage["Dist"] >> distortion_coefficients;
//        file_storage.release();

//        //矫正
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
// * Canny边缘检测器
// */

//int main() {
//    //打开摄像头
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
//    //进行灰度变换
//    Mat dst;
//    cvtColor(src,dst,COLOR_BGR2GRAY);

//    //进行高斯滤波
//    GaussianBlur(dst, dst, Size(7,7), 2, 2);
//    vector<Vec3f>circles;

//    //执行canny边缘检测
//    Mat edges;
//    Canny(dst, edges, 100, 300);

//    //执行膨胀和腐蚀后处理
//    Mat dresult, eresult;
//    //定义结构元素3*3大小的矩形
//    Mat se = getStructuringElement(MORPH_RECT, Size(3, 3));
//    //膨胀
//    dilate(edges, dresult, se);
//    //腐蚀
//    erode(dresult, eresult, se);

//    imshow("dresult",dresult);
//    imshow("eresult",eresult);

////    waitKey(3000);


//    //检测圆并求出圆心与半径
//    vector<vector<Point>>contours;
//    vector<Vec4i>hierarchy;
//    findContours(eresult, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//查找出所有的圆边界
//    int index = 0;
//    for (; index >= 0; index = hierarchy[index][0])
//    {
//        Scalar color(rand() & 255, rand() & 255, rand() & 255);
//        drawContours(eresult, contours, index, color, CV_FILLED, 8, hierarchy);

//    }

//    namedWindow("detected circles", CV_NORMAL);
//    //imshow("edges",edges);
//    //imshow("detected circles", eresult);
//    //标准圆在图片上一般是椭圆，所以采用OpenCV中拟合椭圆的方法求解中心
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

////最新小二乘法拟合直线
//#include <iostream>

//using namespace std;

//void LinearFit(double abr[],double x[],double y[],int n) {//线性拟合ax+b
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
//       dy2sum1 += ((abr[0] * x[i] + abr[1]) - yavg)*((abr[0] * x[i] + abr[1]) - yavg);//r^2的分子
//       dy2sum2 += (y[i] - yavg)*(y[i] - yavg);//r^2的分母
//   }
//   abr[2] = dy2sum1 / dy2sum2;//r^2
//}
//void HalfLogLine(double y[], int n) {//半对数拟合
//   for (int i = 0; i < n; i++)
//   {
//       y[i] = log10(y[i]);

//   }
//}
//void LogtoLine(double x[], double y[], int n) {//对数拟合

//   for (int i = 0; i < n; i++)
//   {
//       y[i] = log(y[i]);
//       x[i] = log(x[i]);

//   }
//}
//int main()
//{
//   int const N = 12;//12;
//   //double x[N] = {0.96,0.94,0.92,0.90,0.88,0.86,0.84,0.82,0.80,0.78,0.76,0.74 };//半对数
//   //double y[N] = {558.0,313.0,174.0,97.0,55.8,31.3,17.4,9.70,5.58,3.13,1.74,1.00 };
//   double x[N] = { 215.0, 230.6, 248.5, 255.6, 277.5, 284.0, 294.8, 296.8, 314.5, 317.5, 332.8, 362.5};//对数
//   double y[N] = { 93.0, 81.0, 71.6, 62.0, 56.3, 51.5, 46.0, 44.0, 37.5, 35.0, 30.5, 19.0};
//   double abr[3];
//   //HalfLogLine(y, N);
//   LogtoLine(x, y, N);
//   LinearFit(abr, x, y, N);
//   abr[1] = exp(abr[1]);
//   cout << showpos;//显示正负号
//   cout <<"相关系数拟合直线:y=" << abr[0] << "x" << abr[1] << endl;
//   cout <<"相关系数:r^2"<< abr[2] << endl;
//   system("pause");
//   return 0;
//}


////椭圆拟合
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


//    //进行灰度变换
//    Mat dst;
//    cvtColor(src,dst,COLOR_BGR2GRAY);

//    //进行高斯滤波
//    GaussianBlur(dst, dst, Size(7,7), 2, 2);
//    vector<Vec3f>circles;

//    //执行canny边缘检测
//    Mat edges;
//    Canny(dst, edges, 100, 300);

//    //执行膨胀和腐蚀后处理
//    Mat dresult, eresult;
//    //定义结构元素3*3大小的矩形
//    Mat se = getStructuringElement(MORPH_RECT, Size(3, 3));
////    //膨胀
////    dilate(edges, dresult, se);
//    //腐蚀
//    erode(edges, eresult, se);
//    //膨胀
//    dilate(edges, dresult, se);

//    imshow("dresult",dresult);
//    imshow("eresult",eresult);
//    imshow("edges",edges);

//    //轮廓
//    vector<vector<Point>> contours;


//    //边缘追踪，没有存储边缘的组织结构
//    findContours(dresult, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//    Mat cimage = Mat::zeros(dresult.size(), CV_8UC3);

//    for(size_t i = 0; i < contours.size(); i++)
//    {
//        //拟合的点至少为200
//        size_t count = contours[i].size();
//        if( count < 200 )
//            continue;

//        //椭圆拟合
//        RotatedRect box = fitEllipse(contours[i]);

//        //如果长宽比大于3，则排除，不做拟合
//        if( MAX(box.size.width, box.size.height) > MIN(box.size.width, box.size.height)*3)
//            continue;
//        cout<<"box.size.height = "<<box.size.height<<endl;
//        cout<<"box.size.width = " <<box.size.width<<endl;

//        cout<<"box.center"<<box.center<<endl;
//        //画出追踪出的轮廓
//        drawContours(cimage, contours, (int)i, Scalar::all(255), 1, 8);

//        //画出拟合的椭圆
//        ellipse(cimage, box, Scalar(0,0,255), 1, CV_AA);


//    }
//    imshow("拟合结果", cimage);

//    waitKey();
//    return 0;
//}

////OpenCV单目视觉定位（测量）
//#include  <iostream>
//#include <include/opencv2/opencv.hpp>
////#include <include/opencv2/highgui.hpp>
////#include <include/opencv2/imgproc.hpp>


//using namespace std;
//using namespace cv;

////全局变量
//Mat src, gray, gray_blur, contours_image,dstThreshold;
////本相机分辨率为640*480，定义图像中心为图像原点，也即对应相机的中心位置，光轴位置
////自定义图像原点坐标
//float oriX = 640.0f;
//float oriY = 360.0f;

//float targetImage_X, targetImage_Y;  //目标物距图像原点的X,Y方向的像素距离
//float mm_per_pixel;                  //像素尺寸
//float targetLength=100.0f;           //目标物实际长度
//float targetActualX, targetActualY;  //二维空间实际数据

////视觉定位函数
//void Location();

//int main()
//{
//    cout <<" 。。。。。。。。。。。。。。。。。按'q'退出程序。。。。。。。。。。。。。。。" << endl;
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


////定位函数
//void Location(){

//    //均值滤波
//    blur(gray, gray_blur, Size(3, 3));

//    //边缘检测提取边缘信息
//    Canny(gray_blur, dstThreshold, 150, 450);
//    imshow("canny边缘检测", dstThreshold);

//    //对边缘图像提取轮廓信息
//    vector<vector<Point> >contours;
//    findContours(dstThreshold, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

//    //画出轮廓
//    drawContours(contours_image, contours, -1, Scalar(0, 0, 255));
//    imshow("contours", contours_image);

//    //画出定义的原点
//    circle(src, Point2f(oriX, oriY), 2, Scalar(0, 0, 255), 3);

//    //定义分别逼近六边形和五边形的轮廓
//    vector< vector<Point> > Contour1_Ok, Contour2_Ok;

//    //轮廓分析
//    vector<Point> approx;
//    for (int i = 0; i < contours.size(); i++){
//        approxPolyDP(Mat(contours[i]), approx, cv::arcLength(cv::Mat(contours[i]), true)*0.04, true);

//        //去除 小轮廓，只提取凸轮廓
//        if (std::fabs(cv::contourArea(contours[i])) < 600 || !cv::isContourConvex(approx))
//            continue;

//        //保存逼近六边形的轮廓 到 Contour1_Ok
//         if (approx.size() == 4){
//            Contour1_Ok.push_back(contours[i]);
//        }
//        //保存逼近五边形的轮廓 到 Contour2_Ok
//         else if (approx.size() == 4){
//             Contour2_Ok.push_back(contours[i]);
//        }

//    }

//    //对所有符合要求的六边形，五边形轮廓进行分析
//    //识别出自定义的物体的关键是：
//    //1.六边形和五边形轮廓的最小外接矩形的中心基本在同一点
//    //2.六边形轮廓的最小外接矩形的任一边长大于五边形轮廓的最小外接矩形的任一边长
//    for (int i = 0; i < Contour1_Ok.size(); i++){
//        for (int j = 0; j < Contour2_Ok.size(); j++){
//            RotatedRect minRect1 = minAreaRect(Mat(Contour1_Ok[i]));  //六边形轮廓的最小外接矩形
//            RotatedRect minRect2 = minAreaRect(Mat(Contour2_Ok[j]));  //五边形轮廓的最小外界矩形
//            //找出符合要求的轮廓的最小外接矩形
//            if ( fabs(minRect1.center.x - minRect2.center.x) < 30 && fabs(minRect1.center.y - minRect2.center.y)<30 && minRect1.size.width > minRect2.size.width){
//                Point2f vtx[4];
//                minRect1.points(vtx);

//                //画出找到的物体的最小外接矩形
//                for (int j = 0; j < 4; j++)
//                    line(src, vtx[j], vtx[(j + 1) % 4], Scalar(0, 0, 255), 2, LINE_AA);

//                //画出目标物中心到图像原点的直线
//                line(src, minRect1.center, Point2f(oriX, oriY), Scalar(0, 255, 0), 1, LINE_AA);

//                //目标物距图像原点的X,Y方向的像素距离
//                targetImage_X = minRect1.center.x - oriX;
//                targetImage_Y = oriY - minRect1.center.y;

//                line(src, minRect1.center, Point2f(minRect1.center.x, oriY), Scalar(255, 0, 0), 1, LINE_AA);
//                line(src, Point2f(oriX, oriY), Point2f(minRect1.center.x, oriY), Scalar(255, 0, 0), 1, LINE_AA);

//                Point2f pointX((oriX + minRect1.center.x) / 2, oriY);
//                Point2f pointY(minRect1.center.x, (oriY + minRect1.center.y) / 2);

//                //找出最大边
//                float a = minRect1.size.height, b = minRect1.size.width;
//                if (a < b) a = b;

//                mm_per_pixel = targetLength / a;               //计算像素尺寸 = 目标物的实际长度（cm）/ 目标物在图像上的像素长度（pixels）
//                targetActualX = mm_per_pixel *targetImage_X;   //计算实际距离X（cm）
//                targetActualY = mm_per_pixel *targetImage_Y;   //计算实际距离Y（cm）

//                //打印信息在图片上
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


////单目测距（摄像机采集的视频流 c++）
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



////轮廓按照面积大小升序排序
//bool ascendSort(vector<Point> a, vector<Point> b)
//{
//    return a.size() < b.size();
//}
////轮廓按照面积大小降序排序
//bool descendSort(vector<Point> a, vector<Point> b) {
//    return a.size() > b.size();
//}
//static inline bool ContoursSortFun(vector<cv::Point> contour1, vector<cv::Point> contour2)
//{
//    return (cv::contourArea(contour1) > cv::contourArea(contour2));
//}
//int main()
//{
//    //从摄像头读入视频
//    VideoCapture capture(0);//打开摄像头
//    if (!capture.isOpened())//没有打开摄像头的话，就返回。
//        return -1;
//    Mat edges; //定义一个Mat变量，用于存储每一帧的图像
//               //循环显示每一帧
//    while (1)
//    {
//        Mat frame; //定义一个Mat变量，用于存储每一帧的图像
//        capture >> frame;  //读取当前帧
//        imshow("Video0", frame);
//        if (frame.empty())
//        {
//            break;
//        }
//        else
//        {
//            //waitKey(2000);可以选择进行处理帧数的时间
//            cvtColor(frame, edges, CV_BGR2GRAY);//彩色转换成灰度
//            GaussianBlur(edges, edges, Size(3, 3), 0, 0);//模糊化
//                                                         //Canny(edges, edges, 35, 125, 3);//边缘化
//            threshold(edges, edges, 220, 255, CV_THRESH_BINARY);
//            imshow("Video1", edges);
//            Mat mask = Mat::zeros(edges.size(), CV_8UC1);
//            vector<vector<Point>>contours;
//            vector<Vec4i>hierarchy;
//            findContours(edges, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);//画出轮廓
//            vector<RotatedRect> rectangle(contours.size()); //最小外接矩形    ***最小外接矩形和最小正外接矩形还是不一样的***
//            Point2f rect[4];
//            float width = 0;//外接矩形的宽和高
//            float height = 0;

//            for (int i = 0; i < contours.size(); i++)
//            {
//                rectangle[i] = minAreaRect(Mat(contours[i]));
//                rectangle[i].points(rect);          //最小外接矩形的4个端点
//                width = rectangle[i].size.width;
//                height = rectangle[i].size.height;
//                if (height >= width)
//                {
//                    float x = 0;
//                    x = height;
//                    height = width;
//                    width = x;
//                }
//                cout << "宽" << width << " " << "高" << height<< endl;
//                for (int j = 0; j < 4; j++)
//                {
//                    cout << "0" << rect[j] << " " << "1" << rect[(j + 1) % 4 ]<< endl;
//                    line(frame, rect[j], rect[(j + 1) % 4], Scalar(0, 0, 255), 1, 8);//绘制最小外接矩形的每条边
//                }
//            }
//            float D = (210 * 509.57) / width;
//            char tam[100];
//            sprintf(tam, "D=:%lf", D);
//            putText(frame, tam, Point(100, 100), FONT_HERSHEY_SIMPLEX, 1, cvScalar(255, 0, 255), 1, 8);
//            imshow("Video2", mask); //显示当前帧
//            imshow("Video3", frame);
//        }
//        waitKey(10); //延时30ms
//    }
//    capture.release();//释放资源
//    destroyAllWindows();//关闭所有窗口
//    return 0;
//}

















////单目测距+椭圆检测
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

////    capture.set(CV_CAP_PROP_FRAME_WIDTH, 1080);//宽度
////    capture.set(CV_CAP_PROP_FRAME_HEIGHT, 960);//高度
//    capture.set(CV_CAP_PROP_FPS, 0.5);//帧率 帧/秒
////    capture.set(CV_CAP_PROP_BRIGHTNESS, 1);//亮度
////    capture.set(CV_CAP_PROP_CONTRAST,40);//对比度 40
////    capture.set(CV_CAP_PROP_SATURATION, 50);//饱和度 50
////    capture.set(CV_CAP_PROP_HUE, 50);//色调 50
////    capture.set(CV_CAP_PROP_EXPOSURE, 50);//曝光 50 获取摄像头参数

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



//        //进行灰度变换
//        cv::Mat dst;
//        cvtColor(frame,dst,cv::COLOR_BGR2GRAY);
//        //进行高斯滤波
//        cv::GaussianBlur(dst, dst, cv::Size(7,7), 2, 2);
//        vector<cv::Vec3f>circles;

//        //执行canny边缘检测
//        cv::Mat edges;
//        Canny(dst, edges, 100, 300);

////        //执行膨胀和腐蚀后处理
////        cv::Mat dresult, eresult;
////        //定义结构元素3*3大小的矩形
////        cv::Mat se = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

////        //腐蚀
////        erode(dst, eresult, se);
////        //膨胀
////        dilate(dst, dresult, se);

//        //imshow("dresult",dresult);
//        //imshow("eresult",eresult);
//        //imshow("edges",dst);

//        //轮廓
//        vector<vector<cv::Point>> contours;


//        //边缘追踪，没有存储边缘的组织结构
//        findContours(edges, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//        cv::Mat cimage = cv::Mat::zeros(edges.size(), CV_8UC3);

//        for(size_t i = 0; i < contours.size(); i++)
//        {
//            //拟合的点至少为200
//            size_t count = contours[i].size();
//            if( count < 200 )
//                continue;

//            //椭圆拟合
//            cv::RotatedRect box = fitEllipse(contours[i]);

//            //如果长宽比大于3，则排除，不做拟合
//            if( MAX(box.size.width, box.size.height) > MIN(box.size.width, box.size.height)*3 )
//                continue;
//            //if( 400 > box.size.height > 250 )
//                //continue;
//            //if( 250 > box.size.width > 150 )
//                //continue;
//            cout<<"box.size.height = "<<box.size.height<<endl;
//            cout<<"box.size.width = " <<box.size.width<<endl;
//            cout<<"box.center"<<box.center<<endl;
//            //画出追踪出的轮廓
//            drawContours(cimage, contours, (int)i, cv::Scalar::all(255), 1, 8);

//            //画出拟合的椭圆
//            ellipse(cimage, box, cv::Scalar(0,0,255), 1, CV_AA);
//            imshow("拟合结果0", cimage);

//        }
//        imshow("拟合结果1", cimage);

//        cv::imshow("Frame", frame);
//        cv::imshow("Gray", dst);
//        if ((cv::waitKey(10) & 0XFF) == 27) break;
//    }
//    cv::destroyAllWindows();
//    capture.release();

//    return EXIT_SUCCESS;
//}




////单目测距+椭圆检测
//#include <iostream>
//#include <include/opencv2/opencv.hpp>
//#include <include/opencv2/highgui.hpp>
//#include <stdio.h>
//#include <vector>

//// 单位像素宽/高(cm/pixel)
//#define UNIT_PIXEL_W 0.03
//#define UNIT_PIXEL_H 0.03

//using namespace std;

//int main(void)
//{
//    cv::Mat frame;
//    cv::VideoCapture capture(0);

////    const double f = 2.8;  // 焦距
////    const double w = 100;   // 被测物体宽度
////    const double h = 60;   // 被测物体高度

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
//        // otsu 可以换用动态阈值
//        cv::threshold(dst, dst, NULL, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

//        vector<vector<cv::Point>> contours;
//        //vector<cv::Point> maxAreaContour;

//        cv::findContours(dst, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//        //cv::drawContours(frame, contours, -1, cv::Scalar(0, 0, 255), 2, 8);

//        //进行灰度变换
//        //cv::Mat dst;
//        //cvtColor(dst,dst,cv::COLOR_BGR2GRAY);
//        //进行高斯滤波
//        cv::GaussianBlur(dst, dst, cv::Size(7,7), 2, 2);
//        vector<cv::Vec3f>circles;

//        //执行canny边缘检测
//        cv::Mat edges;
//        Canny(dst, edges, 100, 300);

//        //执行膨胀和腐蚀后处理
//        cv::Mat dresult, eresult;
//        //定义结构元素3*3大小的矩形
//        cv::Mat se = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
//        //    //膨胀
//        //    dilate(edges, dresult, se);
//        //腐蚀
//        erode(dst, eresult, se);
//        //膨胀
//        dilate(dst, dresult, se);

//        //imshow("dresult",dresult);
//        //imshow("eresult",eresult);
//        //imshow("edges",dst);

////        //轮廓
////        vector<vector<cv::Point>> contours;


//        //边缘追踪，没有存储边缘的组织结构
//        findContours(dresult, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//        cv::Mat cimage = cv::Mat::zeros(dresult.size(), CV_8UC3);

//        for(size_t i = 0; i < contours.size(); i++)
//        {
//            //拟合的点至少为6
//            size_t count = contours[i].size();
//            if( count < 6 )
//                continue;

//            //椭圆拟合
//            cv::RotatedRect box = fitEllipse(contours[i]);

//            //如果长宽比大于30，则排除，不做拟合
//            if( MAX(box.size.width, box.size.height) > MIN(box.size.width, box.size.height)*30 )
//                continue;
//            cout<<"box.size.height = "<<box.size.height<<endl;
//            cout<<"box.size.width = " <<box.size.width<<endl;
//            //画出追踪出的轮廓
//            drawContours(cimage, contours, (int)i, cv::Scalar::all(255), 1, 8);

//            //画出拟合的椭圆
//            ellipse(cimage, box, cv::Scalar(0,0,255), 1, CV_AA);


//        }
//        imshow("拟合结果", cimage);




//        // 提取面积最大轮廓
//        double maxArea = 0;
//        for (size_t i = 0; i < contours.size(); i++) {
//            double area = fabs(cv::contourArea(contours[i]));
//            if (area > maxArea) {
//                maxArea = area;
//                maxAreaContour = contours[i];
//            }
//        }
//        // 轮廓外包正矩形
//        cv::Rect rect = cv::boundingRect(maxAreaContour);


//        cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), cv::Scalar(255, 0, 0), 2, 8);
//        cv::imshow("frame",frame);
//        // 计算成像宽/高
//        double width = rect.width * UNIT_PIXEL_W;
//        double height = rect.height * UNIT_PIXEL_H;
//        // 分别以宽/高为标准计算距离
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





















////单目测距
//#include <iostream>
//#include <include/opencv2/opencv.hpp>
//#include <include/opencv2/highgui.hpp>
//#include <stdio.h>
//#include <vector>

//// 单位像素宽/高(cm/pixel)
//#define UNIT_PIXEL_W 0.0003
//#define UNIT_PIXEL_H 0.0003

//using namespace std;

//int main(void)
//{
//    cv::Mat frame;
//    cv::VideoCapture capture(0);

//    const double f = 0.268;//4.8/1920*1071.83;  // 焦距
//    const double w = 60;   // 被测物体宽度
//    const double h = 100;   // 被测物体高度

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
//        // otsu 可以换用动态阈值
//        cv::threshold(grayImage, grayImage, NULL, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

//        vector<vector<cv::Point>> contours;
//        vector<cv::Point> maxAreaContour;

//        cv::findContours(grayImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//        //cv::drawContours(frame, contours, -1, cv::Scalar(0, 0, 255), 2, 8);

//        // 提取面积最大轮廓
//        double maxArea = 0;
//        for (size_t i = 0; i < contours.size(); i++) {
//            double area = fabs(cv::contourArea(contours[i]));
//            if (area > maxArea) {
//                maxArea = area;
//                maxAreaContour = contours[i];
//            }
//        }
//        // 轮廓外包正矩形
//        cv::Rect rect = cv::boundingRect(maxAreaContour);
//        cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), cv::Scalar(255, 0, 0), 2, 8);

//        // 计算成像宽/高
//        double width = rect.width * UNIT_PIXEL_W;
//        double height = rect.height * UNIT_PIXEL_H;
//        cout<<width<<endl;
//        cout<<height<<endl;    // 分别以宽/高为标准计算距离
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




////边缘+轮廓拟合

////opencv版本:OpenCV3.0
////VS版本:VS2013
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

//    namedWindow("原图");
//    imshow("原图", image);

//    cvtColor(image, image, CV_BGR2GRAY);//转为灰度图像

//    vector<vector<Point>> contours;
//    vector<Vec4i> hierarchy;
//    // 轮廓检测
//    findContours(image, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

//    // 绘制轮廓
//    Mat result(image.size(), CV_8UC3, Scalar(0));
//    drawContours(result, contours, -1, Scalar(255, 255, 255), 1);

//    Mat result_PolyDP = result.clone();
//    Mat result_boundingRect = result.clone();
//    Mat result_Circle = result.clone();


//    //conPoint存储计算得到的外接多边形
//    vector<vector<Point> > conPoint(contours.size());

//    //boundRect存储计算得到的最小立式矩形
//    vector<Rect> boundRect(contours.size());

//    //center和radius存储计算得到的最小外接圆
//    vector<Point2f>center(contours.size());
//    vector<float>radius(contours.size());

//    for (int i = 0; i < contours.size(); i++)
//    {
//        // 计算外接多边形
//        approxPolyDP(Mat(contours[i]), conPoint[i], 3, true);
//        // 计算最小外接立式矩形
//        boundRect[i] = boundingRect(Mat(conPoint[i]));
//        //计算最小外接圆
//        minEnclosingCircle(conPoint[i], center[i], radius[i]);
//    }

//    for (int i = 0; i< contours.size(); i++)
//    {
//        Scalar color = Scalar(0, 0, 255);
//        //绘制外接多边形
//        drawContours(result_PolyDP, conPoint, i, color, 2, 8, vector<Vec4i>(), 0, Point());
//        // 绘制最小外接立式矩形
//        rectangle(result_boundingRect, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
//        // 绘制最小外接圆
//        circle(result_Circle, center[i], (int)radius[i], color, 2, 8, 0);
//    }

//    namedWindow("轮廓图");
//    imshow("轮廓图", result);
//    namedWindow("PolyDP");
//    imshow("PolyDP", result_PolyDP);

//    namedWindow("boundingRect");
//    imshow("boundingRect", result_boundingRect);

//    namedWindow("Circle");
//    imshow("Circle", result_Circle);

//    waitKey(5000);
//    return 0;
//}



















////最大内接矩形
//#include <iostream>
//#include <include/opencv2/opencv.hpp>
//#include <include/opencv2/highgui.hpp>
//#include<vector>

//using namespace cv;
//using namespace std;

///**

//* @brief expandEdge 扩展边界函数

//* @param img:输入图像，单通道二值图，深度为8

//* @param edge  边界数组，存放4条边界值

//* @param edgeID 当前边界号

//* @return 布尔值 确定当前边界是否可以扩展

//*/

//bool expandEdge(const Mat & img, int edge[], const int edgeID)
//{
//    //[1] --初始化参数
//    int nc = img.cols;
//    int nr = img.rows;
//    switch (edgeID) {
//    case 0:
//        if (edge[0]>nr)
//            return false;
//        for (int i = edge[3]; i <= edge[1]; ++i)
//        {
//            if (img.at<uchar>(edge[0], i) == 255)//遇见255像素表明碰到边缘线
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
//            if (img.at<uchar>(i, edge[1]) == 255)//遇见255像素表明碰到边缘线
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
//            if (img.at<uchar>(edge[2], i) == 255)//遇见255像素表明碰到边缘线
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
//            if (img.at<uchar>(i, edge[3]) == 255)//遇见255像素表明碰到边缘线
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

//* @brief 求取连通区域内接矩

//* @param img:输入图像，单通道二值图，深度为8

//* @param center:最小外接矩的中心

//* @return  最大内接矩形

//* 基于中心扩展算法

//*/

//cv::Rect InSquare(Mat &img, const Point center)
//{
//    // --[1]参数检测
//    if (img.empty() ||img.channels()>1|| img.depth()>8)
//        return Rect();
//    // --[2] 初始化变量
//    int edge[4];
//    edge[0] = center.y + 1;//top
//    edge[1] = center.x + 1;//right
//    edge[2] = center.y - 1;//bottom
//    edge[3] = center.x - 1;//left
//                           //[2]
//                           // --[3]边界扩展(中心扩散法)

//    bool EXPAND[4] = { 1,1,1,1 };//扩展标记位
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
//    /// 加载源图像
//    Mat src;
//    src = imread("F:/qt/canny/canny/images/1.jpg", 1);
//    //src = imread("C:\\Users\\Administrator\\Desktop\\测试图片\\xxx\\20190308152516.jpg",1);
//    //src = imread("C:\\Users\\Administrator\\Desktop\\测试图片\\xx\\20190308151912.jpg",1);
//    //src = imread("C:\\Users\\Administrator\\Desktop\\测试图像\\2\\BfImg17(x-247 y--91 z--666)-(492,280).jpg",1);
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










////颜色识别2
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
//            vector<Mat> hsvSplit;   //创建向量容器，存放HSV的三通道数据
//            cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
//            split(imgHSV, hsvSplit);			//分类原图像的HSV三通道
//            equalizeHist(hsvSplit[2], hsvSplit[2]);    //对HSV的亮度通道进行直方图均衡
//            merge(hsvSplit, imgHSV);				   //合并三种通道
//            cvtColor(imgHSV, imgBGR, COLOR_HSV2BGR);    //将HSV空间转回至RGB空间，为接下来的颜色识别做准备
//        }
//        else
//        {
//            imgBGR = imgOriginal.clone();
//        }



//        switch(ctrl)
//        {
//        case 0:
//            {
//                inRange(imgBGR, Scalar(128, 0, 0), Scalar(255, 127, 127), imgThresholded); //蓝色
//                break;
//            }
//        case 1:
//            {
//                inRange(imgBGR, Scalar(128, 128, 128), Scalar(255, 255, 255), imgThresholded); //白色
//                break;
//            }
//        case 2:
//            {
//                inRange(imgBGR, Scalar(128, 128, 0), Scalar(255, 255, 127), imgThresholded); //靛色
//                break;
//            }
//        case 3:
//            {
//                inRange(imgBGR, Scalar(128, 0, 128), Scalar(255, 127, 255), imgThresholded); //紫色
//                break;
//            }
//        case 4:
//            {
//                inRange(imgBGR, Scalar(0, 128, 128), Scalar(127, 255, 255), imgThresholded); //黄色
//                break;
//            }
//        case 5:
//            {
//                inRange(imgBGR, Scalar(0, 128, 0), Scalar(127, 255, 127), imgThresholded); //绿色
//                break;
//            }
//        case 6:
//            {
//                inRange(imgBGR, Scalar(0, 0, 128), Scalar(127, 127, 255), imgThresholded); //红色
//                break;
//            }
//        case 7:
//            {
//                inRange(imgBGR, Scalar(0, 0, 0), Scalar(127, 127, 127), imgThresholded); //黑色
//                break;
//            }
//        }

//        imshow("形态学去噪声前", imgThresholded);

//        Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
//        morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);
//        morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, element);

//        imshow("Thresholded Image", imgThresholded); //show the thresholded image
//        imshow("直方图均衡以后", imgBGR);
//        imshow("Original", imgOriginal); //show the original image

//        char key = (char)waitKey(300);
//        if (key == 27)
//            break;
//    }

//    return 0;

//}










//颜色识别1
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

//   //因为我们读取的是彩色图，直方图均衡化需要在HSV空间做
//   split(imgHSV, hsvSplit);
//   equalizeHist(hsvSplit[2],hsvSplit[2]);
//   merge(hsvSplit,imgHSV);
//   Mat imgThresholded;

//   inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image

//   //开操作 (去除一些噪点)
//   Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
//   morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);

//   //闭操作 (连接一些连通域)
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

////检测矩形
////第一个参数是传入的原始图像，第二是输出的图像。
//void findSquares(const Mat& image, Mat &out)
//{
//    int thresh = 50, N = 5;
//    vector<vector<Point> > squares;
//    squares.clear();

//    Mat src,dst, gray_one, gray;

//    src = image.clone();
//    out = image.clone();
//    gray_one = Mat(src.size(), CV_8U);
//    //滤波增强边缘检测
//    medianBlur(src, dst, 9);
//    //bilateralFilter(src, dst, 25, 25 * 2, 35);

//    vector<vector<Point> > contours;
//    vector<Vec4i> hierarchy;

//    //在图像的每个颜色通道中查找矩形
//    for (int c = 0; c < image.channels(); c++)
//    {
//        int ch[] = { c, 0 };

//        //通道分离
//        mixChannels(&dst, 1, &gray_one, 1, ch, 1);

//        // 尝试几个阈值
//        for (int l = 0; l < N; l++)
//        {
//            // 用canny()提取边缘
//            if (l == 0)
//            {
//                //检测边缘
//                Canny(gray_one, gray, 5, thresh, 5);
//                //膨脹
//                dilate(gray, gray, Mat(), Point(-1, -1));
//                imshow("dilate", gray);
//            }
//            else
//            {
//                gray = gray_one >= (l + 1) * 255 / N;
//            }

//            // 轮廓查找
//            //findContours(gray, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
//            findContours(gray, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

//            vector<Point> approx;

//            // 检测所找到的轮廓
//            for (size_t i = 0; i < contours.size(); i++)
//            {
//                //使用图像轮廓点进行多边形拟合
//                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

//                //计算轮廓面积后，得到矩形4个顶点
//                if (approx.size() == 4 &&fabs(contourArea(Mat(approx))) > 1000 &&isContourConvex(Mat(approx)))
//                {
//                    double maxCosine = 0;

//                    for (int j = 2; j < 5; j++)
//                    {
//                        // 求轮廓边缘之间角度的最大余弦
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

//    // 初始化定义
//    int sampleCount = width * height;
//    int clusterCount = 4;
//    Mat points(sampleCount, dims, CV_32F, Scalar(10));
//    Mat labels;
//    Mat centers(clusterCount, 1, points.type());

//    // RGB 数据类型转化到样本数据
//    int index = 0;
//    for (int row = 0; row < height; row++)
//    {
//        for (int col = 0; col < width; col++)
//        {
//            // 多维转一维
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

//    // 显示图像分割后的结果，一维转多维
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

//    // 中心点显示
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
// * 图像直方图
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
//    // 三通道分离
//    vector<Mat> bgr_plane;
//    split(src, bgr_plane);
//    // 定义参数变量
//    const int channels[1] = {0};
//    const int bins[1] = {256};
//    float hranges[2] = {0, 255};
//    const float *ranges[1] = {hranges};
//    Mat b_hist, g_hist, r_hist;
//    // 计算三通道直方图
//    calcHist(&bgr_plane[0], 1, 0, Mat(), b_hist, 1, bins, ranges);
//    calcHist(&bgr_plane[1], 1, 0, Mat(), g_hist, 1, bins, ranges);
//    calcHist(&bgr_plane[2], 1, 0, Mat(), r_hist, 1, bins, ranges);
//    /*
//     * 显示直方图
//     */
//    int hist_w = 512;
//    int hist_h = 400;
//    int bin_w = cvRound((double) hist_w / bins[0]);
//    Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);
//    // 归一化直方图数据
//    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1);
//    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1);
//    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1);
//    // 绘制直方图曲线
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
//        cv::MSER::create(5, // 局部检测时使用的增量值
//            200, // 允许的最小面积
//            20000); // 允许的最大面积
//    std::vector<std::vector<cv::Point> > points;
//    std::vector<cv::Rect> rects;
//    ptrMSER->detectRegions(image, points, rects);

//    cv::Mat output(image.size(), CV_8UC3);
//    output = cv::Scalar(255, 255, 255);
//    cv::RNG rng;
//    // 针对每个检测到的特征区域，在彩色区域显示 MSER
//    // 反向排序，先显示较大的 MSER
//    for (std::vector<std::vector<cv::Point> >::reverse_iterator
//        it = points.rbegin();
//        it != points.rend(); ++it) {
//        // 生成随机颜色
//        cv::Vec3b c(rng.uniform(0, 254),
//            rng.uniform(0, 254), rng.uniform(0, 254));
//        // 针对 MSER 集合中的每个点
//        for (std::vector<cv::Point>::iterator itPts = it->begin();
//            itPts != it->end(); ++itPts) {
//            // 不重写 MSER 的像素
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
//    imshow("腐蚀", out2);

//    Mat gaussian;
//    GaussianBlur(out2, gaussian, Size(g_nGaussianBlurValue * 2 + 1, g_nGaussianBlurValue * 2 + 1), 0, 0);//模糊化
//    imshow("高斯滤波", gaussian);

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
//    imshow("轮廓", img);

//    waitKey();
//}




//#include <iostream>
//#include <include/opencv2/opencv.hpp>
//#include <include/opencv2/highgui.hpp>

//using namespace std;
//using namespace cv;



//int main( )
//{
//    //【1】载入原始图和Mat变量定义
//    Mat srcImage = imread("F:/qt/canny/canny/images/08.jpg");  //工程目录下应该有一张名为1.jpg的素材图
//    Mat midImage,dstImage;//临时变量和目标图的定义

//    //【2】显示原始图
//    imshow("【原始图】", srcImage);

//    //【3】转为灰度图，进行图像平滑
//    cvtColor(srcImage,midImage, CV_BGR2GRAY);//转化边缘检测后的图为灰度图
//    GaussianBlur( midImage, midImage, Size(9, 9), 2, 2 );

//    //【4】进行霍夫圆变换
//    vector<Vec3f> circles;
//    HoughCircles( midImage, circles, CV_HOUGH_GRADIENT,1.5, 10, 200, 100, 0, 0 );

//    //【5】依次在图中绘制出圆
//    for( size_t i = 0; i < circles.size(); i++ )
//    {
//        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
//        int radius = cvRound(circles[i][2]);
//        //绘制圆心
//        circle( srcImage, center, 3, Scalar(0,255,0), -1, 8, 0 );
//        //绘制圆轮廓
//        circle( srcImage, center, radius, Scalar(155,50,255), 3, 8, 0 );
//    }

//    //【6】显示效果图
//    imshow("【效果图】", srcImage);

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

//    Mat edges; //定义转化的灰度图
//    namedWindow("效果图",CV_WINDOW_NORMAL);


//    Mat frame;
//    Mat img = imread("F:/qt/canny/canny/images/3.jpg");

//    if(!img.data)
//    {
//        return -1;
//    }
//    cvtColor(img, edges, CV_BGR2GRAY);
//    //高斯滤波
//    GaussianBlur(edges, edges, Size(7,7), 2, 2);
//    vector<Vec3f>circles;

//    Mat edges2, edges_src;
//    Canny(img, edges2, 100, 300);
//    // 提取彩色边缘
//    bitwise_and(img, img, edges_src, edges2);
//    imshow("edges2", edges2);
//    imshow("edges_src", edges_src);

//    waitKey(0);
//    return 0;



////    //霍夫圆
////    HoughCircles(edges2, circles, CV_HOUGH_GRADIENT, 1.5, 10, 200, 100, 0, 0);
////    for(size_t i = 0; i < circles.size(); i++)
////    {
////        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
////        int radius = cvRound(circles[i][2]);
////        //绘制圆心
////        circle(img, center, 3, Scalar(0, 255, 0), -1, 8, 0);
////        //绘制圆轮廓
////        circle(img, center, radius, Scalar(155, 50, 255), 2, 8, 0);

////    }

////    imshow("效果图2",img);
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

//    // 去噪声与二值化
//    Mat binary;
//    Canny(src, binary, 80, 160);

//    // 标准霍夫直线检测
//    vector<Vec2f> lines;
//    HoughLines(binary, lines, 1, CV_PI / 180, 150);

//    // 绘制直线
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
// * 二值图像分析(对轮廓圆与椭圆拟合)
// */
//int main() {
//    Mat src = imread("F:/qt/canny/canny/images/06.jpg");
//    if (src.empty()) {
//        cout << "could not load image.." << endl;
//    }
//    imshow("input", src);

//    // 去噪声与二值化
//    Mat dst, gray, binary;
//    Canny(src, binary, 80, 160);

//    Mat k = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
//    dilate(binary, binary, k);

//    // 轮廓发现与绘制
//    vector<vector<Point>> contours;
//    vector<Vec4i> hierarchy;
//    findContours(binary, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());
//    for (size_t t = 0; t < contours.size(); t++) {
//        // 寻找适配圆
//        RotatedRect rrt = fitEllipse(contours[t]);
//        ellipse(src, rrt, Scalar(0,0,255), 2);
//    }

//    imshow("contours", src);

//    waitKey(0);
//    return 0;
//}



//圆拟合
//#include <iostream>
//#include <include/opencv2/opencv.hpp>
//#include <include/opencv2/highgui.hpp>
//#include <include/opencv2/imgproc/imgproc_c.h>
//#include <include/opencv2/imgproc.hpp>

//using namespace std;
//using namespace cv;

//bool circleLeastFit(CvSeq* points, double &center_x, double &center_y, double &radius);//最小二乘法拟合函数


//int main()
//{
//    const char* winname  ="winname";
//    //const char* winname1  ="winname1";
//    //const char* winname2  ="winname2";
//    //const char* winname3  ="winname3";
//    char * picname = "P11.jpg";
//    //加载原图
//    IplImage * pImage = cvLoadImage(picname);

//    //分量图像
//    IplImage *pR = cvCreateImage(cvGetSize(pImage),IPL_DEPTH_8U,1);
//    IplImage *pG = cvCreateImage(cvGetSize(pImage),IPL_DEPTH_8U,1);
//    IplImage *pB = cvCreateImage(cvGetSize(pImage),IPL_DEPTH_8U,1);

//    IplImage *temp = cvCreateImage(cvGetSize(pImage),IPL_DEPTH_8U,1);
//    IplImage *binary = cvCreateImage(cvGetSize(pImage),IPL_DEPTH_8U,1);
//    //trackbar的变量值    //对应各个通道
//    int b_low =20;
//    int b_high = 100;
//    int g_low = 20;
//    int g_high = 100;
//    int r_low = 0;
//    int r_high = 100;

//    //轮廓相关
//    CvMemStorage *storage = cvCreateMemStorage(0);
//    CvSeq * seq = cvCreateSeq(CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint), storage);

//    //窗口
//    cvNamedWindow(winname);
//    cvShowImage(winname, pImage);  //显示原图
//    cvNamedWindow("r",2);
//    cvNamedWindow("g",2);
//    cvNamedWindow("b",2); //各个通道
//    cvNamedWindow("binary",2);//二值化图

//    //在相应的窗口建立滑动条
//    cvCreateTrackbar(  "b1","b", &b_low,  254,   NULL); //H通道分量范围0-180
//    cvSetTrackbarPos("b1","b",0 );                        //设置默认位置
//    cvCreateTrackbar(  "b2","b", &b_high,  254,   NULL);//H通道分量范围0-180
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
//        //各个通道分离
//        cvSplit(pImage,pB,pG,pR,NULL);

//        //阈值化
//        cvThreshold(pB, temp,b_low , 255, CV_THRESH_BINARY);
//        cvThreshold(pB, pB,b_high , 255, CV_THRESH_BINARY_INV);
//        cvAnd(temp,pB,pB,NULL);//与操作，合成一张图

//        cvThreshold(pG, temp,g_low , 255, CV_THRESH_BINARY);
//        cvThreshold(pG, pG,g_high , 255, CV_THRESH_BINARY_INV);
//        cvAnd(temp,pG,pG,NULL);//与操作，合成一张图

//        cvThreshold(pR, temp,r_low , 255, CV_THRESH_BINARY);
//        cvThreshold(pR, pR,r_high , 255, CV_THRESH_BINARY_INV);
//        cvAnd(temp,pR,pR,NULL);//与操作，合成一张图

//        //显示各个通道的图像
//        cvShowImage("b",pB);
//        cvShowImage("g",pG);
//        cvShowImage("r",pR);

//        //合成到一张图里
//        cvAnd(pB, pG, binary, NULL);
//        cvAnd(pR, binary, binary, NULL);

//        //膨胀腐蚀操作去除黑点
//        //cvDilate(binary,binary);
//        //cvErode(binary,binary);

//        //显示合成的二值化图
//        cvShowImage("binary",binary);
//        //cvSaveImage("erzhitu.jpg",binary);

//        // 处理轮廓
//        int cnt = cvFindContours(binary,storage,&seq,sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);//返回轮廓的数目
//        CvSeq* _contour =seq;
//        cout<<"number of contours "<<cnt<<endl;
//////////////////////_
//        //找到长度最大轮廓；
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
//            // CvPoint* p = CV_GET_SEQ_ELEM(CvPoint,cur_cont,i);//输出轮廓上点的坐标
//            // printf("(%d,%d)\n",p->x,p->y);
//         //}
//         //cvWaitKey(0);

//         //建立彩色输出图像
//         IplImage *pOutlineImage = cvCreateImage(cvGetSize(pImage), IPL_DEPTH_8U, 3);
//         cvCopy(pImage,pOutlineImage);

//         //int nLevels = 5;
//         //获取最大轮廓的凸包点集
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

//         //最小二乘法拟合圆
//         double center_x=0;
//         double center_y=0;
//         double radius=0;
//         cout<<"nihe :"<<circleLeastFit(hull, center_x, center_y, radius);
//         cout<<"canshu: "<<center_x<<endl<<center_y<<endl<<radius<<endl;

//         //绘制圆
//         cvCircle(pOutlineImage,Point2f(center_x,center_y),radius,CV_RGB(0,100,100));
//         //cvCircle(pOutlineImage,Point2f(center_x,center_y),radius,CV_RGB(0,100,100));

////////////////////////////////////////////////////////////////////////////

//        //绘制轮廓
//        //cvDrawContours(pOutlineImage, cur_cont, CV_RGB(255,0,0), CV_RGB(0,255,0),0);
//        //cvDrawContours(dst,contour,CV_RGB(255,0,0),CV_RGB(0,255,0),0);
//        cvShowImage(winname, pOutlineImage);  //显示原图jiangshang luokuo

//        if (cvWaitKey(1000) == 27)
//        {
//            cvSaveImage("tutu.jpg",pOutlineImage);

//            break;
//        }
//        cvClearMemStorage( storage );  //清除轮廓所占用的内存
//        cvReleaseImage(&pOutlineImage);//清除彩色输出图像
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

////最小二乘法拟合，输出圆心的xy坐标值和半径大小；
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
// * 本程序演示了如何使用2D-2D的特征匹配估计相机运动
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

//// 像素坐标转相机归一化坐标
//Point2d pixel2cam ( const Point2d& p, const Mat& K );

//int main ( int argc, char** argv )
//{
////    if ( argc != 3 )
////    {
////        cout<<"usage: pose_estimation_2d2d img1 img2"<<endl;
////        return 1;
////    }
////    //-- 读取图像
////    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
////    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );


//    Mat img_1 = imread ("F:/qt/canny/canny/images/12.jpg");
//    Mat img_2 = imread ("F:/qt/canny/canny/images/13.jpg");

//    vector<KeyPoint> keypoints_1, keypoints_2;
//    vector<DMatch> matches;
//    find_feature_matches ( img_1, img_2, keypoints_1, keypoints_2, matches );
//    cout<<"一共找到了"<<matches.size() <<"组匹配点"<<endl;

//    //-- 估计两张图像间运动
//    Mat R,t;
//    pose_estimation_2d2d ( keypoints_1, keypoints_2, matches, R, t );

//    //-- 验证E=t^R*scale
//    Mat t_x = ( Mat_<double> ( 3,3 ) <<
//                0,                      -t.at<double> ( 2,0 ),     t.at<double> ( 1,0 ),
//                t.at<double> ( 2,0 ),      0,                      -t.at<double> ( 0,0 ),
//                -t.at<double> ( 1.0 ),     t.at<double> ( 0,0 ),      0 );

//    cout<<"t^R="<<endl<<t_x*R<<endl;

//    //-- 验证对极约束
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
//    //-- 初始化
//    Mat descriptors_1, descriptors_2;
//    // used in OpenCV3
//    Ptr<FeatureDetector> detector = ORB::create();
//    Ptr<DescriptorExtractor> descriptor = ORB::create();
//    // use this if you are in OpenCV2
//    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
//    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
//    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
//    //-- 第一步:检测 Oriented FAST 角点位置
//    detector->detect ( img_1,keypoints_1 );
//    detector->detect ( img_2,keypoints_2 );

//    //-- 第二步:根据角点位置计算 BRIEF 描述子
//    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
//    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

//    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
//    vector<DMatch> match;
//    //BFMatcher matcher ( NORM_HAMMING );
//    matcher->match ( descriptors_1, descriptors_2, match );

//    //-- 第四步:匹配点对筛选
//    double min_dist=10000, max_dist=0;

//    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
//    for ( int i = 0; i < descriptors_1.rows; i++ )
//    {
//        double dist = match[i].distance;
//        if ( dist < min_dist ) min_dist = dist;
//        if ( dist > max_dist ) max_dist = dist;
//    }

//    printf ( "-- Max dist : %f \n", max_dist );
//    printf ( "-- Min dist : %f \n", min_dist );

//    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
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
//    // 相机内参,TUM Freiburg2
//    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );

//    //-- 把匹配点转换为vector<Point2f>的形式
//    vector<Point2f> points1;
//    vector<Point2f> points2;

//    for ( int i = 0; i < ( int ) matches.size(); i++ )
//    {
//        points1.push_back ( keypoints_1[matches[i].queryIdx].pt );
//        points2.push_back ( keypoints_2[matches[i].trainIdx].pt );
//    }

//    //-- 计算基础矩阵
//    Mat fundamental_matrix;
//    fundamental_matrix = findFundamentalMat ( points1, points2, CV_FM_8POINT );
//    cout<<"fundamental_matrix is "<<endl<< fundamental_matrix<<endl;

//    //-- 计算本质矩阵
//    Point2d principal_point ( 325.1, 249.7 );	//相机光心, TUM dataset标定值
//    double focal_length = 521;			//相机焦距, TUM dataset标定值
//    Mat essential_matrix;
//    essential_matrix = findEssentialMat ( points1, points2, focal_length, principal_point );
//    cout<<"essential_matrix is "<<endl<< essential_matrix<<endl;

//    //-- 计算单应矩阵
//    Mat homography_matrix;
//    homography_matrix = findHomography ( points1, points2, RANSAC, 3 );
//    cout<<"homography_matrix is "<<endl<<homography_matrix<<endl;

//    //-- 从本质矩阵中恢复旋转和平移信息.
//    recoverPose ( essential_matrix, points1, points2, R, t, focal_length, principal_point );
//    cout<<"R is "<<endl<<R<<endl;
//    cout<<"t is "<<endl<<t<<endl;

//}
