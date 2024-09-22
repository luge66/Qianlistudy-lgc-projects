#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <opencv2/opencv.hpp>
#include <pcl/common/common.h>
#include <cmath>  

using namespace cv;
using namespace std;

class PCLToOpenCVProcessingNode : public rclcpp::Node
{
public:
    PCLToOpenCVProcessingNode() : Node("pcl_to_opencv_processing_node")
    {
    	//subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            //"/input/point_cloud", 10,
            //std::bind(&PCLToOpenCVProcessingNode::pointCloudCallback, this, std::placeholders::_1));
            //目前没有实时监控的摄像头所以先用这个具体的pcd文件代替,但要确保publisher的节点名也是"/input/point_cloud"
        this->declare_parameter<std::string>("pcd_file_path", "/home/abc/Desktop/ros/abc_cpp/src/learn15.pcd");
        
        this->get_parameter("pcd_file_path", pcd_file_path_);

        if (!loadPointCloud(pcd_file_path_))
        {
            RCLCPP_ERROR(this->get_logger(), "无法加载点云文件: %s", pcd_file_path_.c_str());
            rclcpp::shutdown();
        }      
        timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&PCLToOpenCVProcessingNode::processImage, this));
    }

private:
    bool loadPointCloud(const std::string &file_path)
    {
        cloud_ = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());

        if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(file_path, *cloud_) == -1)
        {
            return false;
        }

        RCLCPP_INFO(this->get_logger(), "点云数据加载成功，点数量: %zu", cloud_->points.size());
        return true;
    }

    void processImage()
    {
        int width = 640;
        int height = 480;
        Mat rgb_image = Mat::zeros(height, width, CV_8UC3);

        pcl::PointXYZRGB min_pt, max_pt;
        pcl::getMinMax3D(*cloud_, min_pt, max_pt);

        float scale_x = width / (max_pt.x - min_pt.x);
        float scale_y = height / (max_pt.y - min_pt.y);
        float scale = std::min(scale_x, scale_y) * 0.8;

        float center_offset_x = width / 2.0 - scale * (min_pt.x + max_pt.x) / 2.0;
        float center_offset_y = height / 2.0 - scale * (min_pt.y + max_pt.y) / 2.0;

        for (const auto &point : cloud_->points)
        {
            int x = static_cast<int>(scale * point.x + center_offset_x);
            int y = height - static_cast<int>(scale * point.y + center_offset_y);

            if (x >= 0 && x < width && y >= 0 && y < height)
            {
                Vec3b &pixel = rgb_image.at<Vec3b>(y, x);
                pixel[0] = point.b;
                pixel[1] = point.g;
                pixel[2] = point.r;
            }
        }

        performImageProcessing(rgb_image);
    }

    void performImageProcessing(Mat &img)
    {
        Mat imgHSV, imgGray, imgBlur, imgCanny, imgDil, mask;
        vector<Point2f> camera;
        int hmin = 132, smin = 0, vmin = 87;
        int hmax = 179, smax = 86, vmax = 255;

        cvtColor(img, imgHSV, COLOR_BGR2HSV);

        Scalar lower(hmin, smin, vmin);
        Scalar upper(hmax, smax, vmax);
        inRange(imgHSV, lower, upper, mask);

        GaussianBlur(mask, imgBlur, Size(3, 3), 3, 0);
        Canny(imgBlur, imgCanny, 25, 75);
        Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        dilate(imgCanny, imgDil, kernel);

        camera = getContours(imgDil, img);

        Mat cameraMatrix = (Mat_<double>(3, 3) << 1462.3697, 0, 398.59394, 0, 1469.68385, 110.68997, 0, 0, 1);
        Mat distCoeffs = (Mat_<double>(1, 5) << 0.003518, -0.311778, -0.016581, 0.023682, 0);

        vector<Point3f> objectPoints = {
            {-4.0f, -4.0f, 0}, {4.0f, -4.0f, 0},
            {-4.0f, 4.0f, 0}, {4.0f, 4.0f, 0}
        };

        Mat rvec, tvec;

        solvePnP(objectPoints, camera, cameraMatrix, distCoeffs, rvec, tvec);


        std::vector<double> eulerAngles = calculateEulerAngles(rvec);
        std::cout << "Euler Angles (Roll, Pitch, Yaw): "
                  << eulerAngles[0] << ", "
                  << eulerAngles[1] << ", "
                  << eulerAngles[2] << std::endl;

        double distance = norm(tvec);
        std::cout << "物体到相机的距离: " << distance << std::endl;

        imshow("Original Image", img);
        imshow("Mask", mask);
        imshow("Detected Objects", img);

        waitKey(1);
    }

    std::vector<double> calculateEulerAngles(const cv::Mat& rvec)
    {
        cv::Mat R;
        cv::Rodrigues(rvec, R); 

        double sy = sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0)); 
        bool singular = sy < 1e-6;

        double x, y, z;
        if (!singular) {
            x = atan2(R.at<double>(2, 1), R.at<double>(2, 2)); 
            y = atan2(-R.at<double>(2, 0), sy);              
            z = atan2(R.at<double>(1, 0), R.at<double>(0, 0)); 
        } else {
            x = atan2(-R.at<double>(1, 2), R.at<double>(1, 1)); 
            y = atan2(-R.at<double>(2, 0), sy);               
            z = 0; 
        }

        return {x * 180.0 / CV_PI, y * 180.0 / CV_PI, z * 180.0 / CV_PI}; 
    }

    vector<Point2f> getContours(Mat imgDil, Mat img)
    {
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(imgDil, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        vector<vector<Point>> conPoly(contours.size());
        vector<Rect> boundRect(contours.size());
        Point2f topl(10000, 10000);
        Point2f botr(0, 0);
        int maxArea = 0;
        float limitWidth;

        Point2f maxL, maxR;
        for (size_t i = 0; i < contours.size(); i++) {
            int area = contourArea(contours[i]);

            if (area > 80) {
                float peri = arcLength(contours[i], true);
                approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);
                boundRect[i] = boundingRect(conPoly[i]);

                if (area > maxArea) {
                    maxArea = area;
                    limitWidth = boundRect[i].width;
                    maxL = boundRect[i].tl();
                    maxR = boundRect[i].br();
                }
            }
        }

        for (size_t i = 0; i < conPoly.size(); i++) {
            if (abs(boundRect[i].br().x - maxL.x) <= limitWidth || abs(boundRect[i].tl().x - maxR.x) <= limitWidth) {
                if (boundRect[i].tl().x < topl.x) {
                    topl.x = boundRect[i].tl().x;
                }
                if (boundRect[i].br().x > botr.x) {
                    botr.x = boundRect[i].br().x;
                }
                if (boundRect[i].tl().y < topl.y) {
                    topl.y = boundRect[i].tl().y;
                }
                if (boundRect[i].br().y > botr.y) {
                    botr.y = boundRect[i].br().y;
                }
            }
        }

        vector<Point2f> camera2 = { Point2f(topl.x, topl.y), Point2f(botr.x, topl.y), Point2f(topl.x, botr.y), Point2f(botr.x, botr.y) };
        rectangle(img, topl, botr, Scalar(0, 0, 255), 5);
        return camera2;
    }

    std::string pcd_file_path_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_;
    rclcpp::TimerBase::SharedPtr timer_;
    //rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PCLToOpenCVProcessingNode>());
    rclcpp::shutdown();
    return 0;
}

