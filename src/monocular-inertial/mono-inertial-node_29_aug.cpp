#include <iostream>
#include <queue>
#include <thread>
#include <mutex>

#include "rclcpp/rclcpp.hpp"
#include <cv_bridge/cv_bridge.h>
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/image.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include "utility.hpp"

#include "../include/ImuTypes.h"
#include "../include/System.h"
#include "../include/Converter.h"

//#include "/home/bojan/test-orb3/ORB_SLAM3/include/System.h"
//#include "/home/bojan/test-orb3/ORB_SLAM3/include/ImuTypes.h"
//#include "/home/bojan/test-orb3/ORB_SLAM3/include/Converter.h"

#include "tf2_ros/transform_broadcaster.h"
#include "tf2/LinearMath/Transform.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include <geometry_msgs/msg/pose_stamped.hpp>

using namespace std;

class ImuGrabber
{
public:
    ImuGrabber(const rclcpp::Logger& logger)
        : logger_(logger) {
        // Constructor implementation
    }

    void GrabImu(const sensor_msgs::msg::Imu::SharedPtr imu_msg);    
    queue<sensor_msgs::msg::Imu::SharedPtr> imuBuf;
    std::mutex mBufMutex;

private:
    rclcpp::Logger logger_;
};

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM3::System* pSLAM, ImuGrabber *pImuGb, const bool bClahe, rclcpp::Logger logger)
        : mpSLAM(pSLAM), mpImuGb(pImuGb), mbClahe(bClahe), logger_(logger) {}

    void GrabImage(const sensor_msgs::msg::Image::SharedPtr img_msg);
    cv::Mat GetImage(const sensor_msgs::msg::Image::SharedPtr img_msg);
    void SyncWithImu();
    void SetPub(rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub);

    queue<sensor_msgs::msg::Image::SharedPtr> img0Buf;
    std::mutex mBufMutex;

    ORB_SLAM3::System* mpSLAM;
    ImuGrabber *mpImuGb;
    
    bool mbClahe;
    cv::Ptr<cv::CLAHE> mClahe = cv::createCLAHE(3.0, cv::Size(8, 8));

private:
    rclcpp::Logger logger_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr orb_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

class MonoInertial : public rclcpp::Node
{
public:
    MonoInertial(const std::string& vocabulary_path, const std::string& settings_path, bool do_equalize)
        : Node("Mono_Inertial"),
          SLAM_(nullptr),
          imugb_(nullptr),
          igb_(nullptr),
          logger_(get_logger()),
          pose_pub_(nullptr),
          sub_imu_(nullptr),
          sub_img0_(nullptr)
    {
        RCLCPP_INFO(logger_, "Mono_Inertial node started");

        SLAM_ = new ORB_SLAM3::System(vocabulary_path, settings_path, ORB_SLAM3::System::IMU_MONOCULAR, true);
        imugb_ = new ImuGrabber(logger_);
        igb_ = new ImageGrabber(SLAM_, imugb_, do_equalize, logger_);

        RCLCPP_INFO(logger_, "Creating publisher for /orb_pose topic...");
        pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("/orb_pose", rclcpp::SystemDefaultsQoS());
        RCLCPP_INFO(logger_, "Publisher for /orb_pose topic created.");

        sub_imu_ = create_subscription<sensor_msgs::msg::Imu>(
            "/anafi/camera/imu", rclcpp::SystemDefaultsQoS(), [&](const sensor_msgs::msg::Imu::SharedPtr imu_msg) { imugb_->GrabImu(imu_msg); });
        sub_img0_ = create_subscription<sensor_msgs::msg::Image>(
            "/anafi/camera/image", rclcpp::SystemDefaultsQoS(), [&](const sensor_msgs::msg::Image::SharedPtr img_msg) { igb_->GrabImage(img_msg); });

        igb_->SetPub(pose_pub_);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);
        sync_thread_ = std::thread(&ImageGrabber::SyncWithImu, igb_);
    }

private:
    ORB_SLAM3::System* SLAM_;
    ImuGrabber* imugb_;
    ImageGrabber* igb_;
    rclcpp::Logger logger_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img0_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::thread sync_thread_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    bool do_equalize = false;

    if (argc < 3 || argc > 4)
    {
        RCLCPP_ERROR(rclcpp::get_logger("Mono_Inertial"), "Usage: ros2 run ORB_SLAM3 Mono_Inertial path_to_vocabulary path_to_settings [do_equalize]");
        rclcpp::shutdown();
        return 1;
    }

    if (argc == 4)
    {
        std::string sbEqual(argv[3]);
        if (sbEqual == "true")
            do_equalize = true;
    }

    auto node = std::make_shared<MonoInertial>(argv[1], argv[2], do_equalize);

    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}

void ImageGrabber::GrabImage(const sensor_msgs::msg::Image::SharedPtr img_msg)
{
    mBufMutex.lock();
    if (!img0Buf.empty())
        img0Buf.pop();
    img0Buf.push(img_msg);
    mBufMutex.unlock();
    //RCLCPP_INFO(logger_, "ImageGrabber: New image data grabbed.");
}

cv::Mat ImageGrabber::GetImage(const sensor_msgs::msg::Image::SharedPtr img_msg)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
        RCLCPP_ERROR(logger_, "cv_bridge exception: %s", e.what());
        return cv::Mat();
    }

    cv::Mat gray_img;
    cv::cvtColor(cv_ptr->image, gray_img, cv::COLOR_RGB2GRAY);

    if (mbClahe)
    {
        cv::Mat processed_img;
        mClahe->apply(gray_img, processed_img);
        return processed_img;
    }
    else
    {
        return gray_img;
    }
}

void ImageGrabber::SyncWithImu()
{
    while (rclcpp::ok())
    {
        if (!img0Buf.empty() && !mpImuGb->imuBuf.empty())
        {
            cv::Mat im;
            double tIm = 0;

            // Get the timestamp from the front of the image buffer
            //::Time img_time = img0Buf.front()->header.stamp;
            //tIm = img_time.seconds();
            tIm = img0Buf.front()->header.stamp.sec*0.05; // 0.05, 0.01

            // Check if the image timestamp is greater than the last IMU timestamp
            if (tIm > mpImuGb->imuBuf.back()->header.stamp.sec)
                continue;

            this->mBufMutex.lock();
            im = GetImage(img0Buf.front());
            rclcpp::Time msg_time = img0Buf.front()->header.stamp;
            img0Buf.pop();
            this->mBufMutex.unlock();

            std::vector<ORB_SLAM3::IMU::Point> vImuMeas;
            mpImuGb->mBufMutex.lock();
            if (!mpImuGb->imuBuf.empty())
            {
                // Load imu measurements from buffer
                vImuMeas.clear();

                while (!mpImuGb->imuBuf.empty())
                {
                    // Get IMU data and timestamp
                    //sensor_msgs::msg::Imu imu_msg = *mpImuGb->imuBuf.front();
                    //double t = imu_msg.header.stamp.sec;
                    double t = mpImuGb->imuBuf.front()->header.stamp.sec*5; //5, 10

                    // Convert the orientation quaternion to roll, pitch, and yaw
                    tf2::Quaternion quaternion;
                    tf2::fromMsg(mpImuGb->imuBuf.front()->orientation, quaternion);

                    double roll, pitch, yaw;
                    tf2::Matrix3x3(quaternion).getRPY(roll, pitch, yaw);

                    // Create Point3f for orientation
                    cv::Point3f orientation(roll, pitch, yaw);

                    // Extract angular velocity
                    //cv::Point3f gyr(imu_msg.angular_velocity.x * 1, imu_msg.angular_velocity.y * 1, imu_msg.angular_velocity.z * 1);
                    cv::Point3f gyr(mpImuGb->imuBuf.front()->angular_velocity.x, mpImuGb->imuBuf.front()->angular_velocity.y, mpImuGb->imuBuf.front()->angular_velocity.z);

                    // Add IMU data to the vector
                    vImuMeas.push_back(ORB_SLAM3::IMU::Point(orientation, gyr, t));
                    mpImuGb->imuBuf.pop();
                }
            }

            mpImuGb->mBufMutex.unlock();

            // ORB-SLAM3 runs in TrackMonocular()
            Sophus::SE3f Tcw = mpSLAM->TrackMonocular(im, tIm, vImuMeas);

            // Publish pose data
            Eigen::Matrix4f Tcw_eigen = Tcw.matrix().cast<float>();
            Eigen::Matrix3f R_eigen = Tcw_eigen.block<3, 3>(0, 0);
            Eigen::Vector3f t_eigen = Tcw_eigen.block<3, 1>(0, 3);

            geometry_msgs::msg::PoseStamped pose_msg;
            //pose_msg.header.stamp = rclcpp::Time(tIm, msg_time.nanoseconds()).to_msg(); // Preserve nanoseconds
            //pose_msg.header.stamp = rclcpp::Time(tIm, msg_time.nanoseconds()).to_msg(); // Preserve nanoseconds
            //pose_msg.header.stamp = rclcpp::Time(tIm, msg_time.nanoseconds());
            pose_msg.header.stamp = rclcpp::Clock(RCL_ROS_TIME).now();
            pose_msg.header.frame_id = "/camera";
            pose_msg.pose.position.x = t_eigen(0);
            pose_msg.pose.position.y = t_eigen(1);
            pose_msg.pose.position.z = t_eigen(2);

            Eigen::Quaterniond quat(R_eigen.cast<double>());
            pose_msg.pose.orientation.x = quat.x();
            pose_msg.pose.orientation.y = quat.y();
            pose_msg.pose.orientation.z = quat.z();
            pose_msg.pose.orientation.w = quat.w();

            orb_pub_->publish(pose_msg);

            // Control loop frequency if needed
            std::this_thread::sleep_for(std::chrono::nanoseconds(10)); // Adjust sleep duration as needed
        }
    }
}
void ImuGrabber::GrabImu(const sensor_msgs::msg::Imu::SharedPtr imu_msg)
{
    mBufMutex.lock();
    imuBuf.push(imu_msg);
    mBufMutex.unlock();
}


void ImageGrabber::SetPub(rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub)
{
    orb_pub_ = pub;
}




