#ifndef __MONOCULAR_SLAM_NODE_HPP__
#define __MONOCULAR_SLAM_NODE_HPP__

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include <nav_msgs/msg/odometry.h>
#include "geometry_msgs/msg/twist.hpp"

#include <geometry_msgs/msg/pose_stamped.hpp>
#include "geometry_msgs/msg/quaternion.hpp"

#include <cv_bridge/cv_bridge.h>

#include "System.h"
#include "Frame.h"
#include "Map.h"
#include "Tracking.h"

#include "utility.hpp"

class MonocularSlamNode : public rclcpp::Node
{
public:
    MonocularSlamNode(ORB_SLAM3::System* pSLAM);

    ~MonocularSlamNode();

private:
    using ImageMsg = sensor_msgs::msg::Image;
    using ImuMsg = sensor_msgs::msg::Imu;
    void GrabImage(const sensor_msgs::msg::Image::SharedPtr msg);
    void GrabImu(const sensor_msgs::msg::Imu::SharedPtr msg);
    cv::Mat GetImage(const ImageMsg::SharedPtr msg);
    void SyncWithImu();
    
    ORB_SLAM3::System* m_SLAM;
    std::thread *syncThread_;

     // IMU
    queue<ImuMsg::SharedPtr> imuBuf_;
    std::mutex bufImuMutex_;

    // Image
    queue<ImageMsg::SharedPtr> imgBuf_;
    std::mutex bufImgMutex_;

    cv::Mat M1_, M2_;
    bool bClahe_;
    cv::Ptr<cv::CLAHE> clahe_ = cv::createCLAHE(3.0, cv::Size(8, 8));


    cv::Mat prev_pose_;
    //rclcpp::Time prev_pose_timestamp_;
    rclcpp::Time current_pose_timestamp;


    cv_bridge::CvImagePtr m_cvImPtr;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr m_image_subscriber;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr m_imu_subscriber;

    //rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr m_imu_subscriber;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_publisher;

    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::Quaternion>::SharedPtr orientation_publisher_;


    //rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr m_velocity_pub;
    
};

#endif
