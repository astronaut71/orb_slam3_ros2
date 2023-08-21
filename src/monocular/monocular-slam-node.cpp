#include "monocular-slam-node.hpp"

#include<opencv2/core/core.hpp>

using std::placeholders::_1;

MonocularSlamNode::MonocularSlamNode(ORB_SLAM3::System* pSLAM)
:   Node("ORB_SLAM3_ROS2")
{
    m_SLAM = pSLAM;
    // std::cout << "slam changed" << std::endl;
    m_image_subscriber = this->create_subscription<ImageMsg>("/anafi/camera/image",rclcpp::SystemDefaultsQoS(), std::bind(&MonocularSlamNode::GrabImage, this, std::placeholders::_1));
    // Advertise the publisher for camera pose
    m_camera_pose_publisher = this->create_publisher<PoseStampedMsg>("/pose_orb",rclcpp::SystemDefaultsQoS());
    //m_camera_pose_publisher = this->create_publisher<geometry_msgs::msg::PoseStamped>("camera_pose", 10);
    std::cout << "slam changed" << std::endl;
}

MonocularSlamNode::~MonocularSlamNode()
{
    // Stop all threads
    m_SLAM->Shutdown();

    // Save camera trajectory
    m_SLAM->SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
}

void MonocularSlamNode::GrabImage(const ImageMsg::SharedPtr msg)
{
    //Copy the ros image message to cv::Mat.
    try
    {
        // Convert the BGR8 image to grayscale.
        cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
        cv::Mat gray_image;
        cv::cvtColor(cv_ptr->image, gray_image, cv::COLOR_BGR2GRAY);

        // Pass the grayscale image to SLAM.
        //m_SLAM->TrackMonocular(gray_image, Utility::StampToSec(msg->header.stamp));
        // Pass the grayscale image to SLAM and get the camera pose.
        Sophus::SE3f camera_pose = m_SLAM->TrackMonocular(gray_image, Utility::StampToSec(msg->header.stamp));

        // Extract translation and rotation
        Eigen::Vector3f translation = camera_pose.translation();
        Eigen::Matrix3f rotation = camera_pose.rotationMatrix();

        // Create a transformation matrix
        cv::Mat transformation_matrix(4, 4, CV_32F);
        transformation_matrix.setTo(0.0);

        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                transformation_matrix.at<float>(i, j) = rotation(i, j);
            }
            transformation_matrix.at<float>(i, 3) = translation(i);
        }
        transformation_matrix.at<float>(3, 3) = 1.0;

        // Create a PoseStamped message and fill it with camera pose
        PoseStampedMsg pose_msg;
        pose_msg.header.stamp = msg->header.stamp;

        // Assuming the translation is in meters
        pose_msg.pose.position.x = translation(0);
        pose_msg.pose.position.y = translation(1);
        pose_msg.pose.position.z = translation(2);

        Eigen::Quaternionf quaternion(rotation);
        pose_msg.pose.orientation.x = quaternion.x();
        pose_msg.pose.orientation.y = quaternion.y();
        pose_msg.pose.orientation.z = quaternion.z();
        pose_msg.pose.orientation.w = quaternion.w();

        // Publish the camera pose
        m_camera_pose_publisher->publish(pose_msg);

        std::cout << "One frame has been sent" << std::endl;
    }
    catch (cv_bridge::Exception& e)
    {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }
}
