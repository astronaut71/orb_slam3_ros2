#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <cv_bridge/cv_bridge.h>
#include "System.h"
#include "Frame.h"
#include "Map.h"
#include "Tracking.h"
#include "utility.hpp"
#include <opencv2/core/core.hpp>
#include <tuple>

#include "tf2_ros/transform_broadcaster.h"
#include "tf2/LinearMath/Transform.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

class MonocularSlamNode : public rclcpp::Node
{
public:
    MonocularSlamNode(ORB_SLAM3::System* pSLAM);
    ~MonocularSlamNode();

private:
    using ImageMsg = sensor_msgs::msg::Image;
    using ImuMsg = sensor_msgs::msg::Imu;
    using PoseStampedMsg = geometry_msgs::msg::PoseStamped;

    void handleImage(const ImageMsg::SharedPtr image_msg);
    void handleImu(const ImuMsg::SharedPtr imu_msg);
    void processImageData(const cv_bridge::CvImageConstPtr& cv_ptr, const rclcpp::Time& timestamp);
    void processImuData(const cv::Point3f& gyr, const cv::Point3f& acc, const rclcpp::Time& timestamp, float yaw);
    void publishPose(const Sophus::SE3f& camera_pose, const rclcpp::Time& timestamp);

    ORB_SLAM3::System* m_SLAM;
    rclcpp::Subscription<ImageMsg>::SharedPtr m_image_subscriber;
    rclcpp::Subscription<ImuMsg>::SharedPtr m_imu_subscriber;
    rclcpp::Publisher<PoseStampedMsg>::SharedPtr m_camera_pose_publisher;
    bool image_received_;
    bool imu_received_;
    bool pub_tf, pub_pose;
    cv_bridge::CvImageConstPtr m_cvImPtr;
    std::tuple<cv::Point3f, cv::Point3f, rclcpp::Time, float> m_imuData; // Added yaw

    Eigen::Quaterniond calculateRotation(const cv::Point3f& gyr, const cv::Point3f& acc, float yaw);
};

MonocularSlamNode::MonocularSlamNode(ORB_SLAM3::System* pSLAM)
    : Node("ORB_SLAM3_ROS2"),
      m_SLAM(pSLAM),
      image_received_(false),
      imu_received_(false)
{
    m_image_subscriber = this->create_subscription<ImageMsg>(
        "/anafi/camera/image",
        rclcpp::SystemDefaultsQoS(),
        [this](const ImageMsg::SharedPtr image_msg) {
            this->handleImage(image_msg);
        }
    );

    m_imu_subscriber = this->create_subscription<ImuMsg>(
        "/anafi/camera/imu",
        rclcpp::SystemDefaultsQoS(),
        [this](const ImuMsg::SharedPtr imu_msg) {
            this->handleImu(imu_msg);
        }
    );

    m_camera_pose_publisher = this->create_publisher<PoseStampedMsg>(
        "/pose_orb",
        rclcpp::SystemDefaultsQoS()
    );
}

MonocularSlamNode::~MonocularSlamNode()
{
    m_SLAM->Shutdown();
    m_SLAM->SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
}

void MonocularSlamNode::handleImage(const ImageMsg::SharedPtr image_msg)
{
    try {
        cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(image_msg, sensor_msgs::image_encodings::BGR8);
        cv::Mat gray_image;
        cv::cvtColor(cv_ptr->image, gray_image, cv::COLOR_BGR2GRAY);
        processImageData(cv_ptr, image_msg->header.stamp);
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
}

void MonocularSlamNode::handleImu(const ImuMsg::SharedPtr imu_msg)
{
    cv::Point3f gyr(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
    cv::Point3f acc(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);
    float yaw = imu_msg->orientation.z; // Assuming the yaw is stored in the z component of the orientation
    processImuData(gyr, acc, imu_msg->header.stamp, yaw); // Pass yaw
}

Eigen::Quaterniond MonocularSlamNode::calculateRotation(const cv::Point3f& gyr, const cv::Point3f& acc, float yaw) {
    Eigen::Vector3d gyroscope(gyr.x, gyr.y, gyr.z);
    Eigen::Vector3d accelerometer(acc.x, acc.y, acc.z);

    // Perform  rotation calculation here.
    // We may use sensor fusion algorithms like Mahony, Madgwick, or a Kalman filter.
    // For example, we'll use a simple method: tilt-compensated eComplementary filter.
    
    double roll = atan2(accelerometer.y(), sqrt(accelerometer.x() * accelerometer.x() + accelerometer.z() * accelerometer.z()));
    double pitch = atan2(-accelerometer.x(), accelerometer.z());

    Eigen::Quaterniond rotation;
    rotation = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX())
             * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY())
             * Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()); // Include yaw

    return rotation;
}

void MonocularSlamNode::processImageData(const cv_bridge::CvImageConstPtr& cv_ptr, const rclcpp::Time& timestamp)
{
    if (imu_received_) {
        std::vector<ORB_SLAM3::IMU::Point> vImuMeas;

        // Extract yaw from the tuple
        float yaw = std::get<3>(m_imuData);

        // Calculate the rotation from IMU data
        //Eigen::Quaterniond sophus_rotation = calculateRotation(std::get<0>(m_imuData), std::get<1>(m_imuData), std::get<2>(m_imuData));
        Eigen::Quaterniond sophus_rotation = calculateRotation(std::get<0>(m_imuData), std::get<1>(m_imuData), yaw);
        
        // Process image with valid rotation data
        Sophus::SE3f camera_pose = m_SLAM->TrackMonocular(cv_ptr->image, timestamp.seconds(), vImuMeas, ""); // Pass an empty string for the filename
        publishPose(camera_pose, timestamp);

    }

    image_received_ = true;
}

void MonocularSlamNode::processImuData(const cv::Point3f& gyr, const cv::Point3f& acc, const rclcpp::Time& timestamp, float yaw)
{
    // Store the IMU data including yaw
    m_imuData = std::make_tuple(gyr, acc, timestamp, yaw);

    imu_received_ = true;
}

void MonocularSlamNode::publishPose(const Sophus::SE3f& camera_pose, const rclcpp::Time& timestamp)
{
    Eigen::Vector3f translation = camera_pose.translation();
    Eigen::Matrix3f rotation = camera_pose.rotationMatrix();

    cv::Mat transformation_matrix(4, 4, CV_32F);
    transformation_matrix.setTo(0.0);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            transformation_matrix.at<float>(i, j) = rotation(i, j);
        }
        transformation_matrix.at<float>(i, 3) = translation(i);
    }
    transformation_matrix.at<float>(3, 3) = 1.0;

    tf2::Transform tf_transform;
    tf_transform.setOrigin(tf2::Vector3(translation(0), translation(1), translation(2)));
    Eigen::Quaternionf quaternion(rotation);
    tf2::Quaternion tf_quaternion(quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w());
    tf_transform.setRotation(tf_quaternion);

    if (pub_tf) {
        static tf2_ros::TransformBroadcaster tf_broadcaster(this);
        geometry_msgs::msg::TransformStamped transform_stamped; // Corrected message type
        transform_stamped.header.stamp = timestamp;
        transform_stamped.header.frame_id = "/camera";
        transform_stamped.child_frame_id = "ORB_SLAM3_MONO_INERTIAL";
        transform_stamped.transform = tf2::toMsg(tf_transform);
        tf_broadcaster.sendTransform(transform_stamped);
    }

    if (pub_pose) {
        geometry_msgs::msg::PoseStamped pose_msg; // Corrected message type
        pose_msg.header.stamp = timestamp;
        pose_msg.header.frame_id = "ORB_SLAM3_MONO_INERTIAL";
        pose_msg.pose.position.x = translation(0);
        pose_msg.pose.position.y = translation(1);
        pose_msg.pose.position.z = translation(2);
        pose_msg.pose.orientation.x = quaternion.x();
        pose_msg.pose.orientation.y = quaternion.y();
        pose_msg.pose.orientation.z = quaternion.z();
        pose_msg.pose.orientation.w = quaternion.w();

        m_camera_pose_publisher->publish(pose_msg);
    }
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    // Load ORB-SLAM3 parameters
    if (argc != 3) {
        std::cerr << "Usage: ros2 run orb_slam3_ros2 monocular [Vocabulary] [Settings]" << std::endl;
        return 1;
    }

    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, true);

    rclcpp::spin(std::make_shared<MonocularSlamNode>(&SLAM));
    rclcpp::shutdown();

    return 0;
}

