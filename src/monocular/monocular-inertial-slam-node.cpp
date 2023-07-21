#include "monocular-slam-node.hpp"
#include <opencv2/core/core.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include "geometry_msgs/msg/quaternion.hpp"

//#include "nav_msgs/msg/odometry.hpp"

using std::placeholders::_1;

MonocularSlamNode::MonocularSlamNode(ORB_SLAM3::System* pSLAM)
    : Node("ORB_SLAM3_ROS2")
{
    m_SLAM = pSLAM;
    m_image_subscriber = this->create_subscription<ImageMsg>("camera",10, std::bind(&MonocularSlamNode::GrabImage, this, std::placeholders::_1));
    m_imu_subscriber = this->create_subscription<ImuMsg>("imu",10, std::bind(&MonocularSlamNode::GrabImu, this, std::placeholders::_1));
    syncThread_ = new std::thread(&MonocularSlamNode::SyncWithImu, this);
    odometry_publisher = this->create_publisher<nav_msgs::msg::Odometry>("odom", 10);

    pose_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("pose", 10);
    orientation_publisher_ = this->create_publisher<geometry_msgs::msg::Quaternion>("orientation", 10);
    //m_velocity_pub = this->create_publisher<geometry_msgs::Twist>("lin_velocity", 10);
}

MonocularSlamNode::~MonocularSlamNode()
{
    // Delete sync thread
    syncThread_->join();
    delete syncThread_;
    
    // Stop all threads
    m_SLAM->Shutdown();

    // Save camera trajectory
    m_SLAM->SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
}

void MonocularSlamNode::GrabImu(const ImuMsg::SharedPtr msg)
{
    bufImuMutex_.lock();
    imuBuf_.push(msg);
    bufImuMutex_.unlock();
}

void MonocularSlamNode::GrabImage(const ImageMsg::SharedPtr msg)
{
    bufImgMutex_.lock();

    if (!imgBuf_.empty())
        imgBuf_.pop();
    imgBuf_.push(msg);

    bufImgMutex_.unlock();
}
cv::Mat MonocularSlamNode::GetImage(const ImageMsg::SharedPtr msg)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptr;

    try
    {
        cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception &e)
    {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }

    if (cv_ptr->image.type() == 0)
    {
        return cv_ptr->image.clone();
    }
    else
    {
        std::cerr << "Error image type" << std::endl;
        return cv_ptr->image.clone();
    }
}

void MonocularSlamNode::SyncWithImu()
{
    const double maxTimeDiff = 0.01;

    // Initialize variables for velocity estimation
    cv::Mat current_pose = cv::Mat::eye(4, 4, CV_32F);
    //rclcpp::Time current_pose_timestamp;
    while (1)
    {
        cv::Mat im;
        double tIm = 0;
        if (!imgBuf_.empty() && !imuBuf_.empty())
        {
            tIm = Utility::StampToSec(imgBuf_.front()->header.stamp);

            bufImgMutex_.lock();
            while (imgBuf_.size() > 1)
            {
                imgBuf_.pop();
                tIm = Utility::StampToSec(imgBuf_.front()->header.stamp);
            }
            bufImgMutex_.unlock();

            if (tIm > Utility::StampToSec(imuBuf_.back()->header.stamp))
                continue;

            bufImgMutex_.lock();
            im = GetImage(imgBuf_.front());
            imgBuf_.pop();
            bufImgMutex_.unlock();

            vector<ORB_SLAM3::IMU::Point> vImuMeas;
            bufImuMutex_.lock();
            if (!imuBuf_.empty())
            {
                // Load imu measurements from buffer
                auto latest_imu_msg = imuBuf_.back();

                // Convert the IMU message to the ORB_SLAM3 IMU format
                ORB_SLAM3::IMU::Point latest_imu_meas(
                cv::Point3f(latest_imu_msg->linear_acceleration.x, latest_imu_msg->linear_acceleration.y, latest_imu_msg->linear_acceleration.z),
                cv::Point3f(latest_imu_msg->angular_velocity.x, latest_imu_msg->angular_velocity.y, latest_imu_msg->angular_velocity.z),
                Utility::StampToSec(latest_imu_msg->header.stamp));

                // Estimate the time elapsed between the current and previous IMU measurements
                //double deltaTime = (tIm - current_pose_timestamp).seconds();
                 for (size_t i = 1; i < vImuMeas.size(); ++i)
                {
                double deltaTime = vImuMeas[i].t - vImuMeas[i - 1].t;
                           

                // Estimate the linear velocity using Euler integration of the IMU linear acceleration
                cv::Point3f linear_velocity;
                
                linear_velocity.x = current_pose.at<float>(0, 3) + latest_imu_msg->linear_acceleration.x* deltaTime;
                linear_velocity.y = current_pose.at<float>(1, 3) + latest_imu_msg->linear_acceleration.y* deltaTime;
                linear_velocity.z = current_pose.at<float>(2, 3) + latest_imu_msg->linear_acceleration.z* deltaTime;

                // Update the current pose using the linear velocity
                current_pose.at<float>(0, 3) = linear_velocity.x * deltaTime;
                current_pose.at<float>(1, 3) = linear_velocity.y * deltaTime;
                current_pose.at<float>(2, 3) = linear_velocity.z * deltaTime;

                geometry_msgs::msg::PoseStamped pose_msg;
                nav_msgs::msg::Odometry odom;
                odom.header.stamp = this->get_clock()->now(); // Use the current ROS time
                odom.header.frame_id = "odom";
                odom.child_frame_id = "base_footprint";
                pose_msg.pose.position.x = current_pose.at<float>(0, 3);
                pose_msg.pose.position.y = current_pose.at<float>(1, 3);
                pose_msg.pose.position.z = current_pose.at<float>(2, 3);

                

                // Orientation 

                // Publish the pose
                pose_publisher_->publish(pose_msg);

                // Update the current pose timestamp for the next iteration
                current_pose_timestamp = rclcpp::Time(latest_imu_msg->header.stamp);

                }
                //linear_velocity.x = current_pose.at<float>(0, 3) + latest_imu_meas.acc.x * deltaTime;

                vImuMeas.clear();
                while (!imuBuf_.empty() && Utility::StampToSec(imuBuf_.front()->header.stamp) <= tIm)
                {
                    double t = Utility::StampToSec(imuBuf_.front()->header.stamp);
                    cv::Point3f velocity(0, 0, 0);  // Accumulated linear acceleration
                    double prevTime = vImuMeas.front().t;  // Previous IMU timestamp

                    //cv::Point3f linearAcc(0, 0, 0);  // Accumulated linear acceleration
                    double deltaTime = 0;       
                     
                            cv::Point3f acc(imuBuf_.front()->linear_acceleration.x, imuBuf_.front()->linear_acceleration.y, imuBuf_.front()->linear_acceleration.z);
                            
                        
                            cv::Point3f gyr(imuBuf_.front()->angular_velocity.x, imuBuf_.front()->angular_velocity.y, imuBuf_.front()->angular_velocity.z);
                            //vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc, gyr, t));
                            //imuBuf_.pop();
                            /*for (size_t i = 1; i < vImuMeas.size(); ++i)
                                {
                                    deltaTime = vImuMeas[i].t - vImuMeas[i - 1].t;
                                    //velocity.x += (Point[i].acc.x + Pointacc(i-1).x )* deltaTime /2.0;
                                    velocity.x += acc.x  * deltaTime;

                                }*/
                            vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc, gyr, t));
                            imuBuf_.pop();
                }
            }
            bufImuMutex_.unlock();

            if (bClahe_)
            {
                clahe_->apply(im, im);
            }

            cv::remap(im, im, M1_, M2_, cv::INTER_LINEAR);
            
            
            m_SLAM->TrackMonocular(im, tIm, vImuMeas);

            // Get the current pose from ORB_SLAM3
            //Get the latest IMU measurement
            
           

            std::chrono::milliseconds tSleep(1);
            std::this_thread::sleep_for(tSleep);
        }
    }
}

/*void MonocularSlamNode::GrabImage(const ImageMsg::SharedPtr msg)
{
    // Copy the ROS image message to cv::Mat
    try
    {
        m_cvImPtr = cv_bridge::toCvCopy(msg);
    }
    catch (cv_bridge::Exception& e)
    {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    // Process the image with ORB_SLAM3
    m_SLAM->TrackMonocular(m_cvImPtr->image, Utility::StampToSec(msg->header.stamp));

    // Get the current pose from ORB_SLAM3
   //cv::Mat pose = m_SLAM->GetCurrentPose();

    // Publish the inertial odometry

    //m_odom_publisher->publish(odom_msg);
}*/


/*void MonocularSlamNode::GrabImu(const ImuMsg::SharedPtr msg)
{
    // Extract inertial data from IMU message
    double timestamp = Utility::StampToSec(msg->header.stamp);
    cv::Mat imu_data = (cv::Mat_<float>(6, 1) << msg->linear_acceleration.x,
                        msg->linear_acceleration.y, msg->linear_acceleration.z,msg->angular_velocity.x,
                        msg->angular_velocity.y, msg->angular_velocity.z );

    // Pass the inertial data to ORB_SLAM3
    m_SLAM->TrackMonocular(imu_data, timestamp);
}*/
