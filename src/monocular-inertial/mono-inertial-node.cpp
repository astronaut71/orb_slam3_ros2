#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<vector>
#include<queue>
#include<thread>
#include<mutex>

#include "rclcpp/rclcpp.hpp"
#include <cv_bridge/cv_bridge.h>
#include "sensor_msgs/msg/imu.hpp"

#include <opencv2/core/core.hpp>

#include "System.h"
#include "ImuTypes.h"
#include "Converter.h"

//for pubbing
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/point.hpp>
#include "geometry_msgs/msg/quaternion.hpp"
#include "geometry_msgs/msg/transform.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>


using namespace std;

class ImuGrabber
{
public:
    ImuGrabber(){};
    void GrabImu(const sensor_msgs::msg::Imu::SharedPtr imu_msg);
    queue<sensor_msgs::msg::Imu::SharedPtr> imuBuf;
    std::mutex mBufMutex;
};

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM3::System* pSLAM, ImuGrabber *pImuGb, const bool bClahe): mpSLAM(pSLAM), mpImuGb(pImuGb), mbClahe(bClahe){}

    void GrabImage(const sensor_msgs::msg::Image::SharedPtr img_msg);
    cv::Mat GetImage(const sensor_msgs::msg::Image::SharedPtr img_msg);

    void SyncWithImu();

    //method for setting ROS publisher
    void SetPub(rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub);

    queue<sensor_msgs::msg::Image::SharedPtr> img0Buf;
    std::mutex mBufMutex;

    ORB_SLAM3::System* mpSLAM;
    ImuGrabber *mpImuGb;
    //additional variables for publishing pose & broadcasting transform - https://roboticsknowledgebase.com/wiki/state-estimation/orb-slam2-setup/

    cv::Mat m1, m2;
    bool do_rectify, pub_tf, pub_pose;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr orb_pub;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    const bool mbClahe;
    cv::Ptr<cv::CLAHE> mClahe = cv::createCLAHE(3.0, cv::Size(8, 8));
};

class MonoInertial : public rclcpp::Node
{
public:
    MonoInertial(const std::string& vocabulary_path, const std::string& settings_path, bool do_equalize)
        : Node("Mono_Inertial")
    {
        RCLCPP_INFO(this->get_logger(), "Mono_Inertial node started");

        // Create SLAM system. It initializes all system threads and gets ready to process frames.
        SLAM_ = new ORB_SLAM3::System(vocabulary_path, settings_path, ORB_SLAM3::System::IMU_MONOCULAR, true);

        imugb_ = new ImuGrabber();
        igb_ = new ImageGrabber(SLAM_, imugb_, do_equalize);

        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("orb_pose", 100);

        // Maximum delay, 5 seconds
        sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/imu", 1000, [&](const sensor_msgs::msg::Imu::SharedPtr msg) { imugb_->GrabImu(msg); });

        sub_img0_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw", 100, [&](const sensor_msgs::msg::Image::SharedPtr msg) { igb_->GrabImage(msg); });

         // Initialize the transform broadcaster
        // Create publisher
        igb_->SetPub(pose_pub_);

        sync_thread_ = std::thread(&ImageGrabber::SyncWithImu, igb_);
    }

private:
    ORB_SLAM3::System* SLAM_;
    ImuGrabber* imugb_;
    ImageGrabber* igb_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img0_;
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


  if(argc==4)
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

//method for assigning publisher
void ImageGrabber::SetPub(rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub)
{
  orb_pub = pub;
}

void ImageGrabber::GrabImage(const sensor_msgs::msg::Image::SharedPtr img_msg)
{
  mBufMutex.lock();
  if (!img0Buf.empty())
    img0Buf.pop();
    img0Buf.push(img_msg);
    mBufMutex.unlock();
}

cv::Mat ImageGrabber::GetImage(const sensor_msgs::msg::Image::SharedPtr img_msg)
{
  // Copy the ros image message to cv::Mat.
  cv_bridge::CvImageConstPtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::MONO8);
  }
  catch (cv_bridge::Exception& e)
  {
    RCLCPP_ERROR(rclcpp::get_logger("Mono_Inertial"), "cv_bridge exception: %s", e.what());  }

  if(cv_ptr->image.type()==0)
  {
    return cv_ptr->image.clone();
  }
  else
  {
    RCLCPP_ERROR(rclcpp::get_logger("Mono_Inertial"), "Error image type");
    return cv_ptr->image.clone();
  }
}

void ImageGrabber::SyncWithImu()
{
  //while(1)
  while (rclcpp::ok())
  {
    cv::Mat im;
    double tIm = 0;
    if (!img0Buf.empty()&&!mpImuGb->imuBuf.empty())
    {
      tIm = img0Buf.front()->header.stamp.sec + img0Buf.front()->header.stamp.nanosec * 1e-9;
      if (tIm > mpImuGb->imuBuf.back()->header.stamp.sec + mpImuGb->imuBuf.back()->header.stamp.nanosec * 1e-9)
          continue;
      {
        this->mBufMutex.lock();
        im = GetImage(img0Buf.front());
        img0Buf.pop();
        this->mBufMutex.unlock();
      }

      vector<ORB_SLAM3::IMU::Point> vImuMeas;
      mpImuGb->mBufMutex.lock();
      if(!mpImuGb->imuBuf.empty())
      {
        // Load imu measurements from buffer
        vImuMeas.clear();
        while (!mpImuGb->imuBuf.empty() && mpImuGb->imuBuf.front()->header.stamp.sec +
                        mpImuGb->imuBuf.front()->header.stamp.nanosec * 1e-9 <= tIm)
        {
          double t = mpImuGb->imuBuf.front()->header.stamp.sec + mpImuGb->imuBuf.front()->header.stamp.nanosec * 1e-9;          
          cv::Point3f acc(mpImuGb->imuBuf.front()->linear_acceleration.x, mpImuGb->imuBuf.front()->linear_acceleration.y, mpImuGb->imuBuf.front()->linear_acceleration.z);
          cv::Point3f gyr(mpImuGb->imuBuf.front()->angular_velocity.x, mpImuGb->imuBuf.front()->angular_velocity.y, mpImuGb->imuBuf.front()->angular_velocity.z);
          vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc,gyr,t));
          mpImuGb->imuBuf.pop();
        }
      }
      mpImuGb->mBufMutex.unlock();
      if(mbClahe)
        mClahe->apply(im,im);

      cv::Mat T_, R_, t_ ;

      //stores return variable of TrackMonocular
      mpSLAM->TrackMonocular(im, tIm, vImuMeas);

      //this line seems to break things
      //aftermarket publish function

      if (pub_tf || pub_pose)
      {    
        if (!(T_.empty())) {

          cv::Size s = T_.size();
          if ((s.height >= 3) && (s.width >= 3)) {
            R_ = T_.rowRange(0,3).colRange(0,3).t();
            t_ = -R_*T_.rowRange(0,3).col(3);
            std::vector<float> q = ORB_SLAM3::Converter::toQuaternion(R_);
            float scale_factor=1.0;
            tf2::Transform tf_transform;
            geometry_msgs::msg::TransformStamped tf_msg;
            tf_msg.header.stamp = rclcpp::Time(tIm);
            tf_msg.header.frame_id = "world";
            tf_msg.child_frame_id = "ORB_SLAM3_MONO_INERTIAL";
            //geometry_msgs::msg::Transform tf2;

            //tf2::Transform tf_msg1;
            tf_msg.transform.translation.x = t_.at<float>(0, 0) * scale_factor;
            tf_msg.transform.translation.y = t_.at<float>(0, 1) * scale_factor;
            tf_msg.transform.translation.z = t_.at<float>(0, 2) * scale_factor;
            tf_msg.transform.rotation.x = q[0];
            tf_msg.transform.rotation.y = q[1];
            tf_msg.transform.rotation.z = q[2];
            tf_msg.transform.rotation.w = q[3];
           
          if (pub_pose)
            {
              geometry_msgs::msg::PoseStamped pose;
              //pose.header.stamp = img0Buf.front()->header.stamp;
              pose.header.frame_id ="ORB_SLAM3_MONO_INERTIAL";
              tf2::toMsg(tf_transform, pose.pose);
              orb_pub->publish(pose);
            }
            
          }
        }
      }
    }

    std::chrono::milliseconds tSleep(1);
    std::this_thread::sleep_for(tSleep);
  }
}

void ImuGrabber::GrabImu(const sensor_msgs::msg::Imu::SharedPtr imu_msg)
{
  mBufMutex.lock();
  imuBuf.push(imu_msg);
  mBufMutex.unlock();
  return;
}
