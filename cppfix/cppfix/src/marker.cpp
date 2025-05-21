#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/PointCloud2.hpp"
#include "visualization_msgs/msg/Marker.hpp"
#include "graphing/msg/XYcord.hpp"

using namespace std::placeholders;

class Marking : public rclcpp::Node {
  public:
    Marking()
        : Node("zed_depth_tutorial") {

        rclcpp::QoS depth_qos(20);
        depth_qos.keep_last(20);
        depth_qos.reliable();
        depth_qos.durability_volatile();
        uv = create_subscription<graphing::msg::Xycord>("cone_placed",depth_qos,std::bind(&Marking::depthCallback , this, _1) );
        mDepthSub = create_subscription<sensor_msgs::msg::PointCloud2>("/zed/zed_node/pointcloud", depth_qos,std::bind(&Marking::depthCallback2, this, _1) );
        mark = create_publisher<visualization_msgs::msg::Marker>("marking",depth_qos, 10)
    }

  protected:
    void depthCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
     
        float* depths = (float*)(&msg->data[0]);

        // Image coordinates of the center pixel
        int u = msg->width / 2;
        int v = msg->height / 2;

        // Linear index of the center pixel
        int centerIdx = u + msg->width * v;

        // Output the measure
        RCLCPP_INFO(get_logger(), "Center distance : %g m", depths[centerIdx]);
    }
    void depthCallback2(const graphing::msg::XYcord::SharedPtr msg) {
     
        float* depths = (float*)(&msg->data[0]);

        // Image coordinates of the center pixel
        int u = msg->width / 2;
        int v = msg->height / 2;

        // Linear index of the center pixel
        int centerIdx = u + msg->width * v;

        // Output the measure
        RCLCPP_INFO(get_logger(), "Center distance : %g m", depths[centerIdx]);
    }

  private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr mDepthSub;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr mark; 
    rclcpp::Subscription<graphing::msg::XYcord>::SharedPtr uv;
};

// The main function
int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);

    auto depth_node = std::make_shared<Marking>();

    rclcpp::spin(depth_node);
    rclcpp::shutdown();
    return 0;
}

