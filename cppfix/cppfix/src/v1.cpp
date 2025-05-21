#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"


using namespace std::placeholders;

class Subpubimage : public rclcpp::Node {
  public:
    Subpubimage()
        : Node("fixer") {

        /* Note: it is very important to use a QOS profile for the subscriber that is compatible
         * with the QOS profile of the publisher.
         * The ZED component node uses a default QoS profile with reliability set as "RELIABLE"
         * and durability set as "VOLATILE".
         * To be able to receive the subscribed topic the subscriber must use compatible
         * parameters.
         */

        // https://github.com/ros2/ros2/wiki/About-Quality-of-Service-Settings

        rclcpp::QoS fix_qos(20);
        fix_qos.keep_last(20);
        fix_qos.reliable();
        fix_qos.durability_volatile();

      
        left_sub = this->create_subscription<sensor_msgs::msg::Image>("/zed/zed_node/left/image_rect_color", fix_qos, std::bind(&Subpubimage::imageLeftRectifiedCallback, this, _1) );
        publisher_ = this->create_publisher<sensor_msgs::msg::Image>("chaljapls",fix_qos);
        RCLCPP_INFO(get_logger(), "started");
        
     
    }
    
   protected:
  
   void imageLeftRectifiedCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
    	publisher_->publish(*msg);
    	RCLCPP_INFO(get_logger(), "publishing");
        
    }

  private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr left_sub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    
};

// The main function
int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);

    auto subpub_node = std::make_shared<Subpubimage>();

    rclcpp::spin(subpub_node);
    rclcpp::shutdown();
    return 0;
}

