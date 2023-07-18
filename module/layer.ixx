module;
#include <iostream>
#include <Eigen/Dense>

export module layer;

export namespace dnncpp {
class Layer
{
public:
    Layer() {
        std::cout << "create layer" << std::endl;
    }
};

}
