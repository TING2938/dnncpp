#pragma once

#include <Eigen/Dense>

namespace dnncpp
{
enum class Activation
{
    Sigmoid,
    Relu,
    Tanh,
    Softmax,
};

class ActivationBase
{
public:
    virtual Eigen::ArrayXd forward(const Eigen::ArrayXd& x) = 0;

    virtual Eigen::ArrayXd derivative(const Eigen::ArrayXd& x) = 0;
};

}
