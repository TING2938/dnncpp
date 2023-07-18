#pragma once

#include <iostream>
#include <memory>

#include <Eigen/Dense>

#include "dnncpp/activation/activation.hpp"

namespace dnncpp {
class Layer
{
public:
    Layer(int n_input, int n_output, Activation activation)
    {
        std::cout << "create layer" << std::endl;
        this->weights.setRandom(n_input, n_output);
        this->bias.setRandom(n_output);
        this->activation = get_activation(activation);
    }

    auto forward(const Eigen::ArrayXd& x) {
        this->last_activation = this->activation->forward((this->weights.matrix().transpose() * x.matrix()).array() + this->bias);
        return this->last_activation;
    }

public:
    Eigen::ArrayXXd weights; // [in, out]
    Eigen::ArrayXd bias; // [out, 1]
    Eigen::ArrayXd delta; // [out, 1]
    Eigen::ArrayXd last_activation; // [out, 1]
    std::shared_ptr<ActivationBase> activation;
};

}
