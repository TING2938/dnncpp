#pragma once

#include <memory>

#include "core.hpp"
#include "sigmoid.hpp"
#include "linear.hpp"


namespace dnncpp
{
	inline std::shared_ptr<ActivationBase> get_activation(Activation type) {
        switch (type) {
            case Activation::Sigmoid:
                return std::make_shared<Sigmoid>();
            case Activation::Relu:
                break;
            case Activation::Tanh:
                break;
            case Activation::Softmax:
                break;
            case Activation::Linear:
                return std::make_shared<Linear>();
                break;
            default:
                break;
        }
        return std::make_shared<Sigmoid>();
    }
}
