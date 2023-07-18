#pragma once

#include <memory>

#include "core.hpp"
#include "sigmoid.hpp"


namespace dnncpp
{
	inline auto get_activation(Activation type) {
        switch (type) {
            case Activation::Sigmoid:
                return std::make_shared<Sigmoid>();
            case Activation::Relu:
                break;
            case Activation::Tanh:
                break;
            case Activation::Softmax:
                break;
            default:
                break;
        }
        return std::make_shared<Sigmoid>();
    }
}
