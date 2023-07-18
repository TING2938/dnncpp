#pragma once

#include "core.hpp"

namespace dnncpp
{
class Sigmoid : public ActivationBase
{
	
public:

	Eigen::ArrayXd forward(const Eigen::ArrayXd& x) override
    {
        return (x.exp() + 1).inverse();
	}

	Eigen::ArrayXd derivative(const Eigen::ArrayXd& x) override
    {
        return x * (1 - x);
	}
};
}

