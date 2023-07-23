#pragma once

#include "core.hpp"

namespace dnncpp
{
class Linear : public ActivationBase
{
public:
    Eigen::ArrayXd forward(const Eigen::ArrayXd& x) override
    {
        return x;
    }

    Eigen::ArrayXd derivative(const Eigen::ArrayXd& x) override
    {
        return Eigen::ArrayXd::Ones(x.size());
    }
};
}  // namespace dnncpp
