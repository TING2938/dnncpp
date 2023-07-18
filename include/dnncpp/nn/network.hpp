#pragma once

#include <vector>
#include "dnncpp/layer/layer.hpp"

namespace dnncpp
{
class Network
{
public:
    Network() {}

    void add_layer(const Layer& layer)
    {
        this->layers.push_back(layer);
    }

    auto forward(const Eigen::ArrayXd& x)
    {
        Eigen::ArrayXd h = x;
        for (auto&& layer : this->layers) {
            h = layer.forward(h);
        }
        return h;
    }

    auto backward(const Eigen::ArrayXd& output, const Eigen::ArrayXd& y)
    {
        if (this->layers.empty())
            return;

        // output layer
        auto& last_layer = this->layers.back();
        last_layer.delta = last_layer.activation->derivative(output) * (output - y);  // [last_out, 1]

        if (this->layers.size() > 1) {
            for (int i = this->layers.size() - 2; i >= 0; i--) {
                auto& this_layer = this->layers[i];
                this_layer.delta = (this_layer.weights.matrix() * this->layers[i + 1].delta.matrix()).array() *
                                   (this_layer.activation->derivative(this_layer.last_activation));  // [out, 1]
            }
        }
    }

    auto update(const Eigen::ArrayXd& x, double learning_rate)
    {
        // update weights
        for (int i = 0; i < this->layers.size(); i++)
        {
            auto& layer = this->layers[i];
            layer.weights -= learning_rate * layer.delta * (i == 0 ? x : this->layers[i - 1].last_activation);
        }
    }

private:
    std::vector<Layer> layers;
};
}  // namespace dnncpp