#include <format>
#include <iostream>

#include "dnncpp/nn/network.hpp"

auto himmelblau(const Eigen::ArrayX2d& x)
{
    return (x.col(0).pow(2) + x.col(1) - 11).pow(2) + (x.col(0) + x.col(1).pow(2) - 7).pow(2);
}

Eigen::ArrayXd func(const Eigen::ArrayXd& x)
{
    return 2.3 * x + 5.6;
}

int main()
{
    // prepare datasets
    int N = 5000;
    Eigen::ArrayXd x;
    x.setRandom(N);

    auto y = func(x);

    // std::cout << x << "\n\n" << y << std::endl;

    dnncpp::Network nn;
    nn.add_layer(dnncpp::Layer(1, 10, dnncpp::Activation::Sigmoid));
    nn.add_layer(dnncpp::Layer(10, 20, dnncpp::Activation::Sigmoid));
    nn.add_layer(dnncpp::Layer(20, 1, dnncpp::Activation::Linear));

    for (int i_epochs = 0; i_epochs < 1000; i_epochs++) {
        double losses = 0;
        for (int i = 0; i < x.size(); i++) {
            auto x_i    = x.row(i);
            auto y_i    = y.row(i);
            auto y_pred = nn.forward(x_i);
            nn.backward(y_pred, y_i);
            nn.update(x_i, 0.001);
            losses += (y_pred - y_i).square().mean();
        }
        losses /= x.size();
        if (i_epochs % 50 == 0)
            std::cout << std::format("i_epochs={}, loss={}", i_epochs, losses) << std::endl;
    }
}
