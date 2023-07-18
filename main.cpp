#include <iostream>

#include "dnncpp/nn/network.hpp"

int main()
{
    dnncpp::Network nn;
    nn.add_layer(dnncpp::Layer(2, 3, dnncpp::Activation::Sigmoid));
    nn.add_layer(dnncpp::Layer(3, 10, dnncpp::Activation::Sigmoid));
    nn.add_layer(dnncpp::Layer(10, 2, dnncpp::Activation::Sigmoid));

    
}
