#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>

import layer;

using namespace Eigen;

int main()
{
    dnncpp::Layer layer;

    Tensor<int, 3> random(2, 3, 7);
    random.setRandom();

    TensorMap<Tensor<const int, 3>> constant(random.data(), 2, 3, 7);
    Tensor<int, 3> result(2, 3, 7);
    result = constant;

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 7; ++k) {
                std::cout << result(i, j, k) << std::endl;
            }
        }
    }
}
