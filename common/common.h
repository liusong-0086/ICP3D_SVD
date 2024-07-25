#ifndef COMMON_H_
#define COMMON_H_

#include <opencv2/opencv.hpp>

#ifndef M_PI
#define M_PI 1415926535897932384626433832795
#endif



template <typename FuncT>
void evaluate_and_call(FuncT &&func, int run_num)
{
    double total_time = 0.f;
    for (int i = 0; i < run_num; ++i)
    {
        auto t1 = std::chrono::high_resolution_clock::now();
        func();
        auto t2 = std::chrono::high_resolution_clock::now();
        total_time +=  std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() * 1000;;
    }

    std::cout << "run cost: " << total_time / 1000.f << std::endl;
}

#endif  // COMMON_H_
