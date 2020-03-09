#ifndef KERNEL_LIB_TOOLS_TIMER
#define KERNEL_LIB_TOOLS_TIMER

#include <chrono>
#include <iostream>

namespace kernel_lib {
    namespace tools {
        class Timer {
        public:
            Timer()
            {
                _start_time = std::chrono::high_resolution_clock::now();
            }
            ~Timer()
            {
                stop();
            }

            void stop()
            {
                auto end_time = std::chrono::high_resolution_clock::now();

                auto start = std::chrono::time_point_cast<std::chrono::microseconds>(_start_time).time_since_epoch().count();
                auto end = std::chrono::time_point_cast<std::chrono::microseconds>(end_time).time_since_epoch().count();

                auto duration = end - start;
                double ms = duration * 0.001;

                std::cout << duration << "us (" << ms << "ms)" << std::endl;
            }

        private:
            std::chrono::time_point<std::chrono::high_resolution_clock> _start_time;
        };
    } // namespace tools
} // namespace kernel_lib

#endif // KERNEL_LIB_TOOLS_TIMER