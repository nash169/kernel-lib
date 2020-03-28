#ifndef KERNEL_LIB_TOOLS_FILEMANAGER_HPP
#define KERNEL_LIB_TOOLS_FILEMANAGER_HPP

#include <Corrade/Utility/Directory.h>
#include <Eigen/Core>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using namespace Corrade::Utility::Directory;

namespace kernel_lib {
    namespace tools {
        class FileManager {
        public:
            FileManager(const std::string& path_file, const std::string& mode = "", const bool append = false)
            {
                _path_to_file = path(path_file);

                if (!_path_to_file.empty()) {
                    if (!isDirectory(_path_to_file))
                        mkpath(_path_to_file);
                }

                _file_name = filename(path_file);

                if (!mode.compare("w")) {
                    _file.open(path_file, (append) ? std::ios::out | std::ios::app : std::ios::out);
                }
                else if (!mode.compare("r"))
                    _file.open(path_file, std::ios::in);
                else if (!mode.compare("w/r"))
                    _file.open(path_file, (append) ? std::ios::in | std::ios::out | std::ios::app : std::ios::in | std::ios::out);

                _mode = mode;
                _append = append;
            }

            ~FileManager()
            {
                _file.close();
                // appendString(join(_path_to_file, _file_name), "File edited! Corrade fun");
            }

            template <typename EigenType>
            void write(EigenType var)
            {
                _file << var << std::endl;
            }

            template <typename EigenType, typename... Args>
            void write(EigenType first, Args... args)
            {
                _file << first << "\n"
                      << std::endl;
                write(args...);
            }

            // For solving the problem of write Vec in sequence use static variable here

        protected:
            Eigen::MatrixXd mat;
            std::fstream _file;
            std::string _path_to_file, _file_name, _mode;
            bool _append;
        };
    } // namespace tools
} // namespace kernel_lib

#endif // KERNEL_LIB_TOOLS_FILEMANAGER_HPP