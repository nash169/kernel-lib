#ifndef KERNELLIB_TOOLS_FILEMANAGER_HPP
#define KERNELLIB_TOOLS_FILEMANAGER_HPP

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
            // Contructor
            FileManager(const std::string& file_to_write);
            FileManager();

            // Destroyer
            ~FileManager();

            // Get file & path
            std::string fileName() const;
            std::string filePath() const;

            // Set file & path
            void setFile(const std::string& file_to_write);

            // Write file
            template <typename VarType>
            void write(VarType var)
            {
                if (_open)
                    _file.open(join(_path, _name), std::ios::out);

                _file << var << std::endl;

                _file.close();
                _open = true;
            }

            template <typename VarType, typename... Args>
            void write(VarType var, Args... args)
            {
                if (_open) {
                    _file.open(join(_path, _name), std::ios::out);
                    _open = false;
                }

                _file << var << "\n"
                      << std::endl;

                write(args...);
            }

            // Apppend file
            template <typename VarType>
            void append(VarType var)
            {
                if (_open)
                    _file.open(join(_path, _name), std::ios::out | std::ios::app);

                _file << var << "\n"
                      << std::endl;

                _file.close();
                _open = true;
            }

            template <typename VarType, typename... Args>
            void append(VarType var, Args... args)
            {
                if (_open) {
                    _file.open(join(_path, _name), std::ios::out | std::ios::app);
                    _open = false;
                }

                _file << var << std::endl;

                append(args...);
            }

            // Read file
            template <typename VarType>
            VarType read()
            {
                _file.open(join(_path, _name), std::ios::in);
                // to finish
            }

            // For solving the problem of write Vec in sequence use static variable here

        protected:
            bool _open;
            std::fstream _file;
            std::string _path, _name;
        };
    } // namespace tools
} // namespace kernel_lib

#endif // KERNEL_LIB_TOOLS_FILEMANAGER_HPP