#include "kernel_lib/utils/FileManager.hpp"

namespace kernel_lib {
    namespace utils {
        FileManager::FileManager(const std::string& file_to_write)
        {
            setFile(file_to_write);
        }

        FileManager::FileManager() {}

        FileManager::~FileManager() {}

        std::string FileManager::fileName() const
        {
            return _name;
        }

        std::string FileManager::filePath() const
        {
            return _path;
        }

        void FileManager::setFile(const std::string& file_to_write)
        {
            _path = path(file_to_write);

            if (!_path.empty()) {
                if (!isDirectory(_path))
                    mkpath(_path);
            }

            _name = filename(file_to_write);

            _open = true;
        }
    } // namespace utils
} // namespace kernel_lib