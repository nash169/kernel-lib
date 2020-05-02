#include "kernel_lib/utils/FileManager.hpp"

namespace kernel_lib {
    namespace utils {
        FileManager::FileManager(const std::string& file)
        {
            setFile(file);
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

        void FileManager::setFile(const std::string& file)
        {
            _path = path(file);

            if (!_path.empty()) {
                if (!isDirectory(_path))
                    mkpath(_path);
            }

            _name = filename(file);

            _open = false;
        }
    } // namespace utils
} // namespace kernel_lib