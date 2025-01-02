#include "Segmenter.h"
// --------------------------- 测试 main 函数 ---------------------------
int main() {
    try {
        // 1. 创建 Segmenter 对象，并自动完成配置读取 & TensorRT 初始化
        Segmenter segmenter("../config.ini");

        // 2. 执行推理
        // segmenter.doInference(2);

        std::map<std::string, std::string> configMap = segmenter.loadConfig("../config.ini");
        bool saveProcessImage= (configMap["setting.saveProcessImage"] == "true");
        bool saveOutputImage = (configMap["setting.saveOutputImage"] == "true");;
        int loopCount = std::stoi(configMap["setting.loopCount"]);

        std::cout << "[设置项] 保存预处理图像: " << (saveProcessImage ? "是" : "否") << std::endl;
        std::cout << "[设置项] 保存输出图像: " << (saveOutputImage ? "是" : "否") << std::endl;
        std::cout << "[设置项] 循环次数: " << loopCount << std::endl;

        segmenter.doInferenceBatch(saveProcessImage,saveOutputImage,loopCount);

        std::cout << "[main] 推理完成" << std::endl;
    } 
    catch (const std::exception& e) {
        std::cerr << "[main] 错误: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
