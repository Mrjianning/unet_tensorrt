#ifndef SEGMENTER_H
#define SEGMENTER_H

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

// 命名空间 
using namespace nvinfer1;

/**
 * @brief 自定义的 TensorRT 日志器
 */
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

/**
 * @brief Segmenter 类
 */
class Segmenter {
public:
    /**
     * @brief 构造函数
     * @param configPath INI 配置文件路径
     */
    Segmenter(const std::string& configPath);

    /**
     * @brief 析构函数
     */
    ~Segmenter();

    /**
     * @brief 执行推理的对外接口
     * @param loopCount  推理执行次数
     */
    void doInference(int loopCount = 1);

    /**
     * @brief 多batch推理
     * @param saveProcessImage  保存预处理图片
     * @param saveOutputImage   保存输出图片
     * @param loopCount         循环次数
     */
    void doInferenceBatch(bool saveProcessImage,bool saveOutputImage,int loopCount);
    
    /**
     * @brief 从 INI 文件加载配置
     * @param configPath 配置文件路径
     */
    std::map<std::string, std::string> loadConfig(const std::string& configPath);

     /**
     * @brief 文件夹中加载图片
     * @param folderPath 图片路径
     */
    std::vector<cv::Mat> readImagesFromFolder(const std::string& folderPath);

    /**
     * @brief 从文件加载引擎数据
     */
    static std::vector<char> loadEngineFile(const std::string& fileName);

    /**
     * @brief 初始化 TensorRT：创建 runtime、反序列化 engine、创建 context
     */
    void initializeTensorRT();

    /**
     * @brief 推理初始化（绑定维度、申请GPU缓冲区等）
     */
    bool inferInit();

    /**
     * @brief GPU 缓冲区分配
     */
    static void* allocateDeviceBuffer(size_t size);

    /**
     * @brief 打印 TensorRT 的维度信息
     */
    static void printDims(const std::string& name, const Dims& dims);

    /**
     * @brief 图像预处理 (resize -> normalize -> HWC->CHW)
     * @param inputImage 原始图像
     * @param inputDims  模型输入维度
     * @param saveSteps  是否保存中间结果，默认 false
     * @return 预处理后的图像（CHW 格式）
     */
    static cv::Mat preprocessImage(const cv::Mat& inputImage, const Dims& inputDims, bool saveSteps, const std::string& savePrefix);

    /**
     * @brief 图像批量预处理 (resize -> normalize -> HWC->CHW)
     * @param images     原始图像
     * @param inputDims  模型输入维度
     * @param batchSize  批量大小
     * @return 预处理后的图像（CHW 格式）
     */
    static cv::Mat preprocessBatch(const std::vector<cv::Mat> &images, const Dims& inputDims, int batchSize,bool saveMergedBatchImage);
   
    /**
     * @brief 进行一次推理
     */
    void runInference(int sroneNum,const cv::Mat& inputImage, cv::Mat& outputMask);

    /**
     * @brief 后处理：阈值化并保存结果
     */
    static void postprocessOutput(const cv::Mat& outputMask, const std::string& outputPath,bool saveOutputImage);

public:
    
    // 配置项
    std::string mEnginePath;   // 模型引擎路径
    std::string mImageFolder;  // 输入图像文件夹路径
    std::string mImagePath;    // 输入图像路径（也可在类外自定义传入）
    std::string mOutputPath;   // 输出结果路径
    std::string mOutputFolder; // 输出结果文件夹路径
    
    // TensorRT 核心组件
    IRuntime* mRuntime;
    ICudaEngine* mEngine;
    IExecutionContext* mContext;

    // 推理相关
    Dims mInputDims;
    Dims mOutputDims;
    std::vector<void*> mBuffers;
    size_t mInputSize;
    size_t mOutputSize;
    int mInputIndex;
    int mOutputIndex;
    cudaStream_t mStream;

    std::vector<nvinfer1::Dims> m_inputDimsVec;
    std::vector<nvinfer1::Dims> m_outputDimsVec;
    int max_BatchSize=1;
 
};

#endif // SEGMENTER_H
