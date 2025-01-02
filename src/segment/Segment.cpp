#include "Segmenter.h"
#include <filesystem>

Segmenter::Segmenter(const std::string& configPath)
    : mEngine(nullptr)
    , mRuntime(nullptr)
    , mContext(nullptr)
    , mStream(nullptr)
    , mInputSize(0)
    , mOutputSize(0)
    , mInputIndex(-1)
    , mOutputIndex(-1)
{
    // 1. 加载INI配置  
    std::map<std::string, std::string> configMap = loadConfig(configPath);
    mEnginePath = configMap["model.engine_path"];
    mImagePath = configMap["input.image_path"];
    mImageFolder = configMap["input.image_folder"];
    mOutputPath = configMap["output.outputPath"];
    mOutputFolder = configMap["output.outputFolder"];

    std::cout << "[配置读取] 模型路径: " << mEnginePath << std::endl;
    std::cout << "[配置读取] 图像目录: " << mImagePath << std::endl;
    std::cout << "[配置读取] 图像路径: " << mImageFolder << std::endl;
    std::cout << "[配置读取] 图像保存路径: " << mOutputPath << std::endl;
    std::cout << "[配置读取] 结果输出路径: " << mOutputFolder << std::endl;

    // 2. 初始化TensorRT      
    initializeTensorRT();  
    // 3. 推理相关初始化        
    inferInit();                  
}

Segmenter::~Segmenter() {
    // 释放 GPU 内存
    for (void* buf : mBuffers) {
        if (buf) cudaFree(buf);
    }
    mBuffers.clear();

    if (mStream) {
        cudaStreamDestroy(mStream);
        mStream = nullptr;
    }

    // 释放 TensorRT 相关资源
    if (mContext) {
        mContext->destroy();
        mContext = nullptr;
    }
    if (mEngine) {
        mEngine->destroy();
        mEngine = nullptr;
    }
    if (mRuntime) {
        mRuntime->destroy();
        mRuntime = nullptr;
    }

    std::cout << "[Segmenter] 资源已释放" << std::endl;
}

std::vector<cv::Mat> Segmenter::readImagesFromFolder(const std::string& folderPath) {
    std::vector<cv::Mat> images;

    std::cout << "[图像读取] 开始读取图像..." << std::endl;
    std::cout << "[图像读取] 读取图像路径: " << folderPath << std::endl;

    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        if (entry.is_regular_file()) {
            std::string imagePath = entry.path().string();
            cv::Mat readImage = cv::imread(imagePath);

            if (readImage.empty()) {
                throw std::runtime_error("[图像读取] 读取图像失败: " + imagePath);
                continue; // 跳过无法读取的文件
            }

            images.push_back(readImage);
        }
    }

    if (images.empty()) {
        std::cerr << "[图像读取] 没有找到任何图像文件。" << std::endl;
    }

    return images;
}

void Segmenter::doInference(int loopCount) {

    std::string imagePath=mImagePath;
    std::string outputPath=mOutputPath;

    // 1. 读取图像
    std::cout << "[图像读取] 开始读取图像..." << std::endl;
    std::cout << "[图像读取] 读取图像路径: " << imagePath << std::endl;

    cv::Mat readImage = cv::imread(imagePath);
    if (readImage.empty()) {
        throw std::runtime_error("[图像读取] 读取图像失败: " + imagePath);
    }

    // 2. 预处理图像 (HWC -> CHW, resize, normalize)
    cv::Mat inputImage = preprocessImage(readImage, mInputDims, true,"image");

    // 3. 多次推理
    for (int i = 0; i < loopCount; ++i) {
        // 构造一个输出Mat，用于保存推理结果
        cv::Mat outputMask(mOutputDims.d[3], mOutputDims.d[2], CV_32FC1);

        // 执行一次推理
        runInference(1,inputImage, outputMask);

        // 4. 后处理并保存输出结果
        postprocessOutput(outputMask, outputPath,true);
    }
}

void Segmenter::doInferenceBatch(bool saveProcessImage,bool saveOutputImage,int loopCount) {

    std::string imagePath=mImageFolder;
    std::string outputPath=mOutputPath;

    // 1. 读取图像
    std::vector<cv::Mat> images=readImagesFromFolder(imagePath);
    if (images.empty()) {
        throw std::runtime_error("[批量推理] 没有可用的图像进行推理");
    }

    // 2. 预处理图像 (HWC -> CHW, resize, normalize)
    int  batchSize = images.size();
    cv::Mat batchInput = preprocessBatch(images, mInputDims, batchSize,saveProcessImage);

    // 判断输入图像是否为空
    if (batchInput.empty()) {
        throw std::runtime_error("[批量推理] 预处理图像失败");
    }

    // 3. 创建批量输出容器
    int outputVolume = m_outputDimsVec[0].d[1] * m_outputDimsVec[0].d[2] * m_outputDimsVec[0].d[3];
    cv::Mat batchOutput(batchSize * outputVolume, 1, CV_32FC1);

    // 4. 执行批量推理
    for (size_t i = 0; i < loopCount; i++)
    {
        runInference(batchSize, batchInput, batchOutput);
    }
    
    // 5. 后处理每个图像
    for (int i = 0; i < batchSize; ++i) {
        // 每个 batch 的输出数据起始地址
        float* batchStart = batchOutput.ptr<float>() + i * outputVolume;

        // 创建单个 batch 的 Mat
        cv::Mat outputImage(m_outputDimsVec[0].d[2], m_outputDimsVec[0].d[3], CV_32FC1, batchStart);

        // 构建输出路径
        std::string outputFilePath = mOutputFolder + "/batch_" + std::to_string(i) + ".jpg";

        // 后处理并保存
        postprocessOutput(outputImage, outputFilePath,saveOutputImage);
    }
}

std::map<std::string, std::string> Segmenter::loadConfig(const std::string& configPath) {
    std::cout << "[Segmenter] 读取配置文件: " << configPath << std::endl;

    boost::property_tree::ptree pt;
    boost::property_tree::ini_parser::read_ini(configPath, pt);

    // 用于存储配置键值对的 map
    std::map<std::string, std::string> configMap;

    // 遍历配置文件，提取键值对
    for (const auto& section : pt) {
        for (const auto& kv : section.second) {
            std::string key = section.first + "." + kv.first;
            std::string value = kv.second.data();
            configMap[key] = value;
            std::cout << "[配置读取] " << key << ": " << value << std::endl;
        }
    }

    return configMap;
}


std::vector<char> Segmenter::loadEngineFile(const std::string& fileName) {
    std::ifstream file(fileName, std::ios::binary);
    if (!file) {
        throw std::runtime_error("无法打开引擎文件: " + fileName);
    }
    return std::vector<char>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

void Segmenter::initializeTensorRT() {
    std::cout << "[Segmenter] 初始化 TensorRT..." << std::endl;
    // 1. 加载引擎文件
    std::vector<char> engineData = loadEngineFile(mEnginePath);

    // 2. 创建 InferRuntime
    static Logger gLogger;  
    mRuntime = createInferRuntime(gLogger);
    if (!mRuntime) throw std::runtime_error("[初始化 TensorRT] 创建 TensorRT 运行时失败");

    // 3. 反序列化引擎
    mEngine = mRuntime->deserializeCudaEngine(engineData.data(), engineData.size(), nullptr);
    if (!mEngine) throw std::runtime_error("[初始化 TensorRT] 反序列化引擎失败");

    // 4. 创建执行上下文
    mContext = mEngine->createExecutionContext();
    if (!mContext) throw std::runtime_error("[初始化 TensorRT] 创建执行上下文失败");

    std::cout << "[初始化 TensorRT] TensorRT 初始化成功!" << std::endl;
}

bool Segmenter::inferInit() {
    try {

        // 获取bingdings数量
	    int bindingsNum = mEngine->getNbIOTensors();
        std::cout << "[推理初始化] 获取绑定数量: " << bindingsNum << std::endl;

        // 获取绑定索引
        mInputIndex = mEngine->getBindingIndex("input");
        mOutputIndex = mEngine->getBindingIndex("output");

        // 获取绑定维度
        mInputDims = mEngine->getBindingDimensions(mInputIndex);
        mOutputDims = mEngine->getBindingDimensions(mOutputIndex);
        
        // 打印维度信息
        printDims("[模型维度] 输入维度", mInputDims);
        printDims("[模型维度] 输出维度", mOutputDims);
    
        // 如果是动态batch维度，需要设置
        if (mInputDims.d[0] < 0) {

            char const* bindingName = mEngine->getIOTensorName(mInputIndex);
            std::cout << "[模型维度] bindingName : " <<bindingName << std::endl;

            int profileNum = mEngine->getNbOptimizationProfiles();
            std::cout << "[模型维度] Number of optimization profiles: " << profileNum << std::endl;

            if (profileNum > 0)
            {
                nvinfer1::Dims dimsMax = mEngine->getProfileShape(bindingName, 0, nvinfer1::OptProfileSelector::kMAX);
                max_BatchSize = dimsMax.d[0];
                std::cout << "[模型维度] input Maximum batch size: " << max_BatchSize << std::endl;
            }

            // 设置input、output的最大batchsize
            mInputDims.d[0] = max_BatchSize; 
            mOutputDims.d[0]= max_BatchSize;

            // 存储输入输出维度
            m_inputDimsVec.emplace_back(mInputDims);
            m_outputDimsVec.emplace_back(mOutputDims);

            printDims("[模型维度] 设置后的输入维度", mInputDims);
            printDims("[模型维度] 设置后的输出维度", mOutputDims);
        }

        // 计算输入输出的大小
        mInputSize = mInputDims.d[1] * mInputDims.d[2] * mInputDims.d[3] * sizeof(float);
        mOutputSize = mOutputDims.d[1] * mOutputDims.d[2] * mOutputDims.d[3] * sizeof(float);

        size_t totalInputSize = max_BatchSize * mInputDims.d[1] * mInputDims.d[2] * mInputDims.d[3] * sizeof(float);
        size_t totalOutputSize = max_BatchSize * mOutputDims.d[1] * mOutputDims.d[2] * mOutputDims.d[3] * sizeof(float);

        // 分配缓冲区 (input + output)
        mBuffers.resize(2);
        mBuffers[mInputIndex] = allocateDeviceBuffer(totalInputSize);
        mBuffers[mOutputIndex] = allocateDeviceBuffer(totalOutputSize);

        // 创建 CUDA 流
        cudaStreamCreate(&mStream);

        std::cout << "[推理初始化] 初始化成功!" << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "[推理初始化] 错误: " << e.what() << std::endl;
        return false;
    }
}

void* Segmenter::allocateDeviceBuffer(size_t size) {
    void* devicePtr;
    if (cudaMalloc(&devicePtr, size) != cudaSuccess) {
        throw std::runtime_error("无法分配 GPU 缓冲区");
    }
    return devicePtr;
}

void Segmenter::printDims(const std::string& name, const Dims& dims) {
    std::cout << name << ": [";
    for (int i = 0; i < dims.nbDims; ++i) {
        std::cout << dims.d[i];
        if (i < dims.nbDims - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

cv::Mat Segmenter::preprocessImage(const cv::Mat& inputImage, const Dims& inputDims, bool saveSteps, const std::string& savePrefix) 
{
    cv::Mat imgCopy = inputImage.clone();

    std::string fullSavePrefix = "saveImage/" + savePrefix;

    // 1. 通道处理
    if (imgCopy.channels() == 1) {
        cv::cvtColor(imgCopy, imgCopy, cv::COLOR_GRAY2BGR);
    } else if (imgCopy.channels() != 3) {
        throw std::runtime_error("[图像预处理] 不支持的图像格式，仅支持1或3通道图像。");
    }

    // 保存输入原图
    if (saveSteps) {
        cv::imwrite(fullSavePrefix + "_1-input_image_original.jpg", imgCopy);
    }

    // 2. 调整图像大小 (inputDims.d[3], inputDims.d[2])
    cv::Mat resizedImage;
    cv::resize(imgCopy, resizedImage, cv::Size(inputDims.d[3], inputDims.d[2]), 0, 0, cv::INTER_AREA);

    // 保存resize后的图像
    if (saveSteps) {
        cv::imwrite(fullSavePrefix + "_2-input_image_resized.jpg", resizedImage);
    }

    // 3. 归一化到 [0,1]
    cv::Mat normalizedImage;
    resizedImage.convertTo(normalizedImage, CV_32FC3, 1.0f / 255.0f);

    // 保存归一化后的图像（需要先转回 8U）
    if (saveSteps) {
        cv::Mat normalized8U;
        normalizedImage.convertTo(normalized8U, CV_8UC3, 255.0f);
        cv::imwrite(fullSavePrefix + "_3-input_image_normalized.jpg", normalized8U);
    }

    // 4. HWC -> CHW
    std::vector<cv::Mat> chwChannels(3);
    cv::split(normalizedImage, chwChannels);

    // 将三个通道垂直拼接 (vconcat) 到一起
    cv::Mat chwImage;
    cv::vconcat(chwChannels, chwImage);

    return chwImage;
}

// 预处理多个图像
cv::Mat Segmenter::preprocessBatch(const std::vector<cv::Mat>& images, const Dims& inputDims, int batchSize, bool saveMergedBatchImage) {

    std::cout << "[图像预处理] 多batch预处理 " << images.size() << " images..." << std::endl;

    // 确保 batchSize 与输入图像数量一致
    if (batchSize != images.size()) {
        throw std::runtime_error("[图像预处理] Batch Size 与输入图像数量不一致！");
    }

    int channels = inputDims.d[1];
    int height = inputDims.d[2];
    int width = inputDims.d[3];

    // 创建批量输入矩阵 (BatchSize * Channels x Height * Width)
    cv::Mat batchInput(batchSize * channels * height * width, 1, CV_32FC1);

    for (int i = 0; i < images.size(); ++i) {
        // 检查输入图片是否为空
        if (images[i].empty()) {
            throw std::runtime_error("[图像预处理] 输入图像为空，索引: " + std::to_string(i));
        }

        // 预处理单张图像
        cv::Mat preprocessed = Segmenter::preprocessImage(images[i], inputDims, true, std::to_string(i));
        if (preprocessed.empty()) {
            throw std::runtime_error("[图像预处理] 单图像预处理失败，索引: " + std::to_string(i));
        }

        // 拷贝预处理后的图像数据到 batchInput 的对应位置
        memcpy(batchInput.ptr<float>(i * channels * height * width),
               preprocessed.ptr<float>(0),
               preprocessed.total() * sizeof(float));
    }

    std::cout << "[图像预处理] 批量预处理完成，输出大小: " << batchInput.size() << std::endl;

    // 如果需要保存合并图像
    if (saveMergedBatchImage) {
        cv::Mat mergedBatchImage(batchSize * height, width, CV_8UC3); // 合并为单张 HWC 格式图像
        for (int i = 0; i < batchSize; ++i) {
            // 提取每张图像的通道数据并合并
            std::vector<cv::Mat> chwChannels(channels);
            for (int c = 0; c < channels; ++c) {
                chwChannels[c] = cv::Mat(height, width, CV_32FC1,
                                         batchInput.ptr<float>(i * channels * height * width + c * height * width));
            }
            cv::Mat hwcImage;
            cv::merge(chwChannels, hwcImage); // 合并为 HWC 格式
            hwcImage.convertTo(hwcImage, CV_8UC3, 255.0f); // 转为 8U 格式

            // 复制到合并大图对应位置
            hwcImage.copyTo(mergedBatchImage(cv::Rect(0, i * height, width, height)));
        }

        // 保存合并图像
        std::string outputPath = "batch_merged_image.jpg";
        if (!cv::imwrite(outputPath, mergedBatchImage)) {
            throw std::runtime_error("[图像预处理] 保存合并批次图像失败: " + outputPath);
        }
        std::cout << "[图像预处理] 保存合并批次图像到: " << outputPath << std::endl;
    }

    return batchInput;
}

void Segmenter::runInference(int stoneNum,const cv::Mat& inputImage, cv::Mat& outputMask) {

    nvinfer1::Dims4 inputDims = { stoneNum , m_inputDimsVec[0].d[1], m_inputDimsVec[0].d[2] ,m_inputDimsVec[0].d[3] };
    printDims("[模型推理] 输入信息", inputDims);
    
    // 设置动态batch
    if (!mContext->setBindingDimensions(0, inputDims)) {
        throw std::runtime_error("设置绑定维度失败");
    }

    // 统计推理时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop); 

    size_t batchInputSize = stoneNum * mInputSize;
    size_t batchOutputSize = stoneNum * mOutputSize;

    // 1. 拷贝输入数据到 GPU
    cudaMemcpyAsync(mBuffers[mInputIndex], inputImage.data, batchInputSize, cudaMemcpyHostToDevice, mStream);

    // 2. 执行推理
    cudaEventRecord(start, mStream);
    if (!mContext->enqueueV2(mBuffers.data(), mStream, nullptr)) {
        throw std::runtime_error("[模型推理] 推理执行失败");
    }
    cudaEventRecord(stop, mStream);

    // 3. 从 GPU 拷贝输出数据回 Host
    cudaMemcpyAsync(outputMask.data, mBuffers[mOutputIndex], batchOutputSize, cudaMemcpyDeviceToHost, mStream);
    cudaStreamSynchronize(mStream);

    // 检查输出数据
    double minVal, maxVal;
    cv::minMaxLoc(outputMask, &minVal, &maxVal);
    std::cout << "[推理后检查] 输出范围: [" << minVal << ", " << maxVal << "], 大小: " << outputMask.size() << std::endl;

    // 计算耗时
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "[模型推理] 推理时间: " << elapsedTime << " ms" << std::endl;

    // 销毁事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
} 

void Segmenter::postprocessOutput(const cv::Mat& outputMask, const std::string& outputPath,bool saveOutputImage) {
    // std::cout << "[后处理] 开始后处理..." << std::endl;

    double minVal, maxVal;
    cv::minMaxLoc(outputMask, &minVal, &maxVal);
    // std::cout << "[后处理] 输出掩码范围: [" << minVal << ", " << maxVal << "]" << std::endl;

    // 1. 将数据映射到 [0, 255] 范围
    cv::Mat normalizedMask;
    outputMask.convertTo(normalizedMask, CV_8UC1, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

    // 2. 阈值处理 -> 二值掩码
    cv::Mat binaryMask;
    cv::threshold(normalizedMask, binaryMask, 127, 255, cv::THRESH_BINARY);

    if(saveOutputImage){
        // 3. 保存结果
        if (!cv::imwrite(outputPath, binaryMask)) {
            throw std::runtime_error("保存二值掩码失败: " + outputPath);
        }
        std::cout << "[后处理] 保存二值掩码到: " << outputPath << std::endl;
    }
    
}
