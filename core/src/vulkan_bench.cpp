#include "vulkan_bench.h"
#include "matmul_int8_spv.h"
#include <vulkan/vulkan.h>
#include <vector>
#include <random>
#include <cstring>
#include <chrono>
#include <sstream>
#include <cstdint>

namespace vulkan_bench {

namespace {

uint32_t findMemoryType(VkPhysicalDevice phys, uint32_t typeBits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(phys, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props) return i;
    }
    return UINT32_MAX;
}

struct Buf { VkBuffer buf = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; void* mapped = nullptr; };

// CPU single-thread murni -- referensi baseline, logika identik dengan
// transformer_mixed::matmul_int8_scalar (tidak di-#include langsung supaya
// modul ini tetap berdiri sendiri/independen dari core engine).
void matmul_ref(float* out, const float* x, const int8_t* w, float scale, int d_in, int d_out) {
    for (int o = 0; o < d_out; ++o) {
        float acc = 0.0f;
        const int8_t* row = w + static_cast<size_t>(o) * d_in;
        for (int i = 0; i < d_in; ++i) acc += static_cast<float>(row[i]) * x[i];
        out[o] = acc * scale;
    }
}

} // namespace

std::string run() {
    std::ostringstream report;
    const int d_in = 2048, d_out = 2048; // ukuran wq/wo TinyLlama
    const float scale = 0.02f;
    const int iterations = 30;

    std::mt19937 rng(321);
    std::uniform_real_distribution<float> distf(-1.0f, 1.0f);
    std::uniform_int_distribution<int> disti(-127, 127);
    std::vector<int8_t> w(static_cast<size_t>(d_in) * d_out);
    for (auto& v : w) v = static_cast<int8_t>(disti(rng));

    // --- Baseline CPU (single-thread) ---
    std::vector<float> x_cpu(d_in), out_cpu(d_out);
    for (auto& v : x_cpu) v = distf(rng);
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iterations; ++it) matmul_ref(out_cpu.data(), x_cpu.data(), w.data(), scale, d_in, d_out);
    auto t1 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;

    report << "Vulkan dispatch overhead benchmark -- matmul " << d_in << "x" << d_out
           << " INT8, " << iterations << " iterasi\n\n";
    report << "CPU single-thread (baseline): " << cpu_ms << " ms/matmul\n";

    // --- Setup Vulkan (sekali) ---
    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &appInfo;
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) {
        report << "\n[GAGAL] tidak bisa membuat Vulkan instance -- device/driver tidak mendukung compute.\n";
        return report.str();
    }

    uint32_t devCount = 0;
    vkEnumeratePhysicalDevices(instance, &devCount, nullptr);
    if (devCount == 0) {
        report << "\n[GAGAL] tidak ada physical device Vulkan terdeteksi.\n";
        return report.str();
    }
    std::vector<VkPhysicalDevice> devs(devCount);
    vkEnumeratePhysicalDevices(instance, &devCount, devs.data());
    VkPhysicalDevice phys = devs[0];
    VkPhysicalDeviceProperties pprops;
    vkGetPhysicalDeviceProperties(phys, &pprops);
    report << "Vulkan device: " << pprops.deviceName << "\n\n";

    uint32_t qCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qCount, nullptr);
    std::vector<VkQueueFamilyProperties> qProps(qCount);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qCount, qProps.data());
    uint32_t qFamily = UINT32_MAX;
    for (uint32_t i = 0; i < qCount; ++i) {
        if (qProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qFamily = i; break; }
    }
    if (qFamily == UINT32_MAX) {
        report << "[GAGAL] tidak ada compute queue family.\n";
        return report.str();
    }

    float qPrio = 1.0f;
    VkDeviceQueueCreateInfo dqci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    dqci.queueFamilyIndex = qFamily; dqci.queueCount = 1; dqci.pQueuePriorities = &qPrio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &dqci;
    VkDevice dev = VK_NULL_HANDLE;
    if (vkCreateDevice(phys, &dci, nullptr, &dev) != VK_SUCCESS) {
        report << "[GAGAL] tidak bisa membuat logical device.\n";
        return report.str();
    }
    VkQueue queue; vkGetDeviceQueue(dev, qFamily, 0, &queue);

    auto createHostBuf = [&](VkDeviceSize size, VkBufferUsageFlags usage) -> Buf {
        Buf b{};
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = size; bi.usage = usage; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateBuffer(dev, &bi, nullptr, &b.buf);
        VkMemoryRequirements req; vkGetBufferMemoryRequirements(dev, b.buf, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = findMemoryType(phys, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkAllocateMemory(dev, &ai, nullptr, &b.mem);
        vkBindBufferMemory(dev, b.buf, b.mem, 0);
        vkMapMemory(dev, b.mem, 0, size, 0, &b.mapped);
        return b;
    };

    // Bobot di-upload SEKALI -- mensimulasikan produksi (weights resident di
    // GPU sejak load model, bukan diupload ulang tiap panggilan matmul).
    Buf xBuf = createHostBuf(d_in * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buf wBuf = createHostBuf(w.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buf oBuf = createHostBuf(d_out * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    std::memcpy(wBuf.mapped, w.data(), w.size());

    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = vulkan_backend::kMatmulInt8SpvWords * 4; smci.pCode = vulkan_backend::kMatmulInt8Spv;
    VkShaderModule shader;
    vkCreateShaderModule(dev, &smci, nullptr, &shader);

    VkDescriptorSetLayoutBinding bindings[3]{};
    for (int i = 0; i < 3; ++i) {
        bindings[i].binding = i; bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1; bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 3; dslci.pBindings = bindings;
    VkDescriptorSetLayout dsl;
    vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl);

    VkPushConstantRange pcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, 3 * sizeof(int32_t)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcRange;
    VkPipelineLayout pipeLayout;
    vkCreatePipelineLayout(dev, &plci, nullptr, &pipeLayout);

    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; stage.module = shader; stage.pName = "main";
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = stage; cpci.layout = pipeLayout;
    VkPipeline pipeline;
    vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline);

    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &poolSize;
    VkDescriptorPool dpool;
    vkCreateDescriptorPool(dev, &dpci, nullptr, &dpool);

    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = dpool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
    VkDescriptorSet dset;
    vkAllocateDescriptorSets(dev, &dsai, &dset);

    VkDescriptorBufferInfo bufInfos[3] = {
        {xBuf.buf, 0, VK_WHOLE_SIZE}, {wBuf.buf, 0, VK_WHOLE_SIZE}, {oBuf.buf, 0, VK_WHOLE_SIZE}
    };
    VkWriteDescriptorSet writes[3]{};
    for (int i = 0; i < 3; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = dset; writes[i].dstBinding = i; writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[i].pBufferInfo = &bufInfos[i];
    }
    vkUpdateDescriptorSets(dev, 3, writes, 0, nullptr);

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = qFamily;
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool cmdPool;
    vkCreateCommandPool(dev, &cpi, nullptr, &cmdPool);
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = cmdPool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(dev, &cbai, &cmd);

    // Command buffer direkam SEKALI (bobot & deskriptor set tidak berubah
    // antar panggilan) -- ini pola paling optimal yang bisa dicapai untuk
    // shape matmul yang tetap; kalau ini pun masih lebih lambat dari CPU,
    // integrasi penuh tidak akan menang.
    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cmd, &cbbi);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, 1, &dset, 0, nullptr);
    int32_t pc[3] = { d_in, d_out, 0 };
    std::memcpy(&pc[2], &scale, sizeof(float));
    vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
    vkCmdDispatch(cmd, (d_out + 63) / 64, 1, 1);
    vkEndCommandBuffer(cmd);

    // --- Ukur round-trip nyata: tulis x baru -> submit -> wait -> baca out ---
    std::vector<float> out_vk(d_out);
    auto t2 = std::chrono::steady_clock::now();
    for (int it = 0; it < iterations; ++it) {
        for (auto& v : x_cpu) v = distf(rng); // simulasikan aktivasi berbeda tiap panggilan
        std::memcpy(xBuf.mapped, x_cpu.data(), x_cpu.size() * sizeof(float));

        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
        vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue); // paling sederhana/aman; fence per-frame adalah optimasi lanjutan

        std::memcpy(out_vk.data(), oBuf.mapped, d_out * sizeof(float));
    }
    auto t3 = std::chrono::steady_clock::now();
    double vk_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / iterations;

    // Verifikasi korektnes sekali di akhir (bukan tiap iterasi, supaya tidak
    // membebani angka timing).
    std::vector<float> out_ref(d_out);
    matmul_ref(out_ref.data(), x_cpu.data(), w.data(), scale, d_in, d_out);
    float max_diff = 0.0f;
    for (int i = 0; i < d_out; ++i) max_diff = std::max(max_diff, std::abs(out_vk[i] - out_ref[i]));

    report << "Vulkan (bobot resident, upload x + dispatch + readback tiap panggilan): "
           << vk_ms << " ms/matmul\n";
    report << "max_diff vs referensi CPU: " << max_diff
           << (max_diff < 1e-3f ? " [PASS korektnes]\n" : " [FAIL korektnes -- JANGAN dipakai]\n");

    report << "\n";
    if (max_diff >= 1e-3f) {
        report << "-> Shader salah, jangan lanjut integrasi sebelum ini diperbaiki.\n";
    } else if (vk_ms < cpu_ms) {
        report << "-> Vulkan MENANG (" << (cpu_ms / vk_ms) << "x lebih cepat). Layak diintegrasikan penuh.\n";
    } else {
        report << "-> Vulkan KALAH (" << (vk_ms / cpu_ms) << "x lebih lambat dari CPU). "
                  "Overhead submit/wait per-dispatch terlalu mahal untuk ukuran kerja ini di GPU device ini -- "
                  "TIDAK direkomendasikan lanjut ke integrasi penuh.\n";
    }

    return report.str();
}

} // namespace vulkan_bench
