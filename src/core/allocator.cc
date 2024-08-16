#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        // 找到最小的可用内存
        // 使用 std::upper_bound 查找第一个大小大于 size 的块
        auto it = std::upper_bound(freeBlocks.begin(), freeBlocks.end(),
                                   size,
                                   [](size_t size, const std::pair<size_t, size_t> &block)
                                   {
                                       return size < block.second;
                                   });

        size_t addr = 0;
        if (it != freeBlocks.end())
        {
            // 从找到的块分配内存
            addr = it->first + it->second - size;
            it->second -= size;
            if (it->second == 0)
            {
                freeBlocks.erase(it);
            }
        }
        else
        {
            // 如果没有找到合适的块，从原始内存中分配
            addr = used;
            used += size;
        }

        peak = used;
        return addr;
        // return 0;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        // 查找紧邻的前一个和后一个块
        auto next = freeBlocks.lower_bound(addr);
        auto prev = (next != freeBlocks.begin()) ? std::prev(next) : freeBlocks.end();

        if (prev != freeBlocks.end() && (prev->first + prev->second == addr))
        {
            // 合并前一个块
            prev->second += size;
            if (next != freeBlocks.end() && (addr + size == next->first))
            {
                // 如果与下一个块也相邻，继续合并
                prev->second += next->second;
                freeBlocks.erase(next);
            }
        }
        else
        {
            if (next != freeBlocks.end() && (addr + size == next->first))
            {
                // 只与下一个块相邻，合并
                size += next->second;
                freeBlocks.erase(next);
            }
            // 插入新的空闲块
            freeBlocks[addr] = size;
        }
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
