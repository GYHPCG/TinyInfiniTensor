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
        // auto it = std::upper_bound(freeBlocks.begin(), freeBlocks.end(),
        //                            size,
        //                            [](size_t size, const std::pair<size_t, size_t> &block)
        //                            {
        //                                return size < block.second;
        //                            });

        // size_t addr = 0;
        // if (it != freeBlocks.end())
        // {
        //     // 从找到的块分配内存
        //     addr = it->first + it->second - size;
        //     it->second -= size;
        //     if (it->second == 0)
        //     {
        //         freeBlocks.erase(it);
        //     }
        // }
        // else
        // {
        //     // 如果没有找到合适的块，从原始内存中分配
        //     addr = used;
        //     used += size;
        // }

        // peak = used;
        // return addr;
        // return 0;
        auto target{freeBlocks.end()};
        for (auto it = freeBlocks.begin(); it != freeBlocks.end(); ++it)
        {
            if (it->second >= size)
            {
                // valid block
                if (target == freeBlocks.end())
                {
                    // the first available block
                    target = it;
                }
                else if (target->second > it->second)
                {
                    // try to find the smallest block
                    target = it;
                }
            }
        }

        size_t addr{0};
        if (target != freeBlocks.end())
        {
            // alloc from recycled blocks
            addr = target->first + target->second - size;
            target->second -= size;
            if (target->second == 0)
            {
                freeBlocks.erase(target);
            }
        }
        else
        {
            // alloc from raw memory bank
            addr = used;
            used += size;
        }

        peak = used;
        return addr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        // 查找紧邻的前一个和后一个块
        freeBlocks.insert(std::make_pair(addr, size));

        // 尝试合并相邻的空闲块
        for (auto it = freeBlocks.begin(); it != freeBlocks.end();)
        {
            // 查找当前块后面紧挨的块
            auto found = freeBlocks.find(it->first + it->second);
            if (found != freeBlocks.end())
            {
                // 如果找到紧挨的块，将它们合并
                it->second += found->second;
                freeBlocks.erase(found);
            }
            else if (it->first + it->second == used)
            {
                // 如果当前块连接到已使用的内存，将其释放回内存池
                used -= it->second;
                it = freeBlocks.erase(it);
            }
            else
            {
                // 否则，继续检查下一个块
                ++it;
            }
        }

        // 更新峰值
        peak = used;
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
