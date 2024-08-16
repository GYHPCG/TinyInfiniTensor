#include "operators/concat.h"
#include "utils/operator_utils.h"
namespace infini
{
    ConcatObj::ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int _dim)
        : OperatorObj(OpType::Concat, inputs, {output})
    {
        int rank = inputs[0]->getRank();
        dim = get_real_axis(_dim, rank);
        IT_ASSERT(checkValid(graph));
    }

    optional<vector<Shape>> ConcatObj::inferShape(const TensorVec &inputs)
    {
        Shape dims = inputs[0]->getDims();
        auto rank = inputs[0]->getRank(); //

        // =================================== 作业 ===================================
        // TODO：修改 dims，返回正确的 concat 后的 shape
        // REF: https://onnx.ai/onnx/operators/onnx__Concat.html#concat-13
        // =================================== 作业 ===================================
        // 遍历每个输入的维度并更新dims
        for (size_t j = 1; j < inputs.size(); ++j)
        {
            const auto &inputDims = inputs[j]->getDims();

            // 检查rank是否匹配
            if (inputs[j]->getRank() != rank)
            {
                return std::nullopt; // 如果输入的rank不一致，返回nullopt表示失败
            }

            for (size_t i = 0; i < rank; ++i)
            {
                if (static_cast<int>(i) == dim)
                {
                    // 累加拼接轴上的维度
                    dims[i] += inputDims[i];
                }
                else if (dims[i] != inputDims[i])
                {
                    return std::nullopt; // 如果非拼接轴的维度不一致，返回nullopt表示失败
                }
            }
        }
        return {{dims}};
    }

    std::string ConcatObj::toString() const
    {
        std::ostringstream os;
        os << "Concat[" << getGuid() << "]";
        os << "(";
        for (auto input : inputs)
            os << vecToString(input->getDims()) << ",";
        os << "dim=" << dim << ",";
        os << "input=";
        for (auto input : inputs)
            os << input->getGuid() << ",";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }

} // namespace infini
